//! The expert index and unmapped open, validated.
//!
//! Two kinds of test. The synthetic ones need no model and always run: they
//! prove `Model::open_unmapped` accepts a complete file and builds the same
//! index as the mapped open. The oracle ones need `LLAMD_TEST_MODEL` and prove
//! the index on the real file: every sampled expert extent must be
//! byte-identical to the slice the resident path uploads today, and opening
//! the model unmapped must not cost memory in proportion to its size.
//!
//!   LLAMD_TEST_MODEL=/path/to/model.gguf cargo test --release -p llamd-testing --test expert_index_oracle

use llamd_models::{Model, Proj, synthetic::SyntheticMoe};
use llamd_testing::{oracle_model, skip};

fn rss_bytes() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = s.lines().find(|l| l.starts_with("VmRSS:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[test]
fn a_synthetic_model_opens_unmapped_with_the_same_index() {
    let s = SyntheticMoe::default();
    let f = s.write("oracle").expect("write");
    let m = Model::open(&f.0).expect("mapped open");
    let u = Model::open_unmapped(&f.0).expect("unmapped open");
    assert!(!u.gguf().is_mapped());
    assert_eq!(m.hparams.n_layer, u.hparams.n_layer);
    assert_eq!(m.hparams.n_expert, u.hparams.n_expert);

    let (im, iu) = (m.expert_index().unwrap(), u.expert_index().unwrap());
    for l in 0..s.n_layer {
        for p in Proj::ALL {
            for e in 0..s.n_expert {
                assert_eq!(im.extent(l, e, p).unwrap(), iu.extent(l, e, p).unwrap());
            }
        }
    }
    // Zero-copy access is refused, not faked.
    assert!(u.tensor(&u.layers[0].attn_norm).is_err());
}

#[test]
fn the_real_models_expert_extents_are_the_resident_paths_bytes() {
    let Some(path) = oracle_model() else {
        skip("expert_index_oracle", "LLAMD_TEST_MODEL unset");
        return;
    };
    let mapped = Model::open(&path).expect("mapped open");
    let unmapped = Model::open_unmapped(&path).expect("unmapped open");
    let idx = unmapped.expert_index().expect("index");
    let h = &unmapped.hparams;
    assert_eq!(idx.n_layer, h.n_layer);
    assert_eq!(idx.n_expert, h.n_expert);

    // Every layer and projection; first, second, middle and last expert.
    // Reading all experts would re-read the whole model for no extra proof.
    let ne = h.n_expert;
    let samples = [0, 1, ne / 2, ne - 1];
    let g = mapped.gguf();
    for l in 0..h.n_layer {
        for p in Proj::ALL {
            let t = idx.tensor(l, p).unwrap();
            let whole = g.tensor_data(g.tensor(&t.name).unwrap()).unwrap();
            let per = t.per_expert as usize;
            for &e in &samples {
                let got = unmapped
                    .gguf()
                    .read_extent(idx.extent(l, e, p).unwrap())
                    .unwrap();
                assert!(
                    got == whole[e * per..(e + 1) * per],
                    "{} expert {e}: pread bytes differ from the mapped slice",
                    t.name
                );
            }
        }
    }
    eprintln!(
        "  expert index: {} layers x {} experts, {:.2} GiB of experts of {:.2} GiB file, \
         max expert {:.2} MiB, types {:?}, alignment {:?}",
        idx.n_layer,
        idx.n_expert,
        idx.total_bytes() as f64 / (1u64 << 30) as f64,
        unmapped.gguf().file_size() as f64 / (1u64 << 30) as f64,
        idx.max_expert_bytes() as f64 / (1u64 << 20) as f64,
        idx.qtypes(),
        idx.alignment_census()
    );
}

#[test]
fn opening_unmapped_costs_the_header_not_the_model() {
    let Some(path) = oracle_model() else {
        skip("expert_index_oracle/rss", "LLAMD_TEST_MODEL unset");
        return;
    };
    let Some(before) = rss_bytes() else {
        skip("expert_index_oracle/rss", "/proc/self/status unreadable");
        return;
    };
    let m = Model::open_unmapped(&path).expect("unmapped open");
    let idx = m.expert_index().expect("index");
    let after = rss_bytes().unwrap();
    let grew = after.saturating_sub(before);
    let file = m.gguf().file_size() as u64;
    eprintln!(
        "  unmapped open: RSS +{:.1} MiB for a {:.2} GiB file ({} expert tensors indexed)",
        grew as f64 / (1u64 << 20) as f64,
        file as f64 / (1u64 << 30) as f64,
        idx.n_layer * 3
    );
    // The header (vocabulary, merges) is tens of MB. 256 MiB is a generous
    // bound that a size-proportional loader would blow through at once.
    assert!(
        grew < 256 << 20,
        "RSS grew {grew} bytes opening a {file}-byte model"
    );
}
