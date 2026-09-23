//! Compile the HIP kernels, if ROCm is present.
//!
//! WHY THIS DETECTS RATHER THAN REQUIRES. `cargo build --workspace` must work
//! on a machine with no AMD GPU — CI runners, a laptop, a reviewer's box. So
//! an absent ROCm is not an error: the crate compiles to a stub that reports
//! unavailability, and the tests skip loudly rather than failing for a reason
//! that has nothing to do with the code.
//!
//! The alternative, a cargo feature, would mean the crate silently does
//! nothing when somebody forgets to pass it. Detection is harder to get wrong.

use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=kernels");
    println!("cargo::rerun-if-env-changed=ROCM_PATH");
    println!("cargo::rustc-check-cfg=cfg(rocm)");

    let rocm = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
    let hipcc = Path::new(&rocm).join("bin/hipcc");
    if !hipcc.is_file() {
        println!("cargo::warning=ROCm not found at {rocm}; llamd-rocm builds as a stub");
        return;
    }

    // The GPU architecture. Detected rather than assumed, because a build
    // pinned to gfx1201 silently produces code that will not launch on any
    // other card — and the failure appears at runtime as an empty device list.
    let arch = std::env::var("LLAMD_GFX").ok().or_else(|| {
        let out = Command::new(Path::new(&rocm).join("bin/rocminfo"))
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.split_whitespace().find(|w| w.starts_with("gfx")))
            .find(|g| *g != "gfx000")
            .map(|g| g.trim_end_matches(':').to_string())
    });
    let Some(arch) = arch else {
        println!("cargo::warning=no gfx target detected; llamd-rocm builds as a stub");
        return;
    };

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let obj = Path::new(&out_dir).join("llamd_kernels.o");
    let lib = Path::new(&out_dir).join("libllamd_kernels.a");

    let status = Command::new(&hipcc)
        .args(["-O3", "-c", "-fPIC"])
        .arg(format!("--offload-arch={arch}"))
        .arg("kernels/llamd.hip")
        .arg("-o")
        .arg(&obj)
        .status();
    match status {
        Ok(s) if s.success() => {}
        _ => {
            println!("cargo::warning=hipcc failed; llamd-rocm builds as a stub");
            return;
        }
    }
    let _ = Command::new("ar").arg("crs").arg(&lib).arg(&obj).status();

    println!("cargo::rustc-link-search=native={out_dir}");
    println!("cargo::rustc-link-lib=static=llamd_kernels");
    println!("cargo::rustc-link-search=native={rocm}/lib");
    println!("cargo::rustc-link-lib=dylib=amdhip64");
    println!("cargo::rustc-link-arg=-Wl,-rpath,{rocm}/lib");
    println!("cargo::rustc-cfg=rocm");
    println!("cargo::warning=llamd-rocm: building kernels for {arch}");
}
