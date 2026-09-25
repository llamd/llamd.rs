//! The Vulkan operation set: one function per shader, plus record/submit.
//!
//! A token is recorded as one command buffer and submitted once, which is the
//! Vulkan equivalent of the ROCm path's single synchronise and for the same
//! reason — a stall per operation put a hard ceiling on that engine regardless
//! of how fast the kernels were.
//!
//! Every function here queues work and returns; nothing waits until
//! [`end_and_wait`]. Buffer handles travel inside the command buffer via push
//! descriptors, so there is no descriptor pool to manage.

use crate::{DeviceInfo, Error, PlanarWeights, Result, VkBuf, device_info, init};

/// Pipeline ids, in the order `build.rs` compiles the shaders.
///
/// The discriminant IS the id the C side indexes with, so this enum and the
/// `SHADERS` list in build.rs must stay aligned —
/// `pipeline_ids_match_the_shader_table` asserts it by name rather than trusting
/// two lists to be edited together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Pipeline {
    Q6kMatvec = 0,
    Q6kX3 = 1,
    Q6kGemm = 2,
    Q6kGrouped = 3,
    Q6kGroupedPa = 4,
    MatvecF32 = 5,
    Embed = 6,
    RmsScale = 7,
    NormQuant = 8,
    Quantize = 9,
    RmsNormMul = 10,
    NormRope = 11,
    KNormRopeStore = 12,
    Attention = 13,
    Add = 14,
    SiluMul = 15,
    Router = 16,
    MoeCombineAdd = 17,
    Argmax = 18,
    ArgmaxFinal = 19,
    MoeGroup = 20,
    Q6kExpertMajor = 21,
    Q6kExpertMajorPa = 22,
    Q6kExpertRt = 23,
    AttentionScalar = 24,
    Q6kExpertRt2 = 25,
    Q6kExpertRt8 = 26,
    CoopmatProbe = 27,
    Q6kExpertCm = 28,
    MatvecAblate = 29,
    Q6kExpertLane = 30,
    Q6kExpertRt2c = 31,
    Q6kExpertRt2c8 = 32,
    Q6kExpertRt2c32 = 33,
    Q6kMatvecC = 34,
    Q6kX3C = 35,
    Q6kExpertC16R1 = 36,
    Q6kExpertC16R1v = 37,
    Q6kExpertC16R2 = 38,
    Q6kExpertC16R2v = 39,
    Q6kExpertC16R4 = 40,
    Q6kExpertC16R4v = 41,
    Q6kExpertC16R2a = 42,
    Q6kExpertC16R2b = 43,
}

impl Pipeline {
    /// The shader file stem, for checking this enum against the build order.
    pub fn shader_name(self) -> &'static str {
        match self {
            Pipeline::Q6kMatvec => "q6k_matvec",
            Pipeline::Q6kX3 => "q6k_x3",
            Pipeline::Q6kGemm => "q6k_gemm",
            Pipeline::Q6kGrouped => "q6k_grouped",
            Pipeline::Q6kGroupedPa => "q6k_grouped_pa",
            Pipeline::MatvecF32 => "matvec_f32",
            Pipeline::Embed => "embed",
            Pipeline::RmsScale => "rms_scale",
            Pipeline::NormQuant => "norm_quant",
            Pipeline::Quantize => "quantize",
            Pipeline::RmsNormMul => "rms_norm_mul",
            Pipeline::NormRope => "norm_rope",
            Pipeline::KNormRopeStore => "k_norm_rope_store",
            Pipeline::Attention => "attention",
            Pipeline::Add => "add",
            Pipeline::SiluMul => "silu_mul",
            Pipeline::Router => "router",
            Pipeline::MoeCombineAdd => "moe_combine_add",
            Pipeline::Argmax => "argmax",
            Pipeline::ArgmaxFinal => "argmax_final",
            Pipeline::MoeGroup => "moe_group",
            Pipeline::Q6kExpertMajor => "q6k_expert_major",
            Pipeline::Q6kExpertMajorPa => "q6k_expert_major_pa",
            Pipeline::Q6kExpertRt => "q6k_expert_rt",
            Pipeline::AttentionScalar => "attention_scalar",
            Pipeline::Q6kExpertRt2 => "q6k_expert_rt2",
            Pipeline::Q6kExpertRt8 => "q6k_expert_rt8",
            Pipeline::CoopmatProbe => "coopmat_probe",
            Pipeline::Q6kExpertCm => "q6k_expert_cm",
            Pipeline::MatvecAblate => "matvec_ablate",
            Pipeline::Q6kExpertLane => "q6k_expert_lane",
            Pipeline::Q6kExpertRt2c => "q6k_expert_rt2c",
            Pipeline::Q6kExpertRt2c8 => "q6k_expert_rt2c8",
            Pipeline::Q6kExpertRt2c32 => "q6k_expert_rt2c32",
            Pipeline::Q6kMatvecC => "q6k_matvec_c",
            Pipeline::Q6kX3C => "q6k_x3_c",
            Pipeline::Q6kExpertC16R1 => "q6k_expert_c16_r1",
            Pipeline::Q6kExpertC16R1v => "q6k_expert_c16_r1v",
            Pipeline::Q6kExpertC16R2 => "q6k_expert_c16_r2",
            Pipeline::Q6kExpertC16R2v => "q6k_expert_c16_r2v",
            Pipeline::Q6kExpertC16R4 => "q6k_expert_c16_r4",
            Pipeline::Q6kExpertC16R4v => "q6k_expert_c16_r4v",
            Pipeline::Q6kExpertC16R2a => "q6k_expert_c16_r2a",
            Pipeline::Q6kExpertC16R2b => "q6k_expert_c16_r2b",
        }
    }

    pub const ALL: [Pipeline; 44] = [
        Pipeline::Q6kMatvec,
        Pipeline::Q6kX3,
        Pipeline::Q6kGemm,
        Pipeline::Q6kGrouped,
        Pipeline::Q6kGroupedPa,
        Pipeline::MatvecF32,
        Pipeline::Embed,
        Pipeline::RmsScale,
        Pipeline::NormQuant,
        Pipeline::Quantize,
        Pipeline::RmsNormMul,
        Pipeline::NormRope,
        Pipeline::KNormRopeStore,
        Pipeline::Attention,
        Pipeline::Add,
        Pipeline::SiluMul,
        Pipeline::Router,
        Pipeline::MoeCombineAdd,
        Pipeline::Argmax,
        Pipeline::ArgmaxFinal,
        Pipeline::MoeGroup,
        Pipeline::Q6kExpertMajor,
        Pipeline::Q6kExpertMajorPa,
        Pipeline::Q6kExpertRt,
        Pipeline::AttentionScalar,
        Pipeline::Q6kExpertRt2,
        Pipeline::Q6kExpertRt8,
        Pipeline::CoopmatProbe,
        Pipeline::Q6kExpertCm,
        Pipeline::MatvecAblate,
        Pipeline::Q6kExpertLane,
        Pipeline::Q6kExpertRt2c,
        Pipeline::Q6kExpertRt2c8,
        Pipeline::Q6kExpertRt2c32,
        Pipeline::Q6kMatvecC,
        Pipeline::Q6kX3C,
        Pipeline::Q6kExpertC16R1,
        Pipeline::Q6kExpertC16R1v,
        Pipeline::Q6kExpertC16R2,
        Pipeline::Q6kExpertC16R2v,
        Pipeline::Q6kExpertC16R4,
        Pipeline::Q6kExpertC16R4v,
        Pipeline::Q6kExpertC16R2a,
        Pipeline::Q6kExpertC16R2b,
    ];
}

/// The subgroup width the driver gives compute shaders — 64 on RADV, where the
/// HIP kernels use 32. Shaders adapt via `gl_SubgroupSize`; the host needs it to
/// size dispatches, since one subgroup handles one output row.
pub fn subgroup_size() -> u32 {
    init().ok();
    #[cfg(not(vulkan))]
    {
        0
    }
    #[cfg(vulkan)]
    {
        // SAFETY: returns a value queried at init.
        #[allow(unsafe_code)]
        unsafe {
            crate::ffi::llamd_vk_subgroup_size()
        }
    }
}

/// Rows covered per workgroup by the row-per-subgroup shaders.
pub fn rows_per_workgroup() -> usize {
    let sg = subgroup_size();
    if sg == 0 { 1 } else { (256 / sg) as usize }
}

fn groups_for_rows(rows: usize) -> u32 {
    let per = rows_per_workgroup().max(1);
    rows.div_ceil(per) as u32
}

/// Begin recording. Every `op` call after this queues into the same command
/// buffer until [`end_and_wait`].
pub fn begin() -> Result<()> {
    init()?;
    #[cfg(not(vulkan))]
    {
        Err(Error::unsupported(
            "vulkan",
            "not built with Vulkan support",
        ))
    }
    #[cfg(vulkan)]
    {
        // SAFETY: no arguments; reports failure by return value.
        #[allow(unsafe_code)]
        let rc = unsafe { crate::ffi::llamd_vk_begin() };
        if rc != 0 {
            return Err(Error::other("could not begin recording"));
        }
        Ok(())
    }
}

/// Submit everything recorded and wait. THE ONE WAIT per token.
pub fn end_and_wait() -> Result<()> {
    #[cfg(not(vulkan))]
    {
        Err(Error::unsupported(
            "vulkan",
            "not built with Vulkan support",
        ))
    }
    #[cfg(vulkan)]
    {
        // SAFETY: no arguments; reports failure by return value.
        #[allow(unsafe_code)]
        let rc = unsafe { crate::ffi::llamd_vk_end_and_wait() };
        if rc != 0 {
            return Err(Error::other("submission failed"));
        }
        Ok(())
    }
}

/// Push constants: sixteen 32-bit slots, interpreted per shader.
#[derive(Default, Clone, Copy)]
pub struct Push([u32; 32]);

impl Push {
    pub fn new() -> Self {
        Push([0; 32])
    }
    pub fn u(mut self, i: usize, v: usize) -> Self {
        self.0[i] = v as u32;
        self
    }
    pub fn f(mut self, i: usize, v: f32) -> Self {
        self.0[i] = v.to_bits();
        self
    }
}

fn dispatch(p: Pipeline, bufs: &[&VkBuf], push: Push, gx: u32, gy: u32) -> Result<()> {
    dispatch_acc(p, bufs, push, gx, gy, 0)
}

/// `acc_n` selects the pipeline variant whose accumulator is at least that wide.
/// Shaders with no accumulator pass 0 and ignore it.
fn dispatch_acc(
    p: Pipeline,
    bufs: &[&VkBuf],
    push: Push,
    gx: u32,
    gy: u32,
    acc_n: usize,
) -> Result<()> {
    // 0 asks the C side for the widest variant, which is the pre-specialization
    // behaviour and what the A/B compares against.
    let acc_n = if acc_specialized() { acc_n } else { 0 };
    if bufs.len() > 12 {
        return Err(Error::malformed(
            "dispatch",
            format!("{} buffers exceeds the 12 bindings", bufs.len()),
        ));
    }
    #[cfg(not(vulkan))]
    {
        let _ = (p, push, gx, gy);
        Err(Error::unsupported(
            "vulkan",
            "not built with Vulkan support",
        ))
    }
    #[cfg(vulkan)]
    {
        let handles: Vec<*mut std::ffi::c_void> = bufs.iter().map(|b| b.handle()).collect();
        // SAFETY: every handle comes from a live VkBuf, the count is checked
        // against the 8 declared bindings, and `push` is exactly 16 u32.
        #[allow(unsafe_code)]
        let rc = unsafe {
            crate::ffi::llamd_vk_dispatch(
                p as i32,
                handles.as_ptr(),
                handles.len() as i32,
                push.0.as_ptr(),
                gx,
                gy,
                acc_n as i32,
            )
        };
        if rc != 0 {
            return Err(Error::other(format!(
                "dispatch of {} failed",
                p.shader_name()
            )));
        }
        Ok(())
    }
}

/// Whether a dispatch picks the accumulator variant matching its token count, or
/// always takes the widest.
///
/// Exists so the specialization can be A/B'd in one process. The narrow variants
/// should be strictly better — fewer registers, more waves — and if a measurement
/// says otherwise that is worth knowing rather than arguing about.
static ACC_SPECIALIZED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_acc_specialized(on: bool) {
    ACC_SPECIALIZED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn acc_specialized() -> bool {
    ACC_SPECIALIZED.load(std::sync::atomic::Ordering::Relaxed)
}

/// How many tokens one accumulator chunk holds.
///
/// The matvec-shaped shaders loop the token dimension in chunks of their `ACC_N`,
/// so this is what the host asks for and therefore which specialized variant runs.
/// It is the whole knob that decides the prefill trade: a chunk is how many tokens
/// share one weight read, and also how many floats a lane must hold. 32 measured
/// best — 16 moves at 87% of peak bandwidth but doubles expert traffic, 64 halves
/// the traffic again but drops to 33% of peak. Swept in `vulkan_prefill.rs`.
static TOKEN_CHUNK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(32);

/// Set the chunk width. 0 means "no chunking": ask for the full token count, which
/// is the pre-chunking behaviour and what the A/B arm uses.
pub fn set_token_chunk(n: usize) {
    TOKEN_CHUNK.store(n, std::sync::atomic::Ordering::Relaxed);
}

pub fn token_chunk() -> usize {
    TOKEN_CHUNK.load(std::sync::atomic::Ordering::Relaxed)
}

/// Per-dispatch GPU timing.
///
/// WHY THIS AND NOT AN EXTERNAL PROFILER. Every dispatch in this engine is
/// followed by a full memory barrier, so nothing overlaps: the sum of the
/// per-dispatch intervals IS the pass. A timestamp per dispatch is therefore an
/// exact attribution, not a sampling estimate — and it answers the one question
/// two counting models failed to answer, which is where the time actually goes.
///
/// It must be enabled outside a recording, and reports the LAST pass only.
pub fn profile(on: bool) -> Result<()> {
    init()?;
    #[cfg(not(vulkan))]
    {
        let _ = on;
        Err(Error::unsupported("vulkan profile", "built without Vulkan"))
    }
    #[cfg(vulkan)]
    {
        #[allow(unsafe_code)]
        let rc = unsafe { crate::ffi::llamd_vk_profile(i32::from(on)) };
        if rc != 0 {
            return Err(Error::other("vulkan profile: device cannot timestamp"));
        }
        Ok(())
    }
}

/// `(pipeline, milliseconds)` for every dispatch of the last pass, in order.
pub fn profile_read() -> Vec<(Pipeline, f64)> {
    #[cfg(not(vulkan))]
    {
        Vec::new()
    }
    #[cfg(vulkan)]
    {
        const CAP: usize = 4096;
        let mut pipes = vec![0i32; CAP];
        let mut ms = vec![0f64; CAP];
        #[allow(unsafe_code)]
        let n = unsafe {
            crate::ffi::llamd_vk_profile_read(pipes.as_mut_ptr(), ms.as_mut_ptr(), CAP as i32)
        };
        let n = n.max(0) as usize;
        (0..n)
            .filter_map(|i| {
                Pipeline::ALL
                    .iter()
                    .copied()
                    .find(|p| *p as i32 == pipes[i])
                    .map(|p| (p, ms[i]))
            })
            .collect()
    }
}

static ATTN_SCALAR: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Run the PREVIOUS attention kernel instead of the rewritten one.
///
/// Exists so the rewrite can be A/B'd with both arms alternating in one process.
/// A before/after across two builds cannot decide a few percent on this card: four
/// runs of one build spanned 98.82 to 117.18 tok/s at batch 1.
pub fn set_attention_scalar(on: bool) {
    ATTN_SCALAR.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn attention_scalar() -> bool {
    ATTN_SCALAR.load(std::sync::atomic::Ordering::Relaxed)
}

/// Weight rows one workgroup of the cooperative-matrix expert kernel covers: four
/// subgroups, one 16-row matrix tile each. Must match SGS * 16 in
/// `shaders/q6k_expert_cm.comp`.
pub const CM_ROWS_PER_WG: usize = 64;

static EXPERT_CM: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Use the cooperative-matrix expert kernel.
///
/// OFF by default, and §25 is why: it reaches parity with the dot kernel (0.987x)
/// and no more, because the expert path is dequantisation-bound rather than
/// arithmetic-bound. Matrix units accelerate the ~25% of that kernel which is
/// arithmetic. Kept because it is correct, measured, and the right kernel the moment
/// the unpack is amortised over more tokens per expert than a 512-token tile gives.
///
/// Ignored where the device has no `VK_KHR_cooperative_matrix` — the pipeline does
/// not exist there.
pub fn set_expert_coopmat(on: bool) {
    EXPERT_CM.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Is it both requested and usable for this shape?
///
/// The row count must be a multiple of [`CM_ROWS_PER_WG`]: a subgroup with no rows
/// would have to return early, and it cannot — the staging barriers need every
/// invocation in the workgroup. Both expert shapes here qualify (768 and 2048), and
/// one that does not falls back to the dot-product kernel rather than being refused.
pub fn expert_coopmat_for(rows: usize) -> bool {
    EXPERT_CM.load(std::sync::atomic::Ordering::Relaxed)
        && crate::has_coopmat()
        && rows % CM_ROWS_PER_WG == 0
}

static MATVEC_CLUSTER: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Use the 16-lane clustered variants of the attention projections.
///
/// The same fix §27 landed on the expert kernel, applied to the same pathology here.
/// Q, K and V are K = 2048, so `n_u4 = 64` against a 64-lane subgroup — one loop
/// iteration per lane to amortise a 32-accumulator epilogue over, which is exactly
/// the shape the expert gate had.
pub fn set_matvec_cluster(on: bool) {
    MATVEC_CLUSTER.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn matvec_cluster() -> bool {
    MATVEC_CLUSTER.load(std::sync::atomic::Ordering::Relaxed)
}

/// Rows one workgroup covers when 16 lanes share a row.
fn groups_for_rows_clustered(rows: usize) -> u32 {
    let per = rows_per_workgroup().max(1) * (subgroup_size() as usize / 16).max(1);
    rows.div_ceil(per.max(1)) as u32
}

/// Rows one LANE of the lane-per-row expert kernel owns. Must match `LROWS` in
/// `shaders/q6k_expert_lane.comp`.
pub const LANE_ROWS: usize = 2;

static EXPERT_LANE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Use the lane-per-row expert kernel: each lane walks the whole K for its rows, so
/// there is no cross-lane reduction. §26 measured the reduction at 57% unaccounted
/// time on the expert shape, where K = 2048 leaves each lane exactly one loop
/// iteration to amortise a 32-accumulator `subgroupAdd` epilogue over.
pub fn set_expert_lane(on: bool) {
    EXPERT_LANE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn expert_lane_enabled() -> bool {
    EXPERT_LANE.load(std::sync::atomic::Ordering::Relaxed)
}

/// The routed expert projections, one lane per output row. No shared memory and no
/// barriers, so a row count that does not divide the workgroup is fine — a lane with
/// no rows returns.
#[allow(clippy::too_many_arguments)]
pub fn expert_lane(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    g: (&VkBuf, &VkBuf, &VkBuf, &VkBuf),
    rows: usize,
    n_used: usize,
    nseq: usize,
    n_expert: usize,
    by_slot: bool,
) -> Result<()> {
    need(out, rows * n_used * nseq, "expert_lane out")?;
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, n_used)
        .u(7, nseq)
        .u(8, usize::from(by_slot));
    dispatch_acc(
        Pipeline::Q6kExpertLane,
        &[w.buf(), qb, sb, ad, out, g.0, g.1, g.2, g.3],
        p,
        rows.div_ceil(256 * LANE_ROWS) as u32,
        n_expert as u32,
        // LANE_ROWS * ACC_N accumulators is the register budget; §23 measured 32.
        nseq.min(32 / LANE_ROWS),
    )
}

/// The routed expert projections on the matrix units. `by_slot` picks the
/// activation index, as in [`expert_row_tiled`]. Note it does NOT take the
/// group-sum buffer: the signed shape needs no correction terms.
#[allow(clippy::too_many_arguments)]
pub fn expert_coopmat(
    w: &PlanarWeights,
    qb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    g: (&VkBuf, &VkBuf, &VkBuf, &VkBuf),
    rows: usize,
    n_used: usize,
    nseq: usize,
    n_expert: usize,
    by_slot: bool,
) -> Result<()> {
    need(out, rows * n_used * nseq, "expert_coopmat out")?;
    if rows % CM_ROWS_PER_WG != 0 {
        return Err(Error::malformed(
            "expert_coopmat",
            format!("{rows} rows is not a multiple of {CM_ROWS_PER_WG}"),
        ));
    }
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, n_used)
        .u(7, nseq)
        .u(8, usize::from(by_slot));
    dispatch(
        Pipeline::Q6kExpertCm,
        &[w.buf(), qb, ad, out, g.0, g.1, g.2, g.3],
        p,
        (rows / CM_ROWS_PER_WG) as u32,
        n_expert as u32,
    )
}

/// The ablation matvec: the production arithmetic with one class of work removed.
/// `mode` 0 full, 1 no dot, 2 no unpack, 3 no weight load, 4 no activation load.
/// See `shaders/matvec_ablate.comp`.
#[allow(clippy::too_many_arguments)]
pub fn matvec_ablate(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    rows: usize,
    nseq: usize,
    mode: usize,
) -> Result<()> {
    let p = planes(w).u(4, w.blocks()).u(5, rows).u(6, nseq).u(7, mode);
    dispatch_acc(
        Pipeline::MatvecAblate,
        &[w.buf(), qb, sb, ad, out],
        p,
        groups_for_rows(rows),
        1,
        32,
    )
}

/// The cooperative-matrix microbenchmark. `mode` 0 is MMA with no fold, 1 is MMA
/// plus the per-group fold Q6_K forces, 2 is the same MAC count through
/// `dotPacked4x8AccSatEXT`. See `shaders/coopmat_probe.comp`.
#[allow(clippy::too_many_arguments)]
pub fn coopmat_probe(
    a: &VkBuf,
    b: &VkBuf,
    a4: &VkBuf,
    b4: &VkBuf,
    out: &VkBuf,
    groups: u32,
    iters: usize,
    mode: usize,
) -> Result<()> {
    let p = Push::new().u(0, iters).u(1, mode);
    dispatch(Pipeline::CoopmatProbe, &[a, b, a4, b4, out], p, groups, 1)
}

/// Rows one lane of the row-tiled expert kernel accumulates, and the tokens per
/// chunk. Sweepable, because they trade against each other through the register
/// file and the best point is a measurement.
///
/// `ROWS * ACC` is the accumulator count and the dominant register cost. At 4 x 16
/// the kernel compiles to **252 VGPRs — two waves per SIMD**, which is why row
/// tiling was worth only 3% (§20): what it saved in instructions it paid back in
/// lost latency hiding, on a kernel that is at 19% of memory peak and 15% of the
/// dot-product ceiling and therefore bound by neither.
///
/// `rows` of 1 means the untiled kernel; 2, 4 and 8 select the wrappers around
/// `expert_rt_body.glsl`.
/// 2 x 16 measured best (§23): 32 accumulators is the sweet spot on this part, and
/// 64 loses more to occupancy than the extra reuse wins.
/// Lanes per output row group in the expert kernel. 0 uses the whole subgroup, which
/// is the original decomposition; 16 uses the clustered variant. The reduction costs
/// log2(lanes) steps over that many lanes, so a smaller cluster is cheaper — but the
/// lanes of a cluster are what keep the weight loads coalesced, and taking it to 1
/// (`q6k_expert_lane`) was 2.15x slower. 16 is the middle.
/// 16 measured best, and the curve is a clean inverted U: 64 pays too much reduction
/// (1364.7 tok/s at pp512), 32 gives 1529.8, 16 gives 1547.1, 8 gives 1530.4, and 1 —
/// no reduction at all — collapses to 644.5 because the weight loads stop coalescing.
static RT_CLUSTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(16);

pub fn set_expert_cluster(lanes: usize) {
    RT_CLUSTER.store(lanes, std::sync::atomic::Ordering::Relaxed);
}

pub fn expert_cluster() -> usize {
    RT_CLUSTER.load(std::sync::atomic::Ordering::Relaxed)
}

/// Read the expert kernel's two hot buffers as `uvec4`.
///
/// Separate from the projections' switch because the same change has opposite signs
/// on the two kernels (§28), and it trades through the same register file as ROWS and
/// ACC — so it is swept with them rather than decided once.
static RT_VEC4: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_expert_vec4(on: bool) {
    RT_VEC4.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn expert_vec4() -> bool {
    RT_VEC4.load(std::sync::atomic::Ordering::Relaxed)
}

static RT_ADDR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Hoist the expert kernel's address arithmetic out of the K loop.
///
/// §30 put scalar work at 44-49% of the instruction stream against 9-11% for the dot
/// products. This lifts the per-token `pslot`/`ptoken` GLOBAL LOAD and the three
/// `a * blocks * N` multiplies out of the inner loop — one fetch and one multiply per
/// token per chunk instead of per token per K-group — and the per-row weight plane
/// bases, which are loop-invariant in K.
///
/// It uses shared memory, so it needs a barrier, so every lane group in the workgroup
/// must have rows: usable only when the row count divides what a workgroup covers.
/// 0 off, 1 hoisted through shared memory (needs a barrier), 2 hoisted into registers
/// (costs ACC_N uints of pressure, no barrier).
pub fn set_expert_addr(mode: usize) {
    RT_ADDR.store(mode, std::sync::atomic::Ordering::Relaxed);
}

pub fn expert_addr() -> usize {
    RT_ADDR.load(std::sync::atomic::Ordering::Relaxed)
}

static RT_ROWS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(2);
static RT_ACC: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(16);

pub fn set_expert_tile(rows: usize, acc: usize) {
    RT_ROWS.store(rows, std::sync::atomic::Ordering::Relaxed);
    RT_ACC.store(acc, std::sync::atomic::Ordering::Relaxed);
}

pub fn expert_tile() -> (usize, usize) {
    (
        RT_ROWS.load(std::sync::atomic::Ordering::Relaxed),
        RT_ACC.load(std::sync::atomic::Ordering::Relaxed),
    )
}

/// On by default: measured faster at every tile width and bit-exact against the
/// kernel it replaces (`row_tiled_experts_equal_the_untiled_kernel`). It is a
/// modest 1.03x — see benchmarks.md §20 for why it cannot be more than ~1.2x.
static EXPERT_RT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

/// Use the row-tiled expert kernel instead of the one-row-per-subgroup one.
pub fn set_expert_rt(on: bool) {
    EXPERT_RT.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn expert_rt_enabled() -> bool {
    EXPERT_RT.load(std::sync::atomic::Ordering::Relaxed)
}

/// Workgroups to cover `rows` when each lane group owns `RT_ROWS` of them.
fn groups_for_rows_tiled(rows: usize, rt_rows: usize) -> u32 {
    let per = rows_per_workgroup().max(1) * rt_rows.max(1);
    rows.div_ceil(per) as u32
}

/// The accumulator width to request for a dispatch carrying `nseq` tokens.
fn acc_for(nseq: usize) -> usize {
    match token_chunk() {
        0 => nseq,
        c => nseq.min(c),
    }
}

/// Guard an output buffer against what a shader will write. An undersized
/// storage buffer is not a fault in Vulkan — the write is simply dropped or
/// clamped — so it would surface as wrong numbers, not an error.
fn need(b: &VkBuf, floats: usize, what: &str) -> Result<()> {
    if b.bytes() < floats * 4 {
        return Err(Error::malformed(
            "vulkan buffer",
            format!("{what}: holds {} floats, needs {floats}", b.bytes() / 4),
        ));
    }
    Ok(())
}

/// Plane offsets in uints, the first four push slots every Q6_K shader expects.
fn planes(w: &PlanarWeights) -> Push {
    let o = w.offsets();
    Push::new()
        .u(0, (o[0] / 4) as usize)
        .u(1, (o[1] / 4) as usize)
        .u(2, (o[2] / 4) as usize)
        .u(3, (o[3] / 4) as usize)
}

// ---------------------------------------------------------------------------
// The operations. Argument order follows the shader's binding order so a
// mismatch is visible at the call site rather than as wrong output.
// ---------------------------------------------------------------------------

pub fn embed(
    w: &PlanarWeights,
    meta: &VkBuf,
    out: &VkBuf,
    blocks: usize,
    nseq: usize,
) -> Result<()> {
    let p = planes(w).u(4, blocks).u(6, nseq);
    dispatch(
        Pipeline::Embed,
        &[w.buf(), meta, out],
        p,
        blocks as u32,
        nseq as u32,
    )
}

pub fn rms_scale(x: &VkBuf, inv: &VkBuf, n: usize, nseq: usize, eps: f32) -> Result<()> {
    dispatch(
        Pipeline::RmsScale,
        &[x, inv],
        Push::new().u(0, n).f(1, eps),
        nseq as u32,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn norm_quant(
    x: &VkBuf,
    w: &VkBuf,
    inv: &VkBuf,
    out: &VkBuf,
    qb: &VkBuf,
    sb: &VkBuf,
    d: &VkBuf,
    n: usize,
    nseq: usize,
) -> Result<()> {
    let nblocks = n / 256;
    dispatch(
        Pipeline::NormQuant,
        &[x, w, inv, out, qb, sb, d],
        Push::new().u(0, nblocks),
        (nseq * nblocks) as u32,
        1,
    )
}

pub fn quantize(x: &VkBuf, qb: &VkBuf, sb: &VkBuf, d: &VkBuf, nblocks_total: usize) -> Result<()> {
    dispatch(
        Pipeline::Quantize,
        &[x, qb, sb, d],
        Push::new().u(0, nblocks_total),
        nblocks_total as u32,
        1,
    )
}

pub fn rms_norm_mul(
    x: &VkBuf,
    w: &VkBuf,
    out: &VkBuf,
    n: usize,
    nseq: usize,
    eps: f32,
) -> Result<()> {
    dispatch(
        Pipeline::RmsNormMul,
        &[x, w, out],
        Push::new().u(0, n).f(1, eps),
        nseq as u32,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn matvec(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    rows: usize,
    nseq: usize,
) -> Result<()> {
    let p = planes(w).u(4, w.blocks()).u(5, rows).u(6, nseq);
    let (pipe, gx) = if matvec_cluster() {
        (Pipeline::Q6kMatvecC, groups_for_rows_clustered(rows))
    } else {
        (Pipeline::Q6kMatvec, groups_for_rows(rows))
    };
    dispatch_acc(pipe, &[w.buf(), qb, sb, ad, out], p, gx, 1, acc_for(nseq))
}

/// Rows a single lane accumulates in the GEMM kernel. Must match
/// `ROWS_PER_LANE` in `shaders/q6k_gemm.comp`.
pub const GEMM_ROWS_PER_LANE: usize = 2;

/// Rows covered by one GEMM workgroup: one subgroup owns `GEMM_ROWS_PER_LANE`
/// rows, and a 256-thread workgroup holds `256 / subgroup_size` subgroups.
pub fn gemm_rows_per_workgroup() -> usize {
    let sg = subgroup_size();
    let subgroups = if sg == 0 { 1 } else { (256 / sg) as usize };
    subgroups * GEMM_ROWS_PER_LANE
}

/// The tiled GEMM: 32 output rows per workgroup with the activation staged in
/// shared memory.
///
/// Same arithmetic as [`matvec`], different work decomposition. Worth using when
/// there are many tokens to share each staged activation across — which is
/// prefill. Whether it beats `matvec` at a given token count is measured, not
/// assumed.
#[allow(clippy::too_many_arguments)]
pub fn matvec_gemm(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    rows: usize,
    nseq: usize,
) -> Result<()> {
    if nseq == 0 || nseq > crate::MAX_SEQ {
        return Err(Error::malformed("matvec_gemm", format!("nseq {nseq}")));
    }
    let p = planes(w).u(4, w.blocks()).u(5, rows).u(6, nseq);
    dispatch(
        Pipeline::Q6kGemm,
        &[w.buf(), qb, sb, ad, out],
        p,
        rows.div_ceil(gemm_rows_per_workgroup()) as u32,
        1,
    )
}

/// Q, K and V in one dispatch.
///
/// All three multiply the same activation. Issued separately they are 1024, 128
/// and 128 subgroups, and the two small ones leave most of a 32-WGP card idle;
/// fused, the row space is 5120 and they fill it alongside Q. Each matrix carries
/// its OWN plane offsets — sharing one set would silently use Q's planes for K
/// and V.
#[allow(clippy::too_many_arguments)]
pub fn matvec_x3(
    w: (&PlanarWeights, &PlanarWeights, &PlanarWeights),
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: (&VkBuf, &VkBuf, &VkBuf),
    rows: (usize, usize, usize),
    nseq: usize,
) -> Result<()> {
    if w.1.blocks() != w.0.blocks() || w.2.blocks() != w.0.blocks() {
        return Err(Error::malformed(
            "matvec_x3",
            "all three matrices must share the activation's block count",
        ));
    }
    let mut p = Push::new();
    for (i, m) in [w.0, w.1, w.2].iter().enumerate() {
        let o = m.offsets();
        for j in 0..4 {
            p = p.u(i * 4 + j, (o[j] / 4) as usize);
        }
    }
    let p = p
        .u(12, w.0.blocks())
        .u(13, rows.0)
        .u(14, rows.1)
        .u(15, rows.2)
        .u(16, nseq);
    let (pipe, gx) = if matvec_cluster() {
        (
            Pipeline::Q6kX3C,
            groups_for_rows_clustered(rows.0 + rows.1 + rows.2),
        )
    } else {
        (Pipeline::Q6kX3, groups_for_rows(rows.0 + rows.1 + rows.2))
    };
    dispatch_acc(
        pipe,
        &[
            w.0.buf(),
            w.1.buf(),
            w.2.buf(),
            qb,
            sb,
            ad,
            out.0,
            out.1,
            out.2,
        ],
        p,
        gx,
        1,
        acc_for(nseq),
    )
}

/// Expert gate or up: shared activation, per-(sequence, expert) weights.
#[allow(clippy::too_many_arguments)]
pub fn grouped(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    ids: &VkBuf,
    rows: usize,
    n_used: usize,
    nseq: usize,
) -> Result<()> {
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, nseq)
        .u(7, n_used);
    dispatch(
        Pipeline::Q6kGrouped,
        &[w.buf(), qb, sb, ad, out, ids],
        p,
        groups_for_rows(rows * n_used * nseq),
        1,
    )
}

/// The down projection: activation indexed by (sequence, expert).
#[allow(clippy::too_many_arguments)]
pub fn grouped_per_act(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    ids: &VkBuf,
    rows: usize,
    n_used: usize,
    nseq: usize,
) -> Result<()> {
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, nseq)
        .u(7, n_used);
    dispatch(
        Pipeline::Q6kGroupedPa,
        &[w.buf(), qb, sb, ad, out, ids],
        p,
        groups_for_rows(rows * n_used * nseq),
        1,
    )
}

pub fn matvec_f32(
    w: &VkBuf,
    x: &VkBuf,
    out: &VkBuf,
    k: usize,
    rows: usize,
    nseq: usize,
) -> Result<()> {
    dispatch(
        Pipeline::MatvecF32,
        &[w, x, out],
        Push::new().u(0, k).u(1, rows).u(2, nseq),
        (rows * nseq) as u32,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn norm_rope(
    x: &VkBuf,
    w: Option<&VkBuf>,
    meta: &VkBuf,
    n_head: usize,
    hd: usize,
    nseq: usize,
    freq_base: f32,
    eps: f32,
) -> Result<()> {
    // With no QK-norm there is still a weight binding to fill; the shader is
    // told not to read it. Reusing `x` keeps the binding valid without a dummy
    // allocation, and `has_norm = 0` means it is never touched.
    let wb = w.unwrap_or(x);
    let p = Push::new()
        .u(0, n_head)
        .u(1, hd)
        .u(2, nseq)
        .u(3, usize::from(w.is_some()))
        .f(4, freq_base)
        .f(5, eps);
    dispatch(
        Pipeline::NormRope,
        &[x, wb, meta],
        p,
        (nseq * n_head) as u32,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn k_norm_rope_store(
    k: &VkBuf,
    w: Option<&VkBuf>,
    v: &VkBuf,
    kc: &VkBuf,
    vc: &VkBuf,
    meta: &VkBuf,
    n_head_kv: usize,
    hd: usize,
    max_kv: usize,
    nseq: usize,
    freq_base: f32,
    eps: f32,
    kv_stride: usize,
) -> Result<()> {
    let wb = w.unwrap_or(k);
    let p = Push::new()
        .u(0, n_head_kv)
        .u(1, hd)
        .u(2, n_head_kv * hd)
        .u(3, max_kv)
        .u(4, nseq)
        .u(5, usize::from(w.is_some()))
        .f(6, freq_base)
        .f(7, eps)
        .u(8, kv_stride);
    dispatch(
        Pipeline::KNormRopeStore,
        &[k, wb, v, kc, vc, meta],
        p,
        (nseq * n_head_kv) as u32,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn attention(
    q: &VkBuf,
    kc: &VkBuf,
    vc: &VkBuf,
    out: &VkBuf,
    scores: &VkBuf,
    meta: &VkBuf,
    n_head: usize,
    n_head_kv: usize,
    hd: usize,
    max_kv: usize,
    nseq: usize,
    kv_stride: usize,
) -> Result<()> {
    if n_head % n_head_kv != 0 {
        return Err(Error::malformed(
            "attention",
            format!("{n_head} query heads do not divide into {n_head_kv} kv heads"),
        ));
    }
    // The shader reads Q, K, V and the output through a vec4 view of the same
    // buffers — four times fewer load instructions and a 16-byte request per lane,
    // which is most of why it is no longer 61% of prefill. That requires every base
    // it computes to divide by 4, and they all reduce to head_dim. An indivisible
    // head_dim is REFUSED rather than read wrong: the alternative is a shader that
    // silently reads across element boundaries and returns plausible nonsense.
    //
    // 512 is the `shared vec4 sq[128]` staging bound.
    if hd % 4 != 0 || hd == 0 || hd > 512 {
        return Err(Error::unsupported(
            "attention",
            format!("head_dim {hd} must be a non-zero multiple of 4 and at most 512"),
        ));
    }
    let p = Push::new()
        .u(0, n_head)
        .u(1, n_head_kv)
        .u(2, hd)
        .u(3, max_kv)
        .u(4, nseq)
        .u(5, n_head / n_head_kv)
        .u(6, kv_stride);
    dispatch(
        if attention_scalar() {
            Pipeline::AttentionScalar
        } else {
            Pipeline::Attention
        },
        &[q, kc, vc, out, scores, meta],
        p,
        (nseq * n_head) as u32,
        1,
    )
}

pub fn add(a: &VkBuf, b: &VkBuf, out: &VkBuf, n: usize) -> Result<()> {
    dispatch(
        Pipeline::Add,
        &[a, b, out],
        Push::new().u(0, n),
        n.div_ceil(256) as u32,
        1,
    )
}

pub fn silu_mul(g: &VkBuf, u: &VkBuf, out: &VkBuf, n: usize) -> Result<()> {
    dispatch(
        Pipeline::SiluMul,
        &[g, u, out],
        Push::new().u(0, n),
        n.div_ceil(256) as u32,
        1,
    )
}

pub fn router(
    logits: &VkBuf,
    ids: &VkBuf,
    wt: &VkBuf,
    probs: &VkBuf,
    n_expert: usize,
    n_used: usize,
    nseq: usize,
) -> Result<()> {
    if n_used == 0 || n_used > n_expert || n_expert > 1024 {
        return Err(Error::malformed(
            "moe routing",
            format!("{n_used} of {n_expert}"),
        ));
    }
    dispatch(
        Pipeline::Router,
        &[logits, ids, wt, probs],
        Push::new().u(0, n_expert).u(1, n_used).u(2, nseq),
        nseq as u32,
        1,
    )
}

pub fn moe_combine_add(
    down: &VkBuf,
    wt: &VkBuf,
    res: &VkBuf,
    out: &VkBuf,
    n_used: usize,
    n_embd: usize,
    nseq: usize,
) -> Result<()> {
    dispatch(
        Pipeline::MoeCombineAdd,
        &[down, wt, res, out],
        Push::new().u(0, n_used).u(1, n_embd).u(2, nseq),
        (n_embd * nseq).div_ceil(256) as u32,
        1,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn argmax(
    x: &VkBuf,
    bv: &VkBuf,
    bi: &VkBuf,
    out: &VkBuf,
    n: usize,
    nchunk: usize,
    nseq: usize,
) -> Result<()> {
    dispatch(
        Pipeline::Argmax,
        &[x, bv, bi],
        Push::new().u(0, n).u(1, nchunk).u(2, nseq),
        nchunk as u32,
        nseq as u32,
    )?;
    dispatch(
        Pipeline::ArgmaxFinal,
        &[bv, bi, out],
        Push::new().u(0, nchunk).u(1, nseq),
        nseq as u32,
        1,
    )
}

/// What the device reports, for benchmarks that must record which backend and
/// driver produced a number.
pub fn info() -> Result<DeviceInfo> {
    device_info()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipeline_ids_match_the_shader_table() {
        if !crate::built_with_vulkan() {
            return;
        }
        // The enum discriminant is the id the C side indexes with, and build.rs
        // owns the order. Checking by NAME means neither list can drift silently.
        #[allow(unsafe_code)]
        let n = unsafe { crate::ffi::llamd_vk_pipeline_count() };
        assert_eq!(
            n as usize,
            Pipeline::ALL.len(),
            "the shader table has {n} entries but the enum has {}",
            Pipeline::ALL.len()
        );
        for (i, p) in Pipeline::ALL.iter().enumerate() {
            assert_eq!(i as i32, *p as i32, "{p:?} is not at index {i}");
            #[allow(unsafe_code)]
            let raw = unsafe { crate::ffi::llamd_vk_pipeline_name(i as i32) };
            // SAFETY: the C side returns a static string or "".
            #[allow(unsafe_code)]
            let name = unsafe { std::ffi::CStr::from_ptr(raw) }
                .to_str()
                .expect("utf8");
            assert_eq!(
                name,
                p.shader_name(),
                "index {i}: shader table says {name:?}, enum says {:?}",
                p.shader_name()
            );
        }
    }

    #[test]
    fn dispatch_refuses_more_buffers_than_bindings() {
        if !crate::available() {
            return;
        }
        let b = VkBuf::new(256).expect("buf");
        let many: Vec<&VkBuf> = (0..13).map(|_| &b).collect();
        assert!(dispatch(Pipeline::Add, &many, Push::new(), 1, 1).is_err());
    }

    #[test]
    fn rows_per_workgroup_is_consistent_with_the_subgroup_size() {
        if !crate::available() {
            return;
        }
        let sg = subgroup_size();
        assert!(sg == 32 || sg == 64, "unexpected subgroup size {sg}");
        assert_eq!(rows_per_workgroup(), (256 / sg) as usize);
        // And the dispatch sizing must cover every row.
        for rows in [1usize, 7, 255, 4096, 151936] {
            let g = groups_for_rows(rows);
            assert!(
                (g as usize) * rows_per_workgroup() >= rows,
                "{rows} rows need more than {g} workgroups"
            );
        }
    }
}

#[cfg(test)]
mod recording_tests {
    use super::*;

    #[test]
    fn uploads_are_refused_while_recording() {
        // There is one command buffer; an upload mid-recording would reset it
        // and discard a half-built token, which would show up as wrong output
        // rather than as an error. Verified, not just commented.
        if !crate::available() {
            return;
        }
        let b = VkBuf::new(1024).expect("buf");
        begin().expect("begin");
        let r = b.upload(&[0u8; 16]);
        let mut out = [0u8; 16];
        let r2 = b.download(&mut out);
        // Close the recording either way, so a failure here does not wedge the
        // device for later tests.
        let _ = end_and_wait();
        assert!(r.is_err(), "upload was allowed while recording");
        assert!(r2.is_err(), "download was allowed while recording");
        // And the same buffer works again once recording has ended.
        b.upload(&[7u8; 16]).expect("upload after end");
    }

    #[test]
    fn a_dispatch_outside_recording_is_refused() {
        if !crate::available() {
            return;
        }
        let b = VkBuf::new(1024).expect("buf");
        assert!(
            add(&b, &b, &b, 16).is_err(),
            "dispatch was accepted with no recording open"
        );
    }

    #[test]
    fn begin_twice_is_refused() {
        if !crate::available() {
            return;
        }
        begin().expect("first begin");
        let second = begin();
        let _ = end_and_wait();
        assert!(second.is_err(), "nested begin was allowed");
    }
}

#[cfg(test)]
mod cap_tests {
    /// The Rust cap and the shader's `MAX_SEQ` must agree: the shaders size their
    /// accumulator arrays by it, so a Rust side that allowed more would write past
    /// them. Parsed from the shader source rather than restated.
    #[test]
    fn max_seq_matches_the_shaders() {
        let src = include_str!("../shaders/common.glsl");
        let line = src
            .lines()
            .find(|l| l.trim_start().starts_with("#define MAX_SEQ"))
            .expect("MAX_SEQ is defined in common.glsl");
        let n: usize = line
            .split_whitespace()
            .nth(2)
            .expect("a value")
            .parse()
            .expect("a number");
        assert_eq!(
            n,
            crate::MAX_SEQ,
            "shaders/common.glsl says MAX_SEQ {n}, lib.rs says {}",
            crate::MAX_SEQ
        );
    }
}

// ---------------------------------------------------------------------------
// Expert-major routing: group the tile's tokens by expert, then read each
// expert's weights once for all of them.
// ---------------------------------------------------------------------------

/// Sort the tile's `(token, slot)` pairs by expert, on the device.
///
/// Runs between the router and the expert matmuls, inside the same recording:
/// the routing only exists on the GPU, so reading it back to sort on the host
/// would cost a submit-and-wait per layer.
#[allow(clippy::too_many_arguments)]
pub fn moe_group(
    ids: &VkBuf,
    gstart: &VkBuf,
    gcount: &VkBuf,
    ptoken: &VkBuf,
    pslot: &VkBuf,
    n_expert: usize,
    n_used: usize,
    nseq: usize,
) -> Result<()> {
    if n_expert > 1024 {
        return Err(Error::malformed(
            "moe_group",
            format!("{n_expert} experts exceeds the kernel's 1024 shared slots"),
        ));
    }
    let npair = nseq * n_used;
    if gstart.bytes() < n_expert * 4 || gcount.bytes() < n_expert * 4 {
        return Err(Error::malformed("moe_group", "group buffers too small"));
    }
    if ptoken.bytes() < npair * 4 || pslot.bytes() < npair * 4 {
        return Err(Error::malformed("moe_group", "pair buffers too small"));
    }
    dispatch(
        Pipeline::MoeGroup,
        &[ids, gstart, gcount, ptoken, pslot],
        Push::new().u(0, n_expert).u(1, n_used).u(2, nseq),
        1,
        1,
    )
}

/// Expert gate or up, expert-major. The activation is gathered per TOKEN.
#[allow(clippy::too_many_arguments)]
pub fn expert_major(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    g: (&VkBuf, &VkBuf, &VkBuf, &VkBuf), // gstart, gcount, ptoken, pslot
    rows: usize,
    n_used: usize,
    nseq: usize,
    n_expert: usize,
) -> Result<()> {
    need(out, rows * n_used * nseq, "expert_major out")?;
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, n_used)
        .u(7, nseq);
    dispatch_acc(
        Pipeline::Q6kExpertMajor,
        &[w.buf(), qb, sb, ad, out, g.0, g.1, g.2, g.3],
        p,
        groups_for_rows(rows),
        n_expert as u32,
        // A group can hold every token in the tile, and the shader chunks it, so
        // what is requested here is the CHUNK width, not the group's size.
        acc_for(nseq),
    )
}

/// The expert down projection, expert-major. The activation is gathered per
/// `(token, slot)` PAIR, because each expert consumed its own silu(gate)*up —
/// indexing it by token instead is the bug that produces fluent nonsense.
#[allow(clippy::too_many_arguments)]
pub fn expert_major_per_act(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    g: (&VkBuf, &VkBuf, &VkBuf, &VkBuf),
    rows: usize,
    n_used: usize,
    nseq: usize,
    n_expert: usize,
) -> Result<()> {
    need(out, rows * n_used * nseq, "expert_major_pa out")?;
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, n_used)
        .u(7, nseq);
    dispatch_acc(
        Pipeline::Q6kExpertMajorPa,
        &[w.buf(), qb, sb, ad, out, g.0, g.1, g.2, g.3],
        p,
        groups_for_rows(rows),
        n_expert as u32,
        acc_for(nseq),
    )
}

/// The routed expert projections, expert-major AND row-tiled: one activation load
/// feeds `RT_ROWS` rows of dots.
///
/// `by_slot` picks which index the activation is gathered by — the token for gate
/// and up, the (token, slot) PAIR for down, because each expert consumed its own
/// silu(gate)*up. One shader serves both; getting this flag wrong produces fluent
/// nonsense rather than an error.
#[allow(clippy::too_many_arguments)]
pub fn expert_row_tiled(
    w: &PlanarWeights,
    qb: &VkBuf,
    sb: &VkBuf,
    ad: &VkBuf,
    out: &VkBuf,
    g: (&VkBuf, &VkBuf, &VkBuf, &VkBuf),
    rows: usize,
    n_used: usize,
    nseq: usize,
    n_expert: usize,
    by_slot: bool,
) -> Result<()> {
    need(out, rows * n_used * nseq, "expert_row_tiled out")?;
    let (rt_rows, rt_acc) = expert_tile();
    // One row per lane group IS the untiled kernel; there is no point compiling a
    // fourth wrapper that duplicates it.
    if rt_rows <= 1 {
        return if by_slot {
            expert_major_per_act(w, qb, sb, ad, out, g, rows, n_used, nseq, n_expert)
        } else {
            expert_major(w, qb, sb, ad, out, g, rows, n_used, nseq, n_expert)
        };
    }
    let cluster = expert_cluster();
    let (pipe, rows_per_wg) = if cluster == 16 {
        // The swept grid: ROWS x vec4, all at sixteen lanes per row group.
        let per = rows_per_workgroup().max(1) * (subgroup_size() as usize / 16) * rt_rows;
        // The hoisted variant shares memory and therefore barriers, so it is only
        // correct when no lane group returns early for want of rows.
        let hoist = if rt_rows == 2 && !expert_vec4() {
            match expert_addr() {
                // Shared memory means a barrier, so every lane group must have rows.
                1 if rows % per == 0 => Some(Pipeline::Q6kExpertC16R2a),
                2 => Some(Pipeline::Q6kExpertC16R2b),
                _ => None,
            }
        } else {
            None
        };
        let pipe = if let Some(p) = hoist {
            p
        } else {
            match (rt_rows, expert_vec4()) {
                (1, false) => Pipeline::Q6kExpertC16R1,
                (1, true) => Pipeline::Q6kExpertC16R1v,
                (4, false) => Pipeline::Q6kExpertC16R4,
                (4, true) => Pipeline::Q6kExpertC16R4v,
                (_, true) => Pipeline::Q6kExpertC16R2v,
                (_, false) => Pipeline::Q6kExpertC16R2,
            }
        };
        (pipe, per)
    } else if cluster == 8 || cluster == 32 {
        (
            if cluster == 8 {
                Pipeline::Q6kExpertRt2c8
            } else {
                Pipeline::Q6kExpertRt2c32
            },
            rows_per_workgroup().max(1) * (subgroup_size() as usize / cluster) * rt_rows,
        )
    } else {
        (
            match rt_rows {
                2 => Pipeline::Q6kExpertRt2,
                8 => Pipeline::Q6kExpertRt8,
                _ => Pipeline::Q6kExpertRt,
            },
            rows_per_workgroup().max(1) * rt_rows,
        )
    };
    let p = planes(w)
        .u(4, w.blocks())
        .u(5, rows)
        .u(6, n_used)
        .u(7, nseq)
        .u(8, usize::from(by_slot));
    dispatch_acc(
        pipe,
        &[w.buf(), qb, sb, ad, out, g.0, g.1, g.2, g.3],
        p,
        rows.div_ceil(rows_per_wg.max(1)) as u32,
        n_expert as u32,
        // The shader's ACC_N is its token chunk, and the register budget is
        // ROWS * ACC_N — so this asks for the chunk, never the tile width.
        nseq.min(rt_acc),
    )
}
