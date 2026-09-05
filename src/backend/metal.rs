//! Native Metal compute backend for Apple silicon.
//!
//! # Buffer strategy
//!
//! Connectivity (`row_ptr`, `col`) is uploaded once at
//! [`crate::SparseOp::prepare`] and never touched again. Everything else —
//! weights, `x`, `y`, membrane state — lands in scratch buffers allocated once
//! per operator and refilled by `memcpy` per call.
//!
//! That split is measured, not assumed. On an M5 Pro at N = 20000 / 20M nnz,
//! uploading the operator costs ~23 ms while a dispatch costs ~1 ms, so caching
//! connectivity is worth ~23 dispatches. Copying `x` in and `y` out per call,
//! by contrast, is free inside the noise: dispatch-only and copy-in/copy-out
//! timings were within 1% of each other at every size measured. Unified memory
//! means these are memcpys into shared storage, not bus transfers — so there is
//! nothing to win from zero-copy tricks, and the simple, obviously-correct
//! version is also the fast one.
//!
//! # Thread safety
//!
//! objc2's `Retained<ProtocolObject<…>>` is deliberately neither `Send` nor
//! `Sync`: Objective-C object thread-safety is per-class, so the bindings
//! cannot assume it. metal-rs asserted it blanket-wide. The assertion lives
//! here instead, narrowed to the two types that need it and justified against
//! what Apple actually documents — see the `unsafe impl`s below.
//!
//! Given that, a [`crate::SparseOp`] can be shared across threads. The scratch
//! buffers cannot be: two threads writing the same host-visible allocation is a
//! data race regardless of what Metal guarantees about command queues. All
//! scratch lives behind a [`Mutex`], which serialises dispatch per operator.
//! Parallelism across *operators* is unaffected.

use std::ffi::c_void;
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLLibrary, MTLMathMode, MTLResourceOptions,
    MTLSharedEvent, MTLSize,
};

// Aliases so the structs and signatures below read the same as they did under
// the gfx-rs `metal` crate. objc2 spells every Metal type as a `Retained`
// protocol object; naming them once keeps that at the boundary instead of
// spread through every field and argument.
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type MtlDevice = Retained<ProtocolObject<dyn MTLDevice>>;
type SharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;
/// Borrowed encoder, for the `set_*` helpers at the bottom of this file.
type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

/// A GPU-timeline event plus the next value to signal on it.
///
/// Every submission encodes `signal(event, value)` after its last encoder and
/// the host then blocks in `waitUntilSignaledValue:timeoutMS:` for exactly
/// that value. That is a bounded wait with a prompt wake-up, where the earlier
/// status poll slept in a doubling back-off that reached a 1 ms period: a
/// kernel finishing at 1.4 ms was not observed until 2.4 ms.
///
/// One of these per serialised submission stream, never one shared by
/// streams that submit concurrently. `waitUntilSignaledValue` returns once the
/// event's value is *at least* the requested one, and a shared event's value is
/// simply overwritten by whichever signal executes next. Two streams assigning
/// values out of execution order could therefore wake a waiter early; the
/// status re-check after the wait keeps that safe, but not prompt. Owning the
/// counter alongside the scratch it serialises makes the values monotonic in
/// execution order by construction.
struct Completion {
    event: SharedEvent,
    /// Last value handed out; the next submission signals `last + 1`.
    last_value: u64,
}

impl Completion {
    fn new(device: &ProtocolObject<dyn MTLDevice>) -> Option<Self> {
        Some(Self {
            event: device.newSharedEvent()?,
            last_value: 0,
        })
    }

    /// Reserve the next timeline value. Refuses rather than wrapping: a wrapped
    /// value would satisfy `waitUntilSignaledValue` immediately and let the
    /// host read scratch the GPU is still writing.
    fn next_value(&mut self) -> Result<u64, OpError> {
        let value = self.last_value.checked_add(1).ok_or(OpError::Backend {
            reason: "Metal completion timeline exhausted its u64 values",
        })?;
        self.last_value = value;
        Ok(value)
    }
}

/// A command buffer created through the retained-reference path below.
///
/// The newtype makes the resource-lifetime check structural: the bounded wait
/// only accepts command buffers that were created by [`command_buffer`] and
/// proved that Metal will retain every encoded resource.
struct RetainedCommandBuffer(Retained<ProtocolObject<dyn MTLCommandBuffer>>);

impl Deref for RetainedCommandBuffer {
    type Target = ProtocolObject<dyn MTLCommandBuffer>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

use super::{LifParams, OpError, SparsePlanError, SparseShape};
use crate::scan::State;
use crate::sparse::{Csc, Csr};

const KERNEL_SOURCE: &str = include_str!("../kernels/spmv.metal");

/// Threads per threadgroup for the one-thread-per-row kernels, capped by the
/// pipeline's own limit.
const PREFERRED_THREADGROUP: usize = 256;

/// SIMD-group width on Apple silicon. The fused kernel hard-codes this in its
/// strided load and its `simd_sum`, so it is a contract, not a tunable.
const SIMD_WIDTH: usize = 32;

/// Mean stored non-zeros per line at or above which a line gets an eight-lane
/// team instead of a single thread. Below it the extra lanes idle more than
/// they coalesce; see [`RowKernel`] for the measurement behind both
/// thresholds.
const VEC_ROW_MIN_MEAN_NNZ: usize = 12;

/// Mean stored non-zeros per line at or above which a line gets a whole
/// 32-lane simdgroup instead of an eight-lane team.
const SIMD_ROW_MIN_MEAN_NNZ: usize = 64;

/// How many lanes a line gets in the line-parallel kernels.
///
/// Decided once per operator from the CSR it was prepared with, and stored,
/// so that plain SpMV, the spike path, the batched product and the fused LIF
/// step cannot disagree. That is load-bearing: the spike path is documented
/// and tested as bit-identical to the dense one and every SpMM column to the
/// single-vector product, and they are identical only while they traverse a
/// line the same way. A per-call choice, or a caller-supplied flag, could
/// route one to a different width and break that quietly.
///
/// Every tier is one template in `kernels/spmv.metal` instantiated at its
/// lane count, so the tiers differ in nothing but the lane partition and the
/// depth of the butterfly reduction.
///
/// # Why three tiers, and where the thresholds sit
///
/// A team of `LPR` lanes coalesces `LPR` adjacent entries per issued load and
/// idles `LPR - k` lanes on a line of `k < LPR` entries, so the right width
/// follows the mean line length. Measured on an M5 Pro with the total
/// non-zero count held at 8M and the line length swept -- so what moves
/// between points is lane occupancy and coalescing, not the amount of work --
/// and with twenty dispatches per command buffer so the clock sees the kernel
/// rather than the submission (2026-09-04). Six paired, order-balanced rounds
/// of ten calls; each entry is the previous one-thread-per-row kernel's time
/// divided by the width's time at the same shape, so above 1 is faster:
///
/// ```text
///   nnz/row     2     4     6     8    12    16    24    32    48    64   128   256   512
///   1 lane   1.00  1.00  1.05  1.08  1.24  1.43  1.45  1.50  1.49  1.49  1.40  1.06  1.18
///   8 lanes  0.98  1.01  1.08  1.00  1.30  1.70  1.80  1.92  1.95  1.95  1.55  1.18  1.35
///   32 lanes 0.41  0.59  0.67  0.67  1.02  1.36  1.55  1.70  1.78  1.97  1.89  1.47  1.73
/// ```
///
/// The one-lane kernel is the old scalar loop under uniform dispatch and is
/// never slower than it. Eight lanes trail one lane by 8% at 8 entries and
/// overtake it at 12, then hold a 1.3-1.95x win to 64; the full simdgroup
/// ties them at 64 and pulls ahead from 128 (1.89 against 1.55). The previous
/// `simd_sum` kernel measured within 2% of the 32-lane butterfly at every
/// point, so unifying the tiers gave nothing up. Both thresholds sit at the
/// first measured point where the wider team wins rather than at the first
/// tie: the asymmetric error is choosing a width too eagerly, which silently
/// regresses every short-line workload, while choosing it too late only
/// leaves throughput unclaimed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowKernel {
    /// One thread per line. Wastes no lanes on short lines.
    Scalar = 0,
    /// Eight lanes per line: four lines per simdgroup.
    Vec8 = 1,
    /// One simdgroup per line. Coalesces the streamed CSR arrays fully.
    Simd = 2,
}

impl RowKernel {
    /// Every tier, in pipeline-table order.
    const ALL: [RowKernel; 3] = [RowKernel::Scalar, RowKernel::Vec8, RowKernel::Simd];

    /// Lanes a line gets: the `LPR` this tier's kernels were instantiated with.
    const fn lanes(self) -> usize {
        match self {
            RowKernel::Scalar => 1,
            RowKernel::Vec8 => 8,
            RowKernel::Simd => SIMD_WIDTH,
        }
    }

    /// Kernel-name suffix of this tier's instantiations.
    const fn suffix(self) -> &'static str {
        match self {
            RowKernel::Scalar => "l1",
            RowKernel::Vec8 => "l8",
            RowKernel::Simd => "l32",
        }
    }

    /// Choose from the operator's own shape.
    ///
    /// The metric is the *mean* non-zeros per line, not the longest line. What
    /// is being traded is lane occupancy against coalescing across the whole
    /// dispatch: a team assigned a line of `k < lanes` entries leaves
    /// `lanes - k` idle, so the question is how much of the total work sits in
    /// lines long enough to fill a team, and that is what the mean answers.
    /// The longest line is the right statistic for error bounds — which is
    /// what `SparseShape::max_row_nnz` serves — and the wrong one here: a
    /// single dense line in an otherwise empty matrix must not drag every
    /// other line onto a kernel that wastes 31 of its 32 lanes on it.
    ///
    /// An empty operator gets `Scalar`: there is nothing to coalesce, and it
    /// keeps the zero-line case on the same path the rest of the crate takes.
    fn for_shape(lines: usize, nnz: usize) -> Self {
        if lines == 0 {
            return Self::Scalar;
        }
        let mean = nnz / lines;
        if mean < VEC_ROW_MIN_MEAN_NNZ {
            Self::Scalar
        } else if mean < SIMD_ROW_MIN_MEAN_NNZ {
            Self::Vec8
        } else {
            Self::Simd
        }
    }
}

/// Trailing sentinel elements appended to every writable scratch buffer.
///
/// A mutation campaign found that deleting the `row >= n_rows` guard from the
/// fused kernel changed no test result. The kernel was writing past the end of
/// the logical data — the fused dispatch is uniform, so the final threadgroup
/// covers up to `rows_per_group - 1` rows that do not exist — but Metal rounds
/// allocations up to a page, so the overrun landed in slack inside the same
/// buffer. Nothing crashed, nothing was corrupted that anyone read back, and
/// the bug was invisible.
///
/// It is invisible only by luck of the allocator. So every writable buffer now
/// carries a sentinel tail, checked after each dispatch. An out-of-bounds write
/// within this many elements becomes a loud panic instead of silent luck.
///
/// 64 covers the largest realistic tail: `max_total_threads_per_threadgroup`
/// is 1024 on Apple silicon, so a fused threadgroup spans at most 32 rows.
/// Threadgroup scratch depth in the scan kernels.
const SCAN_MAX_TG: usize = 1024;

const CANARY_ELEMS: usize = 64;

/// Sentinel bit pattern. A signalling NaN payload, chosen so that a kernel that
/// wrote a plausible float here is still caught.
const CANARY_BITS: u32 = 0x7FA5_C0DE;

/// Maximum time completion polling accepts a non-terminal command after
/// `commit()` returns.
///
/// This is an execution support boundary, not a claim that every shape
/// representable by the kernels finishes within it. Workloads needing longer
/// must be chunked: they are indistinguishable from a wedged driver. Metal
/// exposes no cancellation API, and an individual Objective-C `commit()` or
/// status selector is opaque to this process, so those calls cannot themselves
/// be pre-empted by an in-process wall-clock deadline.
const METAL_COMMAND_TIMEOUT: Duration = Duration::from_secs(60);

/// Returned after a non-terminal command exceeds [`METAL_COMMAND_TIMEOUT`].
const METAL_QUARANTINED_REASON: &str =
    "Metal is quarantined after a command buffer timed out; restart the process before submitting more work";

/// Linearizes command admission against process-wide quarantine without ever
/// blocking the timeout path on an opaque driver call.
///
/// The high bit records permanent quarantine; the remaining bits count
/// submissions admitted before it was published. A successful increment is a
/// command's admission linearization point. Once the high bit is set every
/// later increment is refused, while already-issued permits remain valid until
/// their `commit()` call returns.
///
/// That distinction is load-bearing. Metal exposes no cancellation primitive,
/// so an already-entered `commit()` may itself stall. Waiting for every permit
/// before publishing quarantine would make the timeout path unbounded; revoking
/// a permit that another thread may already be using would be a false safety
/// claim. Publication is therefore immediate and fail-closed for all *new*
/// submissions, while pre-admitted work keeps its retained ownership.
struct MetalAdmission {
    state: AtomicUsize,
}

impl MetalAdmission {
    const fn new() -> Self {
        Self {
            state: AtomicUsize::new(0),
        }
    }

    fn try_admit(&self) -> Result<SubmissionPermit<'_>, &'static str> {
        let mut observed = self.state.load(Ordering::Acquire);
        loop {
            if observed & METAL_QUARANTINED_BIT != 0 {
                return Err(METAL_QUARANTINED_REASON);
            }
            if observed == METAL_ACTIVE_ADMISSIONS_MASK {
                return Err("Metal command admission counter is exhausted");
            }
            match self.state.compare_exchange_weak(
                observed,
                observed + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(SubmissionPermit { admission: self }),
                Err(actual) => observed = actual,
            }
        }
    }

    fn publish_quarantine(&self) {
        self.state.fetch_or(METAL_QUARANTINED_BIT, Ordering::AcqRel);
    }

    #[cfg(test)]
    fn active_admissions(&self) -> usize {
        self.state.load(Ordering::Acquire) & METAL_ACTIVE_ADMISSIONS_MASK
    }
}

const METAL_QUARANTINED_BIT: usize = 1usize << (usize::BITS - 1);
const METAL_ACTIVE_ADMISSIONS_MASK: usize = METAL_QUARANTINED_BIT - 1;

struct SubmissionPermit<'a> {
    admission: &'a MetalAdmission,
}

impl Drop for SubmissionPermit<'_> {
    fn drop(&mut self) {
        let previous = self.admission.state.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(
            previous & METAL_ACTIVE_ADMISSIONS_MASK > 0,
            "Metal admission permit count underflowed"
        );
    }
}

static METAL_ADMISSION: MetalAdmission = MetalAdmission::new();

/// Process-wide Metal device initialisation, attempted at most once.
///
/// Compiling the MSL library and building eleven pipeline states costs real
/// milliseconds, and nothing about it varies per caller. This records initial
/// availability; [`METAL_ADMISSION`] layers dynamic timeout quarantine over it.
fn shared() -> &'static Result<Arc<MetalDevice>, &'static str> {
    static SHARED: OnceLock<Result<Arc<MetalDevice>, &'static str>> = OnceLock::new();
    SHARED.get_or_init(|| MetalDevice::open_uncached().map(Arc::new))
}

/// The shared device, or why it could not be opened.
pub fn shared_device() -> Result<Arc<MetalDevice>, &'static str> {
    ensure_admission_healthy(&METAL_ADMISSION)?;
    let device = shared().clone();
    // `shared()` may perform cold shader compilation. A timeout can quarantine
    // Metal while that happens, so do not hand a newly initialised device to a
    // caller after quarantine was published.
    ensure_admission_healthy(&METAL_ADMISSION)?;
    device
}

/// `None` when Metal can execute here.
pub fn unavailable_reason() -> Option<&'static str> {
    quarantine_reason(&METAL_ADMISSION).or_else(|| shared().as_ref().err().copied())
}

fn quarantine_reason(admission: &MetalAdmission) -> Option<&'static str> {
    (admission.state.load(Ordering::Acquire) & METAL_QUARANTINED_BIT != 0)
        .then_some(METAL_QUARANTINED_REASON)
}

fn ensure_admission_healthy(admission: &MetalAdmission) -> Result<(), &'static str> {
    quarantine_reason(admission).map_or(Ok(()), Err)
}

fn ensure_device_healthy_reason() -> Result<(), &'static str> {
    ensure_admission_healthy(&METAL_ADMISSION)
}

fn ensure_device_healthy() -> Result<(), OpError> {
    ensure_device_healthy_reason().map_err(|reason| OpError::Backend { reason })
}

fn ensure_device_healthy_for_plan() -> Result<(), SparsePlanError> {
    ensure_device_healthy_reason().map_err(|reason| SparsePlanError::Backend { reason })
}

/// Run one submission action using a permit linearized before quarantine.
fn admit_submission<R>(
    admission: &MetalAdmission,
    submit: impl FnOnce() -> R,
) -> Result<R, &'static str> {
    let permit = admission.try_admit()?;
    let result = submit();
    drop(permit);
    Ok(result)
}

/// Turn a one-off dynamic error into a `&'static str`.
///
/// Called at most once per process, from inside the `OnceLock` initialiser, so
/// the leak is bounded by the number of distinct failure paths (one).
fn leak(reason: String) -> &'static str {
    Box::leak(reason.into_boxed_str())
}

/// Device, queue and compiled pipelines.
/// One tier's kernels: the same templates instantiated at one lane count.
struct LinePipelines {
    spmv: ComputePipelineState,
    spmv_f16: ComputePipelineState,
    spmv_bf16: ComputePipelineState,
    spmv_spikes: ComputePipelineState,
    spmv_t: ComputePipelineState,
    /// The batched product. For [`RowKernel::Scalar`] this is the
    /// two-dimensional one-thread-per-output kernel, which reduces a row in
    /// the same order as the one-lane SpMV.
    spmm: ComputePipelineState,
    fused: ComputePipelineState,
}

pub struct MetalDevice {
    device: MtlDevice,
    queue: CommandQueue,
    /// The line-parallel pipelines, one set per [`RowKernel`] tier, indexed
    /// by the tier's discriminant.
    tiers: [LinePipelines; 3],
    scan_chunk: ComputePipelineState,
    scan_offsets: ComputePipelineState,
    scan_apply: ComputePipelineState,
    lif: ComputePipelineState,
    /// Grow-only device buffers for [`MetalDevice::assoc_scan`].
    ///
    /// A scan has no persistent operator to hang buffers off, so these were
    /// allocated per call. Measured at n = 4.2M that cost ~2.9 ms of a ~11.7 ms
    /// round trip and was the single largest line item after the kernels
    /// themselves: `newBufferWithBytes` faults in a fresh 33.6 MB mapping every
    /// call, and the guarded output another. Kept and grown instead, exactly as
    /// `BatchScratch` is for `spmm`, and never shrunk — a caller alternating
    /// lengths should pay the larger allocation once rather than on every step
    /// back up.
    ///
    /// Behind a mutex because the device is shared by every operator, so unlike
    /// the per-operator scratch this serialises concurrent scans. That costs
    /// nothing real: `assoc_scan` already blocks on `submit_and_wait`, so two
    /// threads scanning at once were contending for the same GPU regardless.
    scan_scratch: Mutex<Option<ScanScratch>>,
    /// Completion timeline for the two device-level paths that own no
    /// operator scratch, `assoc_scan` and dense `lif_integrate`. Held across
    /// submit-and-wait, so those two serialise with each other; a scan already
    /// serialises on `scan_scratch`, and the dense LIF path is documented as a
    /// correctness path rather than a throughput one.
    completion: Mutex<Completion>,
}

/// Reusable device buffers for one associative scan.
struct ScanScratch {
    xs: Buffer,
    out: Guarded,
    totals: Guarded,
    /// Elements `xs` and `out` were sized for.
    n: usize,
    /// Group totals `totals` was sized for.
    groups: usize,
}

// SAFETY: every field is a Metal object Apple documents as safe to use from
// multiple threads — `MTLDevice`, `MTLCommandQueue` and `MTLComputePipelineState`
// are all thread-safe, and the pipelines and queue are built once in
// `open_uncached` and never mutated afterwards. The struct is reached only
// through an `Arc` handed out by `shared()`.
//
// `scan_scratch` and `completion` are the mutable fields, and they are mutable
// only behind a `Mutex`: every access takes the lock, so the buffers and the
// timeline counter inside are reached by one thread at a time and no dispatch
// reads them outside the guard that submitted it. That is the same discipline
// the per-operator `Scratch` already relies on. `MTLSharedEvent` itself is
// documented by Apple as safe to signal and wait on from any thread — it
// exists for cross-queue and cross-process synchronisation.
//
// This is the narrow form of what metal-rs asserted for its whole type set.
unsafe impl Send for MetalDevice {}
unsafe impl Sync for MetalDevice {}

// SAFETY: three classes of field, each safe for a different reason.
//
// `device` is the `Arc<MetalDevice>` justified directly above. The resident
// buffers — `row_ptr`, `col`, `values` and `transpose` — are written once at
// `prepare`, or through `set_weights`, which takes `&mut self` and so cannot
// run while any `&self` method does. Everything a kernel writes lives in
// `scratch`, behind a `Mutex`, so two threads dispatching the same operator
// serialise on it rather than sharing a host-visible allocation.
//
// Concurrent *encoding* from separate operators onto the one shared queue is
// what `MTLCommandQueue`'s documented thread-safety covers. `tests/stress.rs`
// exercises the case this exists for: `THREADS` workers calling `spmv` on one
// `Arc<SparseOp>` and asserting each gets bit-identical results.
unsafe impl Send for MetalSparse {}
unsafe impl Sync for MetalSparse {}

impl MetalDevice {
    fn open_uncached() -> Result<Self, &'static str> {
        let device = MTLCreateSystemDefaultDevice().ok_or("no Metal device on this system")?;
        let queue = device
            .newCommandQueue()
            .ok_or("Metal device returned no command queue")?;

        let options = MTLCompileOptions::new();
        // Apple defines `Safe` as disabling unsafe floating-point
        // optimisations. That is the contract requested here; it is not a
        // stronger promise of operation-for-operation correspondence with a
        // CPU implementation.
        //
        // `mathMode`, not the deprecated `fastMathEnabled`. metal-rs 0.29 did
        // not expose it, which is why this used to set the older property; a
        // comment here recorded that as a known compromise. objc2-metal does
        // expose it, so the compromise is gone.
        //
        // The swap was measured, not assumed equivalent: an FNV hash over the
        // bits of a 512-row SpMV plus eight fused LIF steps is
        // 0x927D9E2BD2C836B2 under both settings on this host. That is one
        // machine and one Metal version — it says the two are the same choice
        // here, not that they must agree everywhere.
        //
        // Neither setting stops the compiler contracting `a * b + c` into a
        // single `fma`, and that too is measured rather than assumed —
        // `tests/fma_contraction.rs` compares every non-spiking GPU membrane
        // against both roundings and finds the fused one used.
        //
        // The consequence is a real one, not a curiosity: one rounding instead
        // of two shifts the membrane by up to an ulp, and that value is then
        // compared against a threshold — so contraction can flip a spike, not
        // merely perturb a float. Cross-backend comparisons must budget for it;
        // see `tolerance_for_elementwise`.
        options.setMathMode(MTLMathMode::Safe);
        let library = device
            .newLibraryWithSource_options_error(&NSString::from_str(KERNEL_SOURCE), Some(&options))
            .map_err(|e| leak(format!("sparsl MSL failed to compile: {e}")))?;

        let pipeline = |name: &str| -> Result<ComputePipelineState, &'static str> {
            let function = library
                .newFunctionWithName(&NSString::from_str(name))
                .ok_or_else(|| leak(format!("kernel `{name}` not found in library")))?;
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| leak(format!("pipeline for `{name}` failed to build: {e}")))
        };

        let tier_pipelines = |tier: RowKernel| -> Result<LinePipelines, &'static str> {
            let s = tier.suffix();
            Ok(LinePipelines {
                spmv: pipeline(&format!("csr_spmv_{s}_kernel"))?,
                spmv_f16: pipeline(&format!("csr_spmv_f16_{s}_kernel"))?,
                spmv_bf16: pipeline(&format!("csr_spmv_bf16_{s}_kernel"))?,
                spmv_spikes: pipeline(&format!("csr_spmv_spikes_{s}_kernel"))?,
                spmv_t: pipeline(&format!("csc_spmv_t_{s}_kernel"))?,
                spmm: match tier {
                    RowKernel::Scalar => pipeline("csr_spmm_kernel")?,
                    RowKernel::Vec8 | RowKernel::Simd => pipeline(&format!("csr_spmm_{s}_kernel"))?,
                },
                fused: pipeline(&format!("fused_spmv_lif_{s}_kernel"))?,
            })
        };
        let tiers = [
            tier_pipelines(RowKernel::Scalar)?,
            tier_pipelines(RowKernel::Vec8)?,
            tier_pipelines(RowKernel::Simd)?,
        ];
        let scan_chunk = pipeline("scan_chunk")?;
        let scan_offsets = pipeline("scan_block_offsets")?;
        let scan_apply = pipeline("scan_apply_offsets")?;
        let lif = pipeline("lif_integrate_kernel")?;

        // `SIMD_WIDTH` is baked into both sides of every team kernel: the host
        // sizes threadgroups as whole multiples of it so an `LPR`-lane team
        // never straddles a simdgroup, and the kernels fold a team with
        // `simd_shuffle_xor`, which only stays inside the team while the
        // hardware's simdgroup is 32 wide.
        //
        // On a device with a 64-wide execution width nothing else would catch
        // that: teams the host believed sat in separate simdgroups would share
        // one, and no canary trips because nothing is written out of bounds.
        // The result is simply wrong, quietly.
        //
        // `Backend::Metal`'s availability check only asks whether a Metal
        // device exists, which an Intel Mac with an AMD GPU satisfies. So the
        // assumption is checked here rather than assumed from the marketing
        // name of the platform. Every pipeline is listed rather than one
        // checked as a proxy: they are separately compiled functions.
        let mut widths: Vec<(String, &ComputePipelineState)> =
            vec![("lif_integrate_kernel".to_string(), &lif)];
        for tier in RowKernel::ALL {
            let s = tier.suffix();
            let p = &tiers[tier as usize];
            widths.extend([
                (format!("csr_spmv_{s}_kernel"), &p.spmv),
                (format!("csr_spmv_f16_{s}_kernel"), &p.spmv_f16),
                (format!("csr_spmv_bf16_{s}_kernel"), &p.spmv_bf16),
                (format!("csr_spmv_spikes_{s}_kernel"), &p.spmv_spikes),
                (format!("csc_spmv_t_{s}_kernel"), &p.spmv_t),
                (format!("csr_spmm_{s}_kernel"), &p.spmm),
                (format!("fused_spmv_lif_{s}_kernel"), &p.fused),
            ]);
        }
        for (name, pipeline) in widths {
            let width = pipeline.threadExecutionWidth();
            if width != SIMD_WIDTH {
                return Err(leak(format!(
                    "`{name}` reports a SIMD execution width of {width}, but sparsl's \
                     kernels are written for {SIMD_WIDTH}. Refusing to run rather than \
                     produce silently wrong rows."
                )));
            }
        }

        let completion =
            Mutex::new(Completion::new(&device).ok_or("Metal device returned no shared event")?);

        Ok(Self {
            device,
            queue,
            tiers,
            scan_chunk,
            scan_offsets,
            scan_apply,
            lif,
            scan_scratch: Mutex::new(None),
            completion,
        })
    }

    /// A fresh completion timeline for one serialised submission stream.
    fn new_completion(&self) -> Result<Completion, SparsePlanError> {
        Completion::new(&self.device).ok_or(SparsePlanError::Allocation {
            what: "completion event",
        })
    }

    /// Name of the physical GPU.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    fn threadgroup_for(&self, pipeline: &ComputePipelineState) -> usize {
        PREFERRED_THREADGROUP.min(pipeline.maxTotalThreadsPerThreadgroup())
    }

    /// Threadgroup size for a line-parallel kernel: a multiple of the SIMD
    /// width, so no lane team straddles a simdgroup and
    /// `threads_per_threadgroup / lanes` is the exact number of lines per group.
    fn simd_threadgroup(&self, pipeline: &ComputePipelineState) -> usize {
        let cap = self.threadgroup_for(pipeline);
        // Round down to a whole number of SIMD groups, then clamp: the `.max(1)`
        // alone would round a sub-32 cap *up* to 32 and dispatch a threadgroup
        // larger than the pipeline permits.
        ((cap / SIMD_WIDTH).max(1) * SIMD_WIDTH).min(cap.max(1))
    }

    /// This tier's kernels.
    fn tier(&self, kernel: RowKernel) -> &LinePipelines {
        &self.tiers[kernel as usize]
    }

    /// Uniform dispatch geometry for a line-parallel kernel: `(threadgroups,
    /// threads per group)`.
    ///
    /// Uniform threadgroups, never `dispatchThreads`: these kernels derive the
    /// line they own from `threads_per_threadgroup`, which non-uniform dispatch
    /// shrinks for the tail group — shifting every line index inside it. The
    /// padding that buys is bounded by the guard each kernel opens with, and
    /// the sentinel tail on every writable buffer is what proves the guard is
    /// still there. The group is a whole number of simdgroups, so a team of
    /// `kernel.lanes()` never straddles one and `tptg / lanes` is exactly the
    /// lines a group covers.
    fn line_geometry(
        &self,
        pipeline: &ComputePipelineState,
        lines: usize,
        kernel: RowKernel,
    ) -> (usize, usize) {
        let tptg = self.simd_threadgroup(pipeline);
        let lines_per_group = (tptg / kernel.lanes()).max(1);
        (lines.div_ceil(lines_per_group), tptg)
    }

    /// Allocate a zeroed device buffer of `len` elements of `T`.
    ///
    /// Metal rejects zero-length allocations, so empty operands get a
    /// one-element placeholder. Nothing reads it: every kernel is bounded by an
    /// explicit count, and the dispatch is skipped entirely when the count is 0.
    fn alloc<T>(&self, len: usize, what: &'static str) -> Result<Buffer, SparsePlanError> {
        let bytes = len
            .max(1)
            .checked_mul(std::mem::size_of::<T>())
            .filter(|&bytes| bytes > 0)
            .ok_or(SparsePlanError::Allocation { what })?;
        let buffer = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(SparsePlanError::Allocation { what })?;
        if buffer.length() < bytes {
            return Err(SparsePlanError::Allocation { what });
        }
        Ok(buffer)
    }

    /// Allocate `len` usable elements followed by [`CANARY_ELEMS`] sentinels.
    fn alloc_guarded<T: Copy>(
        &self,
        len: usize,
        what: &'static str,
    ) -> Result<Guarded, SparsePlanError> {
        let elem = std::mem::size_of::<T>();
        let total = len
            .checked_add(CANARY_ELEMS)
            .ok_or(SparsePlanError::Allocation { what })?;
        let bytes = total
            .checked_mul(elem)
            .filter(|&bytes| bytes > 0)
            .ok_or(SparsePlanError::Allocation { what })?;
        let len_bytes = len
            .checked_mul(elem)
            .ok_or(SparsePlanError::Allocation { what })?;
        let canary_bytes = CANARY_ELEMS
            .checked_mul(elem)
            .ok_or(SparsePlanError::Allocation { what })?;
        let buffer = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(SparsePlanError::Allocation { what })?;
        if buffer.length() < bytes {
            return Err(SparsePlanError::Allocation { what });
        }
        let guarded = Guarded {
            buffer,
            len_bytes,
            canary_bytes,
            what,
        };
        guarded.arm();
        Ok(guarded)
    }

    fn upload<T>(&self, data: &[T], what: &'static str) -> Result<Buffer, SparsePlanError> {
        if data.is_empty() {
            return self.alloc::<T>(0, what);
        }
        let bytes = std::mem::size_of_val(data);
        // SAFETY: `data` is a live borrow of at least `bytes` initialised
        // bytes, which is exactly what `newBufferWithBytes` copies from. The
        // copy happens during the call, so the pointer does not outlive it.
        let buffer = unsafe {
            self.device.newBufferWithBytes_length_options(
                std::ptr::NonNull::new(data.as_ptr() as *mut c_void)
                    .ok_or(SparsePlanError::Allocation { what })?,
                bytes,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(SparsePlanError::Allocation { what })?;
        if buffer.length() < bytes {
            return Err(SparsePlanError::Allocation { what });
        }
        Ok(buffer)
    }

    /// Validate-then-upload. `shape` comes from `SparseOp::prepare`, which has
    /// already proven every column index is in range.
    pub fn prepare(
        self: &Arc<Self>,
        csr: &Csr,
        shape: SparseShape,
        weights: &[f32],
        csc: Option<&Csc>,
        precision: super::WeightPrecision,
    ) -> Result<MetalSparse, SparsePlanError> {
        ensure_device_healthy_for_plan()?;
        let row_ptr = self.upload(&csr.row_ptr, "row_ptr")?;
        let col = self.upload(&csr.col, "col")?;
        // Uploaded as raw `u16`; the selected compact SpMV kernel declares the
        // same memory as `half` or `bfloat`. The host encoders are cross-checked
        // against Metal's own widening in `tests/narrow_backend.rs`, so "same
        // bits, two spellings" is tested rather than assumed.
        let values_narrow = match precision.narrow_bits(weights) {
            Some(bits) => Some(self.upload(&bits, "values_narrow")?),
            None => None,
        };
        let values = self.upload(weights, "values")?;
        let transpose = match csc {
            Some(c) => Some(TransposeIndex {
                col_ptr: self.upload(&c.col_ptr, "csc_col_ptr")?,
                row: self.upload(&c.row, "csc_row")?,
                edge_idx: self.upload(&c.edge_idx, "csc_edge_idx")?,
            }),
            None => None,
        };
        let scratch = Scratch {
            x: self.alloc::<f32>(shape.ncols, "x")?,
            x_spikes: self.alloc::<u32>(crate::spikes::packed_len(shape.ncols), "x_spikes")?,
            y: self.alloc_guarded::<f32>(shape.nrows, "y")?,
            // Allocated only alongside the reverse index, so a forward-only
            // operator pays neither the index nor the scratch.
            yt: match csc {
                Some(_) => Some(self.alloc_guarded::<f32>(shape.ncols, "yt")?),
                None => None,
            },
            xt: match csc {
                Some(_) => Some(self.alloc::<f32>(shape.nrows, "xt")?),
                None => None,
            },
            // Allocated on the first `spmm`, because nothing here knows
            // `n_vec` yet.
            batch: None,
            v: self.alloc_guarded::<f32>(shape.nrows, "v")?,
            theta: self.alloc_guarded::<f32>(shape.nrows, "theta")?,
            spikes: self.alloc_guarded::<u8>(shape.nrows, "spikes")?,
            spikes_host: vec![0u8; shape.nrows],
            completion: self.new_completion()?,
        };
        Ok(MetalSparse {
            device: Arc::clone(self),
            row_ptr,
            col,
            values,
            values_narrow,
            precision,
            transpose,
            shape,
            row_kernel: RowKernel::for_shape(shape.nrows(), shape.nnz()),
            col_kernel: RowKernel::for_shape(shape.ncols(), shape.nnz()),
            scratch: Mutex::new(scratch),
        })
    }

    /// Inclusive scan over affine maps, in three dispatches.
    ///
    /// Not bit-identical to [`crate::assoc_scan`]: this reassociates, which is
    /// what makes it parallel. That is within this crate's rule rather than an
    /// exception to it — reproducibility holds inside a backend and never
    /// across one. Two runs here agree byte for byte.
    ///
    /// Allocates the device buffers per call. A scan has no persistent operator
    /// to hang buffers off the way `SparseOp` does, and pre-sizing them would
    /// mean guessing a length that changes every call.
    ///
    /// It does **not** reformat on the host. `State` is `repr(C)` and exactly
    /// two packed `f32`s, so `&[State]` already *is* the `[a0, b0, a1, b1, …]`
    /// buffer the kernel indexes: the upload reads the caller's slice directly
    /// and the readback fills the output `Vec<State>` directly. This used to
    /// flatten into a `Vec<f32>` and rebuild pairs afterwards, which — together
    /// with the `State`/tuple conversions its caller did — put four full host
    /// passes over the data inside the timed region. Measured at n = 4.2M those
    /// conversions alone were 2.64 ms against a 13.5 ms total.
    pub fn assoc_scan(&self, xs: &[State]) -> Result<Vec<State>, OpError> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        ensure_device_healthy()?;
        let n = xs.len();
        // The Hillis-Steele tree doubles its offset each round and reads
        // `lid + offset`; a non-power-of-two width would drop elements
        // silently rather than failing, so round down.
        let (groups, tptg) = scan_geometry(n, self.scan_chunk.maxTotalThreadsPerThreadgroup())
            .map_err(|reason| OpError::Backend { reason })?;

        let mut slot = self.scan_scratch.lock().map_err(|_| OpError::Backend {
            reason: "sparsl scan scratch mutex is poisoned",
        })?;
        // A caller can enter before another thread times out, then block on
        // this mutex. Recheck after acquiring it so stalled GPU work never
        // releases scratch into a waiter that would immediately reuse it — the
        // same ordering the per-operator scratch uses.
        ensure_device_healthy()?;
        let needs_growth = match slot.as_ref() {
            Some(existing) => existing.n < n || existing.groups < groups,
            None => true,
        };
        if needs_growth {
            // Drop the old buffers before requesting new ones, so a grow does
            // not need both allocations resident at once.
            *slot = None;
            *slot = Some(ScanScratch {
                xs: self
                    .alloc::<State>(n, "scan xs")
                    .map_err(|_| OpError::Backend {
                        reason: "scan: could not allocate input",
                    })?,
                out: self
                    .alloc_guarded::<State>(n, "scan out")
                    .map_err(|_| OpError::Backend {
                        reason: "scan: could not allocate output",
                    })?,
                totals: self
                    .alloc_guarded::<State>(groups, "scan totals")
                    .map_err(|_| OpError::Backend {
                        reason: "scan: could not allocate group totals",
                    })?,
                n,
                groups,
            });
        }
        let scratch = slot
            .as_ref()
            .expect("scan scratch was just ensured present");
        let xs_buf = &scratch.xs;
        let out = &scratch.out;
        let totals = &scratch.totals;

        // Sentinels go immediately after the prefix THIS dispatch uses, not at
        // the physical end of a possibly larger allocation. After a long scan,
        // checking only the far tail would let a short one overrun into the
        // unused capacity unnoticed — the same reason `spmm` arms its batch
        // output per call.
        let out_bytes = std::mem::size_of_val(xs);
        let totals_bytes = groups * std::mem::size_of::<State>();
        out.arm_at(out_bytes);
        totals.arm_at(totals_bytes);
        write_from(xs_buf, xs);

        let cb = command_buffer(&self.queue, "scan: Metal returned no command buffer")?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "scan: Metal returned no compute encoder",
        })?;

        enc.setComputePipelineState(&self.scan_chunk);
        set_buf(&enc, 0, xs_buf);
        set_buf(&enc, 1, &out.buffer);
        set_buf(&enc, 2, &totals.buffer);
        set_u32(&enc, 3, n as u32);
        // Uniform threadgroups, not `dispatchThreads`: the tree needs every
        // lane of a group present, and the kernel pads past `n` with the
        // monoid identity so the extra lanes contribute nothing.
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(tptg));

        enc.setComputePipelineState(&self.scan_offsets);
        set_buf(&enc, 0, &totals.buffer);
        set_u32(&enc, 1, groups as u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(1));

        enc.setComputePipelineState(&self.scan_apply);
        set_buf(&enc, 0, &out.buffer);
        set_buf(&enc, 1, &totals.buffer);
        set_u32(&enc, 2, n as u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(tptg));
        enc.endEncoding();
        // Lock order: `scan_scratch` (held) then `completion`; the dense LIF
        // path takes only `completion`, so the order cannot invert.
        let mut completion = self.completion.lock().map_err(|_| OpError::Backend {
            reason: "sparsl completion mutex is poisoned",
        })?;
        submit_and_wait(cb, "associative scan", &mut completion)?;
        drop(completion);

        out.assert_intact_at(out_bytes);
        totals.assert_intact_at(totals_bytes);
        let mut states = vec![State::identity(); n];
        read_into(&out.buffer, &mut states);
        drop(slot);
        Ok(states)
    }

    /// Dense LIF integrate with no prepared operator.
    ///
    /// Allocates its operands on every call. That is deliberate: without a CSR
    /// there is nothing to bind to, and W0's measurements say allocation is the
    /// expensive part of a GPU call. This exists so the Metal backend has a
    /// dense arm the differential suite can check against the CPU reference —
    /// it is a correctness path, not a throughput path. For throughput, prepare
    /// an operator and call `fused_spmv_lif`.
    pub fn lif_integrate(
        &self,
        v: &mut [f32],
        theta: &mut [f32],
        currents: &[f32],
        spikes: &mut [bool],
        params: LifParams,
    ) -> Result<(), OpError> {
        let n = v.len();
        if n == 0 {
            return Ok(());
        }
        ensure_device_healthy()?;
        if n > u32::MAX as usize {
            return Err(OpError::Backend {
                reason: "LIF length exceeds the Metal kernel's u32 index range",
            });
        }
        let v_buf = self
            .alloc_guarded::<f32>(n, "v")
            .map_err(|_| OpError::Backend {
                reason: "LIF: could not allocate voltage",
            })?;
        v_buf.write(v);
        let theta_buf = self
            .alloc_guarded::<f32>(n, "theta")
            .map_err(|_| OpError::Backend {
                reason: "LIF: could not allocate threshold",
            })?;
        theta_buf.write(theta);
        let currents_buf = self
            .upload(currents, "currents")
            .map_err(|_| OpError::Backend {
                reason: "LIF: could not upload currents",
            })?;
        let spikes_buf = self
            .alloc_guarded::<u8>(n, "spikes")
            .map_err(|_| OpError::Backend {
                reason: "LIF: could not allocate spike output",
            })?;

        let tg = self.threadgroup_for(&self.lif);
        let cb = command_buffer(&self.queue, "LIF: Metal returned no command buffer")?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "LIF: Metal returned no compute encoder",
        })?;
        enc.setComputePipelineState(&self.lif);
        set_buf(&enc, 0, &v_buf.buffer);
        set_buf(&enc, 1, &theta_buf.buffer);
        set_buf(&enc, 2, &currents_buf);
        set_buf(&enc, 3, &spikes_buf.buffer);
        set_f32(&enc, 4, params.decay());
        set_f32(&enc, 5, params.v_reset());
        set_f32(&enc, 6, params.delta_theta());
        set_u32(&enc, 7, n as u32);
        enc.dispatchThreads_threadsPerThreadgroup(size(n), size(tg));
        enc.endEncoding();
        let mut completion = self.completion.lock().map_err(|_| OpError::Backend {
            reason: "sparsl completion mutex is poisoned",
        })?;
        submit_and_wait(cb, "LIF integrate", &mut completion)?;
        drop(completion);

        v_buf.assert_intact();
        theta_buf.assert_intact();
        spikes_buf.assert_intact();
        read_into(&v_buf.buffer, v);
        read_into(&theta_buf.buffer, theta);
        let mut host = vec![0u8; n];
        read_into(&spikes_buf.buffer, &mut host);
        for (dst, &src) in spikes.iter_mut().zip(host.iter()) {
            *dst = src != 0;
        }
        Ok(())
    }
}

/// A device buffer whose tail is a sentinel region, so that a kernel writing
/// past the logical end is detected instead of being absorbed by page padding.
struct Guarded {
    buffer: Buffer,
    len_bytes: usize,
    canary_bytes: usize,
    what: &'static str,
}

impl Guarded {
    /// Copy `src` into the usable region, refusing to spill into the sentinel.
    ///
    /// `write_from` bounds-checks against `buffer.length()`, which for a
    /// guarded buffer includes the sentinel tail. A host write of up to
    /// `CANARY_ELEMS` too many elements therefore passes that check, lands in
    /// the sentinel, and is then reported by `assert_intact` as *"a Metal
    /// kernel wrote past the end"* — blaming the GPU for the host's mistake.
    /// Checking the logical length here keeps the canary's accusation truthful.
    fn write<T: Copy>(&self, src: &[T]) {
        let bytes = std::mem::size_of_val(src);
        assert!(
            bytes <= self.len_bytes,
            "sparsl: `{}` holds {} usable bytes, tried to write {bytes}",
            self.what,
            self.len_bytes
        );
        write_from(&self.buffer, src);
    }

    /// Fill the sentinel region. Called once at allocation and again after any
    /// detected trip, so a second dispatch cannot report a stale failure.
    fn arm(&self) {
        self.arm_at(self.len_bytes);
    }

    /// Fill a sentinel immediately after the logical prefix used by the next
    /// dispatch. This matters for grow-only scratch: after a large SpMM batch,
    /// checking only the allocation's physical tail would let a smaller batch
    /// overrun into the unused capacity without touching the far-end canary.
    fn arm_at(&self, logical_len_bytes: usize) {
        assert!(
            logical_len_bytes <= self.len_bytes,
            "sparsl: logical canary for `{}` starts outside its usable region",
            self.what
        );
        let pattern = CANARY_BITS.to_ne_bytes();
        // SAFETY: `logical_len_bytes <= len_bytes`, and the range
        // `[len_bytes, len_bytes + canary_bytes)` is inside the allocation by
        // construction. Therefore the same-sized range beginning at the
        // logical end is also inside it. The caller holds the scratch mutex.
        unsafe {
            let base = (self.buffer.contents().as_ptr() as *mut u8).add(logical_len_bytes);
            for i in 0..self.canary_bytes {
                base.add(i).write(pattern[i % 4]);
            }
        }
    }

    /// Panic if anything wrote into the sentinel region.
    fn assert_intact(&self) {
        self.assert_intact_at(self.len_bytes);
    }

    /// Check the sentinel beginning at a per-dispatch logical endpoint.
    fn assert_intact_at(&self, logical_len_bytes: usize) {
        assert!(
            logical_len_bytes <= self.len_bytes,
            "sparsl: logical canary for `{}` starts outside its usable region",
            self.what
        );
        let pattern = CANARY_BITS.to_ne_bytes();
        // SAFETY: as `arm_at`.
        let corrupted = unsafe {
            let base = (self.buffer.contents().as_ptr() as *const u8).add(logical_len_bytes);
            (0..self.canary_bytes).any(|i| base.add(i).read() != pattern[i % 4])
        };
        if corrupted {
            self.arm_at(logical_len_bytes);
            panic!(
                "sparsl: a Metal kernel wrote past the end of `{}`. The dispatch grid \
                 covers threads the data does not, and a bounds guard is missing or wrong. \
                 This is an out-of-bounds device write, not a numerical problem.",
                self.what
            );
        }
    }
}

/// Host-visible operands, serialised by the owning operator's mutex.
///
/// `x` is read-only to the kernels and needs no sentinel. Every buffer a kernel
/// writes carries one.
struct Scratch {
    x: Buffer,
    /// Packed spike vector, `packed_len(ncols)` words. Allocated with the rest
    /// so the spike path costs no per-call allocation either.
    x_spikes: Buffer,
    y: Guarded,
    /// Transposed output, `ncols` long. Separate from `y` because the two
    /// directions have different lengths and a shared buffer sized for the
    /// larger would let a length bug read the other's stale tail.
    yt: Option<Guarded>,
    /// Row-length input for the transposed product.
    xt: Option<Buffer>,
    /// Batched operands for `spmm`, and the `n_vec` they were sized for.
    ///
    /// `n_vec` is a per-call quantity, so unlike every other buffer here these
    /// cannot be sized at `prepare` time. They grow on demand and are then
    /// kept: W0's measurements put allocation, not dispatch, at the expensive
    /// end of a Metal call, so reallocating per call would give back exactly
    /// the advantage batching is meant to buy.
    batch: Option<BatchScratch>,
    v: Guarded,
    theta: Guarded,
    spikes: Guarded,
    spikes_host: Vec<u8>,
    /// This operator's completion timeline. It lives with the scratch because
    /// the scratch mutex is what serialises this operator's submissions, and
    /// serialised submission is what keeps the signalled values monotonic.
    completion: Completion,
}

/// A CSR operator resident on the GPU.
pub struct MetalSparse {
    device: Arc<MetalDevice>,
    row_ptr: Buffer,
    col: Buffer,
    /// The operator's quantised values widened to f32, resident for its
    /// lifetime. Uploading these per call is what made the GPU arm lose to
    /// rayon at every size. Every weighted path except plain SpMV (and the
    /// one-vector SpMM that delegates to it) reads this mirror.
    values: Buffer,
    /// Narrow weights — binary16 or bfloat16 — present only for an operator
    /// built by [`crate::Device::prepare_f16`] or
    /// [`crate::Device::prepare_bf16`]. When set, `values` holds the same
    /// quantised weights widened back to f32. Only plain SpMV reads this
    /// compact buffer; the other weighted kernels read `values`.
    values_narrow: Option<Buffer>,
    /// The resident quantisation, which also selects plain SpMV's compact
    /// pipeline when `values_narrow` is present.
    precision: super::WeightPrecision,
    /// Reverse index, present only when the operator was prepared for it.
    /// `edge_idx` points into `values`, so both directions read one table.
    transpose: Option<TransposeIndex>,
    shape: SparseShape,
    /// Row-parallel shape for the forward kernels, fixed at prepare time so
    /// plain SpMV and the spike path always traverse a row identically.
    row_kernel: RowKernel,
    /// The same decision for the transposed direction, taken over columns
    /// because that is what `csc_spmv_t` assigns a simdgroup to.
    col_kernel: RowKernel,
    scratch: Mutex<Scratch>,
}

/// Batched `spmm` operands, sized for `n_vec` vectors.
struct BatchScratch {
    x: Buffer,
    y: Guarded,
    /// The `n_vec` `x` and `y` were allocated for. A request above this
    /// reallocates; at or below it reuses, binding only the prefix in use.
    n_vec: usize,
}

/// CSC device buffers backing `csc_spmv_t_kernel`.
struct TransposeIndex {
    col_ptr: Buffer,
    row: Buffer,
    edge_idx: Buffer,
}

impl MetalSparse {
    /// Overwrite the resident values. Length is checked by the caller.
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), OpError> {
        ensure_device_healthy()?;
        write_from(&self.values, weights);
        // Both representations, or the narrow kernel would keep dispatching the
        // weights the operator was built with while `values` reported the new
        // ones — the exact "two copies that drift" failure the transpose index
        // was designed to avoid by sharing one table.
        if let (Some(buffer), Some(bits)) = (
            self.values_narrow.as_ref(),
            self.precision.narrow_bits(weights),
        ) {
            write_from(buffer, &bits);
        }
        Ok(())
    }

    /// `y += A · x`. Lengths are checked by the caller in `SparseOp::spmv`.
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) -> Result<(), OpError> {
        let mut guard = self.scratch.lock().map_err(|_| OpError::Backend {
            reason: "sparsl scratch mutex is poisoned",
        })?;
        let scratch = &mut *guard;
        // A caller can enter before another thread times out, then block on
        // this mutex. Recheck after acquiring it so stalled GPU work never
        // releases scratch into a waiter that would immediately reuse it.
        ensure_device_healthy()?;
        write_from(&scratch.x, &x[..self.shape.ncols]);
        scratch.y.write(y);

        // One f32 and two narrow pipelines. The compact one is selected by the
        // operator's resident quantisation, not by a caller-supplied flag that
        // could disagree with what was uploaded.
        let tier = self.device.tier(self.row_kernel);
        let (pipeline, values) = match (self.precision, self.values_narrow.as_ref()) {
            (super::WeightPrecision::F16, Some(narrow)) => (&tier.spmv_f16, narrow),
            (super::WeightPrecision::Bf16, Some(narrow)) => (&tier.spmv_bf16, narrow),
            _ => (&tier.spmv, &self.values),
        };
        let cb = command_buffer(&self.device.queue, "SpMV: Metal returned no command buffer")?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "SpMV: Metal returned no compute encoder",
        })?;
        enc.setComputePipelineState(pipeline);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        set_buf(&enc, 2, values);
        set_buf(&enc, 3, &scratch.x);
        set_buf(&enc, 4, &scratch.y.buffer);
        set_u32(&enc, 5, self.shape.nrows as u32);
        dispatch_lines(
            &enc,
            self.row_kernel,
            &self.device,
            pipeline,
            self.shape.nrows,
        );
        enc.endEncoding();
        submit_and_wait(cb, "SpMV", &mut scratch.completion)?;

        scratch.y.assert_intact();
        read_into(&scratch.y.buffer, y);
        drop(guard);
        Ok(())
    }

    /// `Y += A · X` for `n_vec` vectors. Lengths are checked by the caller in
    /// `SparseOp::spmm`.
    ///
    /// Grows the batch scratch when `n_vec` exceeds what it was sized for and
    /// reuses it otherwise, binding only the prefix in use. The buffers are not
    /// shrunk: a caller that alternates batch sizes should pay the larger
    /// allocation once, not on every downward step.
    pub fn spmm(&self, x: &[f32], n_vec: usize, y: &mut [f32]) -> Result<(), OpError> {
        // A batch of one gets no reuse from the two-dimensional kernel and
        // would still pay for batch scratch and a different dispatch shape.
        // Dispatching to the scalar kernel also makes the documented
        // bit-identity structural on this backend rather than a property the
        // two kernels happen to share.
        if n_vec == 1 {
            return self.spmv(x, y);
        }
        if n_vec > u32::MAX as usize {
            return Err(OpError::Backend {
                reason: "spmm n_vec exceeds the Metal kernel's u32 index range",
            });
        }
        let mut guard = self.scratch.lock().map_err(|_| OpError::Backend {
            reason: "sparsl scratch mutex is poisoned",
        })?;
        let scratch = &mut *guard;
        ensure_device_healthy()?;
        let need_x = self
            .shape
            .ncols
            .checked_mul(n_vec)
            .ok_or(OpError::Backend {
                reason: "SpMM x size overflowed usize",
            })?;
        let need_y = self
            .shape
            .nrows
            .checked_mul(n_vec)
            .ok_or(OpError::Backend {
                reason: "SpMM y size overflowed usize",
            })?;

        let grow = match &scratch.batch {
            Some(b) => b.n_vec < n_vec,
            None => true,
        };
        if grow {
            let x_buf = self
                .device
                .alloc::<f32>(need_x.max(1), "spmm x")
                .map_err(|_| OpError::Backend {
                    reason: "SpMM: could not allocate x scratch",
                })?;
            let y_buf = self
                .device
                .alloc_guarded::<f32>(need_y.max(1), "spmm y")
                .map_err(|_| OpError::Backend {
                    reason: "SpMM: could not allocate y scratch",
                })?;
            scratch.batch = Some(BatchScratch {
                x: x_buf,
                y: y_buf,
                n_vec,
            });
        }
        let batch = scratch
            .batch
            .as_ref()
            .expect("batch scratch allocated directly above");

        write_from(&batch.x, &x[..need_x]);
        batch.y.write(&y[..need_y]);
        let logical_y_bytes =
            need_y
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or(OpError::Backend {
                    reason: "SpMM y byte size overflowed usize",
                })?;
        batch.y.arm_at(logical_y_bytes);

        let max_tg = self
            .device
            .threadgroup_for(&self.device.tier(RowKernel::Scalar).spmm);
        let cb = command_buffer(&self.device.queue, "SpMM: Metal returned no command buffer")?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "SpMM: Metal returned no compute encoder",
        })?;
        // The batch kernel must match whatever plain SpMV selected. Every
        // column of a batch is documented and tested as bit-identical to the
        // single-vector product on the same backend, and that holds only while
        // both reduce a row the same way.
        let pipeline = &self.device.tier(self.row_kernel).spmm;
        enc.setComputePipelineState(pipeline);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        set_buf(&enc, 2, &self.values);
        set_buf(&enc, 3, &batch.x);
        set_buf(&enc, 4, &batch.y.buffer);
        set_u32(&enc, 5, self.shape.nrows as u32);
        set_u32(&enc, 6, n_vec as u32);
        match self.row_kernel {
            RowKernel::Scalar => {
                // The two-dimensional one-thread-per-output kernel takes its
                // position from the grid, so it is dispatched non-uniformly.
                let (grid, threads) = spmm_geometry(self.shape.nrows, n_vec, max_tg);
                enc.dispatchThreads_threadsPerThreadgroup(grid, threads);
            }
            RowKernel::Vec8 | RowKernel::Simd => {
                // One team per row, batch handled inside the kernel, so this
                // is the same one-dimensional geometry plain SpMV uses.
                dispatch_lines(
                    &enc,
                    self.row_kernel,
                    &self.device,
                    pipeline,
                    self.shape.nrows,
                );
            }
        }
        enc.endEncoding();
        submit_and_wait(cb, "SpMM", &mut scratch.completion)?;

        batch.y.assert_intact_at(logical_y_bytes);
        read_into(&batch.y.buffer, &mut y[..need_y]);
        drop(guard);
        Ok(())
    }

    /// The compact execution representation available to plain SpMV.
    pub fn weight_precision(&self) -> super::WeightPrecision {
        self.precision
    }

    /// `y += A · s` for a bitpacked spike vector. Lengths checked by the caller.
    pub fn spmv_spikes(&self, spikes: &[u32], y: &mut [f32]) -> Result<(), OpError> {
        let mut guard = self.scratch.lock().map_err(|_| OpError::Backend {
            reason: "sparsl scratch mutex is poisoned",
        })?;
        let scratch = &mut *guard;
        ensure_device_healthy()?;
        let words = crate::spikes::packed_len(self.shape.ncols);
        write_from(&scratch.x_spikes, &spikes[..words]);
        scratch.y.write(y);

        let pipeline = &self.device.tier(self.row_kernel).spmv_spikes;
        let cb = command_buffer(
            &self.device.queue,
            "spike SpMV: Metal returned no command buffer",
        )?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "spike SpMV: Metal returned no compute encoder",
        })?;
        enc.setComputePipelineState(pipeline);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        // Deliberately `values`, never `values_narrow`: the spike path is
        // bit-identical to the dense one, and reading quantised weights here
        // would quietly make that false.
        set_buf(&enc, 2, &self.values);
        set_buf(&enc, 3, &scratch.x_spikes);
        set_buf(&enc, 4, &scratch.y.buffer);
        set_u32(&enc, 5, self.shape.nrows as u32);
        dispatch_lines(
            &enc,
            self.row_kernel,
            &self.device,
            pipeline,
            self.shape.nrows,
        );
        enc.endEncoding();
        submit_and_wait(cb, "spike SpMV", &mut scratch.completion)?;

        scratch.y.assert_intact();
        read_into(&scratch.y.buffer, y);
        drop(guard);
        Ok(())
    }

    /// Whether this operator carries a reverse index.
    pub fn has_transpose(&self) -> bool {
        self.transpose.is_some()
    }

    /// `y += Aᵀ · x`. Lengths are checked by the caller in `SparseOp::spmv_t`.
    ///
    /// Returns [`OpError::TransposeNotPrepared`] rather than panicking when the
    /// operator has no reverse index: `SparseOp::spmv_t` checks the CPU arm the
    /// same way, and both arms must refuse identically or the error becomes a
    /// property of which backend you happened to open.
    pub fn spmv_t(&self, x: &[f32], y: &mut [f32]) -> Result<(), OpError> {
        let idx = self
            .transpose
            .as_ref()
            .ok_or(OpError::TransposeNotPrepared)?;
        let mut guard = self.scratch.lock().map_err(|_| OpError::Backend {
            reason: "sparsl scratch mutex is poisoned",
        })?;
        let scratch = &mut *guard;
        ensure_device_healthy()?;
        let (xt, yt) = match (scratch.xt.as_ref(), scratch.yt.as_ref()) {
            (Some(xt), Some(yt)) => (xt, yt),
            // Unreachable: scratch and index are allocated together in
            // `prepare`. Refusing beats unwrapping if that ever stops holding.
            _ => return Err(OpError::TransposeNotPrepared),
        };
        write_from(xt, &x[..self.shape.nrows]);
        yt.write(y);

        let pipeline = &self.device.tier(self.col_kernel).spmv_t;
        let cb = command_buffer(
            &self.device.queue,
            "transpose SpMV: Metal returned no command buffer",
        )?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "transpose SpMV: Metal returned no compute encoder",
        })?;
        enc.setComputePipelineState(pipeline);
        set_buf(&enc, 0, &idx.col_ptr);
        set_buf(&enc, 1, &idx.row);
        set_buf(&enc, 2, &idx.edge_idx);
        set_buf(&enc, 3, &self.values);
        set_buf(&enc, 4, xt);
        set_buf(&enc, 5, &yt.buffer);
        set_u32(&enc, 6, self.shape.ncols as u32);
        dispatch_lines(
            &enc,
            self.col_kernel,
            &self.device,
            pipeline,
            self.shape.ncols,
        );
        enc.endEncoding();
        submit_and_wait(cb, "transpose SpMV", &mut scratch.completion)?;

        yt.assert_intact();
        read_into(&yt.buffer, y);
        drop(guard);
        Ok(())
    }

    /// Fused SpMV + LIF. Lengths are checked by the caller.
    pub fn fused_spmv_lif(
        &self,
        x: &[f32],
        v: &mut [f32],
        theta: &mut [f32],
        spikes: &mut [bool],
        params: LifParams,
    ) -> Result<(), OpError> {
        let mut guard = self.scratch.lock().map_err(|_| OpError::Backend {
            reason: "sparsl scratch mutex is poisoned",
        })?;
        let scratch = &mut *guard;
        ensure_device_healthy()?;
        write_from(&scratch.x, &x[..self.shape.ncols]);
        scratch.v.write(v);
        scratch.theta.write(theta);

        // The fused kernel runs at the operator's SpMV width, so its row
        // reduction is the one plain SpMV performs.
        let pipeline = &self.device.tier(self.row_kernel).fused;

        let cb = command_buffer(
            &self.device.queue,
            "fused SpMV+LIF: Metal returned no command buffer",
        )?;
        let enc = cb.computeCommandEncoder().ok_or(OpError::Backend {
            reason: "fused SpMV+LIF: Metal returned no compute encoder",
        })?;
        enc.setComputePipelineState(pipeline);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        set_buf(&enc, 2, &self.values);
        set_buf(&enc, 3, &scratch.x);
        set_buf(&enc, 4, &scratch.v.buffer);
        set_buf(&enc, 5, &scratch.theta.buffer);
        set_buf(&enc, 6, &scratch.spikes.buffer);
        set_f32(&enc, 7, params.decay());
        set_f32(&enc, 8, params.v_reset());
        set_f32(&enc, 9, params.delta_theta());
        set_u32(&enc, 10, self.shape.nrows as u32);
        dispatch_lines(
            &enc,
            self.row_kernel,
            &self.device,
            pipeline,
            self.shape.nrows,
        );
        enc.endEncoding();
        submit_and_wait(cb, "fused SpMV+LIF", &mut scratch.completion)?;

        scratch.v.assert_intact();
        scratch.theta.assert_intact();
        scratch.spikes.assert_intact();
        read_into(&scratch.v.buffer, v);
        read_into(&scratch.theta.buffer, theta);
        // Split the borrow: the staging vec is `&mut` while the buffer it is
        // filled from is `&`, and both live in the same guard.
        let Scratch {
            spikes: spikes_buf,
            spikes_host,
            ..
        } = &mut *scratch;
        read_buffer_into(&spikes_buf.buffer, spikes_host);
        for (dst, &src) in spikes.iter_mut().zip(spikes_host.iter()) {
            *dst = src != 0;
        }
        drop(guard);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Dispatch a line-parallel kernel at the operator's selected width.
///
/// Uniform threadgroups at every width: the kernels derive the line they own
/// from `threads_per_threadgroup`, which only uniform dispatch reports
/// faithfully. Pairing the tier with its geometry here keeps that from being a
/// fact each call site has to remember separately.
fn dispatch_lines(
    enc: &ComputeEncoder,
    kernel: RowKernel,
    device: &MetalDevice,
    pipeline: &ComputePipelineState,
    lines: usize,
) {
    let (groups, tptg) = device.line_geometry(pipeline, lines, kernel);
    enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(tptg));
}

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

fn scan_geometry(n: usize, max_threads: usize) -> Result<(usize, usize), &'static str> {
    if n > u32::MAX as usize {
        return Err("scan length exceeds the Metal kernel's u32 index range");
    }
    let cap = max_threads.clamp(1, SCAN_MAX_TG);
    let tptg = 1usize << (usize::BITS - 1 - cap.leading_zeros()) as usize;
    let groups = n.div_ceil(tptg);
    let padded_threads = groups
        .checked_mul(tptg)
        .ok_or("scan's padded Metal grid overflowed usize")?;
    if padded_threads > u32::MAX as usize {
        return Err("scan's padded Metal grid exceeds the kernel's u32 thread-index range");
    }
    Ok((groups, tptg))
}

/// Create a command buffer that owns every resource encoded into it.
///
/// `MTLCommandQueue::commandBuffer` is the retained-reference constructor, but
/// the property is checked rather than inferred from the selector name. That
/// guarantee is what lets a timed-out command outlive the operator whose
/// scratch it may still be using.
fn command_buffer(
    queue: &CommandQueue,
    missing_reason: &'static str,
) -> Result<RetainedCommandBuffer, OpError> {
    ensure_device_healthy()?;
    let cb = queue.commandBuffer().ok_or(OpError::Backend {
        reason: missing_reason,
    })?;
    if !cb.retainedReferences() {
        return Err(OpError::Backend {
            reason: "Metal command buffer does not retain encoded resources",
        });
    }
    // Queue allocation can block when Metal has many outstanding buffers. A
    // timeout may publish quarantine while the selector runs, so perform a
    // second health check before exposing the newly allocated buffer.
    ensure_device_healthy()?;
    Ok(RetainedCommandBuffer(cb))
}

fn completion_result(
    operation: &'static str,
    status: MTLCommandBufferStatus,
    detail: Option<String>,
) -> Result<(), OpError> {
    if status == MTLCommandBufferStatus::Completed {
        return Ok(());
    }
    Err(OpError::Execution {
        operation,
        detail: detail.unwrap_or_else(|| format!("command buffer ended with status {status:?}")),
    })
}

#[derive(Debug)]
enum CompletionWaitError {
    /// Metal reached a terminal failure, so retained resources may be released.
    Terminal(OpError),
    /// The final deadline observation was non-terminal. The command buffer
    /// must stay retained without making another potentially blocking selector
    /// call on the timeout return path.
    TimedOut(OpError),
}

/// Completion state machine with injectable observations and time.
///
/// Production supplies Metal status/error selectors plus a monotonic clock;
/// tests drive the exact same deadline-edge branch without relying on a race
/// against a physical GPU.
fn wait_for_completion_core(
    operation: &'static str,
    admission: &MetalAdmission,
    timeout: Duration,
    mut status: impl FnMut() -> MTLCommandBufferStatus,
    mut detail: impl FnMut() -> Option<String>,
    mut elapsed: impl FnMut() -> Duration,
    mut idle: impl FnMut(Duration),
) -> Result<(), CompletionWaitError> {
    loop {
        let observed = status();
        if observed == MTLCommandBufferStatus::Completed {
            return Ok(());
        }
        if observed == MTLCommandBufferStatus::Error {
            return completion_result(operation, observed, detail())
                .map_err(CompletionWaitError::Terminal);
        }

        let waited = elapsed();
        if waited >= timeout {
            // Terminal states are monotonic. Re-sampling at the deadline edge
            // avoids classifying work that completed between the loop sample
            // and the clock sample as a timeout.
            let deadline_status = status();
            if deadline_status == MTLCommandBufferStatus::Completed {
                return Ok(());
            }
            if deadline_status == MTLCommandBufferStatus::Error {
                return completion_result(operation, deadline_status, detail())
                    .map_err(CompletionWaitError::Terminal);
            }
            admission.publish_quarantine();
            return Err(CompletionWaitError::TimedOut(OpError::Execution {
                operation,
                detail: format!(
                    "command buffer completion timed out after {} ms in status \
                     {deadline_status:?}; Metal is quarantined and its resources \
                     remain retained until process exit",
                    timeout.as_millis()
                ),
            }));
        }

        idle(waited);
    }
}

/// Drive a physical Metal command buffer to a terminal state.
///
/// `signal` is the timeline value the command buffer was told to signal after
/// its last encoder. The host spins on `status()` for the first 100 µs — most
/// small dispatches finish inside that window and never pay a scheduler round
/// trip — then blocks in `waitUntilSignaledValue:timeoutMS:` for the rest of
/// the deadline. The event fires when the GPU has retired the buffer's work,
/// so the status re-check that follows is usually the first and last one.
///
/// The bounded status loop still owns the deadline: an event wait that times
/// out, or one that returns before the driver has published a terminal status,
/// simply drops back into the loop, which re-samples the clock and quarantines
/// on the same edge it always did.
fn wait_for_completion_with_timeout(
    cb: &ProtocolObject<dyn MTLCommandBuffer>,
    operation: &'static str,
    admission: &MetalAdmission,
    timeout: Duration,
    (event, value): (&ProtocolObject<dyn MTLSharedEvent>, u64),
) -> Result<(), CompletionWaitError> {
    let started = Instant::now();
    let spin_until = Duration::from_micros(100);
    let mut event_waited = false;
    let mut sleep_us = 10u64;

    wait_for_completion_core(
        operation,
        admission,
        timeout,
        || cb.status(),
        || {
            cb.error()
                .map(|error| error.localizedDescription().to_string())
        },
        || started.elapsed(),
        |waited| {
            if waited < spin_until {
                std::hint::spin_loop();
                return;
            }
            if !event_waited {
                event_waited = true;
                // Whole milliseconds, rounded up so a sub-millisecond remainder
                // is not turned into an immediate return; the outer loop
                // enforces the exact deadline afterwards either way.
                let remaining_ms = timeout.saturating_sub(waited).as_millis() + 1;
                let remaining_ms = u64::try_from(remaining_ms).unwrap_or(u64::MAX);
                event.waitUntilSignaledValue_timeoutMS(value, remaining_ms);
                return;
            }
            // Only reached after the event fired but before the driver
            // published a terminal status: back off to a one-millisecond
            // ceiling so a stalled GPU cannot burn a core.
            std::thread::sleep(Duration::from_micros(sleep_us));
            sleep_us = sleep_us.saturating_mul(2).min(1_000);
        },
    )
}

/// Finish the ownership handoff after completion polling.
///
/// Generic only so the retention invariant can be tested with a drop sentinel;
/// production passes [`RetainedCommandBuffer`].
fn finalize_retained_wait<T>(
    retained: T,
    result: Result<(), CompletionWaitError>,
) -> Result<(), OpError> {
    match result {
        Ok(()) => Ok(()),
        Err(CompletionWaitError::Terminal(error)) => Err(error),
        Err(CompletionWaitError::TimedOut(error)) => {
            std::mem::forget(retained);
            Err(error)
        }
    }
}

/// Signal `completion`'s next timeline value at the end of `cb`, submit it,
/// then reject every terminal state except a successful completion before any
/// host-visible output is read.
///
/// A non-terminal timeout leaks one retained command buffer deliberately. Its
/// resources may still be in use and Metal has no cancellation API, so freeing
/// or reusing them would be unsound. No additional selector is invoked after
/// timeout classification: even if the command becomes terminal immediately
/// afterwards, conservative retention lasts until process exit. Process-wide
/// quarantine prevents all later admission.
fn submit_and_wait(
    cb: RetainedCommandBuffer,
    operation: &'static str,
    completion: &mut Completion,
) -> Result<(), OpError> {
    let value = completion.next_value()?;
    // After the last encoder ended and before commit, which is the only
    // window Metal allows a command-buffer-level event signal in.
    cb.encodeSignalEvent_value(
        ProtocolObject::<dyn MTLEvent>::from_ref(&*completion.event),
        value,
    );
    // A timeout on another operator can be published while this one is being
    // encoded. Refuse at the last safe point before submission as well as at
    // command-buffer creation. A permit linearizes this call before any
    // concurrent quarantine publication without making publication wait for
    // the opaque driver selector to return.
    admit_submission(&METAL_ADMISSION, || cb.commit())
        .map_err(|reason| OpError::Backend { reason })?;
    let wait = wait_for_completion_with_timeout(
        &cb,
        operation,
        &METAL_ADMISSION,
        METAL_COMMAND_TIMEOUT,
        (&*completion.event, value),
    );
    finalize_retained_wait(cb, wait)
}

/// Two-dimensional SpMM dispatch geometry.
///
/// Keeping `(vector, row)` as the actual grid coordinates removes a device
/// integer division and modulo from every output element. The threadgroup packs
/// as many whole rows as possible while retaining adjacent vector lanes.
fn spmm_geometry(nrows: usize, n_vec: usize, max_threads: usize) -> (MTLSize, MTLSize) {
    debug_assert!(nrows > 0 && n_vec > 1 && max_threads > 0);
    let x = n_vec.next_power_of_two().min(max_threads);
    let y = (max_threads / x).max(1);
    (
        MTLSize {
            width: n_vec,
            height: nrows,
            depth: 1,
        },
        MTLSize {
            width: x,
            height: y,
            depth: 1,
        },
    )
}

/// Bind a scalar by value into the encoder's constant space.
///
/// # Safety
///
/// `value` must live until the call returns; `setBytes` copies it, so a
/// borrow of a local is enough. `index` must name a buffer slot the bound
/// pipeline declares — every caller passes the literal index from the kernel
/// signature it is dispatching.
fn set_scalar<T: Copy>(enc: &ComputeEncoder, index: usize, value: T) {
    // SAFETY: `&value` is a live borrow of exactly `size_of::<T>()` initialised
    // bytes for the duration of the call, and Metal copies out of it before
    // returning. The pointer is non-null because it comes from a reference.
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&value as *const T as *mut c_void)
                .expect("a reference is never null"),
            std::mem::size_of::<T>(),
            index,
        );
    }
}

fn set_f32(enc: &ComputeEncoder, index: usize, value: f32) {
    set_scalar(enc, index, value);
}

fn set_u32(enc: &ComputeEncoder, index: usize, value: u32) {
    set_scalar(enc, index, value);
}

/// Bind a buffer at `index`, offset zero.
fn set_buf(enc: &ComputeEncoder, index: usize, buffer: &Buffer) {
    // SAFETY: `buffer` outlives the encoder — every one is owned by the
    // `MetalSparse` or the `Scratch` held under its mutex for the whole
    // dispatch — and `index` is the literal slot from the kernel signature.
    unsafe { enc.setBuffer_offset_atIndex(Some(buffer), 0, index) };
}

/// Copy `src` into the head of a shared buffer.
///
/// # Panics
///
/// If `src` does not fit. Every caller has already validated lengths against
/// the prepared shape, so a failure here means the shape and the allocation
/// disagree — a bug in this module, not bad input, and it must not be written
/// past the end of the allocation.
fn write_from<T: Copy>(buffer: &Buffer, src: &[T]) {
    if src.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(src);
    assert!(
        buffer.length() >= bytes,
        "sparsl: scratch buffer holds {} bytes, tried to write {bytes}",
        buffer.length()
    );
    // SAFETY: `StorageModeShared` buffers are host-visible for the lifetime of
    // the allocation; the length assertion above proves the destination range
    // is inside it; and `T: Copy` has no drop glue.
    //
    // Exclusive access comes from a per-operator scratch mutex, `&mut self`
    // for resident weight replacement, or ownership of per-call transient
    // buffers. A normal dispatch reaches a terminal state before returning. A
    // timed-out one retains its command and resources until process exit, and
    // quarantine makes both queued scratch waiters and later `set_weights`
    // refuse before they write.
    unsafe {
        std::ptr::copy_nonoverlapping(
            src.as_ptr(),
            buffer.contents().as_ptr() as *mut T,
            src.len(),
        );
    }
}

fn read_into<T: Copy>(buffer: &Buffer, dst: &mut [T]) {
    read_buffer_into(buffer, dst);
}

fn read_buffer_into<T: Copy>(buffer: &Buffer, dst: &mut [T]) {
    if dst.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(dst);
    assert!(
        buffer.length() >= bytes,
        "sparsl: scratch buffer holds {} bytes, tried to read {bytes}",
        buffer.length()
    );
    // SAFETY: as `write_from`, in the opposite direction. The dispatch that
    // produced these bytes was waited on before this call.
    unsafe {
        std::ptr::copy_nonoverlapping(
            buffer.contents().as_ptr() as *const T,
            dst.as_mut_ptr(),
            dst.len(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    const CHILD_TIMEOUT: Duration = Duration::from_secs(30);
    const QUARANTINE_CHILD_ENV: &str = "SPARSL_INTERNAL_QUARANTINE_TEST_CHILD";
    const QUARANTINE_CHILD_TOKEN: &str = "sparsl-quarantine-propagation-v2";
    const QUARANTINE_PASS_MARKER: &str = "SPARSL_QUARANTINE_PROPAGATION_EXECUTED";
    const QUARANTINE_UNAVAILABLE_MARKER: &str = "SPARSL_QUARANTINE_METAL_UNAVAILABLE";

    fn run_isolated_test(
        test_name: &str,
        env_name: &str,
        env_token: &str,
        deadline: Duration,
    ) -> std::process::Output {
        let executable = std::env::current_exe().expect("current test executable");
        let mut child = std::process::Command::new(executable)
            .arg(test_name)
            .arg("--exact")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(env_name, env_token)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("spawn isolated regression child");

        let stop_at = Instant::now() + deadline;
        loop {
            if child
                .try_wait()
                .expect("poll isolated regression child")
                .is_some()
            {
                return child
                    .wait_with_output()
                    .expect("collect isolated regression child output");
            }
            if Instant::now() >= stop_at {
                // Never replace one hang with another: `kill` is a signal, but
                // `wait` may itself block on an unresponsive external call.
                // Dropping the child handle after the signal keeps this parent
                // regression bounded even in that failure mode.
                let _ = child.kill();
                panic!("isolated regression `{test_name}` exceeded {deadline:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn child_logs(output: &std::process::Output) -> String {
        format!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    }

    fn buffer_prefix_bytes(buffer: &Buffer, length: usize) -> Vec<u8> {
        let mut bytes = vec![0u8; length];
        read_buffer_into(buffer, &mut bytes);
        bytes
    }

    fn resident_weight_bytes(op: &crate::SparseOp) -> (Vec<u8>, Option<Vec<u8>>) {
        let super::super::OpResident::Metal(resident) = &op.resident else {
            panic!("quarantine fixture did not create a Metal operator");
        };
        let nnz = op.shape().nnz();
        (
            buffer_prefix_bytes(&resident.values, nnz * std::mem::size_of::<f32>()),
            resident
                .values_narrow
                .as_ref()
                .map(|buffer| buffer_prefix_bytes(buffer, nnz * std::mem::size_of::<u16>())),
        )
    }

    #[test]
    fn only_completed_command_buffers_are_successful() {
        assert!(completion_result("test", MTLCommandBufferStatus::Completed, None).is_ok());
        for status in [
            MTLCommandBufferStatus::NotEnqueued,
            MTLCommandBufferStatus::Enqueued,
            MTLCommandBufferStatus::Committed,
            MTLCommandBufferStatus::Scheduled,
            MTLCommandBufferStatus::Error,
        ] {
            let err = completion_result("test operation", status, Some("injected".into()))
                .expect_err("non-completed command buffer must fail closed");
            assert!(
                matches!(
                    err,
                    OpError::Execution {
                        operation: "test operation",
                        ..
                    }
                ),
                "unexpected error: {err}"
            );
        }
    }

    /// A command buffer that was never committed cannot make progress. Run the
    /// physical fixture in a watchdog child so removing the production timeout
    /// makes this regression fail in 30 seconds rather than hang the CI job.
    #[test]
    fn completion_wait_is_bounded() {
        const CHILD_ENV: &str = "SPARSL_INTERNAL_COMPLETION_TIMEOUT_CHILD";
        const CHILD_TOKEN: &str = "sparsl-completion-timeout-v1";
        const PASS_MARKER: &str = "SPARSL_COMPLETION_TIMEOUT_EXECUTED";
        const UNAVAILABLE_MARKER: &str = "SPARSL_COMPLETION_TIMEOUT_UNAVAILABLE";

        if std::env::var(CHILD_ENV).as_deref() != Ok(CHILD_TOKEN) {
            let output = run_isolated_test(
                "backend::metal::tests::completion_wait_is_bounded",
                CHILD_ENV,
                CHILD_TOKEN,
                CHILD_TIMEOUT,
            );
            let logs = child_logs(&output);
            assert!(output.status.success(), "timeout child failed: {logs}");
            if logs.contains(PASS_MARKER) {
                return;
            }
            if logs.contains(UNAVAILABLE_MARKER) {
                eprintln!(
                    "Metal unavailable; physical completion-timeout regression explicitly skipped"
                );
                return;
            }
            panic!("timeout child exited without an execution marker: {logs}");
        }

        let Some(device) = MTLCreateSystemDefaultDevice() else {
            println!("{UNAVAILABLE_MARKER}");
            return;
        };
        let Some(queue) = device.newCommandQueue() else {
            println!("{UNAVAILABLE_MARKER}");
            return;
        };
        let command_buffer = queue.commandBuffer().expect("command buffer");
        // A real event that nothing will ever signal, so the deadline is
        // enforced through the shared-event wait and not only the poll.
        let event = device.newSharedEvent().expect("shared event");
        let admission = MetalAdmission::new();
        let started = Instant::now();

        let failure = wait_for_completion_with_timeout(
            &command_buffer,
            "timeout regression",
            &admission,
            Duration::from_millis(25),
            (&*event, 1),
        )
        .expect_err("an uncommitted command buffer must time out");
        let error = match failure {
            CompletionWaitError::TimedOut(error) => error,
            CompletionWaitError::Terminal(error) => {
                panic!("uncommitted command unexpectedly became terminal: {error}")
            }
        };

        assert!(
            started.elapsed() < Duration::from_secs(1),
            "completion wait exceeded its test deadline"
        );
        assert!(
            quarantine_reason(&admission).is_some(),
            "a non-terminal timeout must quarantine its device"
        );
        assert!(
            error.to_string().contains("timed out"),
            "unexpected timeout error: {error}"
        );
        println!("{PASS_MARKER}");
    }

    #[test]
    fn production_command_buffers_retain_encoded_resources() {
        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("Metal device unavailable; retained-reference check not reachable");
            return;
        };
        let Some(queue) = device.newCommandQueue() else {
            eprintln!("Metal command queue unavailable; retained-reference check not reachable");
            return;
        };
        let command_buffer =
            command_buffer(&queue, "test command buffer").expect("retained command buffer");

        assert!(command_buffer.retainedReferences());
    }

    #[test]
    fn quarantine_reason_is_fail_closed_and_local_to_its_flag() {
        let admission = MetalAdmission::new();
        assert_eq!(quarantine_reason(&admission), None);

        admission.publish_quarantine();
        assert_eq!(
            quarantine_reason(&admission),
            Some(METAL_QUARANTINED_REASON)
        );
    }

    /// A timeout must not wait for a pre-admitted opaque driver call. This
    /// exact interleaving deadlocked when quarantine and `commit()` shared a
    /// blocking mutex, so the child has an outer watchdog as well as bounded
    /// coordination channels.
    #[test]
    fn quarantine_publication_is_bounded_with_an_active_admission() {
        const CHILD_ENV: &str = "SPARSL_INTERNAL_ADMISSION_CONTENTION_CHILD";
        const CHILD_TOKEN: &str = "sparsl-admission-contention-v1";
        const PASS_MARKER: &str = "SPARSL_ADMISSION_CONTENTION_EXECUTED";

        if std::env::var(CHILD_ENV).as_deref() != Ok(CHILD_TOKEN) {
            let output = run_isolated_test(
                "backend::metal::tests::quarantine_publication_is_bounded_with_an_active_admission",
                CHILD_ENV,
                CHILD_TOKEN,
                CHILD_TIMEOUT,
            );
            let logs = child_logs(&output);
            assert!(output.status.success(), "admission child failed: {logs}");
            assert!(
                logs.contains(PASS_MARKER),
                "admission child exited without its execution marker: {logs}"
            );
            return;
        }

        use std::sync::mpsc;

        let admission = Arc::new(MetalAdmission::new());
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let (finished_tx, finished_rx) = mpsc::channel();
        let submitter = {
            let admission = Arc::clone(&admission);
            std::thread::spawn(move || {
                admit_submission(&admission, || {
                    entered_tx.send(()).expect("signal active admission");
                    release_rx
                        .recv_timeout(Duration::from_secs(5))
                        .expect("release active admission");
                })
                .expect("submission admitted before quarantine");
                finished_tx.send(()).expect("signal admission release");
            })
        };

        entered_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("submission reached its admitted action");
        assert_eq!(admission.active_admissions(), 1);

        let started = Instant::now();
        admission.publish_quarantine();
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "quarantine publication waited for an active admission"
        );
        assert_eq!(admission.active_admissions(), 1);

        let ran = AtomicBool::new(false);
        let refusal = admit_submission(&admission, || ran.store(true, Ordering::Release))
            .expect_err("submission admitted after quarantine");
        assert_eq!(refusal, METAL_QUARANTINED_REASON);
        assert!(!ran.load(Ordering::Acquire));

        release_tx.send(()).expect("release pre-admitted action");
        finished_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("pre-admitted action did not release its permit");
        submitter.join().expect("submission thread panicked");
        assert_eq!(admission.active_admissions(), 0);
        assert_eq!(
            quarantine_reason(&admission),
            Some(METAL_QUARANTINED_REASON)
        );
        println!("{PASS_MARKER}");
    }

    #[test]
    fn production_wait_rechecks_terminal_state_at_the_deadline() {
        for terminal in [
            MTLCommandBufferStatus::Completed,
            MTLCommandBufferStatus::Error,
        ] {
            let admission = MetalAdmission::new();
            let mut statuses = [MTLCommandBufferStatus::Committed, terminal].into_iter();
            let detail_calls = std::cell::Cell::new(0usize);
            let result = wait_for_completion_core(
                "deadline regression",
                &admission,
                Duration::from_millis(25),
                || statuses.next().expect("unexpected extra status sample"),
                || {
                    detail_calls.set(detail_calls.get() + 1);
                    Some("injected terminal detail".into())
                },
                || Duration::from_millis(25),
                |_| panic!("deadline branch must not idle"),
            );
            if terminal == MTLCommandBufferStatus::Completed {
                result.expect("completion at the deadline must succeed");
                assert_eq!(
                    detail_calls.get(),
                    0,
                    "successful completion queried Metal's error selector"
                );
            } else {
                let failure = result.expect_err("device error at the deadline must fail");
                let error = match failure {
                    CompletionWaitError::Terminal(error) => error,
                    CompletionWaitError::TimedOut(error) => {
                        panic!("terminal device error was classified as timeout: {error}")
                    }
                };
                assert!(error.to_string().contains("injected terminal detail"));
                assert_eq!(detail_calls.get(), 1);
            }
            assert_eq!(quarantine_reason(&admission), None);
        }

        let admission = MetalAdmission::new();
        let mut statuses = [
            MTLCommandBufferStatus::Committed,
            MTLCommandBufferStatus::Scheduled,
        ]
        .into_iter();
        let failure = wait_for_completion_core(
            "deadline regression",
            &admission,
            Duration::from_millis(25),
            || statuses.next().expect("unexpected extra status sample"),
            || None,
            || Duration::from_millis(25),
            |_| panic!("deadline branch must not idle"),
        )
        .expect_err("non-terminal command at the deadline must fail");
        let error = match failure {
            CompletionWaitError::TimedOut(error) => error,
            CompletionWaitError::Terminal(error) => {
                panic!("non-terminal deadline was classified as terminal: {error}")
            }
        };
        assert!(error.to_string().contains("completion timed out"));
        assert_eq!(
            quarantine_reason(&admission),
            Some(METAL_QUARANTINED_REASON)
        );
    }

    #[test]
    fn timeout_handoff_retains_resources_without_another_status_query() {
        struct DropSpy<'a>(&'a AtomicBool);

        impl Drop for DropSpy<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Release);
            }
        }

        let timeout_error = || OpError::Execution {
            operation: "retention regression",
            detail: "injected timeout".into(),
        };

        let timed_out_dropped = AtomicBool::new(false);
        let error = finalize_retained_wait(
            DropSpy(&timed_out_dropped),
            Err(CompletionWaitError::TimedOut(timeout_error())),
        )
        .expect_err("injected timeout must fail");
        assert!(error.to_string().contains("injected timeout"));
        assert!(
            !timed_out_dropped.load(Ordering::Acquire),
            "non-terminal timeout released its retained resources"
        );

        let terminal_dropped = AtomicBool::new(false);
        finalize_retained_wait(
            DropSpy(&terminal_dropped),
            Err(CompletionWaitError::Terminal(timeout_error())),
        )
        .expect_err("injected terminal failure must fail");
        assert!(
            terminal_dropped.load(Ordering::Acquire),
            "terminal failure did not release its retained resources"
        );

        let completed_dropped = AtomicBool::new(false);
        finalize_retained_wait(DropSpy(&completed_dropped), Ok(()))
            .expect("injected completion must succeed");
        assert!(
            completed_dropped.load(Ordering::Acquire),
            "completed command did not release its retained resources"
        );
    }

    fn assert_op_quarantined<T>(result: Result<T, OpError>) {
        match result {
            Err(OpError::Backend { reason }) => assert_eq!(reason, METAL_QUARANTINED_REASON),
            _ => panic!("Metal operation did not fail with the quarantine reason"),
        }
    }

    fn assert_plan_quarantined<T>(result: Result<T, SparsePlanError>) {
        match result {
            Err(SparsePlanError::Backend { reason }) => {
                assert_eq!(reason, METAL_QUARANTINED_REASON);
            }
            _ => panic!("Metal preparation did not fail with the quarantine reason"),
        }
    }

    fn run_quarantine_propagation_child() {
        let Ok(raw_device) = shared_device() else {
            println!("{QUARANTINE_UNAVAILABLE_MARKER}");
            return;
        };
        let queue = raw_device.queue.clone();
        let device = crate::Device::try_new(crate::Backend::Metal)
            .expect("shared Metal device opened directly above");
        let csr = Csr::from_adjacency(&[vec![0u32]]);
        let weights = [2.0f32];
        let mut op = device
            .prepare_with_transpose(&csr, 1, &weights)
            .expect("prepare operator before injected quarantine");
        let mut op_f16 = device
            .prepare_f16(&csr, 1, &weights)
            .expect("prepare f16 operator before injected quarantine");
        let mut op_bf16 = device
            .prepare_bf16(&csr, 1, &weights)
            .expect("prepare bf16 operator before injected quarantine");
        let weights_before = resident_weight_bytes(&op);
        let weights_f16_before = resident_weight_bytes(&op_f16);
        let weights_bf16_before = resident_weight_bytes(&op_bf16);
        let empty_csr = Csr::empty(0);
        let empty_op = device
            .prepare(&empty_csr, 0, &[])
            .expect("prepare zero-work operator before injected quarantine");
        // This buffer is deliberately encoded before quarantine. The actual
        // production submission seam must refuse it, leaving Metal status at
        // `NotEnqueued` rather than merely declining a synthetic closure.
        let encoded_before_quarantine =
            command_buffer(&queue, "pre-quarantine command buffer").expect("command buffer");
        let encoded_observer = encoded_before_quarantine.0.clone();

        METAL_ADMISSION.publish_quarantine();

        assert_eq!(unavailable_reason(), Some(METAL_QUARANTINED_REASON));
        assert!(!crate::Backend::Metal.is_available());
        assert!(!crate::available_backends().contains(&crate::Backend::Metal));
        match shared_device() {
            Err(reason) => assert_eq!(reason, METAL_QUARANTINED_REASON),
            Ok(_) => panic!("shared device must honor quarantine"),
        }
        let unavailable = crate::Device::try_new(crate::Backend::Metal)
            .expect_err("new Metal device handles must be refused");
        assert_eq!(unavailable.reason, METAL_QUARANTINED_REASON);

        assert_op_quarantined(command_buffer(&queue, "injected quarantine"));
        let mut completion = raw_device
            .new_completion()
            .expect("completion timeline for the pre-encoded buffer");
        assert_op_quarantined(submit_and_wait(
            encoded_before_quarantine,
            "pre-encoded quarantine regression",
            &mut completion,
        ));
        assert_eq!(
            encoded_observer.status(),
            MTLCommandBufferStatus::NotEnqueued,
            "a command encoded before quarantine was submitted afterwards"
        );
        assert_eq!(METAL_ADMISSION.active_admissions(), 0);
        let submitted = AtomicBool::new(false);
        assert_eq!(
            admit_submission(&METAL_ADMISSION, || {
                submitted.store(true, Ordering::Release);
            }),
            Err(METAL_QUARANTINED_REASON)
        );
        assert!(!submitted.load(Ordering::Acquire));

        for result in [
            device.prepare(&csr, 1, &weights),
            device.prepare_f16(&csr, 1, &weights),
            device.prepare_bf16(&csr, 1, &weights),
            device.prepare_with(&csr, 1, &weights, crate::WeightPrecision::F32),
            device.prepare_with_transpose(&csr, 1, &weights),
        ] {
            assert_plan_quarantined(result);
        }
        assert_op_quarantined(op.set_weights(&[3.0]));
        assert_op_quarantined(op_f16.set_weights(&[3.0]));
        assert_op_quarantined(op_bf16.set_weights(&[3.0]));
        assert_eq!(resident_weight_bytes(&op), weights_before);
        assert_eq!(resident_weight_bytes(&op_f16), weights_f16_before);
        assert_eq!(resident_weight_bytes(&op_bf16), weights_bf16_before);

        let mut y = [7.0f32];
        assert_op_quarantined(op.spmv(&[3.0], &mut y));
        assert_eq!(y, [7.0]);

        let mut single_batch_y = [7.0f32];
        assert_op_quarantined(op.spmm(&[3.0], 1, &mut single_batch_y));
        assert_eq!(single_batch_y, [7.0]);

        let mut batch_y = [7.0f32, 8.0];
        assert_op_quarantined(op.spmm(&[3.0, 4.0], 2, &mut batch_y));
        assert_eq!(batch_y, [7.0, 8.0]);

        let mut spike_y = [7.0f32];
        assert_op_quarantined(op.spmv_spikes(&[1], &mut spike_y));
        assert_eq!(spike_y, [7.0]);

        let mut transpose_y = [7.0f32];
        assert_op_quarantined(op.spmv_t(&[3.0], &mut transpose_y));
        assert_eq!(transpose_y, [7.0]);

        let params = LifParams::new(0.9, 0.0, 0.1).expect("finite LIF parameters");
        let mut v = [0.25f32];
        let mut theta = [1.0f32];
        let mut spikes = [false];
        assert_op_quarantined(op.fused_spmv_lif(&[3.0], &mut v, &mut theta, &mut spikes, params));
        assert_eq!(v, [0.25]);
        assert_eq!(theta, [1.0]);
        assert_eq!(spikes, [false]);

        assert_op_quarantined(device.assoc_scan(&[State { a: 1.0, b: 1.0 }]));
        let mut dense_v = [0.25f32];
        let mut dense_theta = [1.0f32];
        let mut dense_spikes = [false];
        assert_op_quarantined(device.lif_integrate(
            &mut dense_v,
            &mut dense_theta,
            &[0.5],
            &mut dense_spikes,
            params,
        ));
        assert_eq!(dense_v, [0.25]);
        assert_eq!(dense_theta, [1.0]);
        assert_eq!(dense_spikes, [false]);

        // Calls whose validated shape requires no Metal allocation, mutation
        // or submission remain host-side no-ops even after quarantine.
        assert_eq!(device.assoc_scan(&[]).expect("empty scan"), Vec::new());
        device
            .lif_integrate(&mut [], &mut [], &[], &mut [], params)
            .expect("empty LIF");
        empty_op.spmv(&[], &mut []).expect("empty SpMV");
        empty_op
            .spmv_spikes(&[], &mut [])
            .expect("empty spike SpMV");
        empty_op.spmv_t(&[], &mut []).expect("empty transpose");
        empty_op.spmm(&[], 2, &mut []).expect("empty SpMM");
        empty_op
            .fused_spmv_lif(&[], &mut [], &mut [], &mut [], params)
            .expect("empty fused operation");
        println!("{QUARANTINE_PASS_MARKER}");
    }

    /// Runs the destructive process-wide quarantine assertions in an isolated
    /// copy of this test binary so parallel tests keep their healthy backend.
    #[test]
    fn quarantine_propagates_through_every_public_metal_entry_point() {
        if std::env::var(QUARANTINE_CHILD_ENV).as_deref() == Ok(QUARANTINE_CHILD_TOKEN) {
            run_quarantine_propagation_child();
            return;
        }

        let output = run_isolated_test(
            "backend::metal::tests::quarantine_propagates_through_every_public_metal_entry_point",
            QUARANTINE_CHILD_ENV,
            QUARANTINE_CHILD_TOKEN,
            Duration::from_secs(90),
        );
        let logs = child_logs(&output);
        assert!(output.status.success(), "quarantine child failed: {logs}");
        if logs.contains(QUARANTINE_PASS_MARKER) {
            return;
        }
        if logs.contains(QUARANTINE_UNAVAILABLE_MARKER) {
            eprintln!("Metal unavailable; physical quarantine propagation explicitly skipped");
            return;
        }
        panic!("quarantine child exited without an execution marker: {logs}");
    }

    #[test]
    fn spmm_geometry_maps_vectors_to_x_and_rows_to_y_without_overflow() {
        for &(rows, vectors, cap) in &[
            (1usize, 2usize, 256usize),
            (17, 3, 256),
            (10_000, 8, 256),
            (10_000, 32, 256),
            (7, 1_000, 256),
        ] {
            let (grid, threads) = spmm_geometry(rows, vectors, cap);
            assert_eq!((grid.width, grid.height, grid.depth), (vectors, rows, 1));
            assert!(threads.width >= vectors.min(cap));
            assert!(threads.width * threads.height <= cap);
            assert!(threads.width > 0 && threads.height > 0 && threads.depth == 1);
        }
    }

    #[test]
    fn spmm_executes_across_the_physical_pipeline_width_boundary() {
        let Ok(physical) = shared_device() else {
            return; // no Metal device here; the physical boundary is unreachable
        };
        let pipeline_cap = physical
            .tier(RowKernel::Scalar)
            .spmm
            .maxTotalThreadsPerThreadgroup();
        let selected_cap = physical.threadgroup_for(&physical.tier(RowKernel::Scalar).spmm);
        eprintln!(
            "SpMM pipeline maxTotalThreadsPerThreadgroup={pipeline_cap}, host-selected cap={selected_cap}"
        );
        assert!(pipeline_cap >= selected_cap && selected_cap > 0);

        let mut widths = vec![
            pipeline_cap.saturating_sub(1),
            pipeline_cap,
            pipeline_cap.saturating_add(1),
        ];
        widths.retain(|&width| width > 1);
        widths.sort_unstable();
        widths.dedup();

        const NROWS: usize = 33;
        const NCOLS: usize = 7;
        let adjacency: Vec<Vec<u32>> = (0..NROWS)
            .map(|r| vec![((r * 5 + 1) % NCOLS) as u32])
            .collect();
        let csr = Csr::from_adjacency(&adjacency);
        let weights: Vec<f32> = (0..NROWS)
            .map(|r| [0.5f32, -0.25, 1.0, -2.0][r % 4])
            .collect();
        let device = crate::Device::try_new(crate::Backend::Metal)
            .expect("shared Metal device opened directly above");
        let op = device
            .prepare(&csr, NCOLS, &weights)
            .expect("prepare exact physical-cap fixture");

        for n_vec in widths {
            let (grid, threads) = spmm_geometry(NROWS, n_vec, selected_cap);
            assert_eq!((grid.width, grid.height, grid.depth), (n_vec, NROWS, 1));
            assert!(threads.width * threads.height <= selected_cap);
            assert!(threads.width > 0 && threads.height > 0);

            let x: Vec<f32> = (0..NCOLS * n_vec)
                .map(|i| {
                    let c = i / n_vec;
                    let v = i % n_vec;
                    (((c * 13 + v * 7) % 17) as i32 - 8) as f32 * 0.125
                })
                .collect();
            let seed: Vec<f32> = (0..NROWS * n_vec)
                .map(|i| {
                    let r = i / n_vec;
                    let v = i % n_vec;
                    (((r * 11 + v * 3) % 13) as i32 - 6) as f32 * 0.0625
                })
                .collect();
            let mut got = seed.clone();
            op.spmm(&x, n_vec, &mut got)
                .expect("SpMM must split widths above the physical pipeline cap");

            for (r, &weight) in weights.iter().enumerate() {
                let col = csr.col[csr.row_ptr[r] as usize] as usize;
                for v in 0..n_vec {
                    let i = r * n_vec + v;
                    let want = seed[i] + weight * x[col * n_vec + v];
                    assert_eq!(
                        got[i].to_bits(),
                        want.to_bits(),
                        "pipeline_cap={pipeline_cap}, selected_cap={selected_cap}, n_vec={n_vec}, row={r}, vector={v}"
                    );
                }
            }
        }
    }

    #[test]
    fn scan_geometry_rejects_padding_that_exceeds_u32_thread_ids() {
        let tptg = SCAN_MAX_TG;
        let largest_safe = (u32::MAX as usize / tptg) * tptg;
        assert!(scan_geometry(largest_safe, tptg).is_ok());
        assert!(
            scan_geometry(largest_safe + 1, tptg).is_err(),
            "rounding the uniform grid to 2^32 threads must be rejected because the MSL gid is uint"
        );
        assert!(scan_geometry(u32::MAX as usize, tptg).is_err());
    }

    #[test]
    fn impossible_allocation_sizes_fail_before_reaching_metal() {
        let Ok(device) = shared_device() else {
            return; // no Metal device here; allocation API is unreachable
        };
        assert!(matches!(
            device.alloc::<u64>(usize::MAX, "overflow"),
            Err(SparsePlanError::Allocation { what: "overflow" })
        ));
        assert!(matches!(
            device.alloc_guarded::<u64>(usize::MAX, "overflow"),
            Err(SparsePlanError::Allocation { what: "overflow" })
        ));
    }

    /// An over-long host write must be blamed on the host.
    ///
    /// `write_from` bounds-checks against `buffer.length()`, which for a guarded
    /// buffer includes the sentinel tail — so a host write of a few elements too
    /// many used to pass, land in the sentinel, and be reported by
    /// `assert_intact` as a Metal kernel writing out of bounds. Debugging a GPU
    /// kernel for a host bug is an expensive wrong turn.
    #[test]
    fn an_over_long_host_write_is_blamed_on_the_host() {
        let Ok(device) = shared_device() else {
            return; // no Metal device here; nothing to guard
        };
        let guarded = device
            .alloc_guarded::<f32>(4, "test")
            .expect("allocation of four elements");

        // Fits exactly: allowed.
        guarded.write(&[1.0f32; 4]);
        guarded.assert_intact();

        // One element too many still fits inside the allocation, because the
        // sentinel tail is part of it. It must be refused anyway.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            guarded.write(&[1.0f32; 5]);
        }));
        let payload = result.expect_err("a 5-element write into 4 usable elements must panic");
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            message.contains("usable bytes"),
            "the panic must name the host write, not a GPU overrun; got: {message}"
        );
    }

    // ===================================================================
    // Team kernels: eight lanes and a whole simdgroup per line.
    //
    // The rest of this suite works at shapes of a few rows with one or two
    // entries each, which `RowKernel::for_shape` sends to the one-lane
    // kernels. Every test below therefore asserts the fixture actually
    // selected the tier it targets before testing anything else: without
    // that, a later change to a threshold would route these onto another
    // tier and they would keep passing while covering none of the code they
    // exist for.
    // ===================================================================

    /// Row lengths for the eight-lane fixture, chosen to attack the striding
    /// rather than to average tidily: an empty row (no lane iterates and the
    /// fold sums eight zeros), a single-entry row, rows straddling the
    /// eight-lane stride below, on and above it, and rows spanning several
    /// strides with a partial last one. 137 rows of these average 20 entries,
    /// inside the eight-lane band.
    const VEC8_ROW_LENGTHS: [usize; 10] = [0, 1, 7, 8, 9, 17, 31, 32, 33, 64];

    /// The same attack on the 32-lane stride, with a mean past the simdgroup
    /// threshold: 137 rows of these hold 9742 entries, 71 per row.
    const SIMD_ROW_LENGTHS: [usize; 10] = [0, 1, 31, 32, 33, 64, 65, 96, 160, 250];

    /// The tiers with a lane team, each with the width its transposed fixture
    /// uses. `col_kernel` is chosen over columns, so the column mean has to
    /// land in the tier under test: row lengths are clamped to the width, and
    /// at 64 and 96 columns the two fixtures average 42 and 72 entries per
    /// column.
    const TEAM_TIERS: [(RowKernel, usize); 2] = [(RowKernel::Vec8, 64), (RowKernel::Simd, 96)];

    /// Deliberately not a multiple of the lines a threadgroup covers, so the
    /// padded tail group — and the `row >= n_rows` guard that bounds it — is
    /// exercised by every dispatch here.
    const COALESCED_NROWS: usize = 137;
    const COALESCED_NCOLS: usize = 256;

    fn lengths_for(tier: RowKernel) -> &'static [usize] {
        match tier {
            RowKernel::Vec8 => &VEC8_ROW_LENGTHS,
            RowKernel::Simd => &SIMD_ROW_LENGTHS,
            RowKernel::Scalar => panic!("the one-lane tier has no team fixture"),
        }
    }

    fn team_fixture(tier: RowKernel) -> (Csr, Vec<f32>) {
        fixture_over(lengths_for(tier), COALESCED_NCOLS)
    }

    fn fixture_over(lengths: &[usize], ncols: usize) -> (Csr, Vec<f32>) {
        let adjacency: Vec<Vec<u32>> = (0..COALESCED_NROWS)
            .map(|r| {
                // Clamped to the width so the columns stay distinct in a row.
                let len = lengths[r % lengths.len()].min(ncols);
                (0..len).map(|j| ((r + j) % ncols) as u32).collect()
            })
            .collect();
        let csr = Csr::from_adjacency(&adjacency);
        let weights: Vec<f32> = (0..csr.nnz())
            .map(|i| ((i * 37 % 23) as f32 - 11.0) * 0.0625)
            .collect();
        (csr, weights)
    }

    fn longest_row(csr: &Csr) -> usize {
        csr.row_ptr
            .windows(2)
            .map(|b| (b[1] - b[0]) as usize)
            .max()
            .unwrap_or(0)
    }

    fn coalesced_input() -> Vec<f32> {
        (0..COALESCED_NCOLS)
            .map(|c| ((c * 29 % 31) as f32 - 15.0) * 0.125)
            .collect()
    }

    fn coalesced_seed() -> Vec<f32> {
        (0..COALESCED_NROWS)
            .map(|r| ((r * 19 % 13) as f32 - 6.0) * 0.25)
            .collect()
    }

    fn resident(op: &crate::SparseOp) -> &MetalSparse {
        match &op.resident {
            crate::backend::OpResident::Metal(metal) => metal,
            _ => panic!("fixture did not produce a Metal-resident operator"),
        }
    }

    #[test]
    fn row_kernel_selection_follows_the_mean_and_not_the_longest_line() {
        // Below, at, and above each threshold.
        assert_eq!(RowKernel::for_shape(100, 100 * 11), RowKernel::Scalar);
        assert_eq!(
            RowKernel::for_shape(100, 100 * VEC_ROW_MIN_MEAN_NNZ),
            RowKernel::Vec8
        );
        assert_eq!(RowKernel::for_shape(100, 100 * 63), RowKernel::Vec8);
        assert_eq!(
            RowKernel::for_shape(100, 100 * SIMD_ROW_MIN_MEAN_NNZ),
            RowKernel::Simd
        );
        assert_eq!(RowKernel::for_shape(100, 100 * 500), RowKernel::Simd);
        // The mean is an integer division: 1199 entries over 100 lines is 11.
        assert_eq!(RowKernel::for_shape(100, 1199), RowKernel::Scalar);

        // One dense line in an otherwise empty matrix must not drag the other
        // 999 onto a kernel that would waste 31 of its 32 lanes on each.
        assert_eq!(RowKernel::for_shape(1_000, 900), RowKernel::Scalar);

        // Degenerate shapes must not divide by zero or select a kernel whose
        // dispatch geometry would be empty.
        assert_eq!(RowKernel::for_shape(0, 0), RowKernel::Scalar);
        assert_eq!(RowKernel::for_shape(0, 10), RowKernel::Scalar);
        assert_eq!(RowKernel::for_shape(10, 0), RowKernel::Scalar);
    }

    #[test]
    fn coalesced_dispatch_geometry_covers_every_line_exactly_once() {
        let Ok(physical) = shared_device() else {
            return; // no Metal device here
        };
        for tier in RowKernel::ALL {
            let pipeline = &physical.tier(tier).spmv;
            for lines in [1usize, 7, 31, 32, 33, COALESCED_NROWS, 4096] {
                let (groups, tptg) = physical.line_geometry(pipeline, lines, tier);
                assert_eq!(tptg % SIMD_WIDTH, 0, "threadgroup must be whole simdgroups");
                assert!(tptg <= pipeline.maxTotalThreadsPerThreadgroup());
                let per_group = tptg / tier.lanes();
                assert!(
                    groups * per_group >= lines,
                    "{tier:?}: geometry leaves lines unassigned: {groups} x {per_group} < {lines}"
                );
                assert!(
                    (groups - 1) * per_group < lines,
                    "{tier:?}: geometry dispatches a group with no line to own"
                );
            }
        }
    }

    /// Prepare the team fixture for `tier` and assert it selected that tier.
    fn prepared_team(tier: RowKernel) -> (crate::SparseOp, Csr, Vec<f32>) {
        let (csr, weights) = team_fixture(tier);
        let op = crate::Device::try_new(crate::Backend::Metal)
            .expect("metal device")
            .prepare(&csr, COALESCED_NCOLS, &weights)
            .expect("prepare team fixture");
        assert_eq!(
            resident(&op).row_kernel,
            tier,
            "fixture no longer selects {tier:?} ({} nnz over {} rows); the team \
             tests would silently cover another tier",
            csr.nnz(),
            COALESCED_NROWS
        );
        (op, csr, weights)
    }

    #[test]
    fn the_team_kernels_are_the_ones_the_fixtures_actually_run() {
        let Ok(_) = shared_device() else {
            return;
        };
        for (tier, _) in TEAM_TIERS {
            prepared_team(tier);
        }
    }

    #[test]
    fn team_spmv_matches_the_cpu_arm_within_the_published_tolerance() {
        let Ok(_) = shared_device() else {
            return;
        };
        let x = coalesced_input();
        let seed = coalesced_seed();
        for (tier, _) in TEAM_TIERS {
            let (gpu, csr, weights) = prepared_team(tier);
            let mut got = seed.clone();
            gpu.spmv(&x, &mut got).expect("team spmv");

            let cpu = crate::Device::cpu_sequential()
                .prepare(&csr, COALESCED_NCOLS, &weights)
                .expect("prepare cpu");
            let mut want = seed.clone();
            cpu.spmv(&x, &mut want).expect("cpu spmv");

            let max_abs_term = weights.iter().fold(0.0f32, |m, w| m.max(w.abs()))
                * x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let max_abs_result = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let tolerance =
                crate::tolerance_for_spmv(longest_row(&csr), max_abs_term, max_abs_result);
            for (r, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= tolerance,
                    "{tier:?} row {r}: gpu {a} vs cpu {b}, tolerance {tolerance}"
                );
            }
        }
    }

    #[test]
    fn the_team_spike_path_stays_bit_identical_to_the_dense_one() {
        let Ok(_) = shared_device() else {
            return;
        };
        let seed = coalesced_seed();
        // A spike pattern with runs of set and clear bits, so whole strides
        // are both fully on and fully off.
        let bits: Vec<bool> = (0..COALESCED_NCOLS).map(|c| (c / 5) % 2 == 0).collect();
        let dense: Vec<f32> = bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let packed = crate::spikes::pack_spikes(&bits);
        for (tier, _) in TEAM_TIERS {
            let (op, _, _) = prepared_team(tier);
            let mut from_dense = seed.clone();
            op.spmv(&dense, &mut from_dense).expect("dense spmv");
            let mut from_spikes = seed.clone();
            op.spmv_spikes(&packed, &mut from_spikes)
                .expect("spike spmv");
            for (r, (&d, &s)) in from_dense.iter().zip(&from_spikes).enumerate() {
                assert_eq!(
                    d.to_bits(),
                    s.to_bits(),
                    "{tier:?} row {r}: the spike kernel diverged from the dense one \
                     ({d:?} vs {s:?}); these are documented as bit-identical"
                );
            }
        }
    }

    #[test]
    fn a_team_batch_of_one_stays_bit_identical_to_spmv() {
        let Ok(_) = shared_device() else {
            return;
        };
        let x = coalesced_input();
        let seed = coalesced_seed();
        for (tier, _) in TEAM_TIERS {
            let (op, _, _) = prepared_team(tier);
            let mut single = seed.clone();
            op.spmv(&x, &mut single).expect("spmv");
            let mut batched = seed.clone();
            op.spmm(&x, 1, &mut batched).expect("spmm n_vec=1");
            for (r, (&a, &b)) in single.iter().zip(&batched).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "{tier:?} row {r}: batch-of-one diverged"
                );
            }
        }
    }

    #[test]
    fn team_spmv_is_bit_stable_across_repeated_dispatches() {
        let Ok(_) = shared_device() else {
            return;
        };
        let x = coalesced_input();
        let seed = coalesced_seed();
        for (tier, _) in TEAM_TIERS {
            let (op, _, _) = prepared_team(tier);
            let mut first = seed.clone();
            op.spmv(&x, &mut first).expect("spmv");
            for round in 1..16 {
                let mut again = seed.clone();
                op.spmv(&x, &mut again).expect("spmv");
                assert_eq!(
                    first, again,
                    "{tier:?}: round {round} differs from round 0; reproducibility \
                     must hold within a backend"
                );
            }
        }
    }

    #[test]
    fn the_team_transposes_match_the_cpu_arm() {
        let Ok(_) = shared_device() else {
            return;
        };
        for (tier, ncols) in TEAM_TIERS {
            // The transposed direction is chosen over columns, so drive it
            // with a width whose columns are long enough to select the tier.
            let (csr, weights) = fixture_over(lengths_for(tier), ncols);
            let x: Vec<f32> = (0..COALESCED_NROWS)
                .map(|r| ((r * 23 % 17) as f32 - 8.0) * 0.125)
                .collect();
            let seed: Vec<f32> = (0..ncols)
                .map(|c| ((c * 11 % 7) as f32 - 3.0) * 0.5)
                .collect();

            let gpu = crate::Device::try_new(crate::Backend::Metal)
                .expect("metal device")
                .prepare_with_transpose(&csr, ncols, &weights)
                .expect("prepare gpu transpose");
            assert_eq!(
                resident(&gpu).col_kernel,
                tier,
                "transpose fixture no longer selects {tier:?} over {ncols} columns"
            );
            let mut got = seed.clone();
            gpu.spmv_t(&x, &mut got).expect("team spmv_t");

            let cpu = crate::Device::cpu_sequential()
                .prepare_with_transpose(&csr, ncols, &weights)
                .expect("prepare cpu transpose");
            let mut want = seed.clone();
            cpu.spmv_t(&x, &mut want).expect("cpu spmv_t");

            let max_abs_term = weights.iter().fold(0.0f32, |m, w| m.max(w.abs()))
                * x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let max_abs_result = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            // Bounded by the longest COLUMN here; the total non-zero count is
            // an upper bound on it and errs wide, which is the safe direction.
            let tolerance = crate::tolerance_for_spmv(csr.nnz(), max_abs_term, max_abs_result);
            for (c, (&a, &b)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (a - b).abs() <= tolerance,
                    "{tier:?} column {c}: gpu {a} vs cpu {b}, tolerance {tolerance}"
                );
            }
        }
    }

    #[test]
    fn the_team_narrow_kernels_agree_with_their_widened_weights() {
        let Ok(_) = shared_device() else {
            return;
        };
        let x = coalesced_input();
        let seed = coalesced_seed();
        let device = crate::Device::try_new(crate::Backend::Metal).expect("metal device");

        for (tier, _) in TEAM_TIERS {
            let (csr, weights) = team_fixture(tier);
            for (label, prepared) in [
                ("f16", device.prepare_f16(&csr, COALESCED_NCOLS, &weights)),
                ("bf16", device.prepare_bf16(&csr, COALESCED_NCOLS, &weights)),
            ] {
                let op = prepared.unwrap_or_else(|e| panic!("prepare {label}: {e}"));
                assert_eq!(
                    resident(&op).row_kernel,
                    tier,
                    "{label} fixture no longer selects {tier:?}"
                );
                let mut got = seed.clone();
                op.spmv(&x, &mut got)
                    .unwrap_or_else(|e| panic!("{label} spmv: {e}"));

                // The operator keeps the same weights widened back to f32; a
                // full-precision arm over those is the right reference,
                // because it isolates the kernel from the quantisation.
                let widened: Vec<f32> = match op.weight_precision() {
                    crate::WeightPrecision::F16 => weights
                        .iter()
                        .map(|&w| crate::half::f16_bits_to_f32(crate::half::f32_to_f16_bits(w)))
                        .collect(),
                    crate::WeightPrecision::Bf16 => weights
                        .iter()
                        .map(|&w| crate::half::bf16_bits_to_f32(crate::half::f32_to_bf16_bits(w)))
                        .collect(),
                    crate::WeightPrecision::F32 => weights.clone(),
                };
                let cpu = crate::Device::cpu_sequential()
                    .prepare(&csr, COALESCED_NCOLS, &widened)
                    .expect("prepare cpu");
                let mut want = seed.clone();
                cpu.spmv(&x, &mut want).expect("cpu spmv");

                let max_abs_term = widened.iter().fold(0.0f32, |m, w| m.max(w.abs()))
                    * x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let max_abs_result = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let tolerance =
                    crate::tolerance_for_spmv(longest_row(&csr), max_abs_term, max_abs_result);
                for (r, (&a, &b)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (a - b).abs() <= tolerance,
                        "{tier:?} {label} row {r}: gpu {a} vs widened cpu {b}, tolerance {tolerance}"
                    );
                }
            }
        }
    }

    /// Randomised differential stress across the kernel-selection boundary.
    ///
    /// The shared `SHAPES` table in `tests/common` documents itself as sitting
    /// on every boundary the backends care about, but it predates this one: of
    /// its eighteen shapes only two reach a mean degree of 24, and one of those
    /// has a single row. So the coalesced kernels were being exercised by a
    /// couple of incidental shapes rather than deliberately.
    ///
    /// This sweeps mean degree from 0 through 100 -- straddling both thresholds
    /// in both directions -- and checks, per shape, everything the three tiers
    /// must agree on: the CPU arm within the published bound, the spike path bit for
    /// bit against the dense one, every SpMM column bit for bit against the
    /// single-vector product, and repeat-dispatch stability. The tally at the
    /// end is what stops the whole sweep from quietly missing a tier.
    #[test]
    fn both_row_kernels_agree_with_the_cpu_across_a_degree_sweep() {
        let Ok(_) = shared_device() else {
            return; // no Metal device here
        };
        let device = crate::Device::try_new(crate::Backend::Metal).expect("metal device");
        let host = crate::Device::cpu_sequential();
        let mut rng = crate::Rng::new(0x5A15_2026);
        let mut seen = [0usize; 3];

        // Rows and columns deliberately unequal and off every power of two, so
        // a tail-group or lane-assignment bug cannot hide behind a round shape.
        for &(nrows, ncols) in &[(137usize, 200usize), (1usize, 96usize), (513usize, 71usize)] {
            for &max_deg in &[0usize, 1, 5, 12, 16, 24, 40, 56, 80, 140, 200] {
                let adjacency: Vec<Vec<u32>> = (0..nrows)
                    .map(|_| {
                        let deg = if max_deg == 0 {
                            0
                        } else {
                            rng.gen_index(max_deg + 1)
                        };
                        (0..deg).map(|_| rng.gen_index(ncols) as u32).collect()
                    })
                    .collect();
                let csr = Csr::from_adjacency(&adjacency);
                let weights: Vec<f32> = (0..csr.nnz()).map(|_| rng.next_f32() - 0.5).collect();
                let x: Vec<f32> = (0..ncols).map(|_| rng.next_f32() - 0.5).collect();
                let seed: Vec<f32> = (0..nrows).map(|_| rng.next_f32() - 0.5).collect();

                let gpu = device
                    .prepare(&csr, ncols, &weights)
                    .expect("prepare gpu operator");
                let cpu = host
                    .prepare(&csr, ncols, &weights)
                    .expect("prepare cpu operator");
                let label = format!(
                    "nrows={nrows} ncols={ncols} max_deg={max_deg} nnz={}",
                    csr.nnz()
                );
                seen[resident(&gpu).row_kernel as usize] += 1;

                // 1. The GPU arm against the host, under the published bound.
                let mut got = seed.clone();
                gpu.spmv(&x, &mut got).expect("gpu spmv");
                let mut want = seed.clone();
                cpu.spmv(&x, &mut want).expect("cpu spmv");
                let term = weights.iter().fold(0.0f32, |m, w| m.max(w.abs()))
                    * x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let tol = crate::tolerance_for_spmv(longest_row(&csr), term, scale);
                for (r, (&a, &b)) in got.iter().zip(&want).enumerate() {
                    assert!(
                        (a - b).abs() <= tol,
                        "{label}: row {r} gpu {a} vs cpu {b}, tolerance {tol}"
                    );
                }

                // 2. Repeat dispatch must be byte-identical within the backend.
                let mut again = seed.clone();
                gpu.spmv(&x, &mut again).expect("gpu spmv repeat");
                assert_eq!(got, again, "{label}: repeated dispatch drifted");

                // 3. The spike path against the dense one, to the bit.
                let bits: Vec<bool> = (0..ncols).map(|_| rng.next_f32() < 0.5).collect();
                let dense: Vec<f32> = bits.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
                let packed = crate::spikes::pack_spikes(&bits);
                let mut from_dense = seed.clone();
                gpu.spmv(&dense, &mut from_dense).expect("dense spmv");
                let mut from_spikes = seed.clone();
                gpu.spmv_spikes(&packed, &mut from_spikes)
                    .expect("spike spmv");
                for (r, (&d, &sp)) in from_dense.iter().zip(&from_spikes).enumerate() {
                    assert_eq!(
                        d.to_bits(),
                        sp.to_bits(),
                        "{label}: row {r} spike path diverged from dense"
                    );
                }

                // 4. Every SpMM column against the single-vector product, to the
                //    bit. Widths chosen around the kernel's 8-wide register tile
                //    so a partial tile is covered as well as a full one.
                for &n_vec in &[1usize, 3, 8, 11] {
                    let batch_x: Vec<f32> = (0..ncols * n_vec)
                        .map(|i| x[i / n_vec] + (i % n_vec) as f32 * 0.125)
                        .collect();
                    let mut batched = vec![0.0f32; nrows * n_vec];
                    gpu.spmm(&batch_x, n_vec, &mut batched).expect("gpu spmm");
                    for v in 0..n_vec {
                        let column: Vec<f32> = (0..ncols).map(|c| batch_x[c * n_vec + v]).collect();
                        let mut single = vec![0.0f32; nrows];
                        gpu.spmv(&column, &mut single).expect("gpu spmv column");
                        for r in 0..nrows {
                            assert_eq!(
                                batched[r * n_vec + v].to_bits(),
                                single[r].to_bits(),
                                "{label}: n_vec={n_vec} row {r} vector {v} \
                                 diverged from the single-vector product"
                            );
                        }
                    }
                }
            }
        }

        assert!(
            seen.iter().all(|&n| n > 0),
            "the sweep missed a tier (scalar={}, vec8={}, simd={}); it is meant \
             to cross both selection thresholds",
            seen[0],
            seen[1],
            seen[2]
        );
    }
}
