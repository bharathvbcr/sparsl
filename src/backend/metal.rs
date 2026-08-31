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
use std::sync::{Arc, Mutex, OnceLock};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLMathMode, MTLResourceOptions, MTLSize,
};

// Aliases so the structs and signatures below read the same as they did under
// the gfx-rs `metal` crate. objc2 spells every Metal type as a `Retained`
// protocol object; naming them once keeps that at the boundary instead of
// spread through every field and argument.
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type CommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type MtlDevice = Retained<ProtocolObject<dyn MTLDevice>>;
/// Borrowed encoder, for the `set_*` helpers at the bottom of this file.
type ComputeEncoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

use super::{LifParams, OpError, SparsePlanError, SparseShape};
use crate::sparse::{Csc, Csr};

const KERNEL_SOURCE: &str = include_str!("../kernels/spmv.metal");

/// Threads per threadgroup for the one-thread-per-row kernels, capped by the
/// pipeline's own limit.
const PREFERRED_THREADGROUP: usize = 256;

/// SIMD-group width on Apple silicon. The fused kernel hard-codes this in its
/// strided load and its `simd_sum`, so it is a contract, not a tunable.
const SIMD_WIDTH: usize = 32;

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

/// Process-wide Metal device, opened at most once.
///
/// Compiling the MSL library and building three pipeline states costs real
/// milliseconds, and nothing about it varies per caller. Availability is
/// defined as "this succeeded", so `Backend::Metal.is_available()` can never
/// disagree with what `Device::try_new(Backend::Metal)` will do.
fn shared() -> &'static Result<Arc<MetalDevice>, &'static str> {
    static SHARED: OnceLock<Result<Arc<MetalDevice>, &'static str>> = OnceLock::new();
    SHARED.get_or_init(|| MetalDevice::open_uncached().map(Arc::new))
}

/// The shared device, or why it could not be opened.
pub fn shared_device() -> Result<Arc<MetalDevice>, &'static str> {
    shared().clone()
}

/// `None` when Metal can execute here.
pub fn unavailable_reason() -> Option<&'static str> {
    shared().as_ref().err().copied()
}

/// Turn a one-off dynamic error into a `&'static str`.
///
/// Called at most once per process, from inside the `OnceLock` initialiser, so
/// the leak is bounded by the number of distinct failure paths (one).
fn leak(reason: String) -> &'static str {
    Box::leak(reason.into_boxed_str())
}

/// Device, queue and compiled pipelines.
pub struct MetalDevice {
    device: MtlDevice,
    queue: CommandQueue,
    spmv: ComputePipelineState,
    spmv_f16: ComputePipelineState,
    spmv_bf16: ComputePipelineState,
    spmv_spikes: ComputePipelineState,
    spmv_t: ComputePipelineState,
    spmm: ComputePipelineState,
    scan_chunk: ComputePipelineState,
    scan_offsets: ComputePipelineState,
    scan_apply: ComputePipelineState,
    lif: ComputePipelineState,
    fused: ComputePipelineState,
}

// SAFETY: every field is a Metal object Apple documents as safe to use from
// multiple threads — `MTLDevice`, `MTLCommandQueue` and `MTLComputePipelineState`
// are all thread-safe, and the pipelines and queue are built once in
// `open_uncached` and never mutated afterwards. The struct is reached only
// through an `Arc` handed out by `shared()`, so there is no interior mutation
// to race on either.
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
        // Requests conservative float semantics: no reciprocal approximation,
        // no reassociation.
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

        let spmv = pipeline("csr_spmv_kernel")?;
        let spmv_f16 = pipeline("csr_spmv_f16_kernel")?;
        let spmv_bf16 = pipeline("csr_spmv_bf16_kernel")?;
        let spmv_spikes = pipeline("csr_spmv_spikes_kernel")?;
        let spmv_t = pipeline("csc_spmv_t_kernel")?;
        let spmm = pipeline("csr_spmm_kernel")?;
        let scan_chunk = pipeline("scan_chunk")?;
        let scan_offsets = pipeline("scan_block_offsets")?;
        let scan_apply = pipeline("scan_apply_offsets")?;
        let lif = pipeline("lif_integrate_kernel")?;
        let fused = pipeline("fused_spmv_lif_simdgroup_kernel")?;

        // `SIMD_WIDTH` is baked into both sides of the fused kernel: the host
        // derives `rows_per_group = threadgroup / 32`, and the kernel computes
        // `row = group_id * (threads_per_group / 32) + simdgroup_id` while
        // striding `i += 32`.
        //
        // On a device with a 64-wide execution width both halves are wrong in
        // ways nothing else would catch: rows past `threadgroup / 64` per group
        // are never written at all — they silently keep their input values —
        // and the stride-32 loop double-counts every entry inside a 64-lane
        // `simd_sum`. No canary trips, because nothing is written out of
        // bounds. The result is simply wrong, quietly.
        //
        // `Backend::Metal`'s availability check only asks whether a Metal
        // device exists, which an Intel Mac with an AMD GPU satisfies. So the
        // assumption is checked here rather than assumed from the marketing
        // name of the platform.
        for (name, pipeline) in [
            ("csr_spmv_kernel", &spmv),
            ("lif_integrate_kernel", &lif),
            ("fused_spmv_lif_simdgroup_kernel", &fused),
        ] {
            let width = pipeline.threadExecutionWidth();
            if width != SIMD_WIDTH {
                return Err(leak(format!(
                    "`{name}` reports a SIMD execution width of {width}, but sparsl's \
                     kernels are written for {SIMD_WIDTH}. Refusing to run rather than \
                     produce silently wrong rows."
                )));
            }
        }

        Ok(Self {
            device,
            queue,
            spmv,
            spmv_f16,
            spmv_bf16,
            spmv_spikes,
            spmv_t,
            spmm,
            scan_chunk,
            scan_offsets,
            scan_apply,
            lif,
            fused,
        })
    }

    /// Name of the physical GPU.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    fn threadgroup_for(&self, pipeline: &ComputePipelineState) -> usize {
        PREFERRED_THREADGROUP.min(pipeline.maxTotalThreadsPerThreadgroup())
    }

    /// Threadgroup size for the fused kernel: a multiple of the SIMD width, so
    /// `threads_per_threadgroup / 32` is the exact number of rows per group.
    fn fused_threadgroup(&self) -> usize {
        let cap = self.threadgroup_for(&self.fused);
        // Round down to a whole number of SIMD groups, then clamp: the `.max(1)`
        // alone would round a sub-32 cap *up* to 32 and dispatch a threadgroup
        // larger than the pipeline permits.
        ((cap / SIMD_WIDTH).max(1) * SIMD_WIDTH).min(cap.max(1))
    }

    /// Allocate a zeroed device buffer of `len` elements of `T`.
    ///
    /// Metal rejects zero-length allocations, so empty operands get a
    /// one-element placeholder. Nothing reads it: every kernel is bounded by an
    /// explicit count, and the dispatch is skipped entirely when the count is 0.
    fn alloc<T>(&self, len: usize, what: &'static str) -> Result<Buffer, SparsePlanError> {
        let bytes = len.max(1) * std::mem::size_of::<T>();
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
        let total = len + CANARY_ELEMS;
        let bytes = total * elem;
        let buffer = self
            .device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(SparsePlanError::Allocation { what })?;
        if buffer.length() < bytes {
            return Err(SparsePlanError::Allocation { what });
        }
        let guarded = Guarded {
            buffer,
            len_bytes: len * elem,
            canary_bytes: CANARY_ELEMS * elem,
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
        let row_ptr = self.upload(&csr.row_ptr, "row_ptr")?;
        let col = self.upload(&csr.col, "col")?;
        // Uploaded as raw `u16`; the kernel declares the same memory as `half`.
        // The host encoder is cross-checked against Metal's own widening in
        // `tests/half_backend.rs`, so "same bits, two spellings" is tested
        // rather than assumed.
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
    /// Allocates per call. A scan has no persistent operator to hang buffers
    /// off the way `SparseOp` does, and pre-sizing them would mean guessing a
    /// length that changes every call.
    pub fn assoc_scan(&self, xs: &[(f32, f32)]) -> Result<Vec<(f32, f32)>, &'static str> {
        if xs.is_empty() {
            return Ok(Vec::new());
        }
        let n = xs.len();
        // The Hillis-Steele tree doubles its offset each round and reads
        // `lid + offset`; a non-power-of-two width would drop elements
        // silently rather than failing, so round down.
        let cap = self
            .scan_chunk
            .maxTotalThreadsPerThreadgroup()
            .clamp(1, SCAN_MAX_TG);
        let tptg = 1usize << (usize::BITS - 1 - cap.leading_zeros()) as usize;
        let groups = n.div_ceil(tptg);

        let flat: Vec<f32> = xs.iter().flat_map(|(a, b)| [*a, *b]).collect();
        let xs_buf = self
            .upload(&flat, "scan xs")
            .map_err(|_| "scan: could not upload input")?;
        let out = self
            .alloc::<f32>(n * 2, "scan out")
            .map_err(|_| "scan: could not allocate output")?;
        let totals = self
            .alloc::<f32>(groups * 2, "scan totals")
            .map_err(|_| "scan: could not allocate group totals")?;

        let cb = self
            .queue
            .commandBuffer()
            .ok_or("scan: Metal returned no command buffer")?;
        let enc = cb
            .computeCommandEncoder()
            .ok_or("scan: Metal returned no compute encoder")?;

        enc.setComputePipelineState(&self.scan_chunk);
        set_buf(&enc, 0, &xs_buf);
        set_buf(&enc, 1, &out);
        set_buf(&enc, 2, &totals);
        set_u32(&enc, 3, n as u32);
        // Uniform threadgroups, not `dispatchThreads`: the tree needs every
        // lane of a group present, and the kernel pads past `n` with the
        // monoid identity so the extra lanes contribute nothing.
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(tptg));

        enc.setComputePipelineState(&self.scan_offsets);
        set_buf(&enc, 0, &totals);
        set_u32(&enc, 1, groups as u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(1), size(1));

        enc.setComputePipelineState(&self.scan_apply);
        set_buf(&enc, 0, &out);
        set_buf(&enc, 1, &totals);
        set_u32(&enc, 2, n as u32);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(tptg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        let mut flat_out = vec![0.0f32; n * 2];
        read_into(&out, &mut flat_out);
        Ok(flat_out.chunks_exact(2).map(|c| (c[0], c[1])).collect())
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
    ) {
        let n = v.len();
        if n == 0 {
            return;
        }
        let v_buf = self
            .upload(v, "v")
            .expect("transient LIF upload cannot exceed a validated shape");
        let theta_buf = self.upload(theta, "theta").expect("theta upload");
        let currents_buf = self.upload(currents, "currents").expect("currents upload");
        let spikes_buf = self.alloc::<u8>(n, "spikes").expect("spikes alloc");

        let tg = self.threadgroup_for(&self.lif);
        let cb = self
            .queue
            .commandBuffer()
            .expect("Metal returned no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal returned no compute encoder");
        enc.setComputePipelineState(&self.lif);
        set_buf(&enc, 0, &v_buf);
        set_buf(&enc, 1, &theta_buf);
        set_buf(&enc, 2, &currents_buf);
        set_buf(&enc, 3, &spikes_buf);
        set_f32(&enc, 4, params.decay());
        set_f32(&enc, 5, params.v_reset());
        set_f32(&enc, 6, params.delta_theta());
        set_u32(&enc, 7, n as u32);
        enc.dispatchThreads_threadsPerThreadgroup(size(n), size(tg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        read_into(&v_buf, v);
        read_into(&theta_buf, theta);
        let mut host = vec![0u8; n];
        read_into(&spikes_buf, &mut host);
        for (dst, &src) in spikes.iter_mut().zip(host.iter()) {
            *dst = src != 0;
        }
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
        let pattern = CANARY_BITS.to_ne_bytes();
        // SAFETY: the range `[len_bytes, len_bytes + canary_bytes)` is inside
        // the allocation by construction in `alloc_guarded`, and the caller
        // holds the scratch mutex.
        unsafe {
            let base = (self.buffer.contents().as_ptr() as *mut u8).add(self.len_bytes);
            for i in 0..self.canary_bytes {
                base.add(i).write(pattern[i % 4]);
            }
        }
    }

    /// Panic if anything wrote into the sentinel region.
    fn assert_intact(&self) {
        let pattern = CANARY_BITS.to_ne_bytes();
        // SAFETY: as `arm`.
        let corrupted = unsafe {
            let base = (self.buffer.contents().as_ptr() as *const u8).add(self.len_bytes);
            (0..self.canary_bytes).any(|i| base.add(i).read() != pattern[i % 4])
        };
        if corrupted {
            self.arm();
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
}

/// A CSR operator resident on the GPU.
pub struct MetalSparse {
    device: Arc<MetalDevice>,
    row_ptr: Buffer,
    col: Buffer,
    /// The operator's values, resident for its lifetime. Uploading these per
    /// call is what made the GPU arm lose to rayon at every size.
    values: Buffer,
    /// Narrow weights — binary16 or bfloat16 — present only for an operator
    /// built by [`crate::Device::prepare_f16`] or
    /// [`crate::Device::prepare_bf16`]. When set, `values` holds the same
    /// weights widened back to f32 so `set_weights` and the CPU-comparable
    /// path stay available; the kernel reads this one.
    values_narrow: Option<Buffer>,
    /// Which format `values_narrow` holds, and therefore which kernel runs.
    precision: super::WeightPrecision,
    /// Reverse index, present only when the operator was prepared for it.
    /// `edge_idx` points into `values`, so both directions read one table.
    transpose: Option<TransposeIndex>,
    shape: SparseShape,
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
    pub fn set_weights(&mut self, weights: &[f32]) {
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
    }

    /// `y += A · x`. Lengths are checked by the caller in `SparseOp::spmv`.
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) {
        let scratch = self.scratch.lock().expect("sparsl scratch mutex poisoned");
        write_from(&scratch.x, &x[..self.shape.ncols]);
        scratch.y.write(y);

        // One operator, two kernels: the narrow one reads `values_f16` and is
        // selected by the operator's own storage, not by a caller-supplied
        // flag that could disagree with what was uploaded.
        let (pipeline, values) = match (self.precision, self.values_narrow.as_ref()) {
            (super::WeightPrecision::F16, Some(narrow)) => (&self.device.spmv_f16, narrow),
            (super::WeightPrecision::Bf16, Some(narrow)) => (&self.device.spmv_bf16, narrow),
            _ => (&self.device.spmv, &self.values),
        };
        let tg = self.device.threadgroup_for(pipeline);
        let cb = self
            .device
            .queue
            .commandBuffer()
            .expect("Metal returned no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal returned no compute encoder");
        enc.setComputePipelineState(pipeline);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        set_buf(&enc, 2, values);
        set_buf(&enc, 3, &scratch.x);
        set_buf(&enc, 4, &scratch.y.buffer);
        set_u32(&enc, 5, self.shape.nrows as u32);
        enc.dispatchThreads_threadsPerThreadgroup(size(self.shape.nrows), size(tg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        scratch.y.assert_intact();
        read_into(&scratch.y.buffer, y);
        drop(scratch);
    }

    /// `Y += A · X` for `n_vec` vectors. Lengths are checked by the caller in
    /// `SparseOp::spmm`.
    ///
    /// Grows the batch scratch when `n_vec` exceeds what it was sized for and
    /// reuses it otherwise, binding only the prefix in use. The buffers are not
    /// shrunk: a caller that alternates batch sizes should pay the larger
    /// allocation once, not on every downward step.
    pub fn spmm(&self, x: &[f32], n_vec: usize, y: &mut [f32]) -> Result<(), OpError> {
        // `csr_spmm_kernel` derives its row from `id / n_vec`, and `n_vec` is a
        // runtime value, so that is a hardware integer division in every
        // thread. At a batch of one it buys nothing and cost 0.92 ms against
        // SpMV's 0.52 ms at n = 10000. Dispatching to the scalar kernel also
        // makes the documented bit-identity structural on this backend rather
        // than a property the two kernels happen to share.
        if n_vec == 1 {
            self.spmv(x, y);
            return Ok(());
        }
        let mut scratch = self.scratch.lock().expect("sparsl scratch mutex poisoned");
        let need_x = self.shape.ncols * n_vec;
        let need_y = self.shape.nrows * n_vec;

        let grow = match &scratch.batch {
            Some(b) => b.n_vec < n_vec,
            None => true,
        };
        if grow {
            let x_buf = self
                .device
                .alloc::<f32>(need_x.max(1), "spmm x")
                .map_err(|_| OpError::Length {
                    what: "spmm x scratch",
                    expected: need_x,
                    got: 0,
                })?;
            let y_buf = self
                .device
                .alloc_guarded::<f32>(need_y.max(1), "spmm y")
                .map_err(|_| OpError::Length {
                    what: "spmm y scratch",
                    expected: need_y,
                    got: 0,
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

        let tg = self.device.threadgroup_for(&self.device.spmm);
        let cb = self
            .device
            .queue
            .commandBuffer()
            .expect("Metal returned no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal returned no compute encoder");
        enc.setComputePipelineState(&self.device.spmm);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        set_buf(&enc, 2, &self.values);
        set_buf(&enc, 3, &batch.x);
        set_buf(&enc, 4, &batch.y.buffer);
        set_u32(&enc, 5, self.shape.nrows as u32);
        set_u32(&enc, 6, n_vec as u32);
        enc.dispatchThreads_threadsPerThreadgroup(size(need_y), size(tg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        batch.y.assert_intact();
        read_into(&batch.y.buffer, &mut y[..need_y]);
        drop(scratch);
        Ok(())
    }

    /// The storage this operator's weights occupy.
    pub fn weight_precision(&self) -> super::WeightPrecision {
        self.precision
    }

    /// `y += A · s` for a bitpacked spike vector. Lengths checked by the caller.
    pub fn spmv_spikes(&self, spikes: &[u32], y: &mut [f32]) {
        let scratch = self.scratch.lock().expect("sparsl scratch mutex poisoned");
        let words = crate::spikes::packed_len(self.shape.ncols);
        write_from(&scratch.x_spikes, &spikes[..words]);
        scratch.y.write(y);

        let tg = self.device.threadgroup_for(&self.device.spmv_spikes);
        let cb = self
            .device
            .queue
            .commandBuffer()
            .expect("Metal returned no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal returned no compute encoder");
        enc.setComputePipelineState(&self.device.spmv_spikes);
        set_buf(&enc, 0, &self.row_ptr);
        set_buf(&enc, 1, &self.col);
        // Deliberately `values`, never `values_narrow`: the spike path is
        // bit-identical to the dense one, and reading quantised weights here
        // would quietly make that false.
        set_buf(&enc, 2, &self.values);
        set_buf(&enc, 3, &scratch.x_spikes);
        set_buf(&enc, 4, &scratch.y.buffer);
        set_u32(&enc, 5, self.shape.nrows as u32);
        enc.dispatchThreads_threadsPerThreadgroup(size(self.shape.nrows), size(tg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        scratch.y.assert_intact();
        read_into(&scratch.y.buffer, y);
        drop(scratch);
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
        let scratch = self.scratch.lock().expect("sparsl scratch mutex poisoned");
        let (xt, yt) = match (scratch.xt.as_ref(), scratch.yt.as_ref()) {
            (Some(xt), Some(yt)) => (xt, yt),
            // Unreachable: scratch and index are allocated together in
            // `prepare`. Refusing beats unwrapping if that ever stops holding.
            _ => return Err(OpError::TransposeNotPrepared),
        };
        write_from(xt, &x[..self.shape.nrows]);
        yt.write(y);

        let tg = self.device.threadgroup_for(&self.device.spmv_t);
        let cb = self
            .device
            .queue
            .commandBuffer()
            .expect("Metal returned no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal returned no compute encoder");
        enc.setComputePipelineState(&self.device.spmv_t);
        set_buf(&enc, 0, &idx.col_ptr);
        set_buf(&enc, 1, &idx.row);
        set_buf(&enc, 2, &idx.edge_idx);
        set_buf(&enc, 3, &self.values);
        set_buf(&enc, 4, xt);
        set_buf(&enc, 5, &yt.buffer);
        set_u32(&enc, 6, self.shape.ncols as u32);
        enc.dispatchThreads_threadsPerThreadgroup(size(self.shape.ncols), size(tg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

        yt.assert_intact();
        read_into(&yt.buffer, y);
        drop(scratch);
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
    ) {
        let mut scratch = self.scratch.lock().expect("sparsl scratch mutex poisoned");
        write_from(&scratch.x, &x[..self.shape.ncols]);
        scratch.v.write(v);
        scratch.theta.write(theta);

        // Uniform dispatch: the kernel derives its row from
        // `threads_per_threadgroup`, which non-uniform dispatch would shrink for
        // the tail group and silently shift every row index in it.
        let tg = self.device.fused_threadgroup();
        let rows_per_group = (tg / SIMD_WIDTH).max(1);
        let groups = self.shape.nrows.div_ceil(rows_per_group);

        let cb = self
            .device
            .queue
            .commandBuffer()
            .expect("Metal returned no command buffer");
        let enc = cb
            .computeCommandEncoder()
            .expect("Metal returned no compute encoder");
        enc.setComputePipelineState(&self.device.fused);
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
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(groups), size(tg));
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();

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
        drop(scratch);
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
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
    // Exclusive access is established differently at each of the four call
    // sites, so it is spelled out rather than asserted generically:
    //   - `MetalSparse::spmv` / `fused_spmv_lif` hold the scratch mutex.
    //   - `MetalSparse::set_weights` takes `&mut self`, which excludes any
    //     concurrent dispatch, and every dispatch calls `wait_until_completed`
    //     before returning, so no GPU work is reading `values` either.
    //   - `MetalDevice::lif_integrate` writes only buffers it allocated inside
    //     the same call, which no other thread can name.
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
}
