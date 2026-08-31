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
//! metal-rs marks its handles `Send + Sync`, so a [`crate::SparseOp`] can be
//! shared across threads. The scratch buffers cannot be: two threads writing
//! the same host-visible allocation is a data race regardless of what Metal
//! guarantees about command queues. All scratch lives behind a [`Mutex`], which
//! serialises dispatch per operator. Parallelism across *operators* is
//! unaffected.

use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device as MtlDevice, MTLResourceOptions, MTLSize,
};

use super::{LifParams, SparsePlanError, SparseShape};
use crate::sparse::Csr;

const KERNEL_SOURCE: &str = include_str!("../kernels/spmv.metal");

/// Threads per threadgroup for the one-thread-per-row kernels, capped by the
/// pipeline's own limit.
const PREFERRED_THREADGROUP: u64 = 256;

/// SIMD-group width on Apple silicon. The fused kernel hard-codes this in its
/// strided load and its `simd_sum`, so it is a contract, not a tunable.
const SIMD_WIDTH: u64 = 32;

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
    lif: ComputePipelineState,
    fused: ComputePipelineState,
}

impl MetalDevice {
    fn open_uncached() -> Result<Self, &'static str> {
        let device = MtlDevice::system_default().ok_or("no Metal device on this system")?;
        let queue = device.new_command_queue();

        let options = metal::CompileOptions::new();
        // Requests conservative float semantics: no reciprocal approximation,
        // no reassociation.
        //
        // It does NOT stop the compiler contracting `a * b + c` into a single
        // `fma`, and that is measured rather than assumed —
        // `tests/fma_contraction.rs` compares every non-spiking GPU membrane
        // against both roundings and finds the fused one used. `fast_math` is
        // also deprecated in recent Metal in favour of `mathMode`, which
        // metal-rs 0.29 does not expose.
        //
        // The consequence is a real one, not a curiosity: one rounding instead
        // of two shifts the membrane by up to an ulp, and that value is then
        // compared against a threshold — so contraction can flip a spike, not
        // merely perturb a float. Cross-backend comparisons must budget for it;
        // see `tolerance_for_elementwise`.
        options.set_fast_math_enabled(false);
        let library = device
            .new_library_with_source(KERNEL_SOURCE, &options)
            .map_err(|e| leak(format!("sparsl MSL failed to compile: {e}")))?;

        let pipeline = |name: &str| -> Result<ComputePipelineState, &'static str> {
            let function = library
                .get_function(name, None)
                .map_err(|e| leak(format!("kernel `{name}` not found in library: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| leak(format!("pipeline for `{name}` failed to build: {e}")))
        };

        let spmv = pipeline("csr_spmv_kernel")?;
        let lif = pipeline("lif_integrate_kernel")?;
        let fused = pipeline("fused_spmv_lif_simdgroup_kernel")?;

        Ok(Self {
            device,
            queue,
            spmv,
            lif,
            fused,
        })
    }

    /// Name of the physical GPU.
    pub fn name(&self) -> String {
        self.device.name().to_string()
    }

    fn threadgroup_for(&self, pipeline: &ComputePipelineState) -> u64 {
        PREFERRED_THREADGROUP.min(pipeline.max_total_threads_per_threadgroup())
    }

    /// Threadgroup size for the fused kernel: a multiple of the SIMD width, so
    /// `threads_per_threadgroup / 32` is the exact number of rows per group.
    fn fused_threadgroup(&self) -> u64 {
        let cap = self.threadgroup_for(&self.fused);
        (cap / SIMD_WIDTH).max(1) * SIMD_WIDTH
    }

    /// Allocate a zeroed device buffer of `len` elements of `T`.
    ///
    /// Metal rejects zero-length allocations, so empty operands get a
    /// one-element placeholder. Nothing reads it: every kernel is bounded by an
    /// explicit count, and the dispatch is skipped entirely when the count is 0.
    fn alloc<T>(&self, len: usize, what: &'static str) -> Result<Buffer, SparsePlanError> {
        let bytes = (len.max(1) * std::mem::size_of::<T>()) as u64;
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
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
        let bytes = (total * elem) as u64;
        let buffer = self
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
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
        let bytes = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
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
    ) -> Result<MetalSparse, SparsePlanError> {
        let row_ptr = self.upload(&csr.row_ptr, "row_ptr")?;
        let col = self.upload(&csr.col, "col")?;
        let values = self.upload(weights, "values")?;
        let scratch = Scratch {
            x: self.alloc::<f32>(shape.ncols, "x")?,
            y: self.alloc_guarded::<f32>(shape.nrows, "y")?,
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
            shape,
            scratch: Mutex::new(scratch),
        })
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
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.lif);
        enc.set_buffer(0, Some(&v_buf), 0);
        enc.set_buffer(1, Some(&theta_buf), 0);
        enc.set_buffer(2, Some(&currents_buf), 0);
        enc.set_buffer(3, Some(&spikes_buf), 0);
        set_f32(enc, 4, params.decay());
        set_f32(enc, 5, params.v_reset());
        set_f32(enc, 6, params.delta_theta());
        set_u32(enc, 7, n as u32);
        enc.dispatch_threads(size(n as u64), size(tg));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

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
    /// Fill the sentinel region. Called once at allocation and again after any
    /// detected trip, so a second dispatch cannot report a stale failure.
    fn arm(&self) {
        let pattern = CANARY_BITS.to_ne_bytes();
        // SAFETY: the range `[len_bytes, len_bytes + canary_bytes)` is inside
        // the allocation by construction in `alloc_guarded`, and the caller
        // holds the scratch mutex.
        unsafe {
            let base = (self.buffer.contents() as *mut u8).add(self.len_bytes);
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
            let base = (self.buffer.contents() as *const u8).add(self.len_bytes);
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
    y: Guarded,
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
    shape: SparseShape,
    scratch: Mutex<Scratch>,
}

impl MetalSparse {
    /// Overwrite the resident values. Length is checked by the caller.
    pub fn set_weights(&mut self, weights: &[f32]) {
        write_from(&self.values, weights);
    }

    /// `y += A · x`. Lengths are checked by the caller in `SparseOp::spmv`.
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) {
        let scratch = self.scratch.lock().expect("sparsl scratch mutex poisoned");
        write_from(&scratch.x, &x[..self.shape.ncols]);
        write_from(&scratch.y.buffer, y);

        let tg = self.device.threadgroup_for(&self.device.spmv);
        let cb = self.device.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.device.spmv);
        enc.set_buffer(0, Some(&self.row_ptr), 0);
        enc.set_buffer(1, Some(&self.col), 0);
        enc.set_buffer(2, Some(&self.values), 0);
        enc.set_buffer(3, Some(&scratch.x), 0);
        enc.set_buffer(4, Some(&scratch.y.buffer), 0);
        set_u32(enc, 5, self.shape.nrows as u32);
        enc.dispatch_threads(size(self.shape.nrows as u64), size(tg));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        scratch.y.assert_intact();
        read_into(&scratch.y.buffer, y);
        drop(scratch);
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
        write_from(&scratch.v.buffer, v);
        write_from(&scratch.theta.buffer, theta);

        // Uniform dispatch: the kernel derives its row from
        // `threads_per_threadgroup`, which non-uniform dispatch would shrink for
        // the tail group and silently shift every row index in it.
        let tg = self.device.fused_threadgroup();
        let rows_per_group = (tg / SIMD_WIDTH).max(1);
        let groups = (self.shape.nrows as u64).div_ceil(rows_per_group);

        let cb = self.device.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.device.fused);
        enc.set_buffer(0, Some(&self.row_ptr), 0);
        enc.set_buffer(1, Some(&self.col), 0);
        enc.set_buffer(2, Some(&self.values), 0);
        enc.set_buffer(3, Some(&scratch.x), 0);
        enc.set_buffer(4, Some(&scratch.v.buffer), 0);
        enc.set_buffer(5, Some(&scratch.theta.buffer), 0);
        enc.set_buffer(6, Some(&scratch.spikes.buffer), 0);
        set_f32(enc, 7, params.decay());
        set_f32(enc, 8, params.v_reset());
        set_f32(enc, 9, params.delta_theta());
        set_u32(enc, 10, self.shape.nrows as u32);
        enc.dispatch_thread_groups(size(groups), size(tg));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();

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

fn size(width: u64) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

fn set_f32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: f32) {
    enc.set_bytes(
        index,
        std::mem::size_of::<f32>() as u64,
        &value as *const f32 as *const c_void,
    );
}

fn set_u32(enc: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    enc.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &value as *const u32 as *const c_void,
    );
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
    let bytes = std::mem::size_of_val(src) as u64;
    assert!(
        buffer.length() >= bytes,
        "sparsl: scratch buffer holds {} bytes, tried to write {bytes}",
        buffer.length()
    );
    // SAFETY: `StorageModeShared` buffers are host-visible for the lifetime of
    // the allocation; the length assertion above proves the destination range
    // is inside it; `T: Copy` has no drop glue; and the caller holds the
    // scratch mutex, so no other thread is touching this range.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), buffer.contents() as *mut T, src.len());
    }
}

fn read_into<T: Copy>(buffer: &Buffer, dst: &mut [T]) {
    read_buffer_into(buffer, dst);
}

fn read_buffer_into<T: Copy>(buffer: &Buffer, dst: &mut [T]) {
    if dst.is_empty() {
        return;
    }
    let bytes = std::mem::size_of_val(dst) as u64;
    assert!(
        buffer.length() >= bytes,
        "sparsl: scratch buffer holds {} bytes, tried to read {bytes}",
        buffer.length()
    );
    // SAFETY: as `write_from`, in the opposite direction. The dispatch that
    // produced these bytes was waited on before this call.
    unsafe {
        std::ptr::copy_nonoverlapping(buffer.contents() as *const T, dst.as_mut_ptr(), dst.len());
    }
}
