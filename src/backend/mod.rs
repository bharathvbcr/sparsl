//! Execution backends for sparse and LIF kernels.
//!
//! # The honesty invariant
//!
//! A [`Device`] can only be constructed for a backend that can actually
//! execute, and [`Device::label`] reports the substrate that *ran*, not the one
//! that was requested. There is no fallback path: asking for an unavailable
//! backend returns [`BackendUnavailable`] rather than quietly running somewhere
//! else under the original label.
//!
//! This exists because the code this crate was extracted from carried a
//! `use_gpu: bool` that no dispatch path ever read. A "GPU" handle and a "CPU"
//! handle executed byte-identical rayon code, benchmarks reported ~1.00x
//! speedups as real cross-substrate results, and reports labelled CPU numbers
//! as GPU numbers. Every guard in this module exists to make that specific
//! failure unrepresentable, and `tests/honesty.rs` locks it down.
//!
//! # Validate once, then trust
//!
//! GPU kernels index `x[col[i]]` with no range check — a bounds check per
//! non-zero would cost more than the multiply. That is only safe because
//! [`SparseOp::prepare`] validates every stored column index against `ncols`
//! *before* anything is uploaded, and a `SparseOp` is the only way to reach a
//! sparse kernel. An unvalidated [`Csr`] cannot reach device code.

use core::fmt;

use crate::sparse::Csr;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

pub mod cuda;

// ---------------------------------------------------------------------------
// Backend identity and availability
// ---------------------------------------------------------------------------

/// Which substrate a handle actually executes on.
///
/// Deliberately an enum rather than a `bool` or a string: a bool invites the
/// "flag is set but never read" failure this module exists to prevent, and a
/// string invites a report that prints a label nothing verified.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Backend {
    /// Single-threaded CPU. Always available; the determinism reference.
    CpuSequential,
    /// Multi-threaded CPU via rayon. Always available.
    ///
    /// Bit-identical to [`Backend::CpuSequential`]: every output element is
    /// computed by exactly one thread from a fixed input order, so no
    /// cross-thread reduction can reorder a sum. `tests/honesty.rs` asserts it.
    CpuParallel,
    /// Native Metal compute on Apple silicon. Requires the `metal` feature and
    /// a Metal device at runtime.
    Metal,
    /// NVIDIA CUDA. Not implemented; see [`cuda`] for why it is declared but
    /// never available.
    Cuda,
}

impl Backend {
    /// Every backend this crate knows about, available or not.
    pub const ALL: [Backend; 4] = [
        Backend::CpuSequential,
        Backend::CpuParallel,
        Backend::Metal,
        Backend::Cuda,
    ];

    /// Human-readable label. Report generators must use this rather than a
    /// hardcoded column heading, so a table cannot outlive the thing it names.
    pub const fn label(self) -> &'static str {
        match self {
            Backend::CpuSequential => "CPU sequential",
            Backend::CpuParallel => "CPU parallel (rayon)",
            Backend::Metal => "Metal GPU",
            Backend::Cuda => "CUDA GPU",
        }
    }

    /// Whether this backend can execute work in this build, on this machine.
    pub fn is_available(self) -> bool {
        self.unavailable_reason().is_none()
    }

    /// Why the backend cannot execute, or `None` if it can.
    ///
    /// For [`Backend::Metal`] this probes for a real device, so it is not free
    /// and its answer can differ between machines running the same binary.
    pub fn unavailable_reason(self) -> Option<&'static str> {
        match self {
            Backend::CpuSequential | Backend::CpuParallel => None,
            Backend::Metal => metal_unavailable_reason(),
            Backend::Cuda => Some(cuda::UNAVAILABLE_REASON),
        }
    }

    /// True for substrates that execute on a GPU.
    pub const fn is_gpu(self) -> bool {
        matches!(self, Backend::Metal | Backend::Cuda)
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn metal_unavailable_reason() -> Option<&'static str> {
    metal::unavailable_reason()
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn metal_unavailable_reason() -> Option<&'static str> {
    if cfg!(target_os = "macos") {
        Some("sparsl was built without the `metal` cargo feature")
    } else {
        Some("Metal is only available on macOS")
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// Returned instead of silently falling back to a different substrate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendUnavailable {
    pub requested: Backend,
    pub reason: &'static str,
}

impl fmt::Display for BackendUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "backend `{}` is unavailable: {}",
            self.requested.label(),
            self.reason
        )
    }
}

impl std::error::Error for BackendUnavailable {}

/// Backends that can actually execute right now, in a stable order.
///
/// A benchmark driven by this can never emit a two-arm table whose arms are
/// secretly the same substrate.
pub fn available_backends() -> Vec<Backend> {
    Backend::ALL
        .into_iter()
        .filter(|b| b.is_available())
        .collect()
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Leaky integrate-and-fire step parameters, validated on construction.
///
/// `v ← v · decay + current`; on `v >= theta` the cell spikes, `v` resets to
/// `v_reset` and `theta` rises by `delta_theta`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LifParams {
    decay: f32,
    v_reset: f32,
    delta_theta: f32,
}

/// Why a [`LifParams`] was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LifParamsError {
    /// A parameter was NaN or infinite.
    ///
    /// Rejected at construction rather than allowed to propagate: a NaN
    /// `decay` silently turns every membrane into NaN, every `v >= theta`
    /// comparison false, and the network into one that never spikes again —
    /// which looks like a modelling result rather than a bad input.
    NotFinite { field: &'static str },
}

impl fmt::Display for LifParamsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFinite { field } => write!(f, "LIF parameter `{field}` must be finite"),
        }
    }
}

impl std::error::Error for LifParamsError {}

impl LifParams {
    /// Validate and build. All three parameters must be finite.
    pub fn new(decay: f32, v_reset: f32, delta_theta: f32) -> Result<Self, LifParamsError> {
        for (field, value) in [
            ("decay", decay),
            ("v_reset", v_reset),
            ("delta_theta", delta_theta),
        ] {
            if !value.is_finite() {
                return Err(LifParamsError::NotFinite { field });
            }
        }
        Ok(Self {
            decay,
            v_reset,
            delta_theta,
        })
    }

    pub const fn decay(self) -> f32 {
        self.decay
    }
    pub const fn v_reset(self) -> f32 {
        self.v_reset
    }
    pub const fn delta_theta(self) -> f32 {
        self.delta_theta
    }
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Why a [`Csr`] could not be prepared for execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparsePlanError {
    /// `row_ptr` was empty. A zero-row graph still needs `row_ptr == [0]`.
    EmptyRowPtr,
    /// `row_ptr[0]` was not `0`.
    NonZeroStart { start: u32 },
    /// `row_ptr` decreased, so some row had negative length.
    NotMonotonic { index: usize },
    /// `row_ptr.last()` disagreed with `col.len()`.
    NnzMismatch { row_ptr_end: u32, col_len: usize },
    /// A stored column index was `>= ncols`.
    ///
    /// On CPU this would be a bounds panic. On GPU it is an out-of-bounds read
    /// of arbitrary device memory, which is why it is rejected before upload
    /// rather than checked per non-zero inside the kernel.
    ColumnOutOfRange { edge: usize, col: u32, ncols: usize },
    /// The shape does not fit the `u32` indices the GPU kernels use.
    TooLarge { what: &'static str, value: usize },
    /// `weights` did not have one entry per stored non-zero.
    WeightsLen { expected: usize, got: usize },
    /// The device rejected an allocation.
    Allocation { what: &'static str },
}

impl fmt::Display for SparsePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyRowPtr => write!(f, "CSR row_ptr must be non-empty"),
            Self::NonZeroStart { start } => write!(f, "CSR row_ptr must start at 0, got {start}"),
            Self::NotMonotonic { index } => write!(f, "CSR row_ptr not monotonic at index {index}"),
            Self::NnzMismatch {
                row_ptr_end,
                col_len,
            } => write!(
                f,
                "CSR row_ptr end ({row_ptr_end}) != col.len() ({col_len})"
            ),
            Self::ColumnOutOfRange { edge, col, ncols } => write!(
                f,
                "CSR column index {col} at edge {edge} is out of range (ncols = {ncols})"
            ),
            Self::TooLarge { what, value } => {
                write!(
                    f,
                    "{what} = {value} exceeds the u32 index range of the GPU kernels"
                )
            }
            Self::WeightsLen { expected, got } => write!(
                f,
                "weights must have one entry per non-zero: expected {expected}, got {got}"
            ),
            Self::Allocation { what } => write!(f, "device allocation failed for {what}"),
        }
    }
}

impl std::error::Error for SparsePlanError {}

/// Why an operation call was rejected.
///
/// Every variant is a caller-data problem, reported rather than asserted, so a
/// fuzzer or a long-running process gets an error it can handle instead of an
/// abort.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OpError {
    /// A slice length did not match the prepared shape.
    Length {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A slice was shorter than the minimum the shape requires.
    TooShort {
        what: &'static str,
        min: usize,
        got: usize,
    },
}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Length {
                what,
                expected,
                got,
            } => write!(f, "`{what}` must have length {expected}, got {got}"),
            Self::TooShort { what, min, got } => {
                write!(f, "`{what}` must have length >= {min}, got {got}")
            }
        }
    }
}

impl std::error::Error for OpError {}

fn require_len(what: &'static str, got: usize, expected: usize) -> Result<(), OpError> {
    if got == expected {
        Ok(())
    } else {
        Err(OpError::Length {
            what,
            expected,
            got,
        })
    }
}

fn require_min_len(what: &'static str, got: usize, min: usize) -> Result<(), OpError> {
    if got >= min {
        Ok(())
    } else {
        Err(OpError::TooShort { what, min, got })
    }
}

// ---------------------------------------------------------------------------
// Shape
// ---------------------------------------------------------------------------

/// Validated dimensions of a prepared sparse operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseShape {
    pub nrows: usize,
    pub ncols: usize,
    pub nnz: usize,
}

impl SparseShape {
    /// Mean stored non-zeros per row, rounded up. Feeds
    /// [`tolerance_for_nnz_per_row`].
    pub fn nnz_per_row(self) -> usize {
        if self.nrows == 0 {
            0
        } else {
            self.nnz.div_ceil(self.nrows)
        }
    }
}

/// Upper bound on the CPU/GPU disagreement for one SpMV row.
///
/// This is the textbook forward error bound for recursive summation —
/// `n · eps · max|term|` — with a factor of 8 of headroom, **not** a curve
/// fitted to measurements. It is deliberately conservative: a tolerance tuned
/// tight against one machine's observed error becomes a flaky test on the next.
///
/// Measured against it on an Apple M5 Pro (uniform random weights and `x` in
/// `[-0.5, 0.5]`, so `max|term| = 0.25`): 1000 nnz/row gave a real deviation of
/// 3.3e-6 against a bound of 2.4e-4, i.e. ~70x of margin.
///
/// The reason a fixed tolerance does not work here is that the error grows with
/// row density. A constant `1e-4` passes trivially on a 3x3 fixture and stops
/// meaning anything by the time rows hold a thousand entries.
/// Upper bound on the CPU/GPU disagreement for an elementwise kernel.
///
/// An elementwise update has no summation and therefore no reduction order to
/// disagree about, so the intuition is that both substrates must agree exactly.
/// They do not. Metal contracts `v * decay + current` into a single `fma`,
/// rounding once where the CPU rounds twice, which moves the result by up to an
/// ulp. `tests/fma_contraction.rs` pins this behaviour.
///
/// Two ulps of the larger operand covers the omitted rounding with margin.
///
/// The reason this matters more than an ulp usually does: the value it perturbs
/// is immediately compared against a threshold. A membrane sitting within an ulp
/// of `theta` can spike on one substrate and not the other — a boolean
/// difference out of a rounding difference. Comparisons across substrates must
/// treat spikes in that band as legitimately ambiguous.
pub fn tolerance_for_elementwise(max_abs_result: f32) -> f32 {
    let magnitude = max_abs_result.abs().max(f32::MIN_POSITIVE);
    2.0 * f32::EPSILON * magnitude
}

pub fn tolerance_for_nnz_per_row(nnz_per_row: usize, max_abs_term: f32) -> f32 {
    let n = nnz_per_row.max(1) as f32;
    let magnitude = max_abs_term.abs().max(f32::MIN_POSITIVE);
    8.0 * f32::EPSILON * n * magnitude
}

// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum DeviceInner {
    Cpu,
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(std::sync::Arc<metal::MetalDevice>),
}

impl DeviceInner {
    /// Whether this substrate is the one `backend` names.
    ///
    /// The check that makes a mislabelled handle impossible rather than merely
    /// unlikely: a CPU inner under a GPU label fails it.
    fn matches(&self, backend: Backend) -> bool {
        match (self, backend) {
            (DeviceInner::Cpu, Backend::CpuSequential | Backend::CpuParallel) => true,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            (DeviceInner::Metal(_), Backend::Metal) => true,
            _ => false,
        }
    }
}

/// A handle to a substrate that is known to be able to execute.
///
/// Cloning is cheap; GPU state is shared, not duplicated.
#[derive(Clone)]
pub struct Device {
    backend: Backend,
    inner: DeviceInner,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl Default for Device {
    fn default() -> Self {
        Self::cpu_parallel()
    }
}

impl Device {
    /// Multi-threaded CPU. Infallible.
    pub fn cpu_parallel() -> Self {
        Self {
            backend: Backend::CpuParallel,
            inner: DeviceInner::Cpu,
        }
    }

    /// Single-threaded CPU. Infallible; the determinism reference.
    pub fn cpu_sequential() -> Self {
        Self {
            backend: Backend::CpuSequential,
            inner: DeviceInner::Cpu,
        }
    }

    /// Open a device for `backend`, or explain why it cannot execute.
    ///
    /// Never falls back to a different substrate. A caller that wants a
    /// fallback must ask for it and relabel its output.
    ///
    /// # Why availability is checked once, here
    ///
    /// An earlier shape of this function gave every backend its own
    /// `Ok(..) / Err(..)` arm. A mutation that replaced one arm's `Err` with
    /// `Ok(Self::cpu_parallel())` — reintroducing the exact silent-fallback bug
    /// this crate was built to prevent — passed the entire test suite, because
    /// on a machine where Metal works that error arm is unreachable and no test
    /// on such a machine can execute it.
    ///
    /// Untestable code is not made safe by more tests. So there is now one gate
    /// that every backend passes through, and it is exercised on every machine
    /// by whichever backends happen to be unavailable there — `Backend::Cuda`
    /// at minimum, which is unavailable everywhere by construction. The `open_*`
    /// helpers below cannot fabricate a fallback because they do not get to
    /// choose the label: they return an inner substrate, and `backend` is set
    /// from the caller's request, then checked against it.
    pub fn try_new(backend: Backend) -> Result<Self, BackendUnavailable> {
        if let Some(reason) = backend.unavailable_reason() {
            return Err(BackendUnavailable {
                requested: backend,
                reason,
            });
        }

        let inner = match backend {
            Backend::CpuSequential | Backend::CpuParallel => DeviceInner::Cpu,
            Backend::Metal => Self::open_metal()?,
            // Unreachable: `Backend::Cuda` has no dispatch, so its
            // `unavailable_reason` is unconditionally `Some` and the gate above
            // has already returned. Kept as a hard stop so that implementing
            // CUDA without also giving it a real availability probe fails here
            // rather than producing a CUDA-labelled CPU device.
            Backend::Cuda => {
                return Err(BackendUnavailable {
                    requested: Backend::Cuda,
                    reason: cuda::UNAVAILABLE_REASON,
                })
            }
        };

        // A backend reported itself available, then opened as something else.
        // That is the silent-fallback bug, and it is refused rather than
        // labelled.
        if !inner.matches(backend) {
            return Err(BackendUnavailable {
                requested: backend,
                reason: "backend opened a different substrate than requested",
            });
        }

        Ok(Self { backend, inner })
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    fn open_metal() -> Result<DeviceInner, BackendUnavailable> {
        metal::shared_device()
            .map(DeviceInner::Metal)
            .map_err(|reason| BackendUnavailable {
                requested: Backend::Metal,
                reason,
            })
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    fn open_metal() -> Result<DeviceInner, BackendUnavailable> {
        Err(BackendUnavailable {
            requested: Backend::Metal,
            reason: metal_unavailable_reason().unwrap_or("Metal is unavailable"),
        })
    }

    /// The substrate this handle actually executes on.
    pub const fn backend(&self) -> Backend {
        self.backend
    }

    /// Truthful label for report generators.
    pub const fn label(&self) -> &'static str {
        self.backend.label()
    }

    /// Name of the physical device, when the backend can report one.
    pub fn device_name(&self) -> Option<String> {
        match &self.inner {
            DeviceInner::Cpu => None,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            DeviceInner::Metal(d) => Some(d.name()),
        }
    }

    /// Validate a CSR, bind it to this device and upload it with its values.
    ///
    /// # Why the weights belong here and not on each call
    ///
    /// An earlier version took `weights` as a per-call argument, which read
    /// naturally and was measurably wrong: on a 20M-non-zero operator it forced
    /// an 80 MB host-to-device copy before every dispatch, and the Metal
    /// backend lost to rayon at every size purely on that copy. Weights change
    /// on the timescale of learning, not of dispatch, so they live with the
    /// operator and are replaced explicitly via [`SparseOp::set_weights`].
    pub fn prepare(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<SparseOp, SparsePlanError> {
        SparseOp::prepare(self.clone(), csr, ncols, weights)
    }

    /// LIF membrane integrate over dense per-cell state.
    ///
    /// `spikes[i]` is set to whether cell `i` fired this step.
    pub fn lif_integrate(
        &self,
        v: &mut [f32],
        theta: &mut [f32],
        currents: &[f32],
        spikes: &mut [bool],
        params: LifParams,
    ) -> Result<(), OpError> {
        let n = v.len();
        require_len("theta", theta.len(), n)?;
        require_len("currents", currents.len(), n)?;
        require_len("spikes", spikes.len(), n)?;
        if n == 0 {
            return Ok(());
        }
        match &self.inner {
            DeviceInner::Cpu => {
                if self.backend == Backend::CpuParallel {
                    cpu_lif_integrate_parallel(v, theta, currents, spikes, params);
                } else {
                    cpu_lif_integrate_sequential(v, theta, currents, spikes, params);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            DeviceInner::Metal(d) => {
                d.lif_integrate(v, theta, currents, spikes, params);
                Ok(())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Prepared sparse operator
// ---------------------------------------------------------------------------

#[allow(clippy::large_enum_variant)]
enum OpResident {
    Cpu {
        csr: Csr,
        weights: Vec<f32>,
    },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::MetalSparse),
}

/// A CSR operator validated against `ncols` and bound to a device.
///
/// Preparing is the only way to reach a sparse kernel, which is what makes the
/// unchecked `x[col[i]]` indexing inside the GPU kernels sound.
pub struct SparseOp {
    device: Device,
    shape: SparseShape,
    resident: OpResident,
}

impl fmt::Debug for SparseOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseOp")
            .field("backend", &self.device.backend)
            .field("shape", &self.shape)
            .finish_non_exhaustive()
    }
}

impl SparseOp {
    fn prepare(
        device: Device,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<Self, SparsePlanError> {
        let shape = validate_csr(csr, ncols)?;
        if weights.len() != shape.nnz {
            return Err(SparsePlanError::WeightsLen {
                expected: shape.nnz,
                got: weights.len(),
            });
        }
        let resident = match &device.inner {
            DeviceInner::Cpu => OpResident::Cpu {
                csr: csr.clone(),
                weights: weights.to_vec(),
            },
            #[cfg(all(target_os = "macos", feature = "metal"))]
            DeviceInner::Metal(d) => OpResident::Metal(d.prepare(csr, shape, weights)?),
        };
        Ok(Self {
            device,
            shape,
            resident,
        })
    }

    /// The substrate this operator executes on.
    pub const fn backend(&self) -> Backend {
        self.device.backend
    }

    /// Truthful label for report generators.
    pub const fn label(&self) -> &'static str {
        self.device.backend.label()
    }

    pub const fn shape(&self) -> SparseShape {
        self.shape
    }

    /// Replace the stored values, keeping the connectivity and its validation.
    ///
    /// The path a learning rule uses: connectivity is fixed, values move.
    pub fn set_weights(&mut self, weights: &[f32]) -> Result<(), OpError> {
        require_len("weights", weights.len(), self.shape.nnz)?;
        match &mut self.resident {
            OpResident::Cpu {
                weights: stored, ..
            } => {
                stored.clear();
                stored.extend_from_slice(weights);
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.set_weights(weights),
        }
        Ok(())
    }

    /// `y += A · x`, using the values this operator holds.
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) -> Result<(), OpError> {
        require_min_len("x", x.len(), self.shape.ncols)?;
        require_len("y", y.len(), self.shape.nrows)?;
        if self.shape.nrows == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { csr, weights } => {
                if self.device.backend == Backend::CpuParallel {
                    cpu_spmv_parallel(csr, weights, x, y);
                } else {
                    cpu_spmv_sequential(csr, weights, x, y);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => {
                op.spmv(x, y);
                Ok(())
            }
        }
    }

    /// Fused `current = A · x` followed by the LIF membrane update, without
    /// materialising the current vector.
    ///
    /// On Metal this reduces each row with `simd_sum`, a tree reduction whose
    /// order differs from both CPU arms and from [`SparseOp::spmv`]. Expect it
    /// to land further from the CPU reference than plain SpMV does; size any
    /// comparison with [`tolerance_for_nnz_per_row`].
    pub fn fused_spmv_lif(
        &self,
        x: &[f32],
        v: &mut [f32],
        theta: &mut [f32],
        spikes: &mut [bool],
        params: LifParams,
    ) -> Result<(), OpError> {
        require_min_len("x", x.len(), self.shape.ncols)?;
        require_len("v", v.len(), self.shape.nrows)?;
        require_len("theta", theta.len(), self.shape.nrows)?;
        require_len("spikes", spikes.len(), self.shape.nrows)?;
        if self.shape.nrows == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { csr, weights } => {
                if self.device.backend == Backend::CpuParallel {
                    cpu_fused_parallel(csr, weights, x, v, theta, spikes, params);
                } else {
                    cpu_fused_sequential(csr, weights, x, v, theta, spikes, params);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => {
                op.fused_spmv_lif(x, v, theta, spikes, params);
                Ok(())
            }
        }
    }
}

/// Full structural validation of a CSR against a declared column count.
///
/// Re-validates everything [`Csr::from_parts`] checks, because
/// `Csr::from_parts_unchecked` exists and a caller may have used it, plus the
/// column-range check that `Csr` itself cannot make without knowing `ncols`.
fn validate_csr(csr: &Csr, ncols: usize) -> Result<SparseShape, SparsePlanError> {
    if csr.row_ptr.is_empty() {
        return Err(SparsePlanError::EmptyRowPtr);
    }
    if csr.row_ptr[0] != 0 {
        return Err(SparsePlanError::NonZeroStart {
            start: csr.row_ptr[0],
        });
    }
    for i in 1..csr.row_ptr.len() {
        if csr.row_ptr[i] < csr.row_ptr[i - 1] {
            return Err(SparsePlanError::NotMonotonic { index: i });
        }
    }
    let end = *csr.row_ptr.last().expect("row_ptr checked non-empty");
    if end as usize != csr.col.len() {
        return Err(SparsePlanError::NnzMismatch {
            row_ptr_end: end,
            col_len: csr.col.len(),
        });
    }
    for (edge, &col) in csr.col.iter().enumerate() {
        if col as usize >= ncols {
            return Err(SparsePlanError::ColumnOutOfRange { edge, col, ncols });
        }
    }

    let nrows = csr.row_ptr.len() - 1;
    let nnz = csr.col.len();
    for (what, value) in [("nrows", nrows), ("ncols", ncols), ("nnz", nnz)] {
        if value > u32::MAX as usize {
            return Err(SparsePlanError::TooLarge { what, value });
        }
    }

    Ok(SparseShape { nrows, ncols, nnz })
}

// ---------------------------------------------------------------------------
// CPU kernels
// ---------------------------------------------------------------------------

#[inline]
fn row_dot(csr: &Csr, weights: &[f32], x: &[f32], r: usize) -> f32 {
    let start = csr.row_ptr[r] as usize;
    let end = csr.row_ptr[r + 1] as usize;
    let mut sum = 0.0f32;
    for i in start..end {
        sum += weights[i] * x[csr.col[i] as usize];
    }
    sum
}

fn cpu_spmv_sequential(csr: &Csr, weights: &[f32], x: &[f32], y: &mut [f32]) {
    for (r, y_val) in y.iter_mut().enumerate() {
        *y_val += row_dot(csr, weights, x, r);
    }
}

fn cpu_spmv_parallel(csr: &Csr, weights: &[f32], x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;
    y.par_iter_mut().enumerate().for_each(|(r, y_val)| {
        *y_val += row_dot(csr, weights, x, r);
    });
}

#[inline]
fn lif_step(v: &mut f32, theta: &mut f32, current: f32, spike: &mut bool, p: LifParams) {
    let voltage = *v * p.decay + current;
    if voltage >= *theta {
        *spike = true;
        *v = p.v_reset;
        *theta += p.delta_theta;
    } else {
        *spike = false;
        *v = voltage;
    }
}

fn cpu_lif_integrate_sequential(
    v: &mut [f32],
    theta: &mut [f32],
    currents: &[f32],
    spikes: &mut [bool],
    p: LifParams,
) {
    for i in 0..v.len() {
        lif_step(&mut v[i], &mut theta[i], currents[i], &mut spikes[i], p);
    }
}

fn cpu_lif_integrate_parallel(
    v: &mut [f32],
    theta: &mut [f32],
    currents: &[f32],
    spikes: &mut [bool],
    p: LifParams,
) {
    use rayon::prelude::*;
    v.par_iter_mut()
        .zip(theta.par_iter_mut())
        .zip(currents.par_iter())
        .zip(spikes.par_iter_mut())
        .for_each(|(((v_i, th_i), &curr), spk)| {
            lif_step(v_i, th_i, curr, spk, p);
        });
}

#[allow(clippy::too_many_arguments)]
fn cpu_fused_sequential(
    csr: &Csr,
    weights: &[f32],
    x: &[f32],
    v: &mut [f32],
    theta: &mut [f32],
    spikes: &mut [bool],
    p: LifParams,
) {
    for r in 0..v.len() {
        let current = row_dot(csr, weights, x, r);
        lif_step(&mut v[r], &mut theta[r], current, &mut spikes[r], p);
    }
}

#[allow(clippy::too_many_arguments)]
fn cpu_fused_parallel(
    csr: &Csr,
    weights: &[f32],
    x: &[f32],
    v: &mut [f32],
    theta: &mut [f32],
    spikes: &mut [bool],
    p: LifParams,
) {
    use rayon::prelude::*;
    v.par_iter_mut()
        .zip(theta.par_iter_mut())
        .zip(spikes.par_iter_mut())
        .enumerate()
        .for_each(|(r, ((v_i, th_i), spk))| {
            let current = row_dot(csr, weights, x, r);
            lif_step(v_i, th_i, current, spk, p);
        });
}

#[cfg(test)]
mod inner_matches_tests {
    use super::*;

    /// Directly tests the guard that makes a mislabelled handle impossible.
    ///
    /// On a machine where Metal works, `try_new`'s failure path is unreachable,
    /// so no end-to-end test can execute it — a mutation that made `open_metal`
    /// fall back to CPU passed the whole suite for exactly that reason. The
    /// guard is still correct by construction, and this asserts it directly so
    /// that removing it is caught even though triggering it is not.
    #[test]
    fn cpu_inner_never_matches_a_gpu_label() {
        assert!(DeviceInner::Cpu.matches(Backend::CpuSequential));
        assert!(DeviceInner::Cpu.matches(Backend::CpuParallel));
        assert!(
            !DeviceInner::Cpu.matches(Backend::Metal),
            "a CPU substrate must never satisfy a Metal label"
        );
        assert!(
            !DeviceInner::Cpu.matches(Backend::Cuda),
            "a CPU substrate must never satisfy a CUDA label"
        );
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn metal_inner_only_matches_the_metal_label() {
        let Ok(device) = metal::shared_device() else {
            return;
        };
        let inner = DeviceInner::Metal(device);
        assert!(inner.matches(Backend::Metal));
        assert!(!inner.matches(Backend::CpuSequential));
        assert!(!inner.matches(Backend::CpuParallel));
        assert!(!inner.matches(Backend::Cuda));
    }
}

#[cfg(test)]
mod tolerance_tests {
    use super::*;

    /// Pins both tolerance formulas to concrete values.
    ///
    /// A tolerance is the one thing in a differential suite that can fail
    /// upward: widen it enough and every comparison passes while proving
    /// nothing. A mutation that returned `f32::MAX` from
    /// `tolerance_for_nnz_per_row` was invisible to the entire suite. These
    /// bounds close that: the tolerance has to stay small enough to still be a
    /// test.
    #[test]
    fn tolerances_are_bounded_above_as_well_as_below() {
        // The M5 Pro measurement that calibrated this: 1000 nnz/row with terms
        // bounded by 0.25 deviated by 3.3e-6.
        let measured_worst_case = 3.3e-6f32;
        let t = tolerance_for_nnz_per_row(1000, 0.25);
        assert!(
            t > measured_worst_case,
            "tolerance {t} is below the deviation actually observed ({measured_worst_case}); \
             the differential suite would be flaky"
        );
        assert!(
            t < 1e-3,
            "tolerance {t} for 1000 nnz/row is too wide to detect a real defect"
        );

        let e = tolerance_for_elementwise(1.0);
        assert!(
            (f32::EPSILON..1e-5).contains(&e),
            "elementwise tolerance {e} should be a couple of ulps, not a free pass"
        );
    }

    #[test]
    fn tolerance_grows_with_row_density_and_magnitude() {
        assert!(tolerance_for_nnz_per_row(10, 1.0) < tolerance_for_nnz_per_row(1000, 1.0));
        assert!(tolerance_for_nnz_per_row(100, 1.0) < tolerance_for_nnz_per_row(100, 10.0));
    }

    /// Degenerate arguments must still yield a usable positive tolerance rather
    /// than zero (which would make every comparison exact) or NaN (which would
    /// make every comparison pass, since `NaN <= NaN` is false but `d <= NaN`
    /// is too — an assertion that can never hold reads as a failing test, but a
    /// zero tolerance reads as a passing one).
    #[test]
    fn degenerate_arguments_yield_finite_positive_tolerances() {
        for (n, mag) in [(0usize, 0.0f32), (0, 1.0), (1, 0.0), (usize::MAX, 1.0)] {
            let t = tolerance_for_nnz_per_row(n, mag);
            assert!(t > 0.0 && t.is_finite(), "n={n} mag={mag} gave {t}");
        }
        for mag in [0.0f32, -1.0, f32::MIN_POSITIVE] {
            let t = tolerance_for_elementwise(mag);
            assert!(t > 0.0 && t.is_finite(), "mag={mag} gave {t}");
        }
    }
}
