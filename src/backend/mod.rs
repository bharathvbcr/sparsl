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
//! [`Device::prepare`] validates every stored column index against `ncols`
//! *before* anything is uploaded, and the [`SparseOp`] it returns is the only
//! way to reach a sparse kernel. An unvalidated [`Csr`] cannot reach device
//! code.

use core::fmt;
use std::sync::Mutex;

use crate::scan::State;
use crate::sparse::{Csc, Csr};

// Crate-private on purpose. `MetalDevice::prepare` takes a `SparseShape` and
// performs no validation of its own — it trusts that the shape came from
// `SparseOp::prepare`, which proved every column index in range. While this
// module was `pub`, that trust was unenforceable: an external caller could pair
// `Csr::from_parts_unchecked` with a hand-built `SparseShape` and reach the
// kernels directly, and the kernels index `x[col[i]]` with no bounds check. An
// audit demonstrated an out-of-bounds device read from safe, `forbid(unsafe_code)`
// caller code that way.
//
// Reachability is now the enforcement: `SparseOp` really is the only route to a
// sparse kernel, because nothing outside this crate can name the type that
// would bypass it.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub(crate) mod metal;

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

/// Why [`Device::reset_metal`] refused to clear Metal quarantine or rebuild the
/// submission path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalResetError {
    /// At least one submission is still holding an admission permit.
    ActiveAdmissions { count: usize },
    /// A previously timed-out command buffer has not reached a terminal status.
    NonTerminalCommand,
    /// Opening or rebuilding Metal failed.
    Backend(&'static str),
    /// Metal is not part of this build or not available on this host.
    Unavailable(&'static str),
}

impl fmt::Display for MetalResetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ActiveAdmissions { count } => write!(
                f,
                "Metal reset refused: {count} active admission(s) still outstanding"
            ),
            Self::NonTerminalCommand => write!(
                f,
                "Metal reset refused: a timed-out command buffer is still non-terminal"
            ),
            Self::Backend(reason) => write!(f, "Metal reset failed: {reason}"),
            Self::Unavailable(reason) => write!(f, "Metal reset unavailable: {reason}"),
        }
    }
}

impl std::error::Error for MetalResetError {}

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
    /// Column indices within a row decreased (duplicates are allowed).
    RowUnsorted { row: usize, edge: usize },
    /// The CSR's stored column count disagreed with the `ncols` passed to prepare.
    NcolsMismatch { stored: usize, declared: usize },
    /// The shape does not fit the `u32` indices the GPU kernels use.
    TooLarge { what: &'static str, value: usize },
    /// `weights` did not have one entry per stored non-zero.
    WeightsLen { expected: usize, got: usize },
    /// The device rejected an allocation.
    Allocation { what: &'static str },
    /// The selected substrate became unavailable after the device handle was
    /// created, so preparing resident state would be unsafe or misleading.
    Backend { reason: &'static str },
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
            Self::RowUnsorted { row, edge } => write!(
                f,
                "CSR row {row} is unsorted at edge {edge} (columns must be non-decreasing)"
            ),
            Self::NcolsMismatch { stored, declared } => write!(
                f,
                "CSR ncols {stored} does not match prepare ncols {declared}"
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
            Self::Backend { reason } => write!(f, "backend refused: {reason}"),
        }
    }
}

impl std::error::Error for SparsePlanError {}

/// Why an operation call was rejected.
///
/// Caller-data problems are reported rather than asserted, so a fuzzer or a
/// long-running process gets an error it can handle instead of an abort.
/// Backend refusal and post-submission execution failure have distinct variants
/// so callers cannot mistake work that did not run for validated output.
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
    /// Deriving an operand length from caller-controlled dimensions overflowed
    /// `usize` before any buffer was read or mutated.
    SizeOverflow {
        what: &'static str,
        lhs: usize,
        rhs: usize,
    },
    /// [`SparseOp::spmv_t`] on an operator built without a reverse index.
    TransposeNotPrepared,
    /// The substrate refused the work and said why. Distinct from the length
    /// errors above: those are caller mistakes, this is the device declining.
    Backend { reason: &'static str },
    /// Work was submitted but the device reported a terminal failure or did
    /// not reach a terminal state before the backend's bounded deadline.
    ///
    /// This is distinct from [`OpError::Backend`]: construction and allocation
    /// succeeded, but reading the output would otherwise turn a failed or
    /// indefinitely stalled command buffer into a plausible successful result.
    Execution {
        operation: &'static str,
        detail: String,
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
            Self::Backend { reason } => write!(f, "backend refused: {reason}"),
            Self::Execution { operation, detail } => {
                write!(f, "{operation}: device execution failed: {detail}")
            }
            Self::TransposeNotPrepared => write!(
                f,
                "this operator has no reverse index; build it with \
                 `Device::prepare_with_transpose` to use `spmv_t`"
            ),
            Self::TooShort { what, min, got } => {
                write!(f, "`{what}` must have length >= {min}, got {got}")
            }
            Self::SizeOverflow { what, lhs, rhs } => {
                write!(f, "{what} overflowed usize: {lhs} * {rhs}")
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
///
/// The fields are private and there is no public constructor, so a value of
/// this type can only have come from the crate-private `validate_csr`. That is
/// deliberate and
/// load-bearing rather than mere encapsulation: the GPU kernels index
/// `x[col[i]]` with no bounds check, and a `SparseShape` is the token that says
/// the indices were checked. If a caller could build one, the token would mean
/// nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseShape {
    nrows: usize,
    ncols: usize,
    nnz: usize,
    max_row_nnz: usize,
}

impl SparseShape {
    /// Number of rows.
    pub const fn nrows(self) -> usize {
        self.nrows
    }

    /// Declared number of columns; every stored index is `< ncols`.
    pub const fn ncols(self) -> usize {
        self.ncols
    }

    /// Total stored non-zeros.
    pub const fn nnz(self) -> usize {
        self.nnz
    }

    /// Stored non-zeros in the longest row.
    ///
    /// Separate from the mean because error bounds depend on the longest row,
    /// not the average one. See [`SparseShape::nnz_per_row`].
    pub const fn max_row_nnz(self) -> usize {
        self.max_row_nnz
    }

    /// Mean stored non-zeros per row, rounded up.
    ///
    /// **Do not size an error tolerance with this.** A ragged operator — most
    /// rows empty, a few long — has a mean that describes none of its rows, and
    /// a bound derived from it is too tight for exactly the rows that need it
    /// most. Use [`SparseShape::max_row_nnz`]. A 50,000-case fuzz found this
    /// the hard way on a 380-row operator whose mean degree rounded to 1 while
    /// its populated rows held two entries.
    pub fn nnz_per_row(self) -> usize {
        if self.nrows == 0 {
            0
        } else {
            self.nnz.div_ceil(self.nrows)
        }
    }
}

/// Upper bound on the CPU/GPU disagreement for an elementwise kernel.
///
/// An elementwise update has no summation and therefore no reduction order to
/// disagree about, so the intuition is that both substrates must agree exactly.
/// They do not. Metal contracts `v * decay + current` into a single `fma`,
/// rounding once where the CPU rounds twice, which moves the result by up to an
/// ulp. `tests/fma_contraction.rs` pins this behaviour.
///
/// # Pass the largest OPERAND, not the result
///
/// The omitted rounding happens at the magnitude of the product `v * decay`,
/// which under cancellation is unboundedly larger than the result. An earlier
/// signature named this parameter `max_abs_result`, and an audit produced the
/// counterexample: `v = 0.99999106`, `decay = 1.1`, `current = -1.0999901`
/// gives exactly `0.0` on the CPU and `-5.96e-8` fused — a real deviation of
/// 5.96e-8 against a bound of 3e-45 computed from the result. Under-bound by
/// some thirty-seven orders of magnitude.
///
/// So `max_abs_operand` must cover `|v * decay|`, `|current|` and `|theta|`,
/// none of which cancellation can shrink.
///
/// Two ulps of that magnitude covers the omitted rounding with margin.
///
/// The reason this matters more than an ulp usually does: the value it perturbs
/// is immediately compared against a threshold. A membrane sitting within an ulp
/// of `theta` can spike on one substrate and not the other — a boolean
/// difference out of a rounding difference. Comparisons across substrates must
/// treat spikes in that band as legitimately ambiguous.
pub fn tolerance_for_elementwise(max_abs_operand: f32) -> f32 {
    let magnitude = max_abs_operand.abs().max(f32::MIN_POSITIVE);
    2.0 * f32::EPSILON * magnitude
}

/// Upper bound on the CPU/GPU disagreement for one `y += A · x` row.
///
/// The operation being bounded is `y_out = y_in + Σ w_i · x_i`, and the bound
/// has to cover *both* halves of that. An earlier version covered only the row
/// sum, taking `max|w·x|` as the magnitude that matters. That is wrong whenever
/// the incoming `y` is larger than any single product: the final accumulation
/// rounds at the magnitude of the *result*, so a row whose terms are tiny but
/// whose running total is not gets a tolerance far below its real error.
///
/// A 50,000-case fuzz found it: 380 rows over one column, terms bounded by
/// 0.06, result 0.756. Predicted 5.7e-8, observed 6e-8. Small, and a genuine
/// failure of the model rather than noise.
///
/// So the bound is the textbook recursive-summation form over both parts:
///
/// ```text
/// 8 · eps · (max_row_nnz · max|w·x|  +  max|y|)
/// ```
///
/// with a factor of 8 of headroom. `max_row_nnz` is the longest row, not the
/// mean — see [`SparseShape::nnz_per_row`] for why that distinction bites.
///
/// It is deliberately a bound, not a curve fitted to one machine's measured
/// error. A tolerance tuned tight against one GPU becomes a flaky test on the
/// next.
pub fn tolerance_for_spmv(max_row_nnz: usize, max_abs_term: f32, max_abs_result: f32) -> f32 {
    let n = max_row_nnz.max(1) as f32;
    let term = max_abs_term.abs().max(f32::MIN_POSITIVE);
    let result = max_abs_result.abs().max(f32::MIN_POSITIVE);
    8.0 * f32::EPSILON * n.mul_add(term, result)
}

/// Quantisation selected when an operator's weights become resident.
///
/// The selection fixes the values used by every operation: `F16` and `Bf16`
/// round initial inputs during [`Device::prepare_with`] and replacement inputs
/// during [`SparseOp::set_weights`]. Arithmetic and accumulation remain f32 —
/// see [`crate::half`] for why a 500-term row sum at 16-bit width is not an
/// option.
///
/// This is not a promise that every resident copy is narrow. CPU operators
/// store the quantised values widened to f32. Metal keeps that same f32 mirror
/// for batched SpMM, transpose, packed spikes and fused SpMV+LIF, alongside a
/// compact buffer used by plain [`SparseOp::spmv`] (and one-vector SpMM, which
/// delegates to it). Consequently, [`WeightPrecision::bytes`] describes the
/// compact encoding, not an operator's total resident footprint.
///
/// This is an enum rather than the boolean it replaced. A second format made
/// the boolean wrong: two flags would have admitted a state meaning "both
/// binary16 and bfloat16", which no operator can be in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeightPrecision {
    /// f32. The default, and the only one with no quantisation error.
    #[default]
    F32,
    /// IEEE binary16: 1+5+10. Finer than bfloat16, but overflows at 65504.
    F16,
    /// bfloat16: 1+8+7. f32's exponent field and nearly its range, 8x coarser.
    Bf16,
}

impl WeightPrecision {
    /// Bytes one weight occupies in this compact encoding.
    ///
    /// This is not total resident memory. In particular, Metal operators with
    /// a narrow encoding also keep a widened f32 mirror for operations other
    /// than plain SpMV, and CPU operators keep only that mirror.
    pub const fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }

    /// Machine epsilon of the storage format, the ulp at 1.0.
    ///
    /// What the quantisation term of a tolerance is built from.
    pub fn epsilon(self) -> f32 {
        match self {
            Self::F32 => f32::EPSILON,
            Self::F16 => crate::half::HALF_EPSILON,
            Self::Bf16 => crate::half::BF16_EPSILON,
        }
    }

    /// Round `weights` through this format when they become resident.
    fn quantise(self, weights: &[f32]) -> Option<Vec<f32>> {
        use crate::half::*;
        match self {
            Self::F32 => None,
            Self::F16 => Some(
                weights
                    .iter()
                    .map(|w| f16_bits_to_f32(f32_to_f16_bits(*w)))
                    .collect(),
            ),
            Self::Bf16 => Some(
                weights
                    .iter()
                    .map(|w| bf16_bits_to_f32(f32_to_bf16_bits(*w)))
                    .collect(),
            ),
        }
    }

    /// The narrow bit patterns a device buffer is filled from.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(crate) fn narrow_bits(self, weights: &[f32]) -> Option<Vec<u16>> {
        match self {
            Self::F32 => None,
            Self::F16 => Some(crate::half::f32_slice_to_f16(weights)),
            Self::Bf16 => Some(crate::half::f32_slice_to_bf16(weights)),
        }
    }
}

/// Upper bound on a `y += A · x` row when weights are quantised as binary16.
///
/// The README used to record "narrower types need their bounds re-derived, not
/// rescaled" as the reason f16 was missing. This is that derivation.
///
/// Three error sources, not one, and they are not the same size:
///
/// 1. **Quantising the weights.** Each stored `w` is `fl16(w)`, so it carries a
///    relative error up to `HALF_EPSILON / 2` before any arithmetic happens.
///    Over a row this contributes `n · HALF_EPSILON · max|w·x|`.
/// 2. **Forming the products** in f32, once widened.
/// 3. **Accumulating** them in f32, the usual recursive-summation term.
///
/// Sources 2 and 3 are exactly [`tolerance_for_spmv`], so this is that bound
/// plus the quantisation term rather than a separate formula — the two cannot
/// drift apart, and the delegation is what says the arithmetic is still f32:
///
/// ```text
/// 8 · n · HALF_EPSILON · max|w·x|   +   tolerance_for_spmv(n, max|w·x|, max|y|)
/// ```
///
/// The first term dominates by roughly 8192x, which is the ratio of
/// [`HALF_EPSILON`] to [`f32::EPSILON`]. Put plainly: with binary16 weights the
/// error is set by how coarsely the weights were stored, and the accumulation
/// is nearly free by comparison. That is the trade being made, and it is why
/// nothing here accumulates in binary16 — see [`crate::half`].
///
/// [`HALF_EPSILON`]: crate::half::HALF_EPSILON
pub fn tolerance_for_spmv_f16(max_row_nnz: usize, max_abs_term: f32, max_abs_result: f32) -> f32 {
    tolerance_for_spmv_narrow(
        WeightPrecision::F16,
        max_row_nnz,
        max_abs_term,
        max_abs_result,
    )
}

/// [`tolerance_for_spmv_f16`] for bfloat16 weights.
///
/// The same derivation with a coarser constant: [`crate::half::BF16_EPSILON`]
/// is 8x [`crate::half::HALF_EPSILON`], so this bound is roughly 8x looser.
/// That is the trade bfloat16 makes — an 8-bit exponent field and nearly f32's
/// range for three fewer significand bits — stated as a number rather than a
/// preference.
pub fn tolerance_for_spmv_bf16(max_row_nnz: usize, max_abs_term: f32, max_abs_result: f32) -> f32 {
    tolerance_for_spmv_narrow(
        WeightPrecision::Bf16,
        max_row_nnz,
        max_abs_term,
        max_abs_result,
    )
}

/// The derivation both narrow bounds share, parameterised by the storage.
///
/// One formula rather than one per format. A third narrow type would add a
/// [`WeightPrecision`] variant and nothing here, which is the point: the
/// analysis does not depend on which 16-bit layout was chosen, only on its
/// epsilon.
pub fn tolerance_for_spmv_narrow(
    precision: WeightPrecision,
    max_row_nnz: usize,
    max_abs_term: f32,
    max_abs_result: f32,
) -> f32 {
    let n = max_row_nnz.max(1) as f32;
    let term = max_abs_term.abs().max(f32::MIN_POSITIVE);
    let accumulation = tolerance_for_spmv(max_row_nnz, max_abs_term, max_abs_result);
    match precision {
        // f32 storage quantises nothing; adding a term for it would double-count
        // rounding that `tolerance_for_spmv` already covers.
        WeightPrecision::F32 => accumulation,
        _ => 8.0 * n * precision.epsilon() * term + accumulation,
    }
}

/// Conservative tolerance for the disagreement between two prefix scans.
///
/// [`Device::assoc_scan`] is the one operation here whose *algorithm* differs
/// by substrate. The CPU arms left-fold, and their output is bit-identical to
/// [`crate::assoc_scan_sequential`]. Metal runs a two-level tree, which
/// reassociates — and must, because bit-identity comes from a complete
/// sequential fold that no tree reproduces. So a CPU/GPU comparison of a scan
/// needs a bound, exactly as SpMV does, and this is it.
///
/// Composing `prefix_len` affine maps is a chain of that many multiply-adds in
/// both `a` and `b`. Two parenthesizations differ in proportion to the chain
/// length and its absolute condition scale, not merely to the final value,
/// which may be far smaller under cancellation:
///
/// ```text
/// 8 · eps · prefix_len · max|intermediate|
/// ```
///
/// `magnitude_envelope` must bound both the multiplier product and the sum of
/// absolute affine contributions. Use [`crate::scan_magnitude_envelope`] to
/// derive it. Passing only the largest `|b|` is not sound for public [`State`]:
/// a chain with `a > 1, b = 0` has a zero `b` throughout while its `a` error
/// grows geometrically. Pass the longest prefix compared, not the mean.
///
/// The first-order model assumes finite f32 inputs and no arithmetic overflow.
/// A non-finite envelope produces a non-finite tolerance rather than silently
/// substituting a small scale. Callers must reject that case; the crate's
/// differential comparator does. This keeps the existing two-argument API
/// while making its required scalar honest for arbitrary finite [`State`]
/// values.
///
/// [`State`]: crate::State
pub fn tolerance_for_scan(prefix_len: usize, magnitude_envelope: f32) -> f32 {
    if !magnitude_envelope.is_finite() {
        return magnitude_envelope.abs();
    }
    let n = prefix_len.max(1) as f32;
    let magnitude = magnitude_envelope.abs().max(1.0);
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

/// A handle to a substrate that was available when the handle was created.
///
/// Cloning is cheap; GPU state is shared, not duplicated. A Metal command
/// timeout dynamically quarantines that backend, after which existing handles
/// fail closed rather than reusing resources that may still be in flight.
/// [`Device::reset_metal`] clears that quarantine only when no admissions are
/// outstanding and every timed-out command buffer is terminal.
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
    /// Clear Metal quarantine and rebuild the shared submission path.
    ///
    /// Succeeds only when no admission permits are outstanding and every
    /// timed-out command buffer is terminal. On success the quarantine bit is
    /// cleared and the shared device's command queue plus completion event are
    /// replaced. If a timed-out command is still non-terminal, quarantine stays
    /// published.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn reset_metal() -> Result<(), MetalResetError> {
        metal::reset_runtime()
    }

    /// Metal is not part of this build.
    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    pub fn reset_metal() -> Result<(), MetalResetError> {
        Err(MetalResetError::Unavailable(
            metal_unavailable_reason().unwrap_or("Metal is unavailable"),
        ))
    }

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
        SparseOp::prepare(
            self.clone(),
            csr,
            ncols,
            weights,
            false,
            WeightPrecision::F32,
        )
    }

    /// [`Device::prepare`] with weights quantised to IEEE binary16 on residency.
    ///
    /// Takes f32 weights and quantises them once. On `Backend::Metal`, plain
    /// [`SparseOp::spmv`] then streams the compact 2-byte values instead of
    /// 4-byte values. `col_ind` stays 4 bytes, so traffic per non-zero goes 8
    /// to 6 — a 25% cut, not the 2x that halving one of the two arrays might
    /// suggest. `examples/narrow_crossover.rs` measures what that is actually
    /// worth.
    ///
    /// Metal also retains the quantised values widened to f32. Batched
    /// [`SparseOp::spmm`] (`n_vec > 1`), [`SparseOp::spmv_t`],
    /// [`SparseOp::spmv_spikes`] and [`SparseOp::fused_spmv_lif`] read that
    /// mirror; one-vector SpMM delegates to plain SpMV. The selected
    /// quantisation therefore applies to every operation, while the compact
    /// bandwidth path applies only to plain SpMV. CPU operators retain only
    /// the widened f32 representation.
    ///
    /// # Both arms store the same values
    ///
    /// The weights are quantised once, before either backend stores anything,
    /// so a CPU operator built this way holds exactly the values the GPU one
    /// does. A CPU/GPU comparison is therefore still bounded by
    /// [`tolerance_for_spmv`]: the binary16 error lives *in the operator*, not
    /// between the backends. Comparing against what the unquantised weights
    /// would have given is the other question, and [`tolerance_for_spmv_f16`]
    /// bounds that one.
    ///
    /// # The CPU arm gains nothing here
    ///
    /// It stores the quantised values back as f32, so it pays the precision and
    /// saves no bandwidth. Stated rather than hidden: this entry point exists
    /// for the GPU, and a CPU operator built with it is for checking the GPU
    /// rather than for going faster. [`SparseOp::weight_precision`] reports the
    /// compact execution representation available to plain SpMV.
    ///
    /// # Range
    ///
    /// binary16 overflows at 65504 and its smallest normal is 6.1e-5. Weights
    /// outside that become infinities or flush toward zero; this does not
    /// rescale them to fit, because a silent rescale would change the operator
    /// the caller asked for.
    pub fn prepare_f16(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<SparseOp, SparsePlanError> {
        SparseOp::prepare(
            self.clone(),
            csr,
            ncols,
            weights,
            false,
            WeightPrecision::F16,
        )
    }

    /// [`Device::prepare`] with an explicit resident quantisation.
    ///
    /// The general form. [`Device::prepare`], [`Device::prepare_f16`] and
    /// [`Device::prepare_bf16`] are named conveniences over it, and exist
    /// because a call site reading `prepare_bf16` says more than one reading
    /// `prepare_with(.., Bf16)`. Reach for this when the format is a variable
    /// — a benchmark sweeping the options, or a caller choosing from config.
    pub fn prepare_with(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
        precision: WeightPrecision,
    ) -> Result<SparseOp, SparsePlanError> {
        SparseOp::prepare(self.clone(), csr, ncols, weights, false, precision)
    }

    /// [`Device::prepare`] with weights quantised to bfloat16 on residency.
    ///
    /// The other narrow option. Against [`Device::prepare_f16`] the trade is
    /// range for precision: bfloat16 carries f32's exponent field, so it does
    /// not hit binary16's 65504 ceiling, and it is 8x coarser in exchange.
    /// Neither is a default — which is right depends on the weights, and
    /// [`tolerance_for_spmv_bf16`] prices the difference.
    ///
    /// Plain Metal SpMV bandwidth is identical: both compact buffers encode 2
    /// bytes per weight. On the measured shapes that is worth a few percent,
    /// not a doubling; see `examples/narrow_crossover.rs`. The f32 mirror and
    /// operation scope documented by [`Device::prepare_f16`] apply unchanged.
    ///
    /// Everything [`Device::prepare_f16`] documents about *where* the error
    /// lives applies unchanged — the weights are quantised once before either
    /// backend stores anything, so a CPU/GPU comparison is still bounded by
    /// [`tolerance_for_spmv`].
    pub fn prepare_bf16(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<SparseOp, SparsePlanError> {
        SparseOp::prepare(
            self.clone(),
            csr,
            ncols,
            weights,
            false,
            WeightPrecision::Bf16,
        )
    }

    /// [`Device::prepare`], plus the reverse index [`SparseOp::spmv_t`] needs.
    ///
    /// Costs a second index of the same size as the forward one — `nnz` row
    /// indices and `nnz` edge indices, plus `ncols + 1` column pointers. That
    /// is why it is a separate entry point rather than always-on: a
    /// forward-only caller should not pay for a gradient path it never walks.
    ///
    /// The reverse index is derived from the same validated CSR and points at
    /// the same weight table, so the two directions cannot disagree about
    /// either structure or values.
    pub fn prepare_with_transpose(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<SparseOp, SparsePlanError> {
        self.prepare_with_transpose_precision(csr, ncols, weights, WeightPrecision::F32)
    }

    /// [`Device::prepare_with_transpose`] at an explicit weight precision.
    ///
    /// The named [`Device::prepare_f16_with_transpose`] /
    /// [`Device::prepare_bf16_with_transpose`] wrappers exist for call sites
    /// that know the format; reach for this when precision is a variable.
    pub fn prepare_with_transpose_precision(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
        precision: WeightPrecision,
    ) -> Result<SparseOp, SparsePlanError> {
        SparseOp::prepare(self.clone(), csr, ncols, weights, true, precision)
    }

    /// Narrow binary16 weights plus the reverse index [`SparseOp::spmv_t`] needs.
    pub fn prepare_f16_with_transpose(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<SparseOp, SparsePlanError> {
        self.prepare_with_transpose_precision(csr, ncols, weights, WeightPrecision::F16)
    }

    /// Narrow bfloat16 weights plus the reverse index [`SparseOp::spmv_t`] needs.
    pub fn prepare_bf16_with_transpose(
        &self,
        csr: &Csr,
        ncols: usize,
        weights: &[f32],
    ) -> Result<SparseOp, SparsePlanError> {
        self.prepare_with_transpose_precision(csr, ncols, weights, WeightPrecision::Bf16)
    }

    /// Inclusive scan over affine maps, on this device's substrate.
    ///
    /// # This is the one place the crate's bit-identity rule bites
    ///
    /// The CPU arms run [`crate::assoc_scan`], which is bit-identical to a
    /// sequential left-fold — it buys that by making its first phase a complete
    /// sequential fold, about `2n` combines to replace `n`. The hardened paired
    /// sampler has not established a throughput crossover for that extra work;
    /// see [`crate::scan`] and `examples/scan_crossover.rs`.
    /// The Metal arm runs a two-level Hillis-Steele scan that reassociates, so
    /// it is genuinely parallel and is **not** bit-identical to the CPU arms.
    ///
    /// That is the rule this crate already states rather than an exception to
    /// it: reproducibility holds *within* a backend and never across one. Two
    /// runs on the same device agree byte for byte; comparing arms takes a
    /// tolerance, exactly as [`SparseOp::spmv`] does.
    ///
    /// If you need a scan whose output is bit-identical to the sequential fold,
    /// ask for a CPU backend explicitly. A GPU one cannot give it to you and
    /// says so here rather than by silently differing.
    ///
    /// Size any cross-backend comparison with [`tolerance_for_scan`].
    ///
    /// # This is slower on Metal, measured
    ///
    /// Not a throughput path. In the latest bounded, paired run on an M5 Pro,
    /// CPU-sequential/Metal median speedups were 0.23x, 0.33x, 0.56x and 0.48x
    /// at 64K, 256K, 1M and 4M elements: Metal lost at every size. Several CPU
    /// arms were noisy, so `examples/scan_crossover.rs` reports max/min spread
    /// and full provenance rather than presenting those ratios as universal
    /// constants. A scan over affine maps is three flops per sixteen bytes
    /// moved, so it is memory-bound, and the three-phase tree makes roughly
    /// five passes over memory where a fold makes one.
    ///
    /// It exists so `Backend::Metal` can run every operation this crate offers
    /// rather than silently falling back to CPU under a GPU label, which is the
    /// thing [`Backend::Cuda`] refuses to do. Reach for it when the data is
    /// already resident on the GPU and moving it back would cost more than the
    /// scan does — not for speed. `examples/scan_crossover.rs` reproduces the
    /// numbers.
    pub fn assoc_scan(&self, xs: &[State]) -> Result<Vec<State>, OpError> {
        match &self.inner {
            DeviceInner::Cpu => Ok(if self.backend == Backend::CpuParallel {
                crate::assoc_scan(xs, |a, b| a.combine(b))
            } else {
                crate::assoc_scan_sequential(xs, |a, b| a.combine(b))
            }),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            DeviceInner::Metal(d) => {
                // No reformatting: `State` is `repr(C)` and two packed `f32`s,
                // so the slice is already the layout the kernel reads.
                d.assoc_scan(xs)
            }
        }
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
            DeviceInner::Metal(d) => d.lif_integrate(v, theta, currents, spikes, params),
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
        /// Present only when the operator was built by
        /// [`Device::prepare_with_transpose`]. Building it always would double
        /// the index memory of every forward-only caller, which is most of
        /// them.
        csc: Option<Csc>,
        /// Host scratch mirroring the Metal resident buffers so the write /
        /// resident / sync API is backend-agnostic.
        scratch: Mutex<CpuScratch>,
    },
    #[cfg(all(target_os = "macos", feature = "metal"))]
    Metal(metal::MetalSparse),
}

/// CPU-side resident operands for the write / step / sync API.
struct CpuScratch {
    x: Vec<f32>,
    y: Vec<f32>,
    v: Vec<f32>,
    theta: Vec<f32>,
    spikes: Vec<bool>,
}

impl CpuScratch {
    fn new(shape: SparseShape) -> Self {
        Self {
            x: vec![0.0; shape.ncols()],
            y: vec![0.0; shape.nrows()],
            v: vec![0.0; shape.nrows()],
            theta: vec![0.0; shape.nrows()],
            spikes: vec![false; shape.nrows()],
        }
    }
}

/// A CSR operator validated against `ncols` and bound to a device.
///
/// Preparing is the only way to reach a sparse kernel, which is what makes the
/// unchecked `x[col[i]]` indexing inside the GPU kernels sound.
pub struct SparseOp {
    device: Device,
    shape: SparseShape,
    /// The quantisation applied whenever this operator stores new weights.
    ///
    /// Kept because `set_weights` has to preserve the contract. Without it a
    /// narrow operator quietly stopped being narrow on the first update: the
    /// CPU arm took the raw f32 while a freshly prepared operator held the
    /// quantised values, so the two disagreed on weights the caller believed
    /// were the same.
    precision: WeightPrecision,
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
        transpose: bool,
        precision: WeightPrecision,
    ) -> Result<Self, SparsePlanError> {
        let shape = validate_csr(csr, ncols)?;
        if weights.len() != shape.nnz {
            return Err(SparsePlanError::WeightsLen {
                expected: shape.nnz,
                got: weights.len(),
            });
        }
        // Derived from the CSR that `validate_csr` just accepted, so every
        // `row` is a valid CSR row and every `edge_idx` a valid weight slot.
        // That is what makes the unchecked indexing in `col_dot` and
        // `csc_spmv_t_kernel` sound, exactly as for the forward direction.
        // `to_csc_rect`, not `to_csc`: the latter assumes a square graph and
        // panics on a rectangular one. `validate_csr` has already proven every
        // column is below `ncols`, so this cannot fail — but it is checked
        // rather than unwrapped, because the alternative is a panic reaching a
        // caller who passed a perfectly ordinary non-square operator.
        // Quantise before either arm stores anything. Both then hold the same
        // values and differ only in how many bytes they spend on them, so a
        // CPU/GPU comparison is still bounded by `tolerance_for_spmv` — the
        // binary16 error is in the operator, not between the backends.
        let quantised = precision.quantise(weights);
        let weights = quantised.as_deref().unwrap_or(weights);

        let csc = match transpose {
            true => Some(csr.to_csc_rect(ncols).map_err(|_| {
                // Unreachable: `validate_csr` above proved every stored column
                // is below `ncols`, so the only error this constructor raises
                // cannot fire. Mapped rather than unwrapped so that a future
                // divergence between the two checks surfaces as the error the
                // caller already handles, not as a panic from inside `prepare`.
                let (edge, col) = csr
                    .col()
                    .iter()
                    .enumerate()
                    .find(|(_, c)| **c as usize >= ncols)
                    .map_or((0, 0), |(i, c)| (i, *c));
                SparsePlanError::ColumnOutOfRange { edge, col, ncols }
            })?),
            false => None,
        };
        let resident = match &device.inner {
            DeviceInner::Cpu => OpResident::Cpu {
                csr: csr.clone(),
                weights: weights.to_vec(),
                csc,
                scratch: Mutex::new(CpuScratch::new(shape)),
            },
            #[cfg(all(target_os = "macos", feature = "metal"))]
            DeviceInner::Metal(d) => {
                OpResident::Metal(d.prepare(csr, shape, weights, csc.as_ref(), precision)?)
            }
        };
        Ok(Self {
            device,
            shape,
            precision,
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
        require_len("weights", weights.len(), self.shape.nnz())?;
        // Same quantisation `prepare` applied, for the same reason: an operator
        // built narrow stays narrow, and every backend keeps storing the same
        // values as every other.
        let quantised = self.precision.quantise(weights);
        let weights = quantised.as_deref().unwrap_or(weights);
        match &mut self.resident {
            OpResident::Cpu {
                weights: stored, ..
            } => {
                stored.clear();
                stored.extend_from_slice(weights);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.set_weights(weights),
        }
    }

    /// `y += A · x`, using the values this operator holds.
    pub fn spmv(&self, x: &[f32], y: &mut [f32]) -> Result<(), OpError> {
        require_min_len("x", x.len(), self.shape.ncols())?;
        require_len("y", y.len(), self.shape.nrows())?;
        if self.shape.nrows() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { csr, weights, .. } => {
                if self.device.backend == Backend::CpuParallel {
                    cpu_spmv_parallel(csr, weights, x, y);
                } else {
                    cpu_spmv_sequential(csr, weights, x, y);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.spmv(x, y),
        }
    }

    /// `y += A · s`, where `s` is a bitpacked spike vector.
    ///
    /// The operand an SNN actually has. A spike is a boolean, and carrying it
    /// as `f32` spends 32 bits to say one — but the saving is not in the bytes
    /// streamed. `s[col[i]]` is a *random* gather, and its cost is set by
    /// whether the vector fits in cache: at 50,000 cells this is 6.25 KB
    /// against 200 KB. That is the operand narrowing the weights did not touch.
    ///
    /// Pack with [`crate::spikes::pack_spikes`], which is what
    /// [`SparseOp::fused_spmv_lif`] output feeds into.
    ///
    /// # Exact, not approximate
    ///
    /// This is bit-identical to calling [`SparseOp::spmv`] with the same spikes
    /// expanded to `0.0`/`1.0`, on every backend. Both paths decode the bit to
    /// a float and multiply, so they perform the same operations in the same
    /// order. Unlike [`Device::prepare_f16`] there is no tolerance here,
    /// because there is no approximation — see [`crate::spikes`].
    pub fn spmv_spikes(&self, spikes: &[u32], y: &mut [f32]) -> Result<(), OpError> {
        let words = crate::spikes::packed_len(self.shape.ncols());
        require_min_len("spikes", spikes.len(), words)?;
        require_len("y", y.len(), self.shape.nrows())?;
        if self.shape.nrows() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { csr, weights, .. } => {
                if self.device.backend == Backend::CpuParallel {
                    cpu_spmv_spikes_parallel(csr, weights, spikes, y);
                } else {
                    cpu_spmv_spikes_sequential(csr, weights, spikes, y);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.spmv_spikes(spikes, y),
        }
    }

    /// `y += Aᵀ · x`, using the values this operator holds.
    ///
    /// The reverse of [`SparseOp::spmv`]: `x` has one entry per **row** and `y`
    /// one per **column**. This is the direction a gradient travels — given
    /// `dy` over the outputs of a sparse layer, this produces `dx` over its
    /// inputs.
    ///
    /// Requires an operator built by [`Device::prepare_with_transpose`]. One
    /// built by [`Device::prepare`] has no reverse index and returns
    /// [`OpError::TransposeNotPrepared`] rather than building one implicitly:
    /// the index costs as much memory as the forward one, and a method that
    /// silently doubles an operator's footprint the first time it is called is
    /// worse than one that says it cannot.
    ///
    /// Both directions read the same weight table, so [`SparseOp::set_weights`]
    /// updates them together.
    pub fn spmv_t(&self, x: &[f32], y: &mut [f32]) -> Result<(), OpError> {
        require_min_len("x", x.len(), self.shape.nrows())?;
        require_len("y", y.len(), self.shape.ncols())?;
        if self.shape.ncols() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { weights, csc, .. } => {
                let csc = csc.as_ref().ok_or(OpError::TransposeNotPrepared)?;
                if self.device.backend == Backend::CpuParallel {
                    cpu_spmv_t_parallel(csc, weights, x, y);
                } else {
                    cpu_spmv_t_sequential(csc, weights, x, y);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.spmv_t(x, y),
        }
    }

    /// `Y += A · X` for a batch of `n_vec` dense vectors.
    ///
    /// # Layout
    ///
    /// `x` and `y` are **batch-minor**: `x[c * n_vec + v]` is column `c` of
    /// vector `v`, so all `n_vec` values for a column are adjacent. That is the
    /// opposite of storing each vector contiguously, and it is deliberate — it
    /// is what lets adjacent GPU threads read and write adjacent addresses. See
    /// `csr_spmm_kernel` for the full argument.
    ///
    /// # Why this exists
    ///
    /// A single-vector [`SparseOp::spmv`] gives the GPU one multiply-add per
    /// loaded index, which is not enough work to cover the load: on this
    /// crate's own measurements Metal does not overtake the rayon arm until
    /// roughly 20M non-zeros. Reusing each `weights[i]` and each `col[i]`
    /// across `n_vec` vectors is what changes that ratio.
    ///
    /// A batch of one is bit-identical to [`SparseOp::spmv`] on the same
    /// backend, not merely within tolerance: both traverse a row's non-zeros in
    /// the same order.
    pub fn spmm(&self, x: &[f32], n_vec: usize, y: &mut [f32]) -> Result<(), OpError> {
        if n_vec == 0 {
            return Err(OpError::Length {
                what: "n_vec",
                expected: 1,
                got: 0,
            });
        }
        let need_x = self
            .shape
            .ncols()
            .checked_mul(n_vec)
            .ok_or(OpError::SizeOverflow {
                what: "SpMM input element count",
                lhs: self.shape.ncols(),
                rhs: n_vec,
            })?;
        let need_y = self
            .shape
            .nrows()
            .checked_mul(n_vec)
            .ok_or(OpError::SizeOverflow {
                what: "SpMM output element count",
                lhs: self.shape.nrows(),
                rhs: n_vec,
            })?;
        require_min_len("x", x.len(), need_x)?;
        require_len("y", y.len(), need_y)?;
        if self.shape.nrows() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { csr, weights, .. } => {
                if self.device.backend == Backend::CpuParallel {
                    cpu_spmm_parallel(csr, weights, x, n_vec, y);
                } else {
                    cpu_spmm_sequential(csr, weights, x, n_vec, y);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.spmm(x, n_vec, y),
        }
    }

    /// The compact execution representation available to plain SpMV.
    ///
    /// This is not the source-value history or total resident footprint. A CPU
    /// operator built by [`Device::prepare_f16`] or [`Device::prepare_bf16`]
    /// holds quantised values in f32 storage, so it reports
    /// [`WeightPrecision::F32`]. A narrow Metal operator reports its compact
    /// plain-SpMV encoding, but also retains the same quantised values in an
    /// f32 mirror used by the other weighted operations.
    pub fn weight_precision(&self) -> WeightPrecision {
        match &self.resident {
            // The CPU arm holds quantised values in f32 storage: it pays the
            // precision and saves no bytes, so reporting anything else would
            // overstate what it does.
            OpResident::Cpu { .. } => WeightPrecision::F32,
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.weight_precision(),
        }
    }

    /// Whether [`SparseOp::spmv_t`] can run on this operator.
    pub fn has_transpose(&self) -> bool {
        match &self.resident {
            OpResident::Cpu { csc, .. } => csc.is_some(),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.has_transpose(),
        }
    }

    /// Fused `current = A · x` followed by the LIF membrane update, without
    /// materialising the current vector.
    ///
    /// On Metal this reduces each row exactly as the operator's
    /// [`SparseOp::spmv`] does — the same lane team and the same butterfly
    /// fold, chosen once per operator — so it lands no further from the CPU
    /// reference than plain SpMV; that order still differs from both CPU
    /// arms, so size any comparison with [`tolerance_for_spmv`].
    pub fn fused_spmv_lif(
        &self,
        x: &[f32],
        v: &mut [f32],
        theta: &mut [f32],
        spikes: &mut [bool],
        params: LifParams,
    ) -> Result<(), OpError> {
        require_min_len("x", x.len(), self.shape.ncols())?;
        require_len("v", v.len(), self.shape.nrows())?;
        require_len("theta", theta.len(), self.shape.nrows())?;
        require_len("spikes", spikes.len(), self.shape.nrows())?;
        if self.shape.nrows() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu { csr, weights, .. } => {
                if self.device.backend == Backend::CpuParallel {
                    cpu_fused_parallel(csr, weights, x, v, theta, spikes, params);
                } else {
                    cpu_fused_sequential(csr, weights, x, v, theta, spikes, params);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.fused_spmv_lif(x, v, theta, spikes, params),
        }
    }

    /// Upload `x` into resident scratch without dispatching.
    ///
    /// Pair with [`SparseOp::spmv_resident`] / [`SparseOp::fused_spmv_lif_resident`]
    /// and an explicit [`SparseOp::sync_y`] / [`SparseOp::sync_lif_state`] so a
    /// BINN tick loop does not memcpy operands every step.
    pub fn write_x(&self, x: &[f32]) -> Result<(), OpError> {
        require_min_len("x", x.len(), self.shape.ncols())?;
        match &self.resident {
            OpResident::Cpu { scratch, .. } => {
                let mut s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                s.x[..self.shape.ncols()].copy_from_slice(&x[..self.shape.ncols()]);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.write_x(x),
        }
    }

    /// Upload `y` into resident scratch without dispatching.
    pub fn write_y(&self, y: &[f32]) -> Result<(), OpError> {
        require_len("y", y.len(), self.shape.nrows())?;
        match &self.resident {
            OpResident::Cpu { scratch, .. } => {
                let mut s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                s.y.copy_from_slice(y);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.write_y(y),
        }
    }

    /// Upload LIF membrane state into resident scratch without dispatching.
    pub fn write_lif_state(&self, v: &[f32], theta: &[f32]) -> Result<(), OpError> {
        require_len("v", v.len(), self.shape.nrows())?;
        require_len("theta", theta.len(), self.shape.nrows())?;
        match &self.resident {
            OpResident::Cpu { scratch, .. } => {
                let mut s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                s.v.copy_from_slice(v);
                s.theta.copy_from_slice(theta);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.write_lif_state(v, theta),
        }
    }

    /// `y += A · x` using resident scratch; no host readback of `y`.
    pub fn spmv_resident(&self) -> Result<(), OpError> {
        if self.shape.nrows() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu {
                csr,
                weights,
                scratch,
                ..
            } => {
                let mut s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                // Split the struct borrow so `x` can be shared while `y` mutates.
                let CpuScratch {
                    x: ref x_res,
                    y: ref mut y_res,
                    ..
                } = &mut *s;
                if self.device.backend == Backend::CpuParallel {
                    cpu_spmv_parallel(csr, weights, x_res, y_res);
                } else {
                    cpu_spmv_sequential(csr, weights, x_res, y_res);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.spmv_resident(),
        }
    }

    /// Fused SpMV+LIF on resident scratch; no host readback.
    ///
    /// This is the BINN tick path: upload state once, step many times, sync
    /// when the host needs the result.
    pub fn fused_spmv_lif_resident(&self, params: LifParams) -> Result<(), OpError> {
        if self.shape.nrows() == 0 {
            return Ok(());
        }
        match &self.resident {
            OpResident::Cpu {
                csr,
                weights,
                scratch,
                ..
            } => {
                let mut s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                let CpuScratch {
                    x: ref x_res,
                    v: ref mut v_res,
                    theta: ref mut th_res,
                    spikes: ref mut sp_res,
                    ..
                } = &mut *s;
                if self.device.backend == Backend::CpuParallel {
                    cpu_fused_parallel(csr, weights, x_res, v_res, th_res, sp_res, params);
                } else {
                    cpu_fused_sequential(csr, weights, x_res, v_res, th_res, sp_res, params);
                }
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.fused_spmv_lif_resident(params),
        }
    }

    /// Canary check (Metal) + read resident `y` back to the host.
    pub fn sync_y(&self, y: &mut [f32]) -> Result<(), OpError> {
        require_len("y", y.len(), self.shape.nrows())?;
        match &self.resident {
            OpResident::Cpu { scratch, .. } => {
                let s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                y.copy_from_slice(&s.y);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.sync_y(y),
        }
    }

    /// Canary check (Metal) + read resident LIF state back to the host.
    pub fn sync_lif_state(
        &self,
        v: &mut [f32],
        theta: &mut [f32],
        spikes: &mut [bool],
    ) -> Result<(), OpError> {
        require_len("v", v.len(), self.shape.nrows())?;
        require_len("theta", theta.len(), self.shape.nrows())?;
        require_len("spikes", spikes.len(), self.shape.nrows())?;
        match &self.resident {
            OpResident::Cpu { scratch, .. } => {
                let s = scratch.lock().map_err(|_| OpError::Backend {
                    reason: "sparsl CPU scratch mutex is poisoned",
                })?;
                v.copy_from_slice(&s.v);
                theta.copy_from_slice(&s.theta);
                spikes.copy_from_slice(&s.spikes);
                Ok(())
            }
            #[cfg(all(target_os = "macos", feature = "metal"))]
            OpResident::Metal(op) => op.sync_lif_state(v, theta, spikes),
        }
    }

    /// Bind a caller-owned `MTLBuffer` as SpMV `x` (Metal only; no host copy).
    ///
    /// See the Metal backend module docs for the SharedEvent handoff
    /// pattern with tessl. Length must cover `ncols` f32s; device `registryID`
    /// must match. CSR/weights stay on the prepared resident path.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn bind_mtl_x(
        &self,
        buffer: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) -> Result<(), OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.bind_mtl_x(buffer),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "bind_mtl_x requires a Metal SparseOp",
            }),
        }
    }

    /// Bind a caller-owned `MTLBuffer` as SpMV `y` (Metal only; no host copy).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn bind_mtl_y(
        &self,
        buffer: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
    ) -> Result<(), OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.bind_mtl_y(buffer),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "bind_mtl_y requires a Metal SparseOp",
            }),
        }
    }

    /// Clear an external `x` binding.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn unbind_mtl_x(&self) -> Result<(), OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.unbind_mtl_x(),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "unbind_mtl_x requires a Metal SparseOp",
            }),
        }
    }

    /// Clear an external `y` binding.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn unbind_mtl_y(&self) -> Result<(), OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.unbind_mtl_y(),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "unbind_mtl_y requires a Metal SparseOp",
            }),
        }
    }

    /// GPU address of the active `x` buffer (external or owned scratch).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn mtl_x_gpu_address(&self) -> Result<u64, OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.mtl_x_gpu_address(),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "mtl_x_gpu_address requires a Metal SparseOp",
            }),
        }
    }

    /// GPU address of the active `y` buffer (external or owned scratch).
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn mtl_y_gpu_address(&self) -> Result<u64, OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.mtl_y_gpu_address(),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "mtl_y_gpu_address requires a Metal SparseOp",
            }),
        }
    }

    /// Allocate a shared f32 `MTLBuffer` on this operator's Metal device.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn alloc_mtl_f32(
        &self,
        data: &[f32],
    ) -> Result<
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>,
        OpError,
    > {
        match &self.resident {
            OpResident::Metal(op) => op.alloc_mtl_f32(data),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "alloc_mtl_f32 requires a Metal SparseOp",
            }),
        }
    }

    /// Wait on a peer `MTLSharedEvent` (e.g. tessl's) before reading shared
    /// buffers. Queues are not merged across crates.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn wait_shared_event(
        &self,
        event: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
        timeout_ms: u64,
    ) -> Result<(), OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.wait_shared_event(event, value, timeout_ms),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "wait_shared_event requires a Metal SparseOp",
            }),
        }
    }

    /// Host-signal a peer `MTLSharedEvent` after this operator's work completed.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub fn signal_shared_event(
        &self,
        event: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLSharedEvent>,
        value: u64,
    ) -> Result<(), OpError> {
        match &self.resident {
            OpResident::Metal(op) => op.signal_shared_event(event, value),
            OpResident::Cpu { .. } => Err(OpError::Backend {
                reason: "signal_shared_event requires a Metal SparseOp",
            }),
        }
    }

    /// Test-only: whether the Metal forward path split on degree skew.
    #[cfg(all(test, target_os = "macos", feature = "metal"))]
    pub(crate) fn is_hybrid_forward(&self) -> bool {
        match &self.resident {
            OpResident::Metal(op) => op.is_hybrid_forward(),
            _ => false,
        }
    }
}

/// Full structural validation of a CSR against a declared column count.
///
/// Re-validates everything [`Csr::from_parts`] checks, because
/// `Csr::from_parts_unchecked` exists and a caller may have used it. Also
/// cross-checks `csr.ncols() == ncols` so a trailing-empty-column CSR cannot
/// be prepared under a narrower width.
fn validate_csr(csr: &Csr, ncols: usize) -> Result<SparseShape, SparsePlanError> {
    let row_ptr = csr.row_ptr();
    let col = csr.col();
    if row_ptr.is_empty() {
        return Err(SparsePlanError::EmptyRowPtr);
    }
    if row_ptr[0] != 0 {
        return Err(SparsePlanError::NonZeroStart { start: row_ptr[0] });
    }
    for i in 1..row_ptr.len() {
        if row_ptr[i] < row_ptr[i - 1] {
            return Err(SparsePlanError::NotMonotonic { index: i });
        }
    }
    let end = *row_ptr.last().expect("row_ptr checked non-empty");
    if end as usize != col.len() {
        return Err(SparsePlanError::NnzMismatch {
            row_ptr_end: end,
            col_len: col.len(),
        });
    }
    if csr.ncols() != ncols {
        return Err(SparsePlanError::NcolsMismatch {
            stored: csr.ncols(),
            declared: ncols,
        });
    }
    for (edge, &c) in col.iter().enumerate() {
        if c as usize >= ncols {
            return Err(SparsePlanError::ColumnOutOfRange {
                edge,
                col: c,
                ncols,
            });
        }
    }
    for row in 0..row_ptr.len().saturating_sub(1) {
        let start = row_ptr[row] as usize;
        let end = row_ptr[row + 1] as usize;
        for edge in start.saturating_add(1)..end {
            if col[edge] < col[edge - 1] {
                return Err(SparsePlanError::RowUnsorted { row, edge });
            }
        }
    }

    let nrows = row_ptr.len() - 1;
    let nnz = col.len();
    // Monotonicity is already established above, so every difference is
    // non-negative and this cannot underflow.
    let max_row_nnz = (0..nrows)
        .map(|r| (row_ptr[r + 1] - row_ptr[r]) as usize)
        .max()
        .unwrap_or(0);
    for (what, value) in [("nrows", nrows), ("ncols", ncols), ("nnz", nnz)] {
        if value > u32::MAX as usize {
            return Err(SparsePlanError::TooLarge { what, value });
        }
    }

    Ok(SparseShape {
        nrows,
        ncols,
        nnz,
        max_row_nnz,
    })
}

// ---------------------------------------------------------------------------
// CPU kernels
// ---------------------------------------------------------------------------

/// Contiguous line ranges with roughly equal total nnz.
///
/// `ptr` is a CSR `row_ptr` or CSC `col_ptr` (`n + 1` entries). Each range is
/// a half-open `[start, end)` over lines. Whole lines are never split, so a
/// single hub longer than `total / parts` occupies one part by itself — the
/// remaining parts then balance the leftover work instead of leaving every
/// other Rayon worker on a two-edge leaf.
fn nnz_balanced_partitions(ptr: &[u32], parts: usize) -> Vec<(usize, usize)> {
    let n = ptr.len().saturating_sub(1);
    if n == 0 {
        return Vec::new();
    }
    let parts = parts.clamp(1, n);
    let mut out = Vec::with_capacity(parts);
    let mut start = 0usize;
    for p in 0..parts {
        if start >= n {
            break;
        }
        let parts_left = parts - p;
        if parts_left == 1 {
            out.push((start, n));
            break;
        }
        let nnz_left = (ptr[n] - ptr[start]) as usize;
        let target = nnz_left.div_ceil(parts_left);
        let mut end = start + 1;
        while end < n && ((ptr[end] - ptr[start]) as usize) < target {
            end += 1;
        }
        out.push((start, end));
        start = end;
    }
    out
}

/// How many nnz-balanced parts to spawn per Rayon worker.
///
/// One part per worker balances hub rows but freezes the plan: Rayon cannot
/// steal unfinished lines from a slow core. On heterogeneous pools (Apple
/// P+E) that turns the wall time into the slowest E-core's share. Measured on
/// M5 Pro (6P+12E), uniform SpMV at n=20k: 18 parts → 1.47 ms vs adaptive
/// per-row `par_iter` at 1.12 ms; 16× oversubscribe recovers to ~1.05 ms while
/// still keeping hub rows in their own part. Partition *build* cost is ~5 µs
/// and is not the regression.
const PARALLEL_PARTS_PER_WORKER: usize = 16;

/// Partition count for a parallel line walk: nnz-balanced parts with enough
/// oversubscribe for work-stealing, capped by the number of lines.
fn parallel_line_partitions(ptr: &[u32], lines: usize) -> Vec<(usize, usize)> {
    let workers = rayon::current_num_threads().max(1);
    let parts = workers
        .saturating_mul(PARALLEL_PARTS_PER_WORKER)
        .clamp(1, lines.max(1));
    let partitions = nnz_balanced_partitions(ptr, parts);
    #[cfg(test)]
    CPU_PARTITION_TASKS.store(partitions.len(), Ordering::Relaxed);
    partitions
}

#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

/// Number of Rayon tasks the last parallel line walk spawned. Test-only probe
/// for the hub-row imbalance regression: per-row `par_iter` would report
/// `nrows` here; nnz-balanced partitions report about
/// `workers * PARALLEL_PARTS_PER_WORKER`.
#[cfg(test)]
static CPU_PARTITION_TASKS: AtomicUsize = AtomicUsize::new(0);

#[inline]
fn row_dot(csr: &Csr, weights: &[f32], x: &[f32], r: usize) -> f32 {
    let start = csr.row_ptr()[r] as usize;
    let end = csr.row_ptr()[r + 1] as usize;
    let mut sum = 0.0f32;
    for i in start..end {
        sum += weights[i] * x[csr.col()[i] as usize];
    }
    sum
}

/// One output column of `Aᵀ · x`.
///
/// The CSC entry `k` names the CSR row it came from (`row[k]`, indexing `x`)
/// and the slot its value occupies in the forward weight table
/// (`edge_idx[k]`). One value table serves both directions, so a
/// [`SparseOp::set_weights`] is visible to the forward and transposed products
/// at once — there is no second copy to fall out of step.
#[inline]
fn col_dot(csc: &Csc, weights: &[f32], x: &[f32], c: usize) -> f32 {
    let start = csc.col_ptr[c] as usize;
    let end = csc.col_ptr[c + 1] as usize;
    let mut sum = 0.0f32;
    for k in start..end {
        sum += weights[csc.edge_idx[k] as usize] * x[csc.row[k] as usize];
    }
    sum
}

fn cpu_spmv_t_sequential(csc: &Csc, weights: &[f32], x: &[f32], y: &mut [f32]) {
    for (c, y_val) in y.iter_mut().enumerate() {
        *y_val += col_dot(csc, weights, x, c);
    }
}

fn cpu_spmv_t_parallel(csc: &Csc, weights: &[f32], x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;
    let partitions = parallel_line_partitions(&csc.col_ptr, y.len());
    let y_addr = y.as_mut_ptr() as usize;
    partitions.into_par_iter().for_each(|(start, end)| {
        for c in start..end {
            // SAFETY: partitions claim disjoint column ranges of `y`.
            let y_val = unsafe { &mut *(y_addr as *mut f32).add(c) };
            *y_val += col_dot(csc, weights, x, c);
        }
    });
}

/// One output row of `Y += A · X` for all `n_vec` vectors at once.
///
/// `x` and `y` are batch-minor (`[n][n_vec]`), matching `csr_spmm_kernel`. The
/// inner loop is over the row's non-zeros and the innermost over the batch, so
/// each `weights[i]` is loaded once and reused `n_vec` times, and both operands
/// are walked contiguously.
///
/// The traversal order over `i` is identical to [`row_dot`], so a batch of one
/// produces bit-identical results to [`SparseOp::spmv`] rather than merely
/// close ones — which is what makes that a usable test rather than a tolerance
/// argument.
///
/// # Why this tiles instead of accumulating straight into `out`
///
/// The obvious shape — `out[v] += w * x[base + v]` over the row's non-zeros —
/// accumulates into *memory*, one load-modify-store per non-zero per vector,
/// where [`row_dot`] keeps its running sum in a register. Measured at
/// `n_vec = 1`, n = 5000, that cost 42.7 ms against SpMV's 0.95 ms: a batched
/// call was 45x slower than the scalar one it was meant to subsume.
///
/// Accumulating a tile of vectors in a stack array fixes it without giving up
/// what batching is for. Each `weights[i]` is still loaded once per tile, `x`
/// is still walked contiguously, and the accumulators stay in registers.
/// [`BATCH_TILE`] is the tile width; a row with fewer vectors than that simply
/// runs one short tile.
#[inline]
fn row_dot_batched(csr: &Csr, weights: &[f32], x: &[f32], n_vec: usize, r: usize, out: &mut [f32]) {
    let start = csr.row_ptr()[r] as usize;
    let end = csr.row_ptr()[r + 1] as usize;
    let mut v0 = 0usize;
    while v0 < n_vec {
        let width = BATCH_TILE.min(n_vec - v0);
        let mut acc = [0.0f32; BATCH_TILE];
        // Zipped rather than indexed by `i`: same order, so the documented
        // bit-identity with `row_dot` holds, and the bounds checks on both
        // slices go away.
        for (&w, &col) in weights[start..end].iter().zip(&csr.col()[start..end]) {
            let base = col as usize * n_vec + v0;
            // The slices make the width known to the optimiser at the point of
            // use, so the tile stays in registers rather than being indexed.
            for (a, xv) in acc[..width].iter_mut().zip(&x[base..base + width]) {
                *a += w * xv;
            }
        }
        for (o, a) in out[v0..v0 + width].iter_mut().zip(&acc[..width]) {
            *o += *a;
        }
        v0 += width;
    }
}

/// Vectors accumulated in registers at once by [`row_dot_batched`].
///
/// Sixteen f32 accumulators is two 8-wide NEON registers, which fits the
/// register file alongside the row pointers without spilling.
const BATCH_TILE: usize = 16;

fn cpu_spmm_sequential(csr: &Csr, weights: &[f32], x: &[f32], n_vec: usize, y: &mut [f32]) {
    // A batch of one has nothing to batch, and the tiled accumulator carries
    // setup that a single vector cannot amortise: measured 39.7 ms against
    // SpMV's 10.6 ms at n = 10000 before this branch existed. The scalar path
    // is already the best implementation of this case, and dispatching to it
    // keeps the documented bit-identity trivially true rather than merely
    // arranged for.
    if n_vec == 1 {
        cpu_spmv_sequential(csr, weights, x, y);
        return;
    }
    for (r, row_out) in y.chunks_mut(n_vec).enumerate() {
        row_dot_batched(csr, weights, x, n_vec, r, row_out);
    }
}

fn cpu_spmm_parallel(csr: &Csr, weights: &[f32], x: &[f32], n_vec: usize, y: &mut [f32]) {
    use rayon::prelude::*;
    if n_vec == 1 {
        cpu_spmv_parallel(csr, weights, x, y);
        return;
    }
    let nrows = y.len() / n_vec;
    let partitions = parallel_line_partitions(csr.row_ptr(), nrows);
    let y_addr = y.as_mut_ptr() as usize;
    partitions.into_par_iter().for_each(|(start, end)| {
        for r in start..end {
            // SAFETY: partitions claim disjoint row ranges of `y`; the address
            // is only used inside those ranges.
            let row_out = unsafe {
                std::slice::from_raw_parts_mut((y_addr as *mut f32).add(r * n_vec), n_vec)
            };
            row_dot_batched(csr, weights, x, n_vec, r, row_out);
        }
    });
}

/// One row of `A · s` for a bitpacked spike vector.
///
/// Deliberately the same shape as [`row_dot`]: decode the bit to `0.0`/`1.0`
/// and multiply, rather than branching on it. That keeps the operations and
/// their order identical to the dense path, so the two agree bit for bit
/// instead of within a tolerance — and it is why there is no
/// `tolerance_for_spmv_spikes`.
#[inline]
fn row_dot_spikes(csr: &Csr, weights: &[f32], spikes: &[u32], r: usize) -> f32 {
    let start = csr.row_ptr()[r] as usize;
    let end = csr.row_ptr()[r + 1] as usize;
    let mut sum = 0.0f32;
    // Zipped rather than index-walked, but still strictly in index order: the
    // bit-identity claim against the dense path is a claim about summation
    // order, so this may not be reassociated or reordered.
    for (w, &col) in weights[start..end].iter().zip(&csr.col()[start..end]) {
        let c = col as usize;
        let bit = (spikes[c / 32] >> (c % 32)) & 1;
        sum += w * bit as f32;
    }
    sum
}

fn cpu_spmv_spikes_sequential(csr: &Csr, weights: &[f32], spikes: &[u32], y: &mut [f32]) {
    for (r, y_val) in y.iter_mut().enumerate() {
        *y_val += row_dot_spikes(csr, weights, spikes, r);
    }
}

fn cpu_spmv_spikes_parallel(csr: &Csr, weights: &[f32], spikes: &[u32], y: &mut [f32]) {
    use rayon::prelude::*;
    let partitions = parallel_line_partitions(csr.row_ptr(), y.len());
    let y_addr = y.as_mut_ptr() as usize;
    partitions.into_par_iter().for_each(|(start, end)| {
        for r in start..end {
            // SAFETY: partitions claim disjoint row ranges of `y`.
            let y_val = unsafe { &mut *(y_addr as *mut f32).add(r) };
            *y_val += row_dot_spikes(csr, weights, spikes, r);
        }
    });
}

fn cpu_spmv_sequential(csr: &Csr, weights: &[f32], x: &[f32], y: &mut [f32]) {
    for (r, y_val) in y.iter_mut().enumerate() {
        *y_val += row_dot(csr, weights, x, r);
    }
}

fn cpu_spmv_parallel(csr: &Csr, weights: &[f32], x: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;
    let partitions = parallel_line_partitions(csr.row_ptr(), y.len());
    let y_addr = y.as_mut_ptr() as usize;
    partitions.into_par_iter().for_each(|(start, end)| {
        for r in start..end {
            // SAFETY: partitions claim disjoint row ranges of `y`.
            let y_val = unsafe { &mut *(y_addr as *mut f32).add(r) };
            *y_val += row_dot(csr, weights, x, r);
        }
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
    let partitions = parallel_line_partitions(csr.row_ptr(), v.len());
    let v_addr = v.as_mut_ptr() as usize;
    let th_addr = theta.as_mut_ptr() as usize;
    let spk_addr = spikes.as_mut_ptr() as usize;
    partitions.into_par_iter().for_each(|(start, end)| {
        for r in start..end {
            // SAFETY: partitions claim disjoint row indices across all three
            // arrays; addresses are only used inside those ranges.
            let v_i = unsafe { &mut *(v_addr as *mut f32).add(r) };
            let th_i = unsafe { &mut *(th_addr as *mut f32).add(r) };
            let spk = unsafe { &mut *(spk_addr as *mut bool).add(r) };
            let current = row_dot(csr, weights, x, r);
            lif_step(v_i, th_i, current, spk, p);
        }
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
    /// `tolerance_for_spmv` was invisible to the entire suite. These
    /// bounds close that: the tolerance has to stay small enough to still be a
    /// test.
    #[test]
    fn tolerances_are_bounded_above_as_well_as_below() {
        // The M5 Pro measurement that calibrated this: 1000 nnz/row with terms
        // bounded by 0.25 deviated by 3.3e-6.
        let measured_worst_case = 3.3e-6f32;
        let t = tolerance_for_spmv(1000, 0.25, 0.25);
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
    fn tolerance_grows_with_row_density_result_and_magnitude() {
        assert!(tolerance_for_spmv(10, 1.0, 1.0) < tolerance_for_spmv(1000, 1.0, 1.0));
        assert!(tolerance_for_spmv(100, 1.0, 1.0) < tolerance_for_spmv(100, 10.0, 1.0));
        // The result magnitude must move the bound too. This is the axis whose
        // absence let the soak failure through.
        assert!(tolerance_for_spmv(100, 1.0, 1.0) < tolerance_for_spmv(100, 1.0, 50.0));
    }

    /// Degenerate arguments must still yield a usable positive tolerance rather
    /// than zero (which would make every comparison exact) or NaN (which would
    /// make every comparison pass, since `NaN <= NaN` is false but `d <= NaN`
    /// is too — an assertion that can never hold reads as a failing test, but a
    /// zero tolerance reads as a passing one).
    #[test]
    fn degenerate_arguments_yield_finite_positive_tolerances() {
        for (n, mag) in [(0usize, 0.0f32), (0, 1.0), (1, 0.0), (usize::MAX, 1.0)] {
            let t = tolerance_for_spmv(n, mag, mag);
            assert!(t > 0.0 && t.is_finite(), "n={n} mag={mag} gave {t}");
        }
        for mag in [0.0f32, -1.0, f32::MIN_POSITIVE] {
            let t = tolerance_for_elementwise(mag);
            assert!(t > 0.0 && t.is_finite(), "mag={mag} gave {t}");
        }
    }
}

#[cfg(test)]
mod nnz_balance_tests {
    use super::*;

    /// Hub + many leaves: per-row Rayon would spawn one task per row. The
    /// nnz-balanced plan must put the leftover leaf work into the non-hub
    /// parts instead of leaving them idle on a two-edge leaf each.
    #[test]
    fn hub_row_partitions_balance_leaf_work() {
        let hub = 12_000u32;
        let leaves = 3_000u32;
        let mut row_ptr = vec![0u32, hub];
        for i in 0..leaves {
            row_ptr.push(hub + i + 1);
        }
        let workers = 8usize;
        let parts = nnz_balanced_partitions(&row_ptr, workers);
        assert_eq!(parts.len(), workers.min(row_ptr.len() - 1));
        assert_eq!(
            parts.first().copied(),
            Some((0, 1)),
            "hub row is its own first part"
        );
        let leaf_nnz: Vec<usize> = parts[1..]
            .iter()
            .map(|&(s, e)| (row_ptr[e] - row_ptr[s]) as usize)
            .collect();
        let max_leaf = *leaf_nnz.iter().max().unwrap();
        let min_leaf = *leaf_nnz.iter().min().unwrap();
        assert!(
            max_leaf - min_leaf <= 1,
            "leaf parts must be nnz-balanced, got {leaf_nnz:?}"
        );
        assert_eq!(
            leaf_nnz.iter().sum::<usize>(),
            leaves as usize,
            "every leaf edge must land in a non-hub part"
        );
    }

    /// The parallel SpMV path must not fall back to one Rayon task per row on
    /// a hub fixture: that is the idle-core failure the partitioner exists to
    /// kill. Task count is recorded by `parallel_line_partitions`.
    #[test]
    fn parallel_spmv_spawns_partition_tasks_not_per_row() {
        let nrows = 2_048usize;
        let hub = 8_000usize;
        let mut adj = vec![vec![0u32]; nrows];
        adj[0] = (0..hub as u32).collect();
        for (r, row) in adj.iter_mut().enumerate().skip(1) {
            *row = vec![(r as u32) % (nrows as u32)];
        }
        let csr = Csr::from_adjacency(&adj);
        let weights = vec![1.0f32; csr.nnz()];
        let x = vec![1.0f32; csr.ncols()];
        let mut y = vec![0.0f32; csr.nrows()];
        CPU_PARTITION_TASKS.store(usize::MAX, Ordering::Relaxed);
        cpu_spmv_parallel(&csr, &weights, &x, &mut y);
        let tasks = CPU_PARTITION_TASKS.load(Ordering::Relaxed);
        let workers = rayon::current_num_threads().max(1);
        let max_parts = workers.saturating_mul(PARALLEL_PARTS_PER_WORKER).min(nrows);
        assert!(
            tasks > 1 && tasks <= max_parts && tasks < nrows,
            "expected nnz-balanced oversubscribe (<= {max_parts}), got {tasks} (nrows={nrows}, workers={workers})"
        );
    }

    #[test]
    fn nnz_balanced_spmv_matches_sequential_on_hub_fixture() {
        let nrows = 512usize;
        let mut adj = vec![vec![0u32]; nrows];
        adj[0] = (0..4_000u32).collect();
        for (r, row) in adj.iter_mut().enumerate().skip(1) {
            *row = vec![0, 1, (r as u32) % 7];
            row.sort();
        }
        let csr = Csr::from_adjacency(&adj);
        let weights: Vec<f32> = (0..csr.nnz()).map(|i| (i % 11) as f32 * 0.1).collect();
        let x: Vec<f32> = (0..csr.ncols()).map(|i| (i % 5) as f32).collect();
        let mut y_seq = vec![0.25f32; csr.nrows()];
        let mut y_par = y_seq.clone();
        cpu_spmv_sequential(&csr, &weights, &x, &mut y_seq);
        cpu_spmv_parallel(&csr, &weights, &x, &mut y_par);
        assert_eq!(y_seq, y_par);
    }

    /// Uniform graphs still oversubscribe so heterogeneous cores can steal;
    /// a 1:1 workers-parts plan is exactly the n=20k Rayon regression.
    #[test]
    fn parallel_spmv_oversubscribes_beyond_worker_count() {
        let nrows = 2_048usize;
        let csr = Csr::from_adjacency(
            &(0..nrows)
                .map(|r| vec![(r as u32) % (nrows as u32); 8])
                .collect::<Vec<_>>(),
        );
        let weights = vec![1.0f32; csr.nnz()];
        let x = vec![1.0f32; csr.ncols()];
        let mut y = vec![0.0f32; csr.nrows()];
        CPU_PARTITION_TASKS.store(0, Ordering::Relaxed);
        cpu_spmv_parallel(&csr, &weights, &x, &mut y);
        let tasks = CPU_PARTITION_TASKS.load(Ordering::Relaxed);
        let workers = rayon::current_num_threads().max(1);
        assert!(
            tasks > workers,
            "expected oversubscribe for stealability, got {tasks} tasks on {workers} workers"
        );
    }
}
