//! Deterministic sparse + scan compute kernels for event-driven simulation.
//!
//! `sparsl` holds the substrate-independent numeric core: structure-of-arrays
//! column buffers, a seeded RNG, CSR/CSC sparse connectivity, a chunked
//! associative scan over affine maps, a lane-shaped leak/integrate kernel, and
//! a [`backend`] layer that runs sparse and LIF work on CPU or GPU.
//!
//! # Two properties the whole crate is built around
//!
//! **1. Bit-reproducibility.** Same seed and same backend produce byte-identical
//! output. [`assoc_scan`] left-folds rather than reassociating, precisely so a
//! parallel scan matches a sequential one bit for bit. Nothing here autotunes,
//! because a kernel chosen by runtime benchmark differs per machine and changes
//! the reduction order — and therefore the floats.
//!
//! Bit-reproducibility holds *within* a backend, never *across* one. CPU and
//! GPU reduce in different orders and will differ in the last few ulps; see
//! [`backend::tolerance_for_spmv`].
//!
//! **2. A backend handle cannot lie about what ran.** [`backend::Device`] is
//! only constructible for a substrate that can actually execute, and
//! [`backend::Device::label`] reports what executed rather than what was asked
//! for. This is not decoration: the code this crate was extracted from had a
//! `use_gpu: bool` that no dispatch path ever read, so "GPU" benchmarks timed
//! CPU code and reported ~1.00x speedups as real. Every unavailable backend
//! here fails loudly at construction instead.
//!
//! # Quickstart
//!
//! ```
//! use sparsl::{Backend, Csr, Device};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Connectivity as adjacency, converted to CSR.
//! let csr = Csr::from_adjacency(&[vec![1, 2], vec![0], vec![0, 1]]);
//! let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! // Prefer the GPU, but never silently pretend to have one.
//! let device = Device::try_new(Backend::Metal)
//!     .unwrap_or_else(|_why| Device::cpu_parallel());
//!
//! // Validates every column index against `ncols` and uploads once.
//! let mut op = device.prepare(&csr, 3, &weights)?;
//!
//! let x = vec![0.5, 1.0, 1.5];
//! let mut y = vec![0.0; 3];
//! op.spmv(&x, &mut y)?;
//!
//! // `label` reports what executed, not what was asked for.
//! println!("ran on {}", device.label());
//! # Ok(())
//! # }
//! ```
//!
//! The fallback above is the shape to copy. [`Device::try_new`] returns
//! [`backend::BackendUnavailable`] with a reason rather than degrading
//! silently, so the caller decides what to do about a missing GPU and the log
//! says which substrate actually ran.
//!
//! # Module map
//!
//! | Module | What it holds |
//! |---|---|
//! | [`backend`] | [`Device`], [`SparseOp`], backend availability, and the tolerance functions |
//! | [`sparse`] | [`Csr`] and [`Csc`] connectivity, validated on construction |
//! | [`scan`] | The chunked associative scan over affine maps |
//! | [`buffer`] | Structure-of-arrays column buffers for cell state |
//! | [`spikes`] | Bitpacked spike vectors, 32 spikes per `u32` |
//! | [`half`] | `binary16` and `bfloat16` weight storage and conversion |
//! | [`rng`] | Seeded, reproducible random streams |
//! | [`simd`] | Portable SIMD helpers behind the CPU kernels |
//! | [`time`] | Simulation clock and step accounting |
//!
//! # Feature flags
//!
//! | Feature | Default | |
//! |---|---|---|
//! | `metal` | off | Native Metal compute on Apple silicon. Verified end to end against the CPU reference; see `tests/differential.rs`. |
//! | `cuda` | off | Declares *intent* to build a CUDA backend. It does **not** make CUDA available: no dispatch is implemented, so [`Backend::Cuda`] stays unconstructible and says so. It is a feature rather than a silent CPU fallback for the reason in property 2 above. |
//!
//! Without `metal` the crate builds and runs everywhere, with fewer arms in
//! [`available_backends`]. That is not a degraded mode: every differential,
//! stress, honesty and golden test runs against the arms that remain.
//!
//! # Notes for contributors
//!
//! **Tolerance functions are part of the contract.** Cross-backend comparisons
//! go through [`backend::tolerance_for_spmv`] and its siblings rather than a
//! literal epsilon. A tolerance that can be widened until a test passes is a
//! test that has stopped checking anything, so those formulas are bounded from
//! above as well as below, and a mutation that widened one to `f32::MAX` is in
//! the suite specifically because it once survived.
//!
//! **The kernels are held down by differential tests, not by golden values
//! alone.** Every available backend is checked against the CPU reference on the
//! same inputs, plus boundary shapes around the SIMD width (31, 32, 33) and the
//! threadgroup size (255, 256, 257), plus the degenerate zero-row, zero-edge and
//! single-row cases a sweep of reasonable sizes never reaches.
//!
//! **Kernel-written buffers carry sentinel tails.** A kernel that writes out of
//! bounds lands in page padding where nothing observes it, so every buffer the
//! GPU writes has a canary region checked after each dispatch. That check exists
//! because deleting a kernel's row bounds guard once passed the entire suite.
//!
//! **A `.metal` edit must reach the build.** `build.rs` declares
//! `src/kernels/spmv.metal` an explicit input. The shader reaches the crate
//! through `include_str!`, which Cargo does not treat as a rebuild trigger, so
//! without that line an edited kernel stays stale while the tests report a pass.

pub mod backend;
pub mod buffer;
pub mod half;
pub mod rng;
pub mod scan;
pub mod simd;
pub mod sparse;
pub mod spikes;
pub mod time;

pub use backend::{
    available_backends, tolerance_for_elementwise, tolerance_for_scan, tolerance_for_spmv,
    tolerance_for_spmv_bf16, tolerance_for_spmv_f16, tolerance_for_spmv_narrow, Backend,
    BackendUnavailable, Device, LifParams, LifParamsError, OpError, SparseOp, SparsePlanError,
    SparseShape, WeightPrecision,
};
pub use buffer::Buffer;
pub use rng::Rng;
pub use scan::{assoc_scan, assoc_scan_chunked, assoc_scan_sequential, State, DEFAULT_CHUNK_SIZE};
pub use simd::{scalar_leak_integrate, simd_leak_integrate, LANES};
pub use sparse::{Csc, Csr, CsrError};
pub use time::Tick;
