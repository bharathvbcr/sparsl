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
//! [`backend::tolerance_for_nnz_per_row`].
//!
//! **2. A backend handle cannot lie about what ran.** [`backend::Device`] is
//! only constructible for a substrate that can actually execute, and
//! [`backend::Device::label`] reports what executed rather than what was asked
//! for. This is not decoration: the code this crate was extracted from had a
//! `use_gpu: bool` that no dispatch path ever read, so "GPU" benchmarks timed
//! CPU code and reported ~1.00x speedups as real. Every unavailable backend
//! here fails loudly at construction instead.

pub mod backend;
pub mod buffer;
pub mod rng;
pub mod scan;
pub mod simd;
pub mod sparse;
pub mod time;

pub use backend::{
    available_backends, tolerance_for_elementwise, tolerance_for_nnz_per_row, Backend,
    BackendUnavailable, Device, LifParams, LifParamsError, OpError, SparseOp, SparsePlanError,
    SparseShape,
};
pub use buffer::Buffer;
pub use rng::Rng;
pub use scan::{assoc_scan, assoc_scan_chunked, assoc_scan_sequential, State, DEFAULT_CHUNK_SIZE};
pub use simd::{scalar_leak_integrate, simd_leak_integrate, LANES};
pub use sparse::{Csc, Csr, CsrError};
pub use time::Tick;
