//! CUDA backend: declared, never available.
//!
//! # Why this module exists and contains no kernels
//!
//! [`crate::Backend::Cuda`] is part of the backend enum so that callers,
//! reports and benchmark tables can name it, and so that adding real dispatch
//! later is a change inside this module rather than a redesign of the public
//! API. What it is *not* is a working backend, and nothing here pretends
//! otherwise.
//!
//! No dispatch is implemented because none could be verified. This crate was
//! developed on Apple silicon, where there is no NVIDIA device and no CUDA
//! toolchain: a CUDA path written here could not be compiled, let alone run
//! against the CPU reference. Shipping it as "supported" would reproduce
//! exactly the defect the honesty invariant in [`crate::backend`] exists to
//! prevent — a GPU-labelled path that nobody proved runs.
//!
//! So [`UNAVAILABLE_REASON`] is returned from
//! [`crate::Backend::unavailable_reason`], `Device::try_new(Backend::Cuda)`
//! fails, and `available_backends()` omits it. A benchmark that asks for CUDA
//! aborts instead of quietly timing rayon.
//!
//! # Landing CUDA
//!
//! The seams are already in place; none of them need the public API to change.
//!
//! 1. Add a `cudarc` (or equivalent) optional dependency behind the existing
//!    `cuda` feature.
//! 2. Add `CudaDevice` / `CudaSparse` here, mirroring
//!    the crate-private `backend::metal` module: `open()`, `prepare()`, `spmv()`,
//!    `lif_integrate()`, `fused_spmv_lif()`.
//! 3. Port `src/kernels/spmv.metal` to CUDA C. The three kernels are small and
//!    the bounds-guard contract is identical; `SparseOp::prepare` already
//!    guarantees `col[i] < ncols` before upload, so the column indexing needs
//!    no in-kernel range check on CUDA either.
//! 4. Add `DeviceInner::Cuda` / `OpResident::Cuda` arms in
//!    `src/backend/mod.rs`. The compiler will list every site.
//! 5. Point `UNAVAILABLE_REASON` at a real runtime probe, exactly as
//!    `Backend::Metal` does.
//!
//! Step 6 is the one that matters: `tests/differential.rs` is written against
//! `available_backends()`, so the moment CUDA reports available it is fuzzed
//! against the CPU reference on every shape in the suite, with no new test
//! code. Do not flip availability before that suite passes.

/// Why [`crate::Backend::Cuda`] cannot execute.
pub const UNAVAILABLE_REASON: &str =
    "CUDA dispatch is not implemented in sparsl; refusing to fall back to CPU under a GPU label";
