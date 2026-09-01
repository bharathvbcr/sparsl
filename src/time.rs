//! Discrete simulation time.
//!
//! Ticks are integers, not floats, and that is a determinism requirement rather
//! than a convenience. Accumulating a floating-point `dt` drifts, and two runs
//! that drift differently stop being comparable however carefully the kernels
//! are pinned. An integer tick multiplied by a fixed step is exact at every
//! horizon.

/// Discrete simulation tick (integer time for determinism).
pub type Tick = u64;
