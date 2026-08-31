//! Chunked associative scan (U03).
//!
//! # Spike reset is a sequential barrier (v7 F1)
//!
//! This primitive parallelizes the **linear sub-threshold** membrane recurrence
//! only. A hard spike **reset is a sequential, data-dependent barrier**: it
//! breaks the affine structure that makes an associative scan valid, and this
//! scan does **NOT** parallelize across reset events.
//!
//! Use [`assoc_scan`] inside reset-free chunks (or between resets). Across a
//! reset, fall back to sequential time. Neuron / area / stream parallelism
//! remains the primary throughput lever; chunked scan is only a partial
//! parallel-in-time assist for the linear dynamics.

use rayon::prelude::*;

/// Default chunk length for the parallel scan (elements per chunk).
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Affine sub-threshold step: `v' = a · v + b`.
///
/// Composing steps is an associative monoid with identity `(a=1, b=0)`, which
/// is what enables a chunked prefix scan over reset-free segments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct State {
    /// Multiplier applied to the incoming voltage.
    pub a: f32,
    /// Additive drive after the leak scale.
    pub b: f32,
}

impl State {
    /// Monoid identity: `v' = v`.
    #[inline]
    pub const fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// One Euler leak/integrate step as an affine map:
    /// `v' = (1 − dt/τ) · v + (dt/τ) · input`.
    #[inline]
    pub fn leak_step(input: f32, tau: f32, dt: f32) -> Self {
        let alpha = dt / tau;
        Self {
            a: 1.0 - alpha,
            b: alpha * input,
        }
    }

    /// Apply this map to a voltage.
    #[inline]
    pub fn apply(self, v: f32) -> f32 {
        self.a * v + self.b
    }

    /// Compose `self` then `next` (left-to-right in time).
    ///
    /// `combine(x, y)` means "apply `x`, then apply `y`":
    /// `v ↦ y.a · (x.a · v + x.b) + y.b`.
    #[inline]
    pub fn combine(self, next: Self) -> Self {
        Self {
            a: next.a * self.a,
            b: next.a * self.b + next.b,
        }
    }
}

/// Inclusive prefix scan of affine [`State`] values using `combine`.
///
/// Chunked: work is split into windows of [`DEFAULT_CHUNK_SIZE`] so independent
/// chunks can run in parallel (via `rayon`). Parenthesization is pure
/// left-fold, so results match [`assoc_scan_sequential`] **exactly** — bit for
/// bit — on the linear recurrence.
///
/// # Spike reset barrier
///
/// **Spike reset is a sequential barrier.** This scan does **not** parallelize
/// across reset events (v7 F1). Only linear sub-threshold segments are valid
/// inputs; callers must split on resets and scan each chunk independently.
pub fn assoc_scan<F>(xs: &[State], combine: F) -> Vec<State>
where
    F: Fn(State, State) -> State + Sync,
{
    assoc_scan_chunked(xs, DEFAULT_CHUNK_SIZE, combine)
}

/// Chunked inclusive scan with an explicit `chunk_size` (primarily for tests).
///
/// Same contract as [`assoc_scan`]: left-fold exact, parallel across chunks,
/// **no** parallelism across spike resets.
///
/// Offsets at chunk boundaries are computed with a sequential left-fold (so
/// parenthesization matches [`assoc_scan_sequential`] bit-for-bit; reassociating
/// pre-folded chunk totals would drift in `f32`). Chunk bodies then run in
/// parallel from those offsets.
pub fn assoc_scan_chunked<F>(xs: &[State], chunk_size: usize, combine: F) -> Vec<State>
where
    F: Fn(State, State) -> State + Sync,
{
    assert!(chunk_size > 0, "chunk_size must be > 0");
    let n = xs.len();
    if n == 0 {
        return Vec::new();
    }
    // Small inputs: sequential left-fold (exact, no thread overhead).
    if n <= chunk_size {
        return assoc_scan_sequential(xs, combine);
    }

    let n_chunks = n.div_ceil(chunk_size);

    // Phase 1 — sequential left-fold, recording the inclusive prefix at each
    // chunk boundary. `offsets[c]` = left-fold(xs[0 .. c * chunk_size]), i.e.
    // the sequential scan value just before chunk `c` begins.
    // Using element-wise left-fold (not reassociated chunk totals) keeps f32
    // results identical to a pure sequential scan.
    let mut offsets = vec![State::identity(); n_chunks];
    let mut acc = xs[0];
    // `chunk_size` is a runtime parameter, so `i % chunk_size` and
    // `i / chunk_size` were emitting a hardware integer division *per element*
    // in this loop — ~20-40 cycles each on arm64, and uncancellable by the
    // compiler since the divisor is not a compile-time constant. That division
    // dominated the loop body, which is otherwise one `combine` call.
    //
    // A running counter gives the same boundaries with a compare and an add.
    // `combine` is applied to exactly the same elements in exactly the same
    // order, so the f32 results are bit-identical.
    let mut since_boundary = 1usize;
    let mut chunk_idx = 0usize;
    // Indexing is deliberate: see the comment above. An iterator rewrite would
    // reassociate `combine`, and bit-identical f32 output is what the
    // `--config-hash` replay property rests on.
    #[allow(clippy::needless_range_loop)]
    for i in 1..n {
        // At the top of iteration `i`, `since_boundary == i - last_boundary`,
        // so this fires exactly when `i % chunk_size == 0`.
        if since_boundary == chunk_size {
            chunk_idx += 1;
            offsets[chunk_idx] = acc;
            since_boundary = 0;
        }
        acc = combine(acc, xs[i]);
        since_boundary += 1;
    }

    // Phase 2 — left-fold each chunk from its offset (parallel across chunks).
    let mut out = vec![State::identity(); n];
    out.par_chunks_mut(chunk_size)
        .zip(offsets.par_iter().copied())
        .enumerate()
        .for_each(|(c, (chunk_out, offset))| {
            let start = c * chunk_size;
            let end = (start + chunk_size).min(n);
            let len = end - start;
            // Chunk 0: start from xs[0] directly (avoid identity⊕x, which can
            // perturb signed zeros / ulps). Later chunks: continue from the
            // exact sequential prefix at the boundary.
            let mut acc = if c == 0 {
                let first = xs[start];
                chunk_out[0] = first;
                first
            } else {
                offset
            };
            let begin = if c == 0 { 1 } else { 0 };
            for i in begin..len {
                acc = combine(acc, xs[start + i]);
                chunk_out[i] = acc;
            }
        });

    out
}

/// Sequential inclusive left-fold scan (reference for parity tests).
pub fn assoc_scan_sequential<F>(xs: &[State], combine: F) -> Vec<State>
where
    F: Fn(State, State) -> State,
{
    if xs.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(xs.len());
    let mut acc = xs[0];
    out.push(acc);
    for &x in &xs[1..] {
        acc = combine(acc, x);
        out.push(acc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{assoc_scan, assoc_scan_chunked, assoc_scan_sequential, State, DEFAULT_CHUNK_SIZE};
    use crate::rng::Rng;

    fn combine(a: State, b: State) -> State {
        a.combine(b)
    }

    fn random_states(n: usize, seed: u64) -> Vec<State> {
        let mut rng = Rng::new(seed);
        (0..n)
            .map(|_| {
                let tau = 0.5 + rng.next_f32() * 4.0;
                let input = rng.next_f32() * 2.0 - 1.0;
                let dt = 1.0 + rng.next_f32();
                State::leak_step(input, tau, dt)
            })
            .collect()
    }

    #[test]
    fn identity_is_neutral() {
        let s = State { a: 0.75, b: -0.25 };
        assert_eq!(State::identity().combine(s), s);
        assert_eq!(s.combine(State::identity()), s);
    }

    #[test]
    fn empty_scan() {
        let out = assoc_scan(&[], combine);
        assert!(out.is_empty());
    }

    #[test]
    fn singleton_scan() {
        let xs = [State::leak_step(1.0, 2.0, 1.0)];
        let out = assoc_scan(&xs, combine);
        assert_eq!(out, xs);
    }

    #[test]
    fn chunked_matches_sequential_exactly() {
        for &n in &[0, 1, 2, 16, 255, 256, 257, 512, 1000, 4096] {
            let xs = random_states(n, 0x5CA1_0000 + n as u64);
            let sequential = assoc_scan_sequential(&xs, combine);
            let chunked = assoc_scan(&xs, combine);
            assert_eq!(
                chunked, sequential,
                "assoc_scan must match sequential left-fold exactly (n={n})"
            );

            // Exercise non-default chunk sizes too.
            for &cs in &[1usize, 3, 7, 64, 128, DEFAULT_CHUNK_SIZE] {
                if cs == 0 {
                    continue;
                }
                let got = assoc_scan_chunked(&xs, cs, combine);
                assert_eq!(
                    got, sequential,
                    "chunk_size={cs} must match sequential (n={n})"
                );
            }
        }
    }

    #[test]
    fn linear_recurrence_matches_sequential_fold() {
        // Prefix states must match a sequential fold exactly. Applying the
        // composed affine map to v0 is algebraically the same as iterating
        // `step.apply`, but f32 may differ by a few ulps — tolerance covers that.
        let xs = random_states(500, 0xB177_5CA1);
        let scanned = assoc_scan(&xs, combine);
        let sequential = assoc_scan_sequential(&xs, combine);
        assert_eq!(
            scanned, sequential,
            "scan must match sequential fold exactly"
        );

        let v0 = 0.125f32;
        let mut v_seq = v0;
        for (t, &step) in xs.iter().enumerate() {
            v_seq = step.apply(v_seq);
            let v_scan = scanned[t].apply(v0);
            let err = (v_scan - v_seq).abs();
            assert!(
                err <= 1e-5,
                "composed map vs iterative apply at t={t}: scan={v_scan} seq={v_seq} err={err}"
            );
        }
    }

    #[test]
    fn combine_is_associative_on_samples() {
        // Algebraic check (f32 may still differ by ulps under reassociation;
        // the scan itself never reassociates — it left-folds only).
        let xs = random_states(3, 99);
        let (x, y, z) = (xs[0], xs[1], xs[2]);
        let left = x.combine(y).combine(z);
        let right = x.combine(y.combine(z));
        // Allow a tiny ulp gap for the algebraic property itself.
        assert!((left.a - right.a).abs() <= 1e-5);
        assert!((left.b - right.b).abs() <= 1e-5);
    }
}
