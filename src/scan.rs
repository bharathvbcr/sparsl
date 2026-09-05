//! Chunked associative scan over affine maps.
//!
//! # Spike reset is a sequential barrier
//!
//! This primitive parallelizes the **linear sub-threshold** membrane recurrence
//! only. A hard spike **reset is a sequential, data-dependent barrier**: it
//! breaks the affine structure that makes an associative scan valid, and this
//! scan does **NOT** parallelize across reset events.
//!
//! Use [`assoc_scan`] inside reset-free chunks (or between resets). Across a
//! reset, fall back to sequential time.
//!
//! # What this actually buys, measured
//!
//! Not speed. Phase 1 of [`assoc_scan_chunked`] is a *complete sequential
//! left-fold over every element* — it has to be, because recording the exact
//! prefix at each chunk boundary is what makes the parallel phase bit-identical
//! to a sequential scan. Phase 2 then redoes that work in parallel. So the
//! total is roughly `2n` `combine` calls to replace `n`.
//!
//! An earlier two-sample idle run suggested 6.92 ms chunked against 7.48 ms
//! sequential at 4M elements. The hardened sampler did not reproduce a CPU
//! crossover: paired median sequential/chunked speedups rose from 0.02x at 257
//! elements to 0.69x at 786K and remained below parity through 8M on the same
//! M5 Pro. Larger arms became noisy under host load, so there is no defensible
//! universal routing threshold in that run; the provenance and spreads are
//! evidence, while the old 1.08x point is only historical.
//!
//! What it buys is the *property*: a chunked, rayon-backed scan whose output is
//! bit-identical to the sequential fold. It does not currently justify choosing
//! this path for throughput; neuron, area and stream parallelism remain that
//! lever.
//!
//! # The tree scan is slower, measured
//!
//! This text used to say a Blelloch-style two-level scan "would deliver a real
//! speedup and would not be bit-identical". Half of that was right. The kernel
//! now exists — [`crate::Device::assoc_scan`] on `Backend::Metal` — and it is
//! **slower than the sequential CPU fold at every size measured**:
//!
//! The latest hardened run on the same M5 Pro used bounded warmup and eight
//! paired, order-alternating rounds. Its paired median CPU-sequential/Metal
//! speedups were 0.23x, 0.33x, 0.56x and 0.48x at 64K, 256K, 1M and 4M
//! elements: Metal did not reach parity anywhere in the measured range. Several
//! CPU arms had high max/min spread, so these are evidence for the direction of
//! the result, not universal latency constants. The benchmark prints those
//! spreads and full host/build provenance rather than hiding them behind one
//! optimistic sample.
//!
//! The reason is arithmetic intensity: composing two affine maps is three flops
//! over eight bytes in and eight out, so this is purely memory-bound, and the
//! three-phase tree makes roughly five passes over memory where the sequential
//! fold makes one. More bandwidth does not rescue an algorithm that spends it
//! on extra passes.
//!
//! So the tree scan is not a throughput lever either. It is there because
//! `Backend::Metal` should be able to run every operation this crate offers
//! rather than silently falling back — and `examples/scan_crossover.rs`
//! reproduces the table above so the claim stays checkable.

use rayon::prelude::*;

/// Default chunk length for the parallel scan (elements per chunk).
pub const DEFAULT_CHUNK_SIZE: usize = 256;

/// Affine sub-threshold step: `v' = a · v + b`.
///
/// Composing steps is an associative monoid with identity `(a=1, b=0)`, which
/// is what enables a chunked prefix scan over reset-free segments.
///
/// `repr(C)` is load-bearing, not decoration. The Metal scan uploads a
/// `&[State]` straight to the device and reads the result straight back, so the
/// in-memory layout *is* the wire format the kernel indexes as
/// `[a0, b0, a1, b1, …]`. Rust's default repr guarantees no field order, so
/// without this the upload could silently transpose every pair. The layout is
/// pinned by a compile-time assertion below and by
/// `state_layout_matches_flat_f32` in the test suite.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
pub struct State {
    /// Multiplier applied to the incoming voltage.
    pub a: f32,
    /// Additive drive after the leak scale.
    pub b: f32,
}

// Compile-time proof that `State` is exactly two packed `f32`s. The Metal
// backend's zero-reformat/direct typed byte-copy upload is sound only if this
// holds. Keep this anonymous:
// a named private const linked from public docs fails rustdoc's
// `private_intra_doc_links` lint, while a named-but-unused const warns on the
// crate's Rust 1.82 MSRV.
const _: () = {
    assert!(core::mem::size_of::<State>() == 2 * core::mem::size_of::<f32>());
    assert!(core::mem::align_of::<State>() == core::mem::align_of::<f32>());
};

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

/// Conservative magnitude envelope for comparing affine prefix scans.
///
/// This is the scale expected by [`crate::backend::tolerance_for_scan`]. It is
/// deliberately not just the largest output `|b|`: [`State`] is public and may
/// contain multipliers with `|a| > 1`, and cancellation can make an output tiny
/// even though the arithmetic that formed it was large.
///
/// The envelope tracks two different quantities:
///
/// ```text
/// A_suffix <- |a| max(1, A_suffix)
/// B_prefix <- |a| B_prefix + |b|
/// ```
///
/// and returns the largest value reached, floored at one. `B_prefix` is the sum
/// of the absolute affine contributions, so it remains conservative under
/// cancellation. `A_suffix` is the largest product of a contiguous multiplier
/// subchain ending at the current element. Tracking only the full prefix is not
/// sufficient: a tree may first multiply two large later states and only then
/// combine that intermediate with a tiny early multiplier. Every intermediate
/// produced by a valid reassociation is a contiguous subchain, so the suffix
/// recurrence covers exactly that missing scale. The recurrences are evaluated
/// in `f64` and rounded upward on conversion to `f32`. A non-finite input or an
/// envelope too large for `f32` returns infinity, signalling that no finite f32
/// comparison scale is available.
pub fn scan_magnitude_envelope(xs: &[State]) -> f32 {
    let (mut suffix_a, mut prefix_b, mut envelope) = (1.0f64, 0.0f64, 1.0f64);

    for state in xs {
        let (a, b) = (f64::from(state.a.abs()), f64::from(state.b.abs()));
        if !a.is_finite() || !b.is_finite() {
            return f32::INFINITY;
        }

        // A tree scan can materialise any contiguous multiplier subchain. If
        // the previous suffix product is below one, starting a new subchain at
        // this state is the larger choice; otherwise extending it is larger.
        suffix_a = a * suffix_a.max(1.0);
        prefix_b = a * prefix_b + b;
        if !suffix_a.is_finite() || !prefix_b.is_finite() {
            return f32::INFINITY;
        }
        envelope = envelope.max(suffix_a).max(prefix_b);
    }

    let rounded = envelope as f32;
    if !rounded.is_finite() || f64::from(rounded) >= envelope {
        rounded
    } else {
        // `envelope` is positive and `rounded` is finite, so the next larger
        // bit pattern is the next representable positive f32.
        f32::from_bits(rounded.to_bits() + 1)
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
/// across reset events. Only linear sub-threshold segments are valid
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
    use super::{
        assoc_scan, assoc_scan_chunked, assoc_scan_sequential, scan_magnitude_envelope, State,
        DEFAULT_CHUNK_SIZE,
    };
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
    fn magnitude_envelope_covers_growth_and_cancellation() {
        let cancelling = [State { a: 2.0, b: 3.0 }, State { a: -4.0, b: 12.0 }];
        // The composed b is exactly zero, but its two absolute contributions
        // sum to 24; the multiplier product contributes 8.
        assert_eq!(cancelling[0].combine(cancelling[1]).b, 0.0);
        assert_eq!(scan_magnitude_envelope(&cancelling), 24.0);

        let a = f32::from_bits(1.0f32.to_bits() + 1);
        let exact_product = f64::from(a) * f64::from(a);
        let rounded = scan_magnitude_envelope(&[State { a, b: 0.0 }; 2]);
        assert!(
            f64::from(rounded) >= exact_product,
            "envelope rounded down: {rounded} < {exact_product}"
        );
    }

    #[test]
    fn magnitude_envelope_covers_tree_local_multiplier_growth() {
        // A tree scan forms the suffix product of the final two states before
        // it combines that result with the tiny first multiplier. Looking only
        // at prefixes sees 1e-20, 1e-10 and ~1, but the actual tree
        // intermediate is ~1e20. The comparison scale must cover every
        // contiguous subchain a valid reassociation can materialise.
        let xs = [
            State { a: 1.0e-20, b: 0.0 },
            State { a: 1.0e10, b: 0.0 },
            State { a: 1.0e10, b: 0.0 },
        ];
        let tree_local_product = f64::from(xs[1].a) * f64::from(xs[2].a);
        let envelope = scan_magnitude_envelope(&xs);
        assert!(
            f64::from(envelope) >= tree_local_product,
            "envelope missed a tree-local product: {envelope} < {tree_local_product}"
        );
    }

    #[test]
    fn magnitude_envelope_refuses_non_finite_inputs() {
        assert!(scan_magnitude_envelope(&[State {
            a: f32::NAN,
            b: 0.0
        }])
        .is_infinite());
        assert!(scan_magnitude_envelope(&[State {
            a: 1.0,
            b: f32::INFINITY
        }])
        .is_infinite());
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
