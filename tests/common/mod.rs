//! Shared fixtures and comparison helpers for the sparsl test suites.
#![allow(dead_code)]

use sparsl::{Backend, Csr, Device, LifParams, Rng, SparseShape};

/// Deterministic random CSR with per-row degree in `0..=max_deg`.
pub fn random_csr(nrows: usize, ncols: usize, max_deg: usize, rng: &mut Rng) -> Csr {
    let mut adj: Vec<Vec<u32>> = Vec::with_capacity(nrows);
    for _ in 0..nrows {
        let deg = if max_deg == 0 {
            0
        } else {
            rng.gen_index(max_deg + 1)
        };
        let mut row = Vec::with_capacity(deg);
        for _ in 0..deg {
            if ncols > 0 {
                row.push(rng.gen_index(ncols) as u32);
            }
        }
        adj.push(row);
    }
    Csr::from_adjacency(&adj)
}

/// Uniform values in `[-scale, scale)`.
pub fn random_vec(n: usize, scale: f32, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect()
}

/// The largest `|weight * x|` any row sum can contain. Feeds the error bound.
pub fn max_abs_term(weights: &[f32], x: &[f32]) -> f32 {
    let w = weights.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let xm = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    (w * xm).max(f32::MIN_POSITIVE)
}

/// Every backend that can execute, minus the CPU reference itself.
pub fn backends_under_test() -> Vec<Backend> {
    sparsl::available_backends()
        .into_iter()
        .filter(|b| *b != Backend::CpuSequential)
        .collect()
}

/// The reference every other backend is checked against.
pub fn reference() -> Device {
    Device::cpu_sequential()
}

pub fn default_params() -> LifParams {
    LifParams::new(0.9, 0.0, 0.1).expect("finite LIF parameters")
}

/// Assert two float slices agree within `tol`, reporting the worst offender.
pub fn assert_close(got: &[f32], want: &[f32], tol: f32, context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: length mismatch");
    let mut worst = 0.0f32;
    let mut worst_i = 0usize;
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let d = if g.is_nan() && w.is_nan() {
            0.0
        } else {
            (g - w).abs()
        };
        if d > worst {
            worst = d;
            worst_i = i;
        }
    }
    assert!(
        worst <= tol,
        "{context}: max |Δ| = {worst} at index {worst_i} \
         (got {}, want {}), tolerance {tol}",
        got[worst_i],
        want[worst_i]
    );
}

/// Outcome of comparing a LIF step across backends.
pub struct LifComparison {
    pub flips: usize,
    pub compared: usize,
}

/// Compare a LIF step, tolerating spike flips only where the membrane landed
/// within `tol` of threshold.
///
/// A float difference of a few ulps in the synaptic current can push a membrane
/// from just-below to just-above threshold, which flips a boolean — an
/// arbitrarily large difference in output from an arbitrarily small difference
/// in input. Demanding exact spike agreement across substrates would therefore
/// be demanding exact float agreement, which no two reduction orders provide.
///
/// What *is* required: a flip may only happen where the reference membrane was
/// within `tol` of its threshold, and wherever the spike decision agrees, the
/// resulting state must agree to `tol`. A flip anywhere else is a real bug.
#[allow(clippy::too_many_arguments)]
pub fn compare_lif(
    v_got: &[f32],
    theta_got: &[f32],
    spikes_got: &[bool],
    v_want: &[f32],
    theta_want: &[f32],
    spikes_want: &[bool],
    v_pre: &[f32],
    theta_pre: &[f32],
    current_ref: &[f32],
    params: LifParams,
    tol: f32,
    context: &str,
) -> LifComparison {
    let n = v_want.len();
    let mut flips = 0usize;
    for i in 0..n {
        if spikes_got[i] == spikes_want[i] {
            let d_v = (v_got[i] - v_want[i]).abs();
            let d_t = (theta_got[i] - theta_want[i]).abs();
            assert!(
                d_v <= tol,
                "{context}: v[{i}] differs by {d_v} (> {tol}) with matching spike decision"
            );
            assert!(
                d_t <= tol,
                "{context}: theta[{i}] differs by {d_t} (> {tol}) with matching spike decision"
            );
            continue;
        }
        // Spike decisions disagree. Legal only in the boundary band.
        let membrane = v_pre[i] * params.decay() + current_ref[i];
        let margin = (membrane - theta_pre[i]).abs();
        assert!(
            margin <= tol,
            "{context}: spike[{i}] flipped ({} vs {}) with membrane {margin} from threshold, \
             which is outside the {tol} band — this is a real disagreement, not a boundary case",
            spikes_got[i],
            spikes_want[i]
        );
        flips += 1;
    }
    LifComparison { flips, compared: n }
}

/// Shapes chosen to sit on every boundary the backends care about.
///
/// Threadgroup width is 256 and the SIMD width is 32, so the sizes straddle
/// both: a bug in tail handling shows up as a wrong row at exactly one of
/// these. The degenerate shapes at the front catch the zero-length allocation
/// and empty-dispatch paths that a "reasonable sizes" sweep never reaches.
pub const SHAPES: &[(usize, usize, usize)] = &[
    // (nrows, ncols, max_degree)
    (0, 0, 0),
    (0, 8, 4),
    (1, 1, 0),
    (1, 1, 1),
    (1, 64, 64),
    (7, 7, 3),
    (31, 31, 5),
    (32, 32, 5),
    (33, 33, 5),
    (63, 64, 8),
    (64, 64, 8),
    (65, 64, 8),
    (255, 128, 16),
    (256, 128, 16),
    (257, 128, 16),
    (512, 512, 1),
    (1000, 1000, 64),
    (2048, 512, 3),
];

pub fn shape_label(shape: SparseShape) -> String {
    format!(
        "nrows={} ncols={} nnz={}",
        shape.nrows, shape.ncols, shape.nnz
    )
}
