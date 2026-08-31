//! Every available backend, checked against the sequential CPU reference on the
//! same inputs.
//!
//! This is the suite that gates availability. `Backend::Cuda` will be fuzzed
//! here the moment it reports available, with no new test code — which is why
//! `src/backend/cuda.rs` says not to flip that flag until this passes.

mod common;

use common::*;
use proptest::prelude::*;
use sparsl::{tolerance_for_elementwise, tolerance_for_nnz_per_row, Backend, Device, Rng};

/// A suite that silently tests nothing must not look like a suite that passed.
///
/// Without this, building without `--features metal` would run every test below
/// against an empty backend list and report success, which is exactly the shape
/// of failure this crate exists to prevent.
#[test]
fn the_suite_actually_has_something_to_test() {
    let arms = backends_under_test();
    assert!(
        arms.contains(&Backend::CpuParallel),
        "the parallel CPU arm is unconditional and must always be under test"
    );

    if cfg!(all(target_os = "macos", feature = "metal")) {
        assert!(
            arms.contains(&Backend::Metal),
            "built with --features metal on macOS, but Metal is not available: {:?}. \
             Every GPU assertion below would silently pass without running.",
            Backend::Metal.unavailable_reason()
        );
    }
}

#[test]
fn spmv_matches_reference_across_shapes() {
    let reference = reference();
    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let mut rng = Rng::new(0xD1FF_0000 ^ backend as u64);

        for &(nrows, ncols, max_deg) in SHAPES {
            let csr = random_csr(nrows, ncols, max_deg, &mut rng);
            let weights = random_vec(csr.nnz(), 1.0, &mut rng);
            let x = random_vec(ncols, 1.0, &mut rng);
            let y0 = random_vec(nrows, 1.0, &mut rng);

            let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
            let op = device.prepare(&csr, ncols, &weights).expect("valid");
            let shape = op.shape();
            let tol = tolerance_for_nnz_per_row(shape.nnz_per_row(), max_abs_term(&weights, &x));

            let mut y_ref = y0.clone();
            let mut y_got = y0;
            op_ref.spmv(&x, &mut y_ref).expect("ref spmv");
            op.spmv(&x, &mut y_got).expect("spmv");

            assert_close(
                &y_got,
                &y_ref,
                tol,
                &format!("{} spmv, {}", backend.label(), shape_label(shape)),
            );
        }
    }
}

/// `spmv` accumulates into `y`. Two calls must land where two reference calls
/// land — this catches a backend that forgets to upload the incoming `y`, which
/// a single call from a zeroed vector would never notice.
#[test]
fn spmv_accumulates_rather_than_overwrites() {
    let reference = reference();
    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let mut rng = Rng::new(0xACC0_0000 ^ backend as u64);
        let (nrows, ncols, max_deg) = (300usize, 200usize, 12usize);
        let csr = random_csr(nrows, ncols, max_deg, &mut rng);
        let weights = random_vec(csr.nnz(), 1.0, &mut rng);
        let x = random_vec(ncols, 1.0, &mut rng);
        let y0 = random_vec(nrows, 1.0, &mut rng);

        let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let tol =
            tolerance_for_nnz_per_row(op.shape().nnz_per_row() * 2, max_abs_term(&weights, &x));

        let mut y_ref = y0.clone();
        let mut y_got = y0;
        for _ in 0..3 {
            op_ref.spmv(&x, &mut y_ref).expect("ref");
            op.spmv(&x, &mut y_got).expect("got");
        }
        assert_close(
            &y_got,
            &y_ref,
            tol,
            &format!("{} repeated spmv accumulation", backend.label()),
        );
    }
}

#[test]
fn fused_spmv_lif_matches_reference_across_shapes() {
    let reference = reference();
    let params = default_params();

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let mut rng = Rng::new(0xF05E_0000 ^ backend as u64);
        let mut total_flips = 0usize;

        for &(nrows, ncols, max_deg) in SHAPES {
            let csr = random_csr(nrows, ncols, max_deg, &mut rng);
            let weights = random_vec(csr.nnz(), 1.0, &mut rng);
            let x = random_vec(ncols, 1.0, &mut rng);
            // Thresholds straddle the reachable membrane range, so a healthy
            // fraction of cells actually spike. Parameters that never fire
            // would leave the reset and threshold-bump branches untested.
            let v0 = random_vec(nrows, 0.5, &mut rng);
            let theta0: Vec<f32> = random_vec(nrows, 0.5, &mut rng)
                .iter()
                .map(|t| t.abs())
                .collect();

            let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
            let op = device.prepare(&csr, ncols, &weights).expect("valid");
            let shape = op.shape();
            let tol = tolerance_for_nnz_per_row(shape.nnz_per_row(), max_abs_term(&weights, &x));

            // Reference synaptic current, needed to judge whether a spike flip
            // sits inside the boundary band.
            let mut current = vec![0.0f32; nrows];
            op_ref.spmv(&x, &mut current).expect("current");

            let (mut v_ref, mut th_ref, mut sp_ref) =
                (v0.clone(), theta0.clone(), vec![false; nrows]);
            let (mut v_got, mut th_got, mut sp_got) =
                (v0.clone(), theta0.clone(), vec![false; nrows]);

            op_ref
                .fused_spmv_lif(&x, &mut v_ref, &mut th_ref, &mut sp_ref, params)
                .expect("ref fused");
            op.fused_spmv_lif(&x, &mut v_got, &mut th_got, &mut sp_got, params)
                .expect("fused");

            let cmp = compare_lif(
                &v_got,
                &th_got,
                &sp_got,
                &v_ref,
                &th_ref,
                &sp_ref,
                &v0,
                &theta0,
                &current,
                params,
                tol,
                &format!("{} fused, {}", backend.label(), shape_label(shape)),
            );
            total_flips += cmp.flips;
        }

        // Boundary-band flips are legal but should be rare. A backend that
        // flipped a large share of cells would still satisfy the per-cell
        // check while being badly wrong.
        assert!(
            total_flips < 64,
            "{}: {total_flips} boundary spike flips across the shape sweep is too many \
             to be float noise",
            backend.label()
        );
    }
}

/// The elementwise arm.
///
/// There is no summation here, so the only thing that can differ is rounding —
/// and it does: Metal contracts the multiply-add. The tolerance is one
/// contraction's worth, and spike flips are permitted only inside that band.
#[test]
fn lif_integrate_matches_reference() {
    let reference = reference();
    let params = default_params();

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let mut rng = Rng::new(0x11F0_0000 ^ backend as u64);

        for &n in &[0usize, 1, 31, 32, 33, 255, 256, 257, 4096] {
            let v0 = random_vec(n, 1.0, &mut rng);
            let theta0: Vec<f32> = random_vec(n, 0.5, &mut rng)
                .iter()
                .map(|t| t.abs())
                .collect();
            let currents = random_vec(n, 1.0, &mut rng);

            let (mut v_ref, mut th_ref, mut sp_ref) = (v0.clone(), theta0.clone(), vec![false; n]);
            let (mut v_got, mut th_got, mut sp_got) = (v0.clone(), theta0.clone(), vec![false; n]);

            reference
                .lif_integrate(&mut v_ref, &mut th_ref, &currents, &mut sp_ref, params)
                .expect("ref lif");
            device
                .lif_integrate(&mut v_got, &mut th_got, &currents, &mut sp_got, params)
                .expect("lif");

            let scale = v_ref
                .iter()
                .chain(th_ref.iter())
                .chain(currents.iter())
                .fold(1.0f32, |m, v| m.max(v.abs()));
            let tol = if backend.is_gpu() {
                tolerance_for_elementwise(scale)
            } else {
                // Both CPU arms evaluate the identical Rust expression; there is
                // nothing left for them to disagree about.
                0.0
            };

            compare_lif(
                &v_got,
                &th_got,
                &sp_got,
                &v_ref,
                &th_ref,
                &sp_ref,
                &v0,
                &theta0,
                &currents,
                params,
                tol,
                &format!("{} lif at n={n}", backend.label()),
            );
        }
    }
}

/// Number of random cases the fuzz below runs.
///
/// `ProptestConfig::with_cases(n)` is built from `Config::default()` and then
/// overwrites `cases`, so an explicit literal silently defeats the standard
/// `PROPTEST_CASES` environment variable — a knob that looks like it works and
/// does nothing. Reading it here restores it.
///
/// 48 by default so an ordinary `cargo test` stays fast. Raise it for a soak:
/// `PROPTEST_CASES=5000 cargo test --features metal --release --test differential`.
fn proptest_cases() -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(48)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(proptest_cases()))]

    /// Random shapes and random data, every available backend against the
    /// reference. The shape table above covers the boundaries we know about;
    /// this covers the ones we do not.
    #[test]
    fn spmv_matches_reference_on_random_shapes(
        nrows in 0usize..400,
        ncols in 1usize..200,
        max_deg in 0usize..24,
        seed in any::<u64>(),
    ) {
        let reference = reference();
        let mut rng = Rng::new(seed);
        let csr = random_csr(nrows, ncols, max_deg, &mut rng);
        let weights = random_vec(csr.nnz(), 4.0, &mut rng);
        let x = random_vec(ncols, 4.0, &mut rng);
        let y0 = random_vec(nrows, 1.0, &mut rng);

        let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
        let mut y_ref = y0.clone();
        op_ref.spmv(&x, &mut y_ref).expect("ref");

        for backend in backends_under_test() {
            let device = Device::try_new(backend).expect("available");
            let op = device.prepare(&csr, ncols, &weights).expect("valid");
            let tol = tolerance_for_nnz_per_row(
                op.shape().nnz_per_row(),
                max_abs_term(&weights, &x),
            );
            let mut y_got = y0.clone();
            op.spmv(&x, &mut y_got).expect("got");
            for (i, (g, r)) in y_got.iter().zip(y_ref.iter()).enumerate() {
                prop_assert!(
                    (g - r).abs() <= tol,
                    "{} spmv row {i}: {g} vs {r} (tol {tol}, nrows={nrows} ncols={ncols})",
                    backend.label()
                );
            }
        }
    }
}
