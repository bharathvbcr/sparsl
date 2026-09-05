//! Every available backend, checked against the sequential CPU reference on the
//! same inputs.
//!
//! This is the suite that gates availability. `Backend::Cuda` will be fuzzed
//! here the moment it reports available, with no new test code — which is why
//! `src/backend/cuda.rs` says not to flip that flag until this passes.

mod common;

use common::*;
use proptest::prelude::*;
use proptest::test_runner::{FailurePersistence, FileFailurePersistence};
use sparsl::spikes::pack_spikes;
use sparsl::{
    scan_magnitude_envelope, tolerance_for_elementwise, tolerance_for_scan, tolerance_for_spmv,
    tolerance_for_spmv_narrow, Backend, Device, LifParams, Rng, State, WeightPrecision,
};

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
            let mut y_ref = y0.clone();
            let mut y_got = y0;
            op_ref.spmv(&x, &mut y_ref).expect("ref spmv");
            op.spmv(&x, &mut y_got).expect("spmv");

            // Sized after the fact, from the reference result: the bound
            // depends on the magnitude `y` actually reached, which is not
            // knowable from the inputs alone.
            let tol = tolerance_for_spmv(
                shape.max_row_nnz(),
                max_abs_term(&weights, &x),
                max_abs(&y_ref),
            );

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
        let mut y_ref = y0.clone();
        let mut y_got = y0;
        for _ in 0..3 {
            op_ref.spmv(&x, &mut y_ref).expect("ref");
            op.spmv(&x, &mut y_got).expect("got");
        }
        // Three accumulations feed one result, so the row work triples and the
        // magnitude to bound is the one `y` actually reached.
        let tol = tolerance_for_spmv(
            op.shape().max_row_nnz() * 3,
            max_abs_term(&weights, &x),
            max_abs(&y_ref),
        );
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
            let tol = tolerance_for_spmv(
                shape.max_row_nnz(),
                max_abs_term(&weights, &x),
                v0.iter()
                    .chain(theta0.iter())
                    .fold(0.0f32, |m, t| m.max(t.abs())),
            );

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
const DEFAULT_PROPTEST_CASES: u32 = 48;
const MAX_PROPTEST_CASES: u32 = 1_000_000;
const FAILURE_CORPUS: &str = "tests/differential.proptest-regressions";

fn parse_proptest_cases(raw: Option<&str>) -> Result<u32, &'static str> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_PROPTEST_CASES);
    };
    let cases = raw
        .parse::<u32>()
        .map_err(|_| "PROPTEST_CASES must be a decimal integer")?;
    if cases == 0 {
        return Err("PROPTEST_CASES must be at least 1");
    }
    if cases > MAX_PROPTEST_CASES {
        return Err("PROPTEST_CASES exceeds the 1,000,000-case safety bound");
    }
    Ok(cases)
}

fn persisted_proptest_config(cases: u32) -> ProptestConfig {
    // `SourceParallel`, proptest's default, searches for `lib.rs` or
    // `main.rs` directly beside an ancestor of this integration-test source.
    // Cargo crates put that file under `src/`, so the search fails here and a
    // minimized GPU failure is not persisted. Use a stable crate-relative
    // destination so soak failures remain replayable instead of disappearing
    // with their random seed.
    ProptestConfig {
        cases,
        failure_persistence: Some(Box::new(FileFailurePersistence::Direct(FAILURE_CORPUS))),
        ..ProptestConfig::default()
    }
}

fn proptest_config() -> ProptestConfig {
    let raw = match std::env::var("PROPTEST_CASES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(std::env::VarError::NotUnicode(_)) => {
            panic!("PROPTEST_CASES must be valid UTF-8")
        }
    };
    let cases = parse_proptest_cases(raw.as_deref())
        .unwrap_or_else(|reason| panic!("invalid PROPTEST_CASES: {reason}"));

    persisted_proptest_config(cases)
}

/// Keep the original SpMV strategy as a small, fixed replay lane.
///
/// Its strategy shape is deliberately unchanged so the two tracked proptest
/// seeds continue to reproduce the cases described in the corpus. The
/// configurable case budget belongs to the multi-operation lane below: a
/// 50,000-case soak should mean 50,000 total randomized operations, not 50,000
/// for every operation that happens to have a property test.
fn spmv_replay_config() -> ProptestConfig {
    persisted_proptest_config(DEFAULT_PROPTEST_CASES)
}

#[test]
fn fuzz_case_count_is_bounded_and_never_vacuous() {
    assert_eq!(parse_proptest_cases(None), Ok(DEFAULT_PROPTEST_CASES));
    assert_eq!(parse_proptest_cases(Some("4096")), Ok(4096));
    assert!(parse_proptest_cases(Some("0")).is_err());
    assert!(parse_proptest_cases(Some("not-a-number")).is_err());
    assert!(parse_proptest_cases(Some("1000001")).is_err());
}

#[test]
fn tracked_failure_corpus_is_loaded_by_the_configured_persistence() {
    let seeds = FileFailurePersistence::Direct(FAILURE_CORPUS).load_persisted_failures2(None);
    assert!(
        seeds.len() >= 2,
        "expected the two curated replay seeds in {FAILURE_CORPUS}, loaded {}",
        seeds.len()
    );
}

/// One operation per generated case keeps the 50k Metal soak bounded while
/// broadening it beyond SpMV. The deterministic selector test below executes
/// every variant, so a small ordinary run never relies on random chance for
/// operation coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RandomOperation {
    Spmv,
    SpmmResize,
    Transpose,
    PackedSpikes,
    FusedLif,
    StandaloneLif,
    AffineScan,
    F16Spmv,
    Bf16Spmv,
}

impl RandomOperation {
    // Pinned independently of `ALL`: deriving this from the array length would
    // let deleting an operation from both randomized and sentinel coverage
    // make the test smaller while it stayed green.
    const COUNT: u8 = 9;
    const ALL: [Self; 9] = [
        Self::Spmv,
        Self::SpmmResize,
        Self::Transpose,
        Self::PackedSpikes,
        Self::FusedLif,
        Self::StandaloneLif,
        Self::AffineScan,
        Self::F16Spmv,
        Self::Bf16Spmv,
    ];

    fn from_selector(selector: u8) -> Self {
        Self::ALL
            .get(selector as usize)
            .copied()
            .unwrap_or_else(|| panic!("random operation selector {selector} is out of range"))
    }
}

fn random_lif_params(rng: &mut Rng) -> LifParams {
    let decay = 0.25 + 0.75 * rng.next_f32();
    let v_reset = rng.next_f32() - 0.5;
    let delta_theta = 0.25 * (rng.next_f32() * 2.0 - 1.0);
    LifParams::new(decay, v_reset, delta_theta).expect("bounded finite LIF parameters")
}

fn assert_bit_identical(got: &[f32], want: &[f32], context: &str) {
    assert_eq!(got.len(), want.len(), "{context}: length mismatch");
    for (i, (got, want)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{context}: index {i} differs ({got} vs {want})"
        );
    }
}

fn run_spmv_case(nrows: usize, ncols: usize, max_deg: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let csr = random_csr(nrows, ncols, max_deg, &mut rng);
    let weights = random_vec(csr.nnz(), 4.0, &mut rng);
    let x = random_vec(ncols, 4.0, &mut rng);
    let y0 = random_vec(nrows, 1.0, &mut rng);

    let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
    let mut y_ref = y0.clone();
    op_ref.spmv(&x, &mut y_ref).expect("reference SpMV");

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let mut y_got = y0.clone();
        op.spmv(&x, &mut y_got).expect("SpMV");
        let tol = tolerance_for_spmv(
            op.shape().max_row_nnz(),
            max_abs_term(&weights, &x),
            max_abs(&y_ref),
        );
        assert_close(
            &y_got,
            &y_ref,
            tol,
            &format!("{} randomized SpMV", backend.label()),
        );
    }
}

fn run_spmm_resize_case(nrows: usize, ncols: usize, max_deg: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let csr = random_csr(nrows, ncols, max_deg, &mut rng);
    let weights = random_vec(csr.nnz(), 4.0, &mut rng);
    let wide = 2 + rng.gen_index(15);
    // Allocate wide scratch, bind a short logical prefix, then force growth.
    // One operator is reused throughout; fresh preparation per width would not
    // exercise the grow-only cache or its logical-end canary.
    let widths = [wide, 1, wide + 1];
    let calls: Vec<_> = widths
        .into_iter()
        .map(|n_vec| {
            (
                n_vec,
                random_vec(ncols * n_vec, 4.0, &mut rng),
                random_vec(nrows * n_vec, 1.0, &mut rng),
            )
        })
        .collect();

    let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        for (n_vec, x, y0) in &calls {
            let mut y_ref = y0.clone();
            let mut y_got = y0.clone();
            op_ref.spmm(x, *n_vec, &mut y_ref).expect("reference SpMM");
            op.spmm(x, *n_vec, &mut y_got).expect("SpMM");
            let tol = tolerance_for_spmv(
                op.shape().max_row_nnz(),
                max_abs_term(&weights, x),
                max_abs(&y_ref),
            );
            assert_close(
                &y_got,
                &y_ref,
                tol,
                &format!(
                    "{} randomized SpMM resize at width {n_vec}",
                    backend.label()
                ),
            );
        }
    }
}

fn run_transpose_case(nrows: usize, ncols: usize, max_deg: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let csr = random_csr(nrows, ncols, max_deg, &mut rng);
    let weights = random_vec(csr.nnz(), 4.0, &mut rng);
    let x = random_vec(nrows, 4.0, &mut rng);
    let y0 = random_vec(ncols, 1.0, &mut rng);
    let max_col_nnz = max_col_nnz(&csr, ncols);

    let op_ref = reference
        .prepare_with_transpose(&csr, ncols, &weights)
        .expect("valid transpose");
    let mut y_ref = y0.clone();
    op_ref
        .spmv_t(&x, &mut y_ref)
        .expect("reference transpose SpMV");

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let op = device
            .prepare_with_transpose(&csr, ncols, &weights)
            .expect("valid transpose");
        let mut y_got = y0.clone();
        op.spmv_t(&x, &mut y_got).expect("transpose SpMV");
        // A transposed output reduces one original matrix column. Its work is
        // bounded by the maximum column degree, not `shape.max_row_nnz()`.
        let tol = tolerance_for_spmv(max_col_nnz, max_abs_term(&weights, &x), max_abs(&y_ref));
        assert_close(
            &y_got,
            &y_ref,
            tol,
            &format!("{} randomized transpose SpMV", backend.label()),
        );
    }
}

fn run_packed_spike_case(nrows: usize, ncols: usize, max_deg: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let csr = random_csr(nrows, ncols, max_deg, &mut rng);
    let weights = random_vec(csr.nnz(), 4.0, &mut rng);
    let spikes: Vec<bool> = (0..ncols).map(|_| rng.next_u32() & 0b11 == 0).collect();
    let packed = pack_spikes(&spikes);
    let y0 = random_vec(nrows, 1.0, &mut rng);

    let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
    let mut y_ref = y0.clone();
    op_ref
        .spmv_spikes(&packed, &mut y_ref)
        .expect("reference packed-spike SpMV");

    // The dense vector the packed bits stand for: the spike path is
    // documented as bit-identical to *this* product on the same backend.
    let dense: Vec<f32> = spikes.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let mut y_got = y0.clone();
        op.spmv_spikes(&packed, &mut y_got)
            .expect("packed-spike SpMV");
        // Across backends the comparison takes the published tolerance, as
        // every other product does: a backend may reduce a row in its own
        // order. This used to demand bit-identity with the reference, which
        // only ever held because every generated shape landed on the Metal
        // backend's one-lane tier, whose order happens to be the CPU's.
        let tol = tolerance_for_spmv(
            op.shape().max_row_nnz(),
            max_abs_term(&weights, &dense),
            max_abs(&y_ref),
        );
        assert_close(
            &y_got,
            &y_ref,
            tol,
            &format!("{} randomized packed-spike SpMV", backend.label()),
        );
        // Within a backend the contract is exact: the packed path must match
        // the dense product of the same bits, whatever tier the shape picked.
        let mut y_dense = y0.clone();
        op.spmv(&dense, &mut y_dense).expect("dense SpMV");
        assert_bit_identical(
            &y_got,
            &y_dense,
            &format!(
                "{} packed-spike SpMV vs its own dense product",
                backend.label()
            ),
        );
    }
}

fn run_fused_lif_case(nrows: usize, ncols: usize, max_deg: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let csr = random_csr(nrows, ncols, max_deg, &mut rng);
    let weights = random_vec(csr.nnz(), 4.0, &mut rng);
    let x = random_vec(ncols, 4.0, &mut rng);
    let v0 = random_vec(nrows, 0.5, &mut rng);
    let theta0: Vec<f32> = random_vec(nrows, 0.5, &mut rng)
        .into_iter()
        .map(f32::abs)
        .collect();
    let params = random_lif_params(&mut rng);

    let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
    let mut current = vec![0.0f32; nrows];
    op_ref
        .spmv(&x, &mut current)
        .expect("reference synaptic current");
    let (mut v_ref, mut theta_ref, mut spikes_ref) =
        (v0.clone(), theta0.clone(), vec![false; nrows]);
    op_ref
        .fused_spmv_lif(&x, &mut v_ref, &mut theta_ref, &mut spikes_ref, params)
        .expect("reference fused LIF");

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let (mut v_got, mut theta_got, mut spikes_got) =
            (v0.clone(), theta0.clone(), vec![false; nrows]);
        op.fused_spmv_lif(&x, &mut v_got, &mut theta_got, &mut spikes_got, params)
            .expect("fused LIF");
        let tol = if backend.is_gpu() {
            tolerance_for_spmv(
                op.shape().max_row_nnz(),
                max_abs_term(&weights, &x),
                v0.iter()
                    .chain(theta0.iter())
                    .fold(1.0f32, |scale, value| scale.max(value.abs())),
            )
        } else {
            0.0
        };
        compare_lif(
            &v_got,
            &theta_got,
            &spikes_got,
            &v_ref,
            &theta_ref,
            &spikes_ref,
            &v0,
            &theta0,
            &current,
            params,
            tol,
            &format!("{} randomized fused LIF", backend.label()),
        );
    }
}

fn run_standalone_lif_case(n: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let v0 = random_vec(n, 1.0, &mut rng);
    let theta0 = random_vec(n, 1.0, &mut rng);
    let currents = random_vec(n, 2.0, &mut rng);
    let params = random_lif_params(&mut rng);
    let (mut v_ref, mut theta_ref, mut spikes_ref) = (v0.clone(), theta0.clone(), vec![false; n]);
    reference
        .lif_integrate(
            &mut v_ref,
            &mut theta_ref,
            &currents,
            &mut spikes_ref,
            params,
        )
        .expect("reference standalone LIF");

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let (mut v_got, mut theta_got, mut spikes_got) =
            (v0.clone(), theta0.clone(), vec![false; n]);
        device
            .lif_integrate(
                &mut v_got,
                &mut theta_got,
                &currents,
                &mut spikes_got,
                params,
            )
            .expect("standalone LIF");
        let scale = v_ref
            .iter()
            .chain(theta_ref.iter())
            .chain(currents.iter())
            .fold(1.0f32, |scale, value| scale.max(value.abs()));
        let tol = if backend.is_gpu() {
            tolerance_for_elementwise(scale)
        } else {
            0.0
        };
        compare_lif(
            &v_got,
            &theta_got,
            &spikes_got,
            &v_ref,
            &theta_ref,
            &spikes_ref,
            &v0,
            &theta0,
            &currents,
            params,
            tol,
            &format!("{} randomized standalone LIF", backend.label()),
        );
    }
}

fn run_affine_scan_case(n: usize, seed: u64) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let states: Vec<State> = (0..n)
        .map(|_| State {
            // Contractive multipliers keep every finite prefix informative.
            a: 0.5 + 0.49 * rng.next_f32(),
            b: rng.next_f32() * 2.0 - 1.0,
        })
        .collect();
    let want = reference
        .assoc_scan(&states)
        .expect("reference affine scan");
    let envelope = scan_magnitude_envelope(&states);

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let got = device.assoc_scan(&states).expect("affine scan");
        assert_eq!(got.len(), want.len(), "{} scan length", backend.label());
        let tol = if backend.is_gpu() {
            tolerance_for_scan(n, envelope)
        } else {
            0.0
        };
        let got_a: Vec<f32> = got.iter().map(|state| state.a).collect();
        let want_a: Vec<f32> = want.iter().map(|state| state.a).collect();
        let got_b: Vec<f32> = got.iter().map(|state| state.b).collect();
        let want_b: Vec<f32> = want.iter().map(|state| state.b).collect();
        assert_close(
            &got_a,
            &want_a,
            tol,
            &format!("{} randomized affine scan multipliers", backend.label()),
        );
        assert_close(
            &got_b,
            &want_b,
            tol,
            &format!("{} randomized affine scan offsets", backend.label()),
        );
    }
}

fn run_narrow_spmv_case(
    precision: WeightPrecision,
    nrows: usize,
    ncols: usize,
    max_deg: usize,
    seed: u64,
) {
    let reference = reference();
    let mut rng = Rng::new(seed);
    let csr = random_csr(nrows, ncols, max_deg, &mut rng);
    // Bounded far below binary16's ceiling: this lane checks ordinary
    // quantisation and accumulation, while overflow policy has dedicated tests.
    let weights = random_vec(csr.nnz(), 4.0, &mut rng);
    let x = random_vec(ncols, 4.0, &mut rng);
    let y0 = random_vec(nrows, 1.0, &mut rng);
    let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
    let mut y_ref = y0.clone();
    op_ref
        .spmv(&x, &mut y_ref)
        .expect("reference full-precision SpMV");
    let tol = tolerance_for_spmv_narrow(
        precision,
        op_ref.shape().max_row_nnz(),
        max_abs_term(&weights, &x),
        max_abs(&y_ref),
    );

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available backend");
        let op = device
            .prepare_with(&csr, ncols, &weights, precision)
            .expect("valid narrow operator");
        let mut y_got = y0.clone();
        op.spmv(&x, &mut y_got).expect("narrow SpMV");
        assert_close(
            &y_got,
            &y_ref,
            tol,
            &format!("{} randomized {precision:?} SpMV", backend.label()),
        );
    }
}

fn run_random_operation(
    operation: RandomOperation,
    nrows: usize,
    ncols: usize,
    max_deg: usize,
    seed: u64,
) {
    match operation {
        RandomOperation::Spmv => run_spmv_case(nrows, ncols, max_deg, seed),
        RandomOperation::SpmmResize => run_spmm_resize_case(nrows, ncols, max_deg, seed),
        RandomOperation::Transpose => run_transpose_case(nrows, ncols, max_deg, seed),
        RandomOperation::PackedSpikes => run_packed_spike_case(nrows, ncols, max_deg, seed),
        RandomOperation::FusedLif => run_fused_lif_case(nrows, ncols, max_deg, seed),
        RandomOperation::StandaloneLif => run_standalone_lif_case(nrows, seed),
        RandomOperation::AffineScan => run_affine_scan_case(nrows, seed),
        RandomOperation::F16Spmv => {
            run_narrow_spmv_case(WeightPrecision::F16, nrows, ncols, max_deg, seed)
        }
        RandomOperation::Bf16Spmv => {
            run_narrow_spmv_case(WeightPrecision::Bf16, nrows, ncols, max_deg, seed)
        }
    }
}

#[test]
fn every_randomized_operation_selector_executes_deterministically() {
    let selected: Vec<_> = (0..RandomOperation::COUNT)
        .map(RandomOperation::from_selector)
        .collect();
    assert_eq!(selected, RandomOperation::ALL);

    // This is the coverage sentinel. Even the 1-case configuration executes
    // every operation once; random selection only broadens its inputs.
    for (selector, operation) in selected.into_iter().enumerate() {
        run_random_operation(operation, 17, 19, 7, 0xC0DE_0000 + selector as u64);
    }
}

proptest! {
    #![proptest_config(spmv_replay_config())]

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
            let mut y_got = y0.clone();
            op.spmv(&x, &mut y_got).expect("got");
            let tol = tolerance_for_spmv(
                op.shape().max_row_nnz(),
                max_abs_term(&weights, &x),
                max_abs(&y_ref),
            );
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

proptest! {
    #![proptest_config(proptest_config())]

    /// One randomly selected operation and shape per generated case, every
    /// available backend against the sequential reference. Selecting one arm
    /// rather than running all nine keeps a 50,000-case Metal soak bounded;
    /// `every_randomized_operation_selector_executes_deterministically` makes
    /// operation coverage non-probabilistic in ordinary runs.
    #[test]
    fn operations_match_reference_on_random_shapes(
        selector in 0u8..RandomOperation::COUNT,
        nrows in 0usize..400,
        ncols in 1usize..200,
        max_deg in 0usize..24,
        seed in any::<u64>(),
    ) {
        run_random_operation(
            RandomOperation::from_selector(selector),
            nrows,
            ncols,
            max_deg,
            seed,
        );
    }
}
