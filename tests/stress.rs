//! Adversarial inputs, degenerate shapes, concurrency and soak.
//!
//! The differential suite checks that the backends agree on well-formed work.
//! This one checks what happens when the input is hostile: malformed
//! connectivity, out-of-range indices, mismatched lengths, non-finite values,
//! empty and extreme shapes, and many threads sharing one operator.
//!
//! The single most important test here is
//! `out_of_range_columns_never_reach_a_device`. Every GPU kernel indexes
//! `x[col[i]]` with no range check, because a per-non-zero bounds check would
//! cost more than the multiply it guards. That is sound only because
//! `SparseOp::prepare` proves every stored column index is in range before a
//! byte is uploaded. If that check regresses, an out-of-range CSR stops being a
//! Rust panic and becomes an out-of-bounds read of arbitrary device memory.

mod common;

use std::sync::Arc;

use common::*;
use sparsl::{
    tolerance_for_spmv, Csr, Device, LifParams, LifParamsError, OpError, Rng, SparseOp,
    SparsePlanError,
};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn operators_are_shareable_across_threads() {
    assert_send_sync::<Device>();
    assert_send_sync::<SparseOp>();
}

// ---------------------------------------------------------------------------
// Malformed connectivity
// ---------------------------------------------------------------------------

/// The memory-safety gate for every GPU backend.
#[test]
fn out_of_range_columns_never_reach_a_device() {
    // `from_parts_unchecked` is the hole: it exists for callers that have
    // already validated, and nothing stops one being wrong. Stored `ncols`
    // matches the prepare argument so the column-range check is what fires.
    let cases: &[(&str, Csr, usize)] = &[
        (
            "column exactly at ncols",
            Csr::from_parts_unchecked(vec![0, 1], vec![4], 4),
            4,
        ),
        (
            "column far past ncols",
            Csr::from_parts_unchecked(vec![0, 2], vec![0, 9999], 8),
            8,
        ),
        (
            "u32::MAX column",
            Csr::from_parts_unchecked(vec![0, 1], vec![u32::MAX], 16),
            16,
        ),
        (
            "in-range rows, one bad edge deep in the middle",
            Csr::from_parts_unchecked(vec![0, 2, 4, 6], vec![0, 1, 2, 77, 0, 1], 4),
            4,
        ),
    ];

    for backend in sparsl::available_backends() {
        let device = Device::try_new(backend).expect("available");
        for (name, csr, ncols) in cases {
            let err = device
                .prepare(csr, *ncols, &vec![1.0f32; csr.nnz()])
                .expect_err(&format!(
                    "{}: `{name}` must be rejected before upload",
                    backend.label()
                ));
            match err {
                SparsePlanError::ColumnOutOfRange { col, ncols: n, .. } => {
                    assert!(
                        col as usize >= n,
                        "{}: reported column {col} is actually in range for ncols={n}",
                        backend.label()
                    );
                }
                other => panic!(
                    "{}: `{name}` gave {other:?}, expected ColumnOutOfRange",
                    backend.label()
                ),
            }
        }
    }
}

/// Stored `ncols` past `u32::MAX` must be rejected at prepare (TooLarge), not
/// accepted into a kernel that indexes with `u32`.
#[test]
fn ncols_does_not_overflow_on_a_maximal_column_index() {
    let ncols = u32::MAX as usize + 1;
    let csr = Csr::from_parts_unchecked(vec![0, 1], vec![u32::MAX], ncols);
    assert_eq!(csr.ncols(), ncols);

    let err = Device::cpu_sequential()
        .prepare(&csr, csr.ncols(), &[1.0])
        .expect_err("an ncols beyond u32::MAX must not be preparable");
    assert!(
        matches!(err, SparsePlanError::TooLarge { .. }),
        "expected TooLarge, got {err:?}"
    );
}

#[test]
fn structurally_invalid_csr_is_rejected() {
    let device = Device::cpu_sequential();
    let cases: Vec<(&str, Csr, usize, SparsePlanError)> = vec![
        (
            "empty row_ptr",
            Csr::from_parts_unchecked(vec![], vec![], 4),
            4,
            SparsePlanError::EmptyRowPtr,
        ),
        (
            "row_ptr does not start at zero",
            Csr::from_parts_unchecked(vec![1, 2], vec![0], 4),
            4,
            SparsePlanError::NonZeroStart { start: 1 },
        ),
        (
            "row_ptr goes backwards",
            Csr::from_parts_unchecked(vec![0, 2, 1], vec![0, 1], 4),
            4,
            SparsePlanError::NotMonotonic { index: 2 },
        ),
        (
            "row_ptr end disagrees with col length",
            Csr::from_parts_unchecked(vec![0, 1], vec![0, 1], 4),
            4,
            SparsePlanError::NnzMismatch {
                row_ptr_end: 1,
                col_len: 2,
            },
        ),
        (
            "unsorted columns within a row",
            Csr::from_parts_unchecked(vec![0, 2], vec![2, 0], 3),
            3,
            SparsePlanError::RowUnsorted { row: 0, edge: 1 },
        ),
        (
            "stored ncols disagrees with prepare",
            Csr::from_parts_unchecked(vec![0, 1], vec![0], 4),
            3,
            SparsePlanError::NcolsMismatch {
                stored: 4,
                declared: 3,
            },
        ),
    ];
    for (name, csr, ncols, expected) in cases {
        let err = device
            .prepare(&csr, ncols, &vec![1.0f32; csr.nnz()])
            .expect_err(&format!("`{name}` must be rejected"));
        assert_eq!(err, expected, "wrong error for `{name}`");
    }
}

/// Any CSR that `Csr::from_parts` accepts must also survive `prepare` with the
/// same stored `ncols`. If these two validators ever disagree, one of them is
/// wrong.
#[test]
fn from_parts_and_prepare_agree_on_valid_shapes() {
    let device = Device::cpu_sequential();
    let mut rng = Rng::new(0x007A_11D8);
    for _ in 0..200 {
        let nrows = rng.gen_index(32);
        let ncols = 1 + rng.gen_index(32);
        let csr = random_csr(nrows, ncols, 6, &mut rng);
        let from_parts = Csr::from_parts(csr.row_ptr().to_vec(), csr.col().to_vec(), csr.ncols());
        let prepared = device.prepare(&csr, csr.ncols(), &vec![1.0f32; csr.nnz()]);
        assert_eq!(
            from_parts.is_ok(),
            prepared.is_ok(),
            "validators disagree on nrows={nrows} ncols={ncols}"
        );
    }
}

/// Trailing empty columns survive prepare when stored on the CSR.
#[test]
fn prepare_preserves_trailing_empty_columns() {
    let csr = Csr::from_parts(vec![0, 1], vec![0], 4).expect("valid");
    let op = Device::cpu_sequential()
        .prepare(&csr, 4, &[1.0])
        .expect("trailing empty columns must be preparable");
    assert_eq!(op.shape().ncols(), 4);
    let mut y = vec![0.0];
    op.spmv(&[1.0, 0.0, 0.0, 0.0], &mut y).expect("spmv");
    assert_eq!(y[0].to_bits(), 1.0f32.to_bits());
}

// ---------------------------------------------------------------------------
// Operand length errors
// ---------------------------------------------------------------------------

#[test]
fn mismatched_operand_lengths_are_reported_not_asserted() {
    for backend in sparsl::available_backends() {
        let device = Device::try_new(backend).expect("available");
        let mut rng = Rng::new(0x1E17_0000 ^ backend as u64);
        let (nrows, ncols) = (16usize, 8usize);
        let csr = random_csr(nrows, ncols, 4, &mut rng);
        let nnz = csr.nnz();
        let weights = vec![1.0f32; nnz];
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let x = vec![1.0f32; ncols];
        let mut y = vec![0.0f32; nrows];
        let label = backend.label();

        // Weights are validated where they are now supplied: at prepare.
        assert!(
            matches!(
                device.prepare(&csr, ncols, &vec![1.0; nnz + 1]),
                Err(SparsePlanError::WeightsLen { .. })
            ),
            "{label}: long weights must be rejected at prepare"
        );
        let mut op_mut = device.prepare(&csr, ncols, &weights).expect("valid");
        assert!(
            matches!(
                op_mut.set_weights(&vec![1.0; nnz + 1]),
                Err(OpError::Length {
                    what: "weights",
                    ..
                })
            ),
            "{label}: set_weights must reject a wrong length"
        );
        assert!(
            matches!(
                op.spmv(&vec![1.0; ncols - 1], &mut y),
                Err(OpError::TooShort { what: "x", .. })
            ),
            "{label}: short x"
        );
        assert!(
            matches!(
                op.spmv(&x, &mut vec![0.0; nrows + 3]),
                Err(OpError::Length { what: "y", .. })
            ),
            "{label}: wrong y"
        );

        // A longer-than-required `x` is legal: only the first `ncols` are read.
        let long_x = {
            let mut v = x.clone();
            v.extend_from_slice(&[f32::NAN; 5]);
            v
        };
        let mut y_long = vec![0.0f32; nrows];
        let mut y_exact = vec![0.0f32; nrows];
        op.spmv(&long_x, &mut y_long).expect("long x is allowed");
        op.spmv(&x, &mut y_exact).expect("exact x");
        assert_eq!(
            y_long.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            y_exact.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "{label}: padding past ncols must not be read — NaNs leaked into the result"
        );

        let params = default_params();
        let mut v = vec![0.0f32; nrows];
        let mut theta = vec![1.0f32; nrows];
        let mut spikes = vec![false; nrows];
        assert!(
            matches!(
                op.fused_spmv_lif(&x, &mut v, &mut vec![1.0; nrows - 1], &mut spikes, params),
                Err(OpError::Length { what: "theta", .. })
            ),
            "{label}: short theta"
        );
        assert!(
            matches!(
                device.lif_integrate(
                    &mut v,
                    &mut theta,
                    &vec![0.0; nrows + 1],
                    &mut spikes,
                    params
                ),
                Err(OpError::Length {
                    what: "currents",
                    ..
                })
            ),
            "{label}: wrong currents"
        );
    }
}

#[test]
fn non_finite_lif_parameters_are_rejected() {
    for (decay, reset, delta, field) in [
        (f32::NAN, 0.0, 0.1, "decay"),
        (f32::INFINITY, 0.0, 0.1, "decay"),
        (0.9, f32::NAN, 0.1, "v_reset"),
        (0.9, 0.0, f32::NEG_INFINITY, "delta_theta"),
    ] {
        assert_eq!(
            LifParams::new(decay, reset, delta),
            Err(LifParamsError::NotFinite { field }),
            "non-finite `{field}` must not build"
        );
    }
    assert!(LifParams::new(0.9, -1.0, 0.0).is_ok());
    // Zero decay is degenerate but legal: a cell with no memory.
    assert!(LifParams::new(0.0, 0.0, 0.0).is_ok());
}

// ---------------------------------------------------------------------------
// Hostile data
// ---------------------------------------------------------------------------

/// NaN and infinity in the data are the caller's business, not an error. What
/// must not happen is a crash, a hang, or one backend turning a NaN into a
/// finite number while another propagates it.
#[test]
fn non_finite_data_propagates_without_crashing() {
    let reference = reference();
    let (nrows, ncols) = (64usize, 32usize);
    let mut rng = Rng::new(0x0BAD_F33D);
    let csr = random_csr(nrows, ncols, 8, &mut rng);

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let nnz = csr.nnz();

        for (name, poison) in [
            ("nan", f32::NAN),
            ("inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            let mut weights = vec![0.5f32; nnz];
            if !weights.is_empty() {
                weights[nnz / 2] = poison;
            }
            let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
            let op = device.prepare(&csr, ncols, &weights).expect("valid");
            let x = vec![1.0f32; ncols];
            let mut y_ref = vec![0.0f32; nrows];
            let mut y_got = vec![0.0f32; nrows];
            op_ref.spmv(&x, &mut y_ref).expect("ref");
            op.spmv(&x, &mut y_got).expect("got");

            for i in 0..nrows {
                assert_same_numeric_class(
                    y_got[i],
                    y_ref[i],
                    &format!("{} row {i} with {name} weight", backend.label()),
                );
            }
        }
    }
}

/// Subnormals are a classic place for a GPU to differ, because flush-to-zero is
/// a legal and common choice. This does not demand equality — it demands that
/// whatever happens is consistent and does not produce garbage.
#[test]
fn subnormal_inputs_do_not_produce_garbage() {
    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let csr = Csr::from_adjacency(&[vec![0, 1], vec![1]]);
        let tiny = f32::from_bits(1); // smallest positive subnormal
        let weights = vec![tiny, tiny, tiny];
        let op = device.prepare(&csr, 2, &weights).expect("valid");
        let x = vec![tiny, tiny];
        let mut y = vec![0.0f32; 2];
        op.spmv(&x, &mut y).expect("spmv");
        for (i, v) in y.iter().enumerate() {
            assert!(
                v.is_finite() && *v >= 0.0 && *v < 1.0,
                "{} row {i}: subnormal product produced {v}",
                backend.label()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Degenerate and extreme shapes
// ---------------------------------------------------------------------------

#[test]
fn degenerate_shapes_are_no_ops_not_crashes() {
    let params = default_params();
    for backend in sparsl::available_backends() {
        let device = Device::try_new(backend).expect("available");
        let label = backend.label();

        // Zero rows, with four declared columns (trailing empties).
        let op = device
            .prepare(&Csr::from_parts_unchecked(vec![0], vec![], 4), 4, &[])
            .expect("zero-row CSR is valid");
        assert_eq!(op.shape().nrows(), 0);
        op.spmv(&[1.0; 4], &mut []).expect("zero-row spmv");
        op.fused_spmv_lif(&[1.0; 4], &mut [], &mut [], &mut [], params)
            .expect("zero-row fused");

        // Rows but no edges.
        let op = device
            .prepare(&Csr::empty(8, 4), 4, &[])
            .expect("edgeless CSR");
        assert_eq!(op.shape().nnz(), 0);
        let mut y = vec![7.0f32; 8];
        op.spmv(&[1.0; 4], &mut y).expect("edgeless spmv");
        assert!(
            y.iter().all(|v| *v == 7.0),
            "{label}: a graph with no edges must leave y untouched, got {y:?}"
        );

        // Zero cells in the dense kernel.
        device
            .lif_integrate(&mut [], &mut [], &[], &mut [], params)
            .expect("empty lif");
    }
}

/// One row holding a very large number of non-zeros. The fused kernel walks a
/// row with a stride-32 loop inside a single SIMD group, so a long row exercises
/// many more loop iterations per thread than the shape sweep ever does.
#[test]
fn a_single_very_long_row_is_handled() {
    let reference = reference();
    let ncols = 4096usize;
    let nnz = 200_000usize;
    let mut rng = Rng::new(0x1006_0A0B);
    let mut col: Vec<u32> = (0..nnz).map(|_| rng.gen_index(ncols) as u32).collect();
    col.sort();
    let csr = Csr::from_parts(vec![0, nnz as u32], col, ncols).expect("valid");
    let weights = random_vec(nnz, 1.0, &mut rng);
    let x = random_vec(ncols, 1.0, &mut rng);

    let op_ref = reference.prepare(&csr, ncols, &weights).expect("valid");
    let mut y_ref = vec![0.0f32; 1];
    op_ref.spmv(&x, &mut y_ref).expect("ref");

    for backend in backends_under_test() {
        let device = Device::try_new(backend).expect("available");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let mut y = vec![0.0f32; 1];
        op.spmv(&x, &mut y).expect("spmv");
        let tol = tolerance_for_spmv(nnz, max_abs_term(&weights, &x), max_abs(&y_ref));
        assert!(
            (y[0] - y_ref[0]).abs() <= tol,
            "{}: single {nnz}-entry row gave {} vs reference {} (tol {tol})",
            backend.label(),
            y[0],
            y_ref[0]
        );
    }
}

/// The team kernels stride through a line by their lane count. A valid line
/// may begin within 31 entries of `u32::MAX`; adding the lane or the next
/// stride in `uint` then wraps to the beginning of the values buffer instead
/// of ending the loop. The fixture is arithmetic-only because allocating four billion
/// edges would make the regression test useless; the source assertion pins the
/// required widening at the device boundary.
#[test]
fn fused_metal_indices_widen_before_offset_arithmetic() {
    let row_start = u32::MAX - 1;
    let lane = 31u32;
    assert!(
        u64::from(row_start) + u64::from(lane) > u64::from(u32::MAX),
        "fixture must cross the uint boundary"
    );

    let source = include_str!("../src/kernels/spmv.metal");
    assert!(
        source.contains("return (ulong)group_id * (threads_per_group / LPR) + (tid / LPR);"),
        "line geometry must widen before multiplying and adding"
    );
    assert!(
        source.contains(
            "for (ulong i = (ulong)line_start + lane; i < (ulong)line_end; i += (ulong)LPR)"
        ),
        "line traversal must widen before adding a lane or the team stride"
    );
}

/// SpMM's team kernel keeps a constant `SIMD_SPMM_TILE` trip count so the
/// partial tail must substitute zero rather than reading past `n_vec`. See
/// `tests/spmm.rs` for the matching pin; duplicated here so a stress-only
/// run still holds the guard down.
#[test]
fn spmm_metal_tile_guard_pins_partial_tail() {
    let source = include_str!("../src/kernels/spmv.metal");
    assert!(
        source.contains("const float xv = (v0 + t < (ulong)n_vec) ? x[base + t] : 0.0f;"),
        "SpMM tile loop must keep the v0+t < n_vec ternary on x loads"
    );
}

// ---------------------------------------------------------------------------
// Repeatability, concurrency, soak
// ---------------------------------------------------------------------------

/// The same operator, the same inputs, many times: bit-identical every time.
///
/// A GPU is allowed to disagree with the CPU. It is not allowed to disagree
/// with itself — that would mean uninitialised scratch, a missing barrier, or a
/// race between the dispatch and the readback.
#[test]
fn repeated_dispatch_is_bit_stable() {
    let mut rng = Rng::new(0x5EED_5A3E);
    let (nrows, ncols) = (777usize, 333usize);
    let csr = random_csr(nrows, ncols, 20, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(ncols, 1.0, &mut rng);
    let params = default_params();

    for backend in sparsl::available_backends() {
        let device = Device::try_new(backend).expect("available");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");

        let mut first: Option<Vec<u32>> = None;
        for round in 0..50 {
            let mut y = vec![0.0f32; nrows];
            let mut v = vec![0.1f32; nrows];
            let mut theta = vec![0.4f32; nrows];
            let mut spikes = vec![false; nrows];
            op.spmv(&x, &mut y).expect("spmv");
            op.fused_spmv_lif(&x, &mut v, &mut theta, &mut spikes, params)
                .expect("fused");

            let mut bits: Vec<u32> = y.iter().map(|f| f.to_bits()).collect();
            bits.extend(v.iter().map(|f| f.to_bits()));
            bits.extend(theta.iter().map(|f| f.to_bits()));
            bits.extend(spikes.iter().map(|b| *b as u32));
            match &first {
                None => first = Some(bits),
                Some(expected) => assert_eq!(
                    &bits,
                    expected,
                    "{} round {round} differs from round 0",
                    backend.label()
                ),
            }
        }
    }
}

/// One operator, many threads. The Metal backend serialises through a mutex
/// around its scratch buffers; without it, two threads would memcpy into the
/// same host-visible allocation and read each other's operands.
#[test]
fn concurrent_use_of_one_operator_matches_serial_use() {
    const THREADS: usize = 8;
    const ROUNDS: usize = 25;

    let mut rng = Rng::new(0xC0C0_FFEE);
    let (nrows, ncols) = (512usize, 256usize);
    let csr = random_csr(nrows, ncols, 16, &mut rng);
    let weights = Arc::new(random_vec(csr.nnz(), 1.0, &mut rng));
    // Each thread uses a distinct `x`, so a leaked operand shows up as a wrong
    // answer rather than as the right answer computed from someone else's data.
    let inputs: Arc<Vec<Vec<f32>>> = Arc::new(
        (0..THREADS)
            .map(|t| random_vec(ncols, 1.0 + t as f32, &mut rng))
            .collect(),
    );

    for backend in sparsl::available_backends() {
        let device = Device::try_new(backend).expect("available");
        let op = Arc::new(device.prepare(&csr, ncols, &weights).expect("valid"));

        // Serial expectation, computed one thread at a time.
        let expected: Vec<Vec<u32>> = (0..THREADS)
            .map(|t| {
                let mut y = vec![0.0f32; nrows];
                op.spmv(&inputs[t], &mut y).expect("serial");
                y.iter().map(|f| f.to_bits()).collect()
            })
            .collect();

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let op = Arc::clone(&op);
                let inputs = Arc::clone(&inputs);
                std::thread::spawn(move || {
                    let mut last = Vec::new();
                    for _ in 0..ROUNDS {
                        let mut y = vec![0.0f32; nrows];
                        op.spmv(&inputs[t], &mut y).expect("concurrent");
                        last = y.iter().map(|f| f.to_bits()).collect();
                    }
                    last
                })
            })
            .collect();

        for (t, handle) in handles.into_iter().enumerate() {
            let got = handle.join().expect("worker thread panicked");
            assert_eq!(
                got,
                expected[t],
                "{}: thread {t} got a different answer under contention — \
                 operands are leaking between concurrent calls",
                backend.label()
            );
        }
    }
}

/// Long run on one operator: state must stay bounded and the operator must stay
/// usable. Catches scratch that is never reset and per-call allocations that
/// are never released.
#[test]
fn soak_many_iterations_without_drift() {
    const ITERS: usize = 2000;
    let mut rng = Rng::new(0x50AC_0000);
    let (nrows, ncols) = (256usize, 128usize);
    let csr = random_csr(nrows, ncols, 8, &mut rng);
    let weights = random_vec(csr.nnz(), 0.05, &mut rng);
    let x = random_vec(ncols, 0.5, &mut rng);
    let params = LifParams::new(0.9, 0.0, 0.01).expect("finite");

    for backend in sparsl::available_backends() {
        let device = Device::try_new(backend).expect("available");
        let op = device.prepare(&csr, ncols, &weights).expect("valid");
        let mut v = vec![0.0f32; nrows];
        let mut theta = vec![0.5f32; nrows];
        let mut spikes = vec![false; nrows];
        let mut total_spikes = 0usize;

        for iter in 0..ITERS {
            op.fused_spmv_lif(&x, &mut v, &mut theta, &mut spikes, params)
                .expect("fused");
            total_spikes += spikes.iter().filter(|s| **s).count();
            assert!(
                v.iter().all(|f| f.is_finite()) && theta.iter().all(|f| f.is_finite()),
                "{} went non-finite at iteration {iter}",
                backend.label()
            );
        }

        // `theta` only ever rises, by `delta_theta` per spike. That gives an
        // exact invariant rather than a vague "looks reasonable" check.
        let expected_total_rise = total_spikes as f32 * params.delta_theta();
        let actual_rise: f32 = theta.iter().map(|t| t - 0.5).sum();
        let tol = 1e-2 * expected_total_rise.max(1.0);
        assert!(
            (actual_rise - expected_total_rise).abs() <= tol,
            "{}: theta rose by {actual_rise} across {total_spikes} spikes, expected \
             {expected_total_rise}",
            backend.label()
        );
    }
}
