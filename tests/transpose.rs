//! `SparseOp::spmv_t` — the transposed product a gradient travels through.
//!
//! The gate here is the inner-product identity `⟨A·x, y⟩ == ⟨x, Aᵀ·y⟩`, which
//! is the *definition* of the transpose. A wrong-but-plausible implementation
//! — indices swapped, the reverse index built against the wrong axis, the
//! weight table indexed by CSC position instead of `edge_idx` — produces a
//! result that looks like a sparse product and fails this identity. Comparing
//! against a dense reference would only restate whichever convention the
//! reference author happened to pick.

mod common;

use common::{max_abs, max_abs_term, random_csr, random_vec};
use sparsl::{available_backends, tolerance_for_spmv, Device, OpError, Rng};

/// `⟨a, b⟩` in f64, so the identity is judged at higher precision than the
/// f32 products it compares.
fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect()
}

#[test]
fn transpose_satisfies_the_inner_product_identity() {
    let mut rng = Rng::new(0x007A_5E11);
    for device in devices() {
        for &(nrows, ncols, deg) in &[
            (1usize, 1usize, 0usize),
            (7, 5, 3),
            (64, 96, 8),
            (129, 63, 5),
        ] {
            let csr = random_csr(nrows, ncols, deg, &mut rng);
            let weights = random_vec(csr.nnz(), 1.0, &mut rng);
            let op = device
                .prepare_with_transpose(&csr, ncols, &weights)
                .expect("prepare_with_transpose");

            let x = random_vec(ncols, 1.0, &mut rng);
            let y = random_vec(nrows, 1.0, &mut rng);

            let mut ax = vec![0.0f32; nrows];
            op.spmv(&x, &mut ax).expect("spmv");
            let mut aty = vec![0.0f32; ncols];
            op.spmv_t(&y, &mut aty).expect("spmv_t");

            let lhs = dot(&ax, &y);
            let rhs = dot(&x, &aty);
            // Both sides accumulate `nnz` products; bound the gap by the same
            // recursive-summation argument the crate uses elsewhere, applied to
            // the whole contraction rather than a single row.
            let scale = max_abs_term(&weights, &x).max(max_abs_term(&weights, &y));
            let bound = 8.0
                * f64::from(f32::EPSILON)
                * (csr.nnz().max(1) as f64)
                * (scale as f64)
                * (max_abs(&x).max(max_abs(&y)) as f64).max(1.0);
            assert!(
                (lhs - rhs).abs() <= bound.max(1e-4),
                "{}: <Ax,y> = {lhs} but <x,A^T y> = {rhs} \
                 (nrows={nrows} ncols={ncols} deg={deg}, bound {bound})",
                op.label()
            );
        }
    }
}

#[test]
fn transpose_matches_a_dense_reference() {
    let mut rng = Rng::new(0xD0_0D);
    let (nrows, ncols, deg) = (48usize, 33usize, 6usize);
    let csr = random_csr(nrows, ncols, deg, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(nrows, 1.0, &mut rng);

    // Dense A^T, built straight from the CSR: column `c` of A^T gathers every
    // stored entry whose CSR column is `c`.
    let mut want = vec![0.0f32; ncols];
    for (r, xr) in x.iter().enumerate() {
        let (s, e) = (csr.row_ptr[r] as usize, csr.row_ptr[r + 1] as usize);
        for i in s..e {
            want[csr.col[i] as usize] += weights[i] * xr;
        }
    }

    for device in devices() {
        let op = device
            .prepare_with_transpose(&csr, ncols, &weights)
            .expect("prepare_with_transpose");
        let mut got = vec![0.0f32; ncols];
        op.spmv_t(&x, &mut got).expect("spmv_t");

        let tol = tolerance_for_spmv(
            op.shape().max_row_nnz().max(1),
            max_abs_term(&weights, &x),
            max_abs(&want),
        );
        for (c, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{}: A^T x [{c}] = {g}, dense reference {w} (tol {tol})",
                op.label()
            );
        }
    }
}

#[test]
fn transpose_accumulates_into_y_rather_than_overwriting() {
    // `spmv` documents `y += A·x`; `spmv_t` must not quietly differ.
    let mut rng = Rng::new(0x000A_CC00);
    let (nrows, ncols) = (16usize, 12usize);
    let csr = random_csr(nrows, ncols, 4, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(nrows, 1.0, &mut rng);

    for device in devices() {
        let op = device
            .prepare_with_transpose(&csr, ncols, &weights)
            .expect("prepare");
        let mut once = vec![0.0f32; ncols];
        op.spmv_t(&x, &mut once).expect("spmv_t");

        let seed = random_vec(ncols, 1.0, &mut rng);
        let mut acc = seed.clone();
        op.spmv_t(&x, &mut acc).expect("spmv_t");
        for c in 0..ncols {
            let want = seed[c] + once[c];
            assert!(
                (acc[c] - want).abs() <= 1e-4,
                "{}: spmv_t overwrote instead of accumulating at {c}",
                op.label()
            );
        }
    }
}

#[test]
fn weight_updates_reach_both_directions() {
    // Forward and transposed share one value table. If they ever get separate
    // copies, this is what catches the one that stops being updated.
    let mut rng = Rng::new(0xBEEF);
    let (nrows, ncols) = (20usize, 14usize);
    let csr = random_csr(nrows, ncols, 5, &mut rng);
    let w0 = random_vec(csr.nnz(), 1.0, &mut rng);
    let w1 = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(nrows, 1.0, &mut rng);

    for device in devices() {
        let mut op = device
            .prepare_with_transpose(&csr, ncols, &w0)
            .expect("prepare");
        let mut before = vec![0.0f32; ncols];
        op.spmv_t(&x, &mut before).expect("spmv_t");

        op.set_weights(&w1).expect("set_weights");
        let mut after = vec![0.0f32; ncols];
        op.spmv_t(&x, &mut after).expect("spmv_t");

        let reference = device
            .prepare_with_transpose(&csr, ncols, &w1)
            .expect("prepare");
        let mut want = vec![0.0f32; ncols];
        reference.spmv_t(&x, &mut want).expect("spmv_t");

        for c in 0..ncols {
            assert!(
                (after[c] - want[c]).abs() <= 1e-4,
                "{}: set_weights did not reach the transposed path at {c}",
                op.label()
            );
        }
        // And it actually changed, so the check above is not vacuous.
        assert!(
            before.iter().zip(&after).any(|(b, a)| (b - a).abs() > 1e-6),
            "{}: new weights produced an identical result — the test proves nothing",
            op.label()
        );
    }
}

#[test]
fn spmv_t_refuses_an_operator_prepared_without_a_reverse_index() {
    let mut rng = Rng::new(0xC0FFEE);
    let (nrows, ncols) = (8usize, 6usize);
    let csr = random_csr(nrows, ncols, 3, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        assert!(
            !op.has_transpose(),
            "{}: unexpected reverse index",
            op.label()
        );
        let mut y = vec![0.0f32; ncols];
        let err = op
            .spmv_t(&vec![1.0f32; nrows], &mut y)
            .expect_err("must refuse without a reverse index");
        assert!(matches!(err, OpError::TransposeNotPrepared), "{err}");
        // Every backend refuses identically — the error is a property of the
        // operator, not of which substrate happened to open.
        assert!(
            err.to_string().contains("prepare_with_transpose"),
            "{}: error should name the fix, got {err}",
            op.label()
        );
    }
}

#[test]
fn transpose_rejects_wrong_operand_lengths() {
    let mut rng = Rng::new(0x1234);
    let (nrows, ncols) = (10usize, 7usize);
    let csr = random_csr(nrows, ncols, 3, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);

    for device in devices() {
        let op = device
            .prepare_with_transpose(&csr, ncols, &weights)
            .expect("prepare");
        // `y` is the column-length side here, the reverse of `spmv`. Passing
        // the forward lengths is the mistake this catches.
        let mut wrong = vec![0.0f32; nrows];
        assert!(op.spmv_t(&vec![0.0f32; nrows], &mut wrong).is_err());
        let mut right = vec![0.0f32; ncols];
        assert!(op.spmv_t(&vec![0.0f32; nrows - 1], &mut right).is_err());
        assert!(op.spmv_t(&vec![0.0f32; nrows], &mut right).is_ok());
    }
}
