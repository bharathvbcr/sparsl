//! `SparseOp::spmm` — the batched product.
//!
//! The primary gate is not a tolerance comparison. A batch of one traverses a
//! row's non-zeros in exactly the order [`SparseOp::spmv`] does, so the two
//! must agree **bit for bit** on the same backend. That is a far sharper test
//! than "within `tolerance_for_spmv`": it fails on any reassociation, any fused
//! multiply-add the scalar path did not have, and any off-by-one in the batch
//! indexing, none of which a tolerance would notice.

mod common;

use common::{max_abs, max_abs_term, random_csr, random_vec};
use sparsl::{available_backends, tolerance_for_spmv, Device, OpError, Rng};

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect()
}

/// Batch-minor: `x[c * n_vec + v]` is column `c` of vector `v`.
fn interleave(vectors: &[Vec<f32>], n: usize) -> Vec<f32> {
    let n_vec = vectors.len();
    let mut out = vec![0.0f32; n * n_vec];
    for (v, vec) in vectors.iter().enumerate() {
        for c in 0..n {
            out[c * n_vec + v] = vec[c];
        }
    }
    out
}

#[test]
fn a_batch_of_one_is_bit_identical_to_spmv() {
    let mut rng = Rng::new(0x05BA_7C401u64);
    for device in devices() {
        for &(nrows, ncols, deg) in &[(1usize, 1usize, 0usize), (9, 7, 3), (128, 96, 6)] {
            let csr = random_csr(nrows, ncols, deg, &mut rng);
            let weights = random_vec(csr.nnz(), 1.0, &mut rng);
            let op = device.prepare(&csr, ncols, &weights).expect("prepare");
            let x = random_vec(ncols, 1.0, &mut rng);

            let mut want = vec![0.0f32; nrows];
            op.spmv(&x, &mut want).expect("spmv");
            let mut got = vec![0.0f32; nrows];
            op.spmm(&x, 1, &mut got).expect("spmm");

            for r in 0..nrows {
                assert_eq!(
                    got[r].to_bits(),
                    want[r].to_bits(),
                    "{}: spmm(n_vec=1) differs from spmv at row {r}: \
                     {} vs {} (nrows={nrows} ncols={ncols} deg={deg})",
                    op.label(),
                    got[r],
                    want[r]
                );
            }
        }
    }
}

#[test]
fn each_batch_column_equals_its_own_spmv() {
    let mut rng = Rng::new(0x0BA7_C4EDu64);
    let (nrows, ncols, deg, n_vec) = (64usize, 48usize, 5usize, 7usize);
    for device in devices() {
        let csr = random_csr(nrows, ncols, deg, &mut rng);
        let weights = random_vec(csr.nnz(), 1.0, &mut rng);
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");

        let vectors: Vec<Vec<f32>> = (0..n_vec)
            .map(|_| random_vec(ncols, 1.0, &mut rng))
            .collect();
        let x = interleave(&vectors, ncols);
        let mut batched = vec![0.0f32; nrows * n_vec];
        op.spmm(&x, n_vec, &mut batched).expect("spmm");

        // Every column must match the single-vector product bit for bit: the
        // batch changes memory layout, not arithmetic.
        for (v, vec) in vectors.iter().enumerate() {
            let mut single = vec![0.0f32; nrows];
            op.spmv(vec, &mut single).expect("spmv");
            for r in 0..nrows {
                assert_eq!(
                    batched[r * n_vec + v].to_bits(),
                    single[r].to_bits(),
                    "{}: batch column {v} differs from its own spmv at row {r}",
                    op.label()
                );
            }
        }
    }
}

#[test]
fn spmm_matches_a_dense_reference() {
    let mut rng = Rng::new(0x0000_D35Eu64);
    let (nrows, ncols, deg, n_vec) = (40usize, 33usize, 6usize, 5usize);
    let csr = random_csr(nrows, ncols, deg, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let vectors: Vec<Vec<f32>> = (0..n_vec)
        .map(|_| random_vec(ncols, 1.0, &mut rng))
        .collect();
    let x = interleave(&vectors, ncols);

    let mut want = vec![0.0f32; nrows * n_vec];
    for r in 0..nrows {
        let (s, e) = (csr.row_ptr[r] as usize, csr.row_ptr[r + 1] as usize);
        for i in s..e {
            for (v, vec) in vectors.iter().enumerate() {
                want[r * n_vec + v] += weights[i] * vec[csr.col[i] as usize];
            }
        }
    }

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let mut got = vec![0.0f32; nrows * n_vec];
        op.spmm(&x, n_vec, &mut got).expect("spmm");
        let tol = tolerance_for_spmv(
            op.shape().max_row_nnz().max(1),
            max_abs_term(&weights, &x),
            max_abs(&want),
        );
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= tol,
                "{}: spmm[{i}] = {g}, dense reference {w} (tol {tol})",
                op.label()
            );
        }
    }
}

#[test]
fn spmm_accumulates_into_y_rather_than_overwriting() {
    let mut rng = Rng::new(0x000A_CC5Bu64);
    let (nrows, ncols, n_vec) = (24usize, 18usize, 4usize);
    let csr = random_csr(nrows, ncols, 4, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(ncols * n_vec, 1.0, &mut rng);

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let mut once = vec![0.0f32; nrows * n_vec];
        op.spmm(&x, n_vec, &mut once).expect("spmm");

        let seed = random_vec(nrows * n_vec, 1.0, &mut rng);
        let mut acc = seed.clone();
        op.spmm(&x, n_vec, &mut acc).expect("spmm");
        for i in 0..nrows * n_vec {
            let want = seed[i] + once[i];
            assert!(
                (acc[i] - want).abs() <= 1e-4,
                "{}: spmm overwrote instead of accumulating at {i}",
                op.label()
            );
        }
    }
}

#[test]
fn batch_scratch_survives_a_shrink_then_regrow() {
    // The Metal arm caches its batch buffers and only reallocates when `n_vec`
    // grows, binding a prefix otherwise. A stale tail from the larger call
    // would show up here and nowhere else.
    let mut rng = Rng::new(0x005C_4A7Cu64);
    let (nrows, ncols) = (32usize, 24usize);
    let csr = random_csr(nrows, ncols, 5, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        for &n_vec in &[8usize, 2, 5, 1, 8] {
            let x = random_vec(ncols * n_vec, 1.0, &mut rng);
            let mut got = vec![0.0f32; nrows * n_vec];
            op.spmm(&x, n_vec, &mut got).expect("spmm");

            // Check against per-column spmv, which allocates nothing shared.
            for v in 0..n_vec {
                let column: Vec<f32> = (0..ncols).map(|c| x[c * n_vec + v]).collect();
                let mut single = vec![0.0f32; nrows];
                op.spmv(&column, &mut single).expect("spmv");
                for r in 0..nrows {
                    assert_eq!(
                        got[r * n_vec + v].to_bits(),
                        single[r].to_bits(),
                        "{}: n_vec={n_vec} column {v} row {r} disagrees after a resize",
                        op.label()
                    );
                }
            }
        }
    }
}

#[test]
fn spmm_rejects_a_zero_batch_and_wrong_lengths() {
    let mut rng = Rng::new(0x0012_345Bu64);
    let (nrows, ncols) = (12usize, 9usize);
    let csr = random_csr(nrows, ncols, 3, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let mut y = vec![0.0f32; nrows * 3];

        // A zero batch is a caller mistake, not a silent no-op: it would leave
        // `y` untouched and look like a product of zeros.
        let err = op
            .spmm(&vec![0.0f32; ncols * 3], 0, &mut y)
            .expect_err("n_vec = 0");
        assert!(
            matches!(err, OpError::Length { what: "n_vec", .. }),
            "{err}"
        );

        // `y` sized for the wrong batch.
        let mut wrong = vec![0.0f32; nrows * 2];
        assert!(op.spmm(&vec![0.0f32; ncols * 3], 3, &mut wrong).is_err());
        // `x` too short for the batch.
        assert!(op.spmm(&vec![0.0f32; ncols * 2], 3, &mut y).is_err());
        // And the correct shapes are accepted.
        assert!(op.spmm(&vec![0.0f32; ncols * 3], 3, &mut y).is_ok());
    }
}
