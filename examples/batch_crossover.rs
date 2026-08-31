//! Does batching move the GPU crossover?
//!
//! `crossover.rs` measures one vector at a time and finds that Metal does not
//! overtake the rayon arm until roughly 20M non-zeros. The reason is arithmetic
//! intensity: a single-vector SpMV does one multiply-add per index it loads,
//! which is not enough work to cover the load. `spmm` reuses each `weights[i]`
//! and each `col[i]` across `n_vec` vectors, so the ratio changes.
//!
//! This measures whether it changes *enough* to move the crossover, comparing
//! `spmm(n_vec)` against `n_vec` repeated `spmv` calls on the same backend.
//! Repeated SpMV is the honest baseline: it is what a caller does today.
//!
//! The measurement rules from `crossover.rs` apply here for the same reasons:
//! every arm is exercised before any is timed, and each is timed twice in
//! opposite orders with the spread reported.
//!
//! Run: `cargo run --release --features metal --example batch_crossover`

use std::time::{Duration, Instant};

use sparsl::{available_backends, Csr, Device, Rng, SparseOp};

const SIZES: &[usize] = &[1_000, 5_000, 10_000];
const BATCHES: &[usize] = &[1, 8, 32];
const DENSITY: f32 = 0.05;
const RAMP: usize = 10;
const ITERS: usize = 10;

fn build(n: usize, rng: &mut Rng) -> (Csr, Vec<f32>) {
    let nnz_per_row = ((n as f32) * DENSITY) as usize;
    let mut adj: Vec<Vec<u32>> = vec![Vec::with_capacity(nnz_per_row); n];
    for (r, row) in adj.iter_mut().enumerate() {
        for i in 0..nnz_per_row {
            row.push(((r + i * 3) % n) as u32);
        }
    }
    let csr = Csr::from_adjacency(&adj);
    let weights = (0..csr.nnz()).map(|_| rng.next_f32() - 0.5).collect();
    (csr, weights)
}

fn ms(total: Duration, iters: usize) -> f64 {
    total.as_secs_f64() * 1000.0 / iters as f64
}

/// `n_vec` separate SpMV calls — what a caller does without `spmm`.
///
/// Takes its vectors already separate, rather than gathering columns out of the
/// batch-minor buffer `spmm` wants. A caller with no `spmm` would store them
/// this way, so charging the baseline for a layout conversion it would never
/// perform would flatter `spmm` for free.
fn time_repeated_spmv(op: &SparseOp, vectors: &[Vec<f32>], n: usize, iters: usize) -> Duration {
    let mut y = vec![0.0f32; n];
    let t0 = Instant::now();
    for _ in 0..iters {
        for vec in vectors {
            y.fill(0.0);
            op.spmv(vec, &mut y).expect("spmv");
        }
    }
    t0.elapsed()
}

fn time_spmm(op: &SparseOp, x: &[f32], n_vec: usize, n: usize, iters: usize) -> Duration {
    let mut y = vec![0.0f32; n * n_vec];
    let t0 = Instant::now();
    for _ in 0..iters {
        y.fill(0.0);
        op.spmm(x, n_vec, &mut y).expect("spmm");
    }
    t0.elapsed()
}

fn main() {
    let arms = available_backends();
    println!(
        "backends: {}\n",
        arms.iter()
            .map(|b| b.label())
            .collect::<Vec<_>>()
            .join(", ")
    );

    for &n in SIZES {
        let mut rng = Rng::new(0x5B_A7C_400 ^ n as u64);
        let (csr, weights) = build(n, &mut rng);
        println!("n = {n}, nnz = {}", csr.nnz());
        println!(
            "  {:<16} {:>7} {:>12} {:>12} {:>9} {:>8}",
            "backend", "n_vec", "spmv x n (ms)", "spmm (ms)", "speedup", "spread"
        );

        for &n_vec in BATCHES {
            // Same numbers in both layouts: separate vectors for the
            // baseline, batch-minor interleaved for `spmm`.
            let vectors: Vec<Vec<f32>> = (0..n_vec)
                .map(|_| (0..n).map(|_| rng.next_f32() - 0.5).collect())
                .collect();
            let mut x = vec![0.0f32; n * n_vec];
            for (v, vec) in vectors.iter().enumerate() {
                for (c, value) in vec.iter().enumerate() {
                    x[c * n_vec + v] = *value;
                }
            }
            for &backend in &arms {
                let Ok(device) = Device::try_new(backend) else {
                    continue;
                };
                let op = device.prepare(&csr, n, &weights).expect("prepare");

                // Ramp both arms before timing either.
                time_repeated_spmv(&op, &vectors, n, RAMP.min(3));
                time_spmm(&op, &x, n_vec, n, RAMP);

                // Timed twice in opposite orders; spread is the disagreement.
                let a_rep = time_repeated_spmv(&op, &vectors, n, ITERS);
                let a_spmm = time_spmm(&op, &x, n_vec, n, ITERS);
                let b_spmm = time_spmm(&op, &x, n_vec, n, ITERS);
                let b_rep = time_repeated_spmv(&op, &vectors, n, ITERS);

                let rep = ms(a_rep, ITERS).min(ms(b_rep, ITERS));
                let batched = ms(a_spmm, ITERS).min(ms(b_spmm, ITERS));
                let spread = (ms(a_spmm, ITERS) / ms(b_spmm, ITERS))
                    .max(ms(b_spmm, ITERS) / ms(a_spmm, ITERS));

                println!(
                    "  {:<16} {:>7} {:>12.3} {:>12.3} {:>8.2}x {:>8.2}",
                    op.label(),
                    n_vec,
                    rep,
                    batched,
                    rep / batched,
                    spread
                );
            }
        }
        println!();
    }
}
