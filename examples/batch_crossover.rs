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
//! Warmup is bounded, then each pair is measured over multiple rounds with its
//! order reversed every round. Medians are reported rather than the faster of
//! two attempts, alongside the spread of both arms and the paired speedup.
//! An untimed bit-exact preflight refuses to sample mismatched results.
//! `SPARSL_BENCH_WARMUP_ROUNDS`, `SPARSL_BENCH_ROUNDS`, and
//! `SPARSL_BENCH_ITERS` tune the bounded sampling plan.
//!
//! Run: `cargo run --release --features metal --example batch_crossover`

mod support;

use std::time::Instant;

use sparsl::{available_backends, Csr, Device, Rng, SparseOp};
use support::{milliseconds_per_iteration, paired_ratio_stats, BenchConfig, SampleStats};

const SIZES: &[usize] = &[1_000, 5_000, 10_000];
const BATCHES: &[usize] = &[1, 8, 32];
const DENSITY: f32 = 0.05;
const SEED_BASE: u64 = 0x5B_A7C_400;
const DEFAULT_ITERS: usize = 10;
const MAX_ITERS: usize = 100;

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

/// `n_vec` separate SpMV calls — what a caller does without `spmm`.
///
/// Takes its vectors already separate, rather than gathering columns out of the
/// batch-minor buffer `spmm` wants. A caller with no `spmm` would store them
/// this way, so charging the baseline for a layout conversion it would never
/// perform would flatter `spmm` for free.
fn time_repeated_spmv(op: &SparseOp, vectors: &[Vec<f32>], n: usize, iters: usize) -> f64 {
    let mut outputs = vec![vec![0.0f32; n]; vectors.len()];
    let t0 = Instant::now();
    for _ in 0..iters {
        for (vec, y) in vectors.iter().zip(&mut outputs) {
            y.fill(0.0);
            op.spmv(vec, y).expect("spmv");
        }
    }
    let elapsed = t0.elapsed();
    std::hint::black_box(&outputs);
    milliseconds_per_iteration(elapsed, iters)
}

fn time_spmm(op: &SparseOp, x: &[f32], n_vec: usize, n: usize, iters: usize) -> f64 {
    let mut y = vec![0.0f32; n * n_vec];
    let t0 = Instant::now();
    for _ in 0..iters {
        y.fill(0.0);
        op.spmm(x, n_vec, &mut y).expect("spmm");
    }
    let elapsed = t0.elapsed();
    std::hint::black_box(&y);
    milliseconds_per_iteration(elapsed, iters)
}

/// Refuse to time two arms unless they produce the same batch bit-for-bit.
///
/// Both kernels traverse each CSR row in the same order on a given backend, so
/// this is an exact contract rather than a tolerance comparison. Without this
/// preflight, a missing SpMM write or a layout regression would merely look
/// faster and the harness would print a persuasive speedup for wrong work.
fn verify_spmm_matches_repeated_spmv(
    op: &SparseOp,
    vectors: &[Vec<f32>],
    x: &[f32],
) -> Result<(), String> {
    let n_vec = vectors.len();
    let shape = op.shape();
    let expected_x = shape
        .ncols()
        .checked_mul(n_vec)
        .ok_or_else(|| "validation input length overflowed usize".to_owned())?;
    if x.len() != expected_x {
        return Err(format!(
            "batch-minor input has length {}, expected {expected_x}",
            x.len()
        ));
    }

    let mut repeated = vec![vec![0.0f32; shape.nrows()]; n_vec];
    for (vector, output) in vectors.iter().zip(&mut repeated) {
        if vector.len() != shape.ncols() {
            return Err(format!(
                "separate input vector has length {}, expected {}",
                vector.len(),
                shape.ncols()
            ));
        }
        op.spmv(vector, output)
            .map_err(|error| format!("repeated SpMV validation failed: {error}"))?;
    }

    let output_len = shape
        .nrows()
        .checked_mul(n_vec)
        .ok_or_else(|| "validation output length overflowed usize".to_owned())?;
    let mut batched = vec![0.0f32; output_len];
    op.spmm(x, n_vec, &mut batched)
        .map_err(|error| format!("SpMM validation failed: {error}"))?;

    for (v, output) in repeated.iter().enumerate() {
        for (row, expected) in output.iter().enumerate() {
            let index = row * n_vec + v;
            let actual = batched[index];
            if actual.to_bits() != expected.to_bits() {
                return Err(format!(
                    "row {row}, vector {v}: SpMM returned {actual:?} (0x{:08X}), repeated SpMV returned {expected:?} (0x{:08X})",
                    actual.to_bits(),
                    expected.to_bits()
                ));
            }
        }
    }
    Ok(())
}

fn main() {
    let config = BenchConfig::from_env(DEFAULT_ITERS, MAX_ITERS).unwrap_or_else(|error| {
        eprintln!("batch_crossover configuration error: {error}");
        std::process::exit(2);
    });
    let devices: Vec<Device> = available_backends()
        .into_iter()
        .map(|backend| {
            Device::try_new(backend).unwrap_or_else(|error| {
                panic!(
                    "advertised backend {} failed to open: {error}",
                    backend.label()
                )
            })
        })
        .collect();
    support::print_provenance("batch_crossover", config);
    println!(
        "  workload: sizes={SIZES:?}, batches={BATCHES:?}, density={DENSITY}, seed=0x{SEED_BASE:X}^n"
    );
    println!(
        "  backends: {}",
        devices
            .iter()
            .map(Device::label)
            .collect::<Vec<_>>()
            .join(", ")
    );
    for device in &devices {
        if let Some(name) = device.device_name() {
            println!("  device.{}: {name}", device.label());
        }
    }
    println!();

    for &n in SIZES {
        let mut rng = Rng::new(SEED_BASE ^ n as u64);
        let (csr, weights) = build(n, &mut rng);
        println!("n = {n}, nnz = {}", csr.nnz());
        println!(
            "  {:<22} {:>5} {:>12} {:>12} {:>11} {:>11} {:>12} {:>12}",
            "backend",
            "batch",
            "spmv median",
            "spmm median",
            "paired x",
            "spmv spread",
            "spmm spread",
            "ratio spread"
        );
        println!("  spread is max/min across samples; paired x is repeated-SpMV/SpMM per round");

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
            for device in &devices {
                let op = device.prepare(&csr, n, &weights).expect("prepare");
                verify_spmm_matches_repeated_spmv(&op, &vectors, &x).unwrap_or_else(|error| {
                    panic!(
                        "{} batch={n_vec} correctness preflight failed: {error}",
                        op.label()
                    )
                });

                // Warm both arms in balanced order. One logical iteration is
                // enough here: repeated SpMV already dispatches `n_vec` times.
                for round in 0..config.warmup_rounds {
                    if round % 2 == 0 {
                        std::hint::black_box(time_repeated_spmv(&op, &vectors, n, 1));
                        std::hint::black_box(time_spmm(&op, &x, n_vec, n, 1));
                    } else {
                        std::hint::black_box(time_spmm(&op, &x, n_vec, n, 1));
                        std::hint::black_box(time_repeated_spmv(&op, &vectors, n, 1));
                    }
                }

                let mut repeated_samples = Vec::with_capacity(config.sample_rounds);
                let mut spmm_samples = Vec::with_capacity(config.sample_rounds);
                for round in 0..config.sample_rounds {
                    if round % 2 == 0 {
                        repeated_samples.push(time_repeated_spmv(
                            &op,
                            &vectors,
                            n,
                            config.iterations,
                        ));
                        spmm_samples.push(time_spmm(&op, &x, n_vec, n, config.iterations));
                    } else {
                        let spmm = time_spmm(&op, &x, n_vec, n, config.iterations);
                        let repeated = time_repeated_spmv(&op, &vectors, n, config.iterations);
                        repeated_samples.push(repeated);
                        spmm_samples.push(spmm);
                    }
                }

                let repeated = SampleStats::from_samples(&repeated_samples)
                    .expect("positive repeated-SpMV samples");
                let batched =
                    SampleStats::from_samples(&spmm_samples).expect("positive SpMM samples");
                let speedup = paired_ratio_stats(&repeated_samples, &spmm_samples)
                    .expect("same number of positive paired samples");

                println!(
                    "  {:<22} {:>5} {:>12.3} {:>12.3} {:>10.2}x {:>11.2} {:>12.2} {:>12.2}",
                    op.label(),
                    n_vec,
                    repeated.median,
                    batched.median,
                    speedup.median,
                    repeated.spread(),
                    batched.spread(),
                    speedup.spread()
                );
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::verify_spmm_matches_repeated_spmv;
    use sparsl::{Csr, Device};

    #[test]
    fn correctness_preflight_accepts_the_layout_and_rejects_a_scramble() {
        let csr = Csr::from_adjacency(&[vec![0], vec![1]]);
        let device = Device::cpu_sequential();
        let op = device
            .prepare(&csr, 2, &[1.0, 1.0])
            .expect("identity operator");
        let vectors = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let batch_minor = vec![1.0, 3.0, 2.0, 4.0];
        verify_spmm_matches_repeated_spmv(&op, &vectors, &batch_minor)
            .expect("correct batch-minor layout");

        let vector_major_scramble = vec![1.0, 2.0, 3.0, 4.0];
        assert!(
            verify_spmm_matches_repeated_spmv(&op, &vectors, &vector_major_scramble).is_err(),
            "the benchmark would time a layout-regressed SpMM result"
        );
    }
}
