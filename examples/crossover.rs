//! Where does each backend start winning?
//!
//! Every advertised backend runs the same CSR workload. Before timing, each
//! arm must match an independent scalar oracle: CPU exactly and GPU within the
//! crate's derived SpMV bound. Warmup is bounded, then multiple paired rounds
//! alternate forward/reverse arm order. The report uses medians rather than an
//! optimistic minimum and exposes both per-arm and paired-ratio spread.
//!
//! `SparseOp::spmv` accumulates into its output, so every timed iteration clears
//! that output. Otherwise later iterations do different work numerically and a
//! missing write can masquerade as speed. The final output is black-boxed.
//! `SPARSL_BENCH_WARMUP_ROUNDS`, `SPARSL_BENCH_ROUNDS`, and
//! `SPARSL_BENCH_ITERS` tune the bounded sampling plan.
//!
//! Run: `cargo run --release --features metal --example crossover`

mod support;

use sparsl::{available_backends, tolerance_for_spmv, Backend, Csr, Device, Rng, SparseOp};
use support::{
    balanced_arm_order, max_abs_finite, max_abs_product, max_row_nnz, paired_ratio_stats,
    reference_spmv, time_zeroed_output, verify_float_output, BenchConfig, SampleStats,
};

const SIZES: &[usize] = &[1_000, 5_000, 10_000, 20_000];
const DENSITY: f32 = 0.05;
const SEED_BASE: u64 = 0x5713_2026;
const DEFAULT_ITERS: usize = 10;
const MAX_ITERS: usize = 100;

fn build(n: usize, rng: &mut Rng) -> (Csr, Vec<f32>, Vec<f32>) {
    let nnz_per_row = ((n as f32) * DENSITY) as usize;
    let mut adj: Vec<Vec<u32>> = vec![Vec::with_capacity(nnz_per_row); n];
    for (r, row) in adj.iter_mut().enumerate() {
        for i in 0..nnz_per_row {
            row.push(((r + i * 3) % n) as u32);
        }
    }
    let csr = Csr::from_adjacency(&adj);
    // Varied values, not constants: constant weights and inputs make every row
    // a sum of identical terms and turn an order-sensitive parity check inert.
    let weights = (0..csr.nnz()).map(|_| rng.next_f32() - 0.5).collect();
    let x = (0..n).map(|_| rng.next_f32() - 0.5).collect();
    (csr, weights, x)
}

fn verify_spmv_preflight(
    label: &str,
    backend: Backend,
    got: &[f32],
    expected: &[f32],
    max_row_nnz: usize,
    max_abs_term: f32,
) -> Result<(), String> {
    let tolerance = if backend.is_gpu() {
        tolerance_for_spmv(
            max_row_nnz,
            max_abs_term,
            max_abs_finite("reference output", expected)?,
        )
    } else {
        0.0
    };
    verify_float_output(label, got, expected, tolerance)
}

struct Arm {
    backend: Backend,
    op: SparseOp,
    prepare_ms: f64,
    output: Vec<f32>,
}

fn main() {
    let config = BenchConfig::from_env(DEFAULT_ITERS, MAX_ITERS).unwrap_or_else(|error| {
        eprintln!("crossover configuration error: {error}");
        std::process::exit(2);
    });
    let backends = available_backends();
    let devices: Vec<(Backend, Device)> = backends
        .iter()
        .copied()
        .map(|backend| {
            let device = Device::try_new(backend).unwrap_or_else(|error| {
                panic!(
                    "advertised backend {} failed to open: {error}",
                    backend.label()
                )
            });
            (backend, device)
        })
        .collect();

    support::print_provenance("crossover", config);
    println!("  workload: sizes={SIZES:?}, density={DENSITY}, seed=0x{SEED_BASE:X}^n");
    println!("  timed_iteration: zero output, then one y += A*x dispatch");
    println!(
        "  backends: {}",
        devices
            .iter()
            .map(|(_, device)| device.label())
            .collect::<Vec<_>>()
            .join(", ")
    );
    for (_, device) in &devices {
        if let Some(name) = device.device_name() {
            println!("  device.{}: {name}", device.label());
        }
    }
    for backend in Backend::ALL {
        if !backends.contains(&backend) {
            let reason = backend
                .unavailable_reason()
                .unwrap_or("backend disappeared during discovery");
            println!("  unavailable.{}: {reason}", backend.label());
        }
    }

    for &n in SIZES {
        let mut rng = Rng::new(SEED_BASE ^ n as u64);
        let (csr, weights, x) = build(n, &mut rng);
        let initial: Vec<f32> = (0..n)
            .map(|row| 0.25 + (row % 17) as f32 * 0.03125)
            .collect();
        let expected = reference_spmv(&csr, &weights, &x, &initial)
            .unwrap_or_else(|error| panic!("reference preflight failed: {error}"));
        let term_scale = max_abs_product(&weights, &x)
            .unwrap_or_else(|error| panic!("preflight scale failed: {error}"));
        let row_nnz = max_row_nnz(&csr);

        let mut arms: Vec<Arm> = devices
            .iter()
            .map(|(backend, device)| {
                let start = std::time::Instant::now();
                let op = device.prepare(&csr, n, &weights).unwrap_or_else(|error| {
                    panic!("{} preparation failed: {error}", device.label())
                });
                let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;

                let mut output = initial.clone();
                op.spmv(&x, &mut output).unwrap_or_else(|error| {
                    panic!("{} correctness dispatch failed: {error}", op.label())
                });
                verify_spmv_preflight(
                    op.label(),
                    *backend,
                    &output,
                    &expected,
                    row_nnz,
                    term_scale,
                )
                .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));

                Arm {
                    backend: *backend,
                    op,
                    prepare_ms,
                    output,
                }
            })
            .collect();

        for round in 0..config.warmup_rounds {
            for index in balanced_arm_order(round, arms.len()) {
                let arm = &mut arms[index];
                std::hint::black_box(
                    time_zeroed_output(&mut arm.output, 1, |output| arm.op.spmv(&x, output))
                        .expect("warmup SpMV"),
                );
            }
        }

        let mut samples = vec![Vec::with_capacity(config.sample_rounds); arms.len()];
        for round in 0..config.sample_rounds {
            for index in balanced_arm_order(round, arms.len()) {
                let arm = &mut arms[index];
                samples[index].push(
                    time_zeroed_output(&mut arm.output, config.iterations, |output| {
                        arm.op.spmv(&x, output)
                    })
                    .expect("sampled SpMV"),
                );
            }
        }

        let reference_index = arms
            .iter()
            .position(|arm| arm.backend == Backend::CpuSequential)
            .expect("CPU sequential is always available");
        println!("\nn = {n}, nnz = {}", csr.nnz());
        println!(
            "  {:<22} {:>10} {:>11} {:>14} {:>14} {:>12}",
            "backend", "median ms", "paired x", "sample spread", "ratio spread", "prepare ms"
        );
        println!("  paired x is CPU-sequential/arm within the same round; spread is max/min");
        for (arm, arm_samples) in arms.iter().zip(&samples) {
            let stats = SampleStats::from_samples(arm_samples).expect("positive timing samples");
            let ratio = paired_ratio_stats(&samples[reference_index], arm_samples)
                .expect("same number of positive paired samples");
            println!(
                "  {:<22} {:>10.3} {:>10.2}x {:>14.2} {:>14.2} {:>12.3}",
                arm.op.label(),
                stats.median,
                ratio.median,
                stats.spread(),
                ratio.spread(),
                arm.prepare_ms
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::verify_spmv_preflight;
    use crate::support::reference_spmv;
    use sparsl::{Backend, Csr};

    #[test]
    fn preflight_oracle_rejects_unwritten_and_non_finite_outputs() {
        let csr = Csr::from_adjacency(&[vec![0], vec![1]]);
        let expected =
            reference_spmv(&csr, &[2.0, -3.0], &[4.0, 5.0], &[0.5, -0.5]).expect("reference");
        verify_spmv_preflight(
            "correct",
            Backend::CpuSequential,
            &expected,
            &expected,
            1,
            15.0,
        )
        .expect("exact CPU output");
        assert!(verify_spmv_preflight(
            "unwritten",
            Backend::CpuSequential,
            &[0.5, -0.5],
            &expected,
            1,
            15.0,
        )
        .is_err());
        assert!(verify_spmv_preflight(
            "nan",
            Backend::Metal,
            &[f32::NAN, expected[1]],
            &expected,
            1,
            15.0,
        )
        .is_err());
    }
}
