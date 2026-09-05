//! What does narrow weight storage actually buy?
//!
//! Narrowing only `values` changes streamed index+weight traffic from 8 bytes
//! to 6 bytes per non-zero, so 1.33x is a ceiling rather than a prediction.
//! Every format is checked against an independent unquantised scalar oracle
//! before timing, using the public bound appropriate to its storage precision.
//! A wrong or non-finite result aborts the benchmark instead of looking fast.
//!
//! Warmup is bounded, followed by paired, order-alternating multi-sample rounds.
//! Medians replace the optimistic minimum of two attempts, and per-arm plus
//! paired-ratio spreads expose instability. Every timed iteration clears the
//! accumulating output and black-boxes the final result.
//! `SPARSL_BENCH_WARMUP_ROUNDS`, `SPARSL_BENCH_ROUNDS`, and
//! `SPARSL_BENCH_ITERS` tune the bounded sampling plan.
//!
//! Run: `cargo run --release --features metal --example narrow_crossover`

mod support;

use sparsl::{
    tolerance_for_spmv, tolerance_for_spmv_narrow, Backend, Csr, Device, Rng, SparseOp,
    WeightPrecision,
};
use support::{
    balanced_arm_order, max_abs_finite, max_abs_product, max_row_nnz, paired_ratio_stats,
    reference_spmv, time_zeroed_output, verify_float_output, BenchConfig, SampleStats,
};

const SHAPES: &[(usize, usize)] = &[(10_000, 500), (20_000, 1_000), (50_000, 400)];
const SEED_BASE: u64 = 0xF16C_2026;
const DEFAULT_ITERS: usize = 20;
const MAX_ITERS: usize = 256;

fn random_csr(nrows: usize, ncols: usize, degree: usize, rng: &mut Rng) -> Csr {
    let adj: Vec<Vec<u32>> = (0..nrows)
        .map(|_| (0..degree).map(|_| rng.gen_index(ncols) as u32).collect())
        .collect();
    Csr::from_adjacency(&adj)
}

fn precision_label(precision: WeightPrecision) -> &'static str {
    match precision {
        WeightPrecision::F32 => "f32",
        WeightPrecision::F16 => "binary16",
        WeightPrecision::Bf16 => "bfloat16",
    }
}

fn verify_narrow_preflight(
    label: &str,
    precision: WeightPrecision,
    got: &[f32],
    expected: &[f32],
    max_row_nnz: usize,
    max_abs_term: f32,
) -> Result<(), String> {
    let result_scale = max_abs_finite("reference output", expected)?;
    let tolerance = match precision {
        WeightPrecision::F32 => tolerance_for_spmv(max_row_nnz, max_abs_term, result_scale),
        WeightPrecision::F16 | WeightPrecision::Bf16 => {
            tolerance_for_spmv_narrow(precision, max_row_nnz, max_abs_term, result_scale)
        }
    };
    verify_float_output(label, got, expected, tolerance)
}

struct Arm {
    precision: WeightPrecision,
    op: SparseOp,
    prepare_ms: f64,
    output: Vec<f32>,
}

fn main() {
    let config = BenchConfig::from_env(DEFAULT_ITERS, MAX_ITERS).unwrap_or_else(|error| {
        eprintln!("narrow_crossover configuration error: {error}");
        std::process::exit(2);
    });
    support::print_provenance("narrow_crossover", config);
    println!("  workload: shapes={SHAPES:?} as (rows, degree), square CSR, seed=0x{SEED_BASE:X}^n");
    println!("  timed_iteration: zero output, then one y += A*x dispatch");
    println!("  traffic_per_nonzero: f32=8 bytes, narrow=6 bytes, ceiling=1.33x");

    let Ok(gpu) = Device::try_new(Backend::Metal) else {
        eprintln!(
            "no Metal device; nothing to measure: {}",
            Backend::Metal
                .unavailable_reason()
                .unwrap_or("device construction failed")
        );
        return;
    };
    if let Some(name) = gpu.device_name() {
        println!("  device.{}: {name}", gpu.label());
    }

    for &(n, degree) in SHAPES {
        let mut rng = Rng::new(SEED_BASE ^ n as u64);
        let csr = random_csr(n, n, degree, &mut rng);
        let weights: Vec<f32> = (0..csr.nnz()).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let input: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let initial: Vec<f32> = (0..n)
            .map(|row| -0.375 + (row % 13) as f32 * 0.0625)
            .collect();
        let expected = reference_spmv(&csr, &weights, &input, &initial)
            .unwrap_or_else(|error| panic!("reference preflight failed: {error}"));
        let term_scale = max_abs_product(&weights, &input)
            .unwrap_or_else(|error| panic!("preflight scale failed: {error}"));
        let row_nnz = max_row_nnz(&csr);

        let mut arms: Vec<Arm> = [
            WeightPrecision::F32,
            WeightPrecision::F16,
            WeightPrecision::Bf16,
        ]
        .into_iter()
        .map(|precision| {
            let start = std::time::Instant::now();
            let op = gpu
                .prepare_with(&csr, n, &weights, precision)
                .unwrap_or_else(|error| {
                    panic!("{} preparation failed: {error}", precision_label(precision))
                });
            let prepare_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(
                op.weight_precision(),
                precision,
                "Metal operator did not retain requested storage precision"
            );

            let mut output = initial.clone();
            op.spmv(&input, &mut output).unwrap_or_else(|error| {
                panic!(
                    "{} correctness dispatch failed: {error}",
                    precision_label(precision)
                )
            });
            verify_narrow_preflight(
                precision_label(precision),
                precision,
                &output,
                &expected,
                row_nnz,
                term_scale,
            )
            .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));

            Arm {
                precision,
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
                    time_zeroed_output(&mut arm.output, 1, |output| arm.op.spmv(&input, output))
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
                        arm.op.spmv(&input, output)
                    })
                    .expect("sampled SpMV"),
                );
            }
        }

        let wide_index = arms
            .iter()
            .position(|arm| arm.precision == WeightPrecision::F32)
            .expect("wide arm is present");
        println!("\nn = {n}, degree = {degree}, nnz = {}", csr.nnz());
        println!(
            "  {:<10} {:>10} {:>11} {:>14} {:>14} {:>12}",
            "storage", "median ms", "paired x", "sample spread", "ratio spread", "prepare ms"
        );
        println!("  paired x is f32/format within the same round; spread is max/min");
        for (arm, arm_samples) in arms.iter().zip(&samples) {
            let stats = SampleStats::from_samples(arm_samples).expect("positive timing samples");
            let ratio = paired_ratio_stats(&samples[wide_index], arm_samples)
                .expect("same number of positive paired samples");
            println!(
                "  {:<10} {:>10.3} {:>10.2}x {:>14.2} {:>14.2} {:>12.3}",
                precision_label(arm.precision),
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
    use super::verify_narrow_preflight;
    use sparsl::WeightPrecision;

    #[test]
    fn narrow_preflight_rejects_optimistic_or_poisoned_results() {
        let expected = [8.5f32, -15.5];
        verify_narrow_preflight("wide", WeightPrecision::F32, &expected, &expected, 2, 15.0)
            .expect("correct wide result");
        assert!(verify_narrow_preflight(
            "unwritten",
            WeightPrecision::F16,
            &[0.5, -0.5],
            &expected,
            2,
            15.0,
        )
        .is_err());
        assert!(verify_narrow_preflight(
            "nan",
            WeightPrecision::Bf16,
            &[f32::NAN, expected[1]],
            &expected,
            2,
            15.0,
        )
        .is_err());
    }
}
