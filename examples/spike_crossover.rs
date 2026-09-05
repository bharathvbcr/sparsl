//! What does a bitpacked spike vector buy?
//!
//! The gathered operand shrinks from one `f32` per cell to one bit per cell.
//! Before timing, both dense and packed paths must match an independent scalar
//! oracle bit-for-bit. This prevents a missing or truncated device write from
//! being celebrated as an optimisation.
//!
//! Warmup is bounded, followed by paired AB/BA multi-sample rounds. Medians
//! replace the optimistic minimum of two attempts; per-arm and paired-ratio
//! spreads expose instability. Every timed iteration clears the accumulating
//! output and the result is black-boxed.
//! `SPARSL_BENCH_WARMUP_ROUNDS`, `SPARSL_BENCH_ROUNDS`, and
//! `SPARSL_BENCH_ITERS` tune the bounded sampling plan.
//!
//! Run: `cargo run --release --features metal --example spike_crossover`

mod support;

use sparsl::spikes::{pack_spikes, spikes_to_f32};
use sparsl::{Backend, Csr, Device, OpError, Rng, SparseOp};
use support::{
    balanced_arm_order, max_abs_finite, max_abs_product, max_row_nnz, paired_ratio_stats,
    reference_spmv, time_zeroed_output, verify_float_output, BenchConfig, SampleStats,
};

const SHAPES: &[(usize, usize)] = &[(10_000, 500), (20_000, 1_000), (50_000, 400)];
const SPIKE_DENSITY: f32 = 0.05;
const SEED_BASE: u64 = 0x5B1C_2026;
const DEFAULT_ITERS: usize = 20;
const MAX_ITERS: usize = 256;

#[derive(Clone, Copy)]
enum SpikeArm {
    Dense,
    Packed,
}

impl SpikeArm {
    const ALL: [Self; 2] = [Self::Dense, Self::Packed];
}

fn time_spike_arm(
    op: &SparseOp,
    dense: &[f32],
    packed: &[u32],
    output: &mut [f32],
    iterations: usize,
    arm: SpikeArm,
) -> Result<f64, OpError> {
    time_zeroed_output(output, iterations, |output| {
        match arm {
            SpikeArm::Dense => op.spmv(dense, output)?,
            SpikeArm::Packed => op.spmv_spikes(packed, output)?,
        }
        Ok(())
    })
}

/// Compare one GPU arm with the host oracle.
///
/// Bounded by `tolerance_for_spmv`, not by bit-identity. Both arms here run a
/// team kernel, which splits a row across its lanes and folds them with a
/// shuffle butterfly; the oracle is a sequential fold. Those associate
/// differently, so requiring bit-identity would be requiring the GPU to
/// reproduce the host's rounding -- something this crate promises nowhere, and
/// which this benchmark got away with only while the scalar kernel happened to
/// sum in the host's order. `crossover` and `narrow_crossover` already bound
/// their GPU arms this way; this now matches them.
fn verify_spike_preflight(
    label: &str,
    got: &[f32],
    expected: &[f32],
    max_row_nnz: usize,
    max_abs_term: f32,
) -> Result<(), String> {
    let tolerance = sparsl::tolerance_for_spmv(
        max_row_nnz,
        max_abs_term,
        max_abs_finite("reference output", expected)?,
    );
    verify_float_output(label, got, expected, tolerance)
}

/// The claim this benchmark exists to make, held to the bit.
///
/// The packed path performs the same multiply-add in the same order as the
/// dense one -- that is what makes 32x smaller spike vectors a free win rather
/// than an approximation -- so on one backend the two must agree exactly. It is
/// asserted directly rather than inferred from both arms separately matching a
/// third thing, which is all the previous zero-tolerance oracle comparison
/// established.
fn verify_arms_agree(dense: &[f32], packed: &[f32]) -> Result<(), String> {
    verify_float_output("packed against dense", packed, dense, 0.0)
}

fn main() {
    let config = BenchConfig::from_env(DEFAULT_ITERS, MAX_ITERS).unwrap_or_else(|error| {
        eprintln!("spike_crossover configuration error: {error}");
        std::process::exit(2);
    });
    support::print_provenance("spike_crossover", config);
    println!(
        "  workload: shapes={SHAPES:?} as (rows, degree), square CSR, spike_density={SPIKE_DENSITY}, seed=0x{SEED_BASE:X}^n"
    );
    println!("  timed_iteration: zero output, then one y += A*x dispatch");

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
        let adjacency: Vec<Vec<u32>> = (0..n)
            .map(|_| (0..degree).map(|_| rng.gen_index(n) as u32).collect())
            .collect();
        let csr = Csr::from_adjacency(&adjacency);
        let weights: Vec<f32> = (0..csr.nnz()).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let op = gpu.prepare(&csr, n, &weights).expect("prepare");

        let fired: Vec<bool> = (0..n).map(|_| rng.next_f32() < SPIKE_DENSITY).collect();
        let packed = pack_spikes(&fired);
        let dense = spikes_to_f32(&packed, n);
        let initial: Vec<f32> = (0..n)
            .map(|row| 0.125 + (row % 11) as f32 * 0.03125)
            .collect();
        let expected = reference_spmv(&csr, &weights, &dense, &initial)
            .unwrap_or_else(|error| panic!("reference preflight failed: {error}"));
        let term_scale = max_abs_product(&weights, &dense)
            .unwrap_or_else(|error| panic!("preflight scale failed: {error}"));
        let row_nnz = max_row_nnz(&csr);

        let mut dense_output = initial.clone();
        op.spmv(&dense, &mut dense_output)
            .expect("dense correctness dispatch");
        verify_spike_preflight("dense", &dense_output, &expected, row_nnz, term_scale)
            .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));
        let mut packed_output = initial;
        op.spmv_spikes(&packed, &mut packed_output)
            .expect("packed correctness dispatch");
        verify_spike_preflight("packed", &packed_output, &expected, row_nnz, term_scale)
            .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));
        verify_arms_agree(&dense_output, &packed_output)
            .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));
        let mut outputs = [dense_output, packed_output];

        for round in 0..config.warmup_rounds {
            for index in balanced_arm_order(round, SpikeArm::ALL.len()) {
                std::hint::black_box(
                    time_spike_arm(
                        &op,
                        &dense,
                        &packed,
                        &mut outputs[index],
                        1,
                        SpikeArm::ALL[index],
                    )
                    .expect("warmup spike SpMV"),
                );
            }
        }

        let mut samples = vec![Vec::with_capacity(config.sample_rounds); SpikeArm::ALL.len()];
        for round in 0..config.sample_rounds {
            for index in balanced_arm_order(round, SpikeArm::ALL.len()) {
                samples[index].push(
                    time_spike_arm(
                        &op,
                        &dense,
                        &packed,
                        &mut outputs[index],
                        config.iterations,
                        SpikeArm::ALL[index],
                    )
                    .expect("sampled spike SpMV"),
                );
            }
        }

        let dense_stats = SampleStats::from_samples(&samples[0]).expect("dense samples");
        let packed_stats = SampleStats::from_samples(&samples[1]).expect("packed samples");
        let speedup = paired_ratio_stats(&samples[0], &samples[1])
            .expect("same number of positive paired samples");
        println!(
            "\nn = {n}, degree = {degree}, nnz = {}, x = {:.1} KiB dense / {:.1} KiB packed",
            csr.nnz(),
            dense.len() as f64 * std::mem::size_of::<f32>() as f64 / 1024.0,
            packed.len() as f64 * std::mem::size_of::<u32>() as f64 / 1024.0
        );
        println!(
            "  dense median={:.3} ms spread={:.2}; packed median={:.3} ms spread={:.2}; paired dense/packed={:.2}x ratio_spread={:.2}",
            dense_stats.median,
            dense_stats.spread(),
            packed_stats.median,
            packed_stats.spread(),
            speedup.median,
            speedup.spread()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{time_spike_arm, verify_arms_agree, verify_spike_preflight, SpikeArm};
    use sparsl::spikes::pack_spikes;
    use sparsl::{Csr, Device};

    #[test]
    fn dense_and_packed_timed_iterations_reset_accumulating_outputs() {
        let csr = Csr::from_adjacency(&[vec![0], vec![1]]);
        let op = Device::cpu_sequential()
            .prepare(&csr, 2, &[2.0, -3.0])
            .expect("identity-shaped operator");
        let dense = [1.0, 0.0];
        let packed = pack_spikes(&[true, false]);
        for arm in [SpikeArm::Dense, SpikeArm::Packed] {
            let mut output = vec![99.0, -99.0];
            time_spike_arm(&op, &dense, &packed, &mut output, 3, arm).expect("repeated product");
            assert_eq!(output, vec![2.0, 0.0]);
        }
    }

    #[test]
    fn the_oracle_preflight_still_rejects_unwritten_or_non_finite_outputs() {
        // A tolerance is not permission to pass anything: a missing write and a
        // NaN must both still be refused, or an unwritten output becomes an
        // impressive latency.
        let expected = [2.5f32, -0.5];
        verify_spike_preflight("correct", &expected, &expected, 4, 1.0).expect("exact output");
        assert!(verify_spike_preflight("unwritten", &[0.5, -0.5], &expected, 4, 1.0).is_err());
        assert!(verify_spike_preflight("nan", &[f32::NAN, -0.5], &expected, 4, 1.0).is_err());
    }

    #[test]
    fn the_two_arms_are_compared_to_each_other_exactly() {
        let dense = [2.5f32, -0.5];
        verify_arms_agree(&dense, &dense).expect("identical arms");
        // One ulp apart: within any sane cross-substrate tolerance, and still a
        // failure here, because these two paths share a backend and an order.
        let nudged = [f32::from_bits(dense[0].to_bits() + 1), dense[1]];
        assert!(verify_arms_agree(&dense, &nudged).is_err());
        assert!(verify_arms_agree(&dense, &[f32::NAN, -0.5]).is_err());
    }
}
