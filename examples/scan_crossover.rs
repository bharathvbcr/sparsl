//! Does the GPU scan actually beat the CPU one?
//!
//! `scan.rs` records that an earlier two-sample run suggested 1.08x for the
//! chunked CPU scan, but the hardened sampler did not reproduce a crossover.
//! Bit-identity forces phase 1 to be a complete sequential left-fold and phase
//! 2 then redoes that work. The same module also records that an old prediction
//! of a tree-scan speedup was wrong after the Metal kernel made it measurable.
//! This command keeps both conclusions independently checkable.
//!
//! Warmup is bounded, then every arm is measured over multiple rounds with the
//! order reversed each round. Medians replace the optimistic minimum of two
//! samples; per-arm and paired-ratio spreads expose unstable measurements.
//! Untimed realistic-data and exact counting comparisons refuse to sample an
//! incorrect arm, including one whose missing tail writes could hide inside a
//! deliberately conservative cross-backend floating-point tolerance.
//! `SPARSL_BENCH_WARMUP_ROUNDS`, `SPARSL_BENCH_ROUNDS`, and
//! `SPARSL_BENCH_ITERS` tune the bounded sampling plan.
//! `SPARSL_BENCH_SCAN_SIZES` accepts a comma-separated, bounded size sweep;
//! `SPARSL_BENCH_CPU_ONLY=1` skips device discovery and measures only the two
//! CPU algorithms, which is useful when GPU work must be serialized.
//!
//! Run: `cargo run --release --features metal --example scan_crossover`

mod support;

use std::{env, time::Instant};

use sparsl::{
    assoc_scan, assoc_scan_sequential, available_backends, scan_magnitude_envelope,
    tolerance_for_scan, Device, Rng, State,
};
use support::{milliseconds_per_iteration, paired_ratio_stats, BenchConfig, SampleStats};

const DEFAULT_SIZES: &[usize] = &[1 << 16, 1 << 18, 1 << 20, 1 << 22];
const MAX_CUSTOM_SIZES: usize = 32;
const MAX_SCAN_SIZE: usize = 1 << 24;
const SEED_BASE: u64 = 0x5CA1_0000;
const DEFAULT_ITERS: usize = 3;
const MAX_ITERS: usize = 4096;

/// Leak steps with bounded `a`, the shape the affine scan exists for.
fn leak_steps(n: usize, seed: u64) -> Vec<State> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let tau = 2.0 + rng.next_f32() * 8.0;
            let input = rng.next_f32() * 2.0 - 1.0;
            State::leak_step(input, tau, 1.0)
        })
        .collect()
}

fn time_scan(mut scan: impl FnMut() -> Vec<State>, iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(scan());
    }
    milliseconds_per_iteration(start.elapsed(), iterations)
}

/// Fail closed before timing if an arm did not perform the scan being measured.
fn verify_scan_result(
    label: &str,
    got: &[State],
    expected: &[State],
    tolerance: f32,
) -> Result<(), String> {
    if got.len() != expected.len() {
        return Err(format!(
            "{label}: output length {} differs from reference length {}",
            got.len(),
            expected.len()
        ));
    }
    if !tolerance.is_finite() || tolerance < 0.0 {
        return Err(format!(
            "{label}: comparison tolerance must be finite and non-negative, got {tolerance}"
        ));
    }

    for (index, (actual, want)) in got.iter().zip(expected).enumerate() {
        for (field, actual, want) in [("a", actual.a, want.a), ("b", actual.b, want.b)] {
            if actual.to_bits() == want.to_bits() {
                continue;
            }
            // The first inclusive prefix is copied, not combined, on every
            // backend. Requiring its bits exactly prevents a large whole-scan
            // tolerance from disguising a kernel that never wrote its output.
            if index == 0 {
                return Err(format!(
                    "{label}: first prefix changed at {field} ({actual:?} versus {want:?})"
                ));
            }
            if !actual.is_finite() || !want.is_finite() {
                return Err(format!(
                    "{label}: non-finite mismatch at {field}[{index}] ({actual:?} versus {want:?})"
                ));
            }
            let delta = (actual - want).abs();
            if !delta.is_finite() || delta > tolerance {
                return Err(format!(
                    "{label}: {field}[{index}] differs by {delta}, above tolerance {tolerance} ({actual:?} versus {want:?})"
                ));
            }
        }
    }
    Ok(())
}

/// Verify every output slot and both scan phases with exactly representable data.
///
/// The realistic leak workload still uses the public floating-point bound, as
/// it must across different parenthesizations. That bound intentionally grows
/// with the full prefix and can be broad at benchmark sizes. A counting scan
/// (`a = 1, b = 1`) complements it: every prefix through 2^24 is an exactly
/// representable integer under either association, so no tolerance can hide a
/// missing tail write, wrong group offset, or truncated dispatch.
fn verify_exact_counting_scan(
    label: &str,
    got: &[State],
    expected_len: usize,
) -> Result<(), String> {
    if expected_len > (1 << 24) {
        return Err(format!(
            "{label}: exact counting preflight length {expected_len} exceeds the f32 integer range"
        ));
    }
    if got.len() != expected_len {
        return Err(format!(
            "{label}: counting output length {} differs from expected length {expected_len}",
            got.len()
        ));
    }
    for (index, actual) in got.iter().enumerate() {
        let expected_b = (index + 1) as f32;
        if actual.a.to_bits() != 1.0f32.to_bits() || actual.b.to_bits() != expected_b.to_bits() {
            return Err(format!(
                "{label}: counting prefix {index} was {actual:?}, expected State {{ a: 1.0, b: {expected_b} }}"
            ));
        }
    }
    Ok(())
}

fn scan_sizes_from_env() -> Result<Vec<usize>, String> {
    match env::var("SPARSL_BENCH_SCAN_SIZES") {
        Ok(raw) => parse_scan_sizes(Some(&raw)),
        Err(env::VarError::NotPresent) => parse_scan_sizes(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err("SPARSL_BENCH_SCAN_SIZES is not valid Unicode".to_owned())
        }
    }
}

fn parse_scan_sizes(raw: Option<&str>) -> Result<Vec<usize>, String> {
    let Some(raw) = raw else {
        return Ok(DEFAULT_SIZES.to_vec());
    };
    let values: Vec<&str> = raw.split(',').map(str::trim).collect();
    if values.is_empty() || values.len() > MAX_CUSTOM_SIZES || values.iter().any(|v| v.is_empty()) {
        return Err(format!(
            "SPARSL_BENCH_SCAN_SIZES must contain 1..={MAX_CUSTOM_SIZES} comma-separated integers"
        ));
    }
    values
        .into_iter()
        .map(|value| {
            let n = value.parse::<usize>().map_err(|_| {
                format!("SPARSL_BENCH_SCAN_SIZES contains a non-integer value {value:?}")
            })?;
            if !(1..=MAX_SCAN_SIZE).contains(&n) {
                return Err(format!(
                    "SPARSL_BENCH_SCAN_SIZES values must be in 1..={MAX_SCAN_SIZE}, got {n}"
                ));
            }
            Ok(n)
        })
        .collect()
}

fn cpu_only_from_env() -> Result<bool, String> {
    match env::var("SPARSL_BENCH_CPU_ONLY") {
        Ok(raw) => parse_cpu_only(Some(&raw)),
        Err(env::VarError::NotPresent) => parse_cpu_only(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err("SPARSL_BENCH_CPU_ONLY is not valid Unicode".to_owned())
        }
    }
}

fn parse_cpu_only(raw: Option<&str>) -> Result<bool, String> {
    match raw.map(str::trim) {
        None | Some("0" | "false" | "no" | "off") => Ok(false),
        Some("1" | "true" | "yes" | "on") => Ok(true),
        Some(value) => Err(format!(
            "SPARSL_BENCH_CPU_ONLY must be one of 0/1, false/true, no/yes, or off/on; got {value:?}"
        )),
    }
}

fn main() {
    let configured = (|| {
        Ok::<_, String>((
            BenchConfig::from_env(DEFAULT_ITERS, MAX_ITERS)?,
            scan_sizes_from_env()?,
            cpu_only_from_env()?,
        ))
    })();
    let (config, sizes, cpu_only) = configured.unwrap_or_else(|error| {
        eprintln!("scan_crossover configuration error: {error}");
        std::process::exit(2);
    });
    let devices: Vec<Device> = if cpu_only {
        Vec::new()
    } else {
        available_backends()
            .into_iter()
            .map(|backend| {
                Device::try_new(backend).unwrap_or_else(|error| {
                    panic!(
                        "advertised backend {} failed to open: {error}",
                        backend.label()
                    )
                })
            })
            .collect()
    };

    support::print_provenance("scan_crossover", config);
    println!("  workload: sizes={sizes:?}, seed=0x{SEED_BASE:X}+n");
    println!("  cpu_only: {cpu_only}");
    println!(
        "  backends: {}",
        if devices.is_empty() {
            "none".to_owned()
        } else {
            devices
                .iter()
                .map(Device::label)
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    for device in &devices {
        if let Some(name) = device.device_name() {
            println!("  device.{}: {name}", device.label());
        }
    }

    for &n in &sizes {
        let xs = leak_steps(n, SEED_BASE + n as u64);

        // A benchmark that never checks its outputs can reward a missing write
        // as an optimisation. Gate every arm once outside the timed region.
        let reference = assoc_scan_sequential(&xs, State::combine);
        let chunked = assoc_scan(&xs, State::combine);
        verify_scan_result("cpu chunked (rayon)", &chunked, &reference, 0.0)
            .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));
        let cross_backend_tolerance = tolerance_for_scan(n, scan_magnitude_envelope(&xs));
        for device in &devices {
            let output = device.assoc_scan(&xs).unwrap_or_else(|error| {
                panic!("{} preflight dispatch failed: {error}", device.label())
            });
            let tolerance = if device.backend().is_gpu() {
                cross_backend_tolerance
            } else {
                0.0
            };
            verify_scan_result(device.label(), &output, &reference, tolerance)
                .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));
        }
        // The preflight outputs are not part of the measured working set. In
        // particular, two retained 4M-state vectors would add 64 MiB of live
        // memory and turn a correctness guard into a benchmark perturbation.
        drop(chunked);
        drop(reference);

        // The cross-backend error bound above is deliberately conservative
        // and grows with n. Exercise the complete GPU grid once more with an
        // exact oracle so a missing tail write cannot hide inside that bound.
        for device in devices.iter().filter(|device| device.backend().is_gpu()) {
            let counting = vec![State { a: 1.0, b: 1.0 }; n];
            let output = device.assoc_scan(&counting).unwrap_or_else(|error| {
                panic!(
                    "{} exact counting preflight failed: {error}",
                    device.label()
                )
            });
            verify_exact_counting_scan(device.label(), &output, n)
                .unwrap_or_else(|error| panic!("correctness preflight failed: {error}"));
        }

        // Warm every arm in alternating order without making warmup itself an
        // unbounded multiplication of the configured sample iteration count.
        for round in 0..config.warmup_rounds {
            if round % 2 == 0 {
                std::hint::black_box(time_scan(|| assoc_scan_sequential(&xs, State::combine), 1));
                std::hint::black_box(time_scan(|| assoc_scan(&xs, State::combine), 1));
                for device in &devices {
                    std::hint::black_box(time_scan(|| device.assoc_scan(&xs).expect("scan"), 1));
                }
            } else {
                for device in devices.iter().rev() {
                    std::hint::black_box(time_scan(|| device.assoc_scan(&xs).expect("scan"), 1));
                }
                std::hint::black_box(time_scan(|| assoc_scan(&xs, State::combine), 1));
                std::hint::black_box(time_scan(|| assoc_scan_sequential(&xs, State::combine), 1));
            }
        }

        let mut seq_samples = Vec::with_capacity(config.sample_rounds);
        let mut chunked_samples = Vec::with_capacity(config.sample_rounds);
        let mut device_samples = vec![Vec::with_capacity(config.sample_rounds); devices.len()];
        for round in 0..config.sample_rounds {
            if round % 2 == 0 {
                seq_samples.push(time_scan(
                    || assoc_scan_sequential(&xs, State::combine),
                    config.iterations,
                ));
                chunked_samples.push(time_scan(
                    || assoc_scan(&xs, State::combine),
                    config.iterations,
                ));
                for (samples, device) in device_samples.iter_mut().zip(&devices) {
                    samples.push(time_scan(
                        || device.assoc_scan(&xs).expect("scan"),
                        config.iterations,
                    ));
                }
            } else {
                for i in (0..devices.len()).rev() {
                    device_samples[i].push(time_scan(
                        || devices[i].assoc_scan(&xs).expect("scan"),
                        config.iterations,
                    ));
                }
                chunked_samples.push(time_scan(
                    || assoc_scan(&xs, State::combine),
                    config.iterations,
                ));
                seq_samples.push(time_scan(
                    || assoc_scan_sequential(&xs, State::combine),
                    config.iterations,
                ));
            }
        }

        let sequential =
            SampleStats::from_samples(&seq_samples).expect("positive sequential samples");

        println!("\nn = {n} ({:.1}M elements)", n as f64 / 1e6);
        println!(
            "  {:<22} {:>10} {:>11} {:>14} {:>14}",
            "arm", "median ms", "paired x", "sample spread", "ratio spread"
        );
        println!("  spread is max/min; paired x is CPU-sequential/arm within each round");
        println!(
            "  {:<22} {:>10.6} {:>10.2}x {:>14.2} {:>14.2}",
            "cpu sequential",
            sequential.median,
            1.0,
            sequential.spread(),
            1.0
        );
        let chunked =
            SampleStats::from_samples(&chunked_samples).expect("positive chunked samples");
        let chunked_ratio = paired_ratio_stats(&seq_samples, &chunked_samples)
            .expect("same number of positive paired samples");
        println!(
            "  {:<22} {:>10.6} {:>10.2}x {:>14.2} {:>14.2}",
            "cpu chunked (rayon)",
            chunked.median,
            chunked_ratio.median,
            chunked.spread(),
            chunked_ratio.spread()
        );
        for (device, samples) in devices.iter().zip(&device_samples) {
            let stats = SampleStats::from_samples(samples).expect("positive device samples");
            let ratio = paired_ratio_stats(&seq_samples, samples)
                .expect("same number of positive paired samples");
            println!(
                "  {:<22} {:>10.6} {:>10.2}x {:>14.2} {:>14.2}",
                device.label(),
                stats.median,
                ratio.median,
                stats.spread(),
                ratio.spread()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        parse_cpu_only, parse_scan_sizes, verify_exact_counting_scan, verify_scan_result,
        DEFAULT_SIZES, MAX_CUSTOM_SIZES, MAX_SCAN_SIZE,
    };
    use sparsl::State;

    #[test]
    fn custom_scan_sizes_are_bounded_and_order_preserving() {
        assert_eq!(parse_scan_sizes(None), Ok(DEFAULT_SIZES.to_vec()));
        assert_eq!(
            parse_scan_sizes(Some("257, 1024,65536")),
            Ok(vec![257, 1024, 65536])
        );
        assert!(parse_scan_sizes(Some("")).is_err());
        assert!(parse_scan_sizes(Some("1,,2")).is_err());
        assert!(parse_scan_sizes(Some("not-a-size")).is_err());
        assert!(parse_scan_sizes(Some("0")).is_err());
        assert!(parse_scan_sizes(Some(&(MAX_SCAN_SIZE + 1).to_string())).is_err());
        let too_many = vec!["1"; MAX_CUSTOM_SIZES + 1].join(",");
        assert!(parse_scan_sizes(Some(&too_many)).is_err());
    }

    #[test]
    fn cpu_only_parser_is_explicit() {
        assert_eq!(parse_cpu_only(None), Ok(false));
        for value in ["1", "true", "yes", "on"] {
            assert_eq!(parse_cpu_only(Some(value)), Ok(true));
        }
        for value in ["0", "false", "no", "off"] {
            assert_eq!(parse_cpu_only(Some(value)), Ok(false));
        }
        assert!(parse_cpu_only(Some("sometimes")).is_err());
    }

    #[test]
    fn correctness_preflight_rejects_wrong_and_non_finite_outputs() {
        let expected = [State { a: 0.5, b: 1.0 }];
        assert!(verify_scan_result("exact", &expected, &expected, 0.0).is_ok());
        assert!(verify_scan_result("wrong", &[State { a: 0.5, b: 2.0 }], &expected, 0.25).is_err());
        assert!(verify_scan_result(
            "nan",
            &[State {
                a: f32::NAN,
                b: 1.0,
            }],
            &expected,
            0.0
        )
        .is_err());
        assert!(verify_scan_result("bad tolerance", &expected, &expected, f32::INFINITY).is_err());

        // At 4M elements the public worst-case bound can legitimately exceed
        // one. The first-prefix invariant must still reject a missing write
        // rather than letting that global tolerance make the preflight vacuous.
        assert!(verify_scan_result(
            "missing first write",
            &[State { a: 1.0, b: 0.0 }],
            &expected,
            40.0
        )
        .is_err());

        let counting = [
            State { a: 1.0, b: 1.0 },
            State { a: 1.0, b: 2.0 },
            State { a: 1.0, b: 3.0 },
        ];
        assert!(verify_exact_counting_scan("counting", &counting, counting.len()).is_ok());
        let mut missing_tail = counting;
        missing_tail[2] = State { a: 1.0, b: 0.0 };
        assert!(
            verify_exact_counting_scan("missing tail", &missing_tail, missing_tail.len()).is_err()
        );
    }
}
