//! Shared measurement and provenance support for the performance examples.

use std::env;
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_WARMUP_ROUNDS: usize = 4;
const MAX_WARMUP_ROUNDS: usize = 16;
const DEFAULT_SAMPLE_ROUNDS: usize = 8;
const MIN_SAMPLE_ROUNDS: usize = 4;
const MAX_SAMPLE_ROUNDS: usize = 32;

/// Bounded benchmark configuration loaded from the environment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchConfig {
    pub warmup_rounds: usize,
    pub sample_rounds: usize,
    pub iterations: usize,
}

impl BenchConfig {
    /// Load the common knobs while letting each workload cap its inner loop.
    pub fn from_env(default_iterations: usize, max_iterations: usize) -> Result<Self, String> {
        let warmup_rounds = env_count(
            "SPARSL_BENCH_WARMUP_ROUNDS",
            DEFAULT_WARMUP_ROUNDS,
            0,
            MAX_WARMUP_ROUNDS,
        )?;
        let sample_rounds = env_count(
            "SPARSL_BENCH_ROUNDS",
            DEFAULT_SAMPLE_ROUNDS,
            MIN_SAMPLE_ROUNDS,
            MAX_SAMPLE_ROUNDS,
        )?;
        require_even_sample_rounds(sample_rounds)?;
        let iterations = env_count("SPARSL_BENCH_ITERS", default_iterations, 1, max_iterations)?;
        Ok(Self {
            warmup_rounds,
            sample_rounds,
            iterations,
        })
    }
}

/// Robust summary of independent positive timing samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleStats {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

impl SampleStats {
    pub fn from_samples(samples: &[f64]) -> Result<Self, String> {
        validate_samples(samples)?;

        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        let median = if sorted.len() % 2 == 0 {
            let lower = sorted[middle - 1];
            lower + (sorted[middle] - lower) / 2.0
        } else {
            sorted[middle]
        };
        Ok(Self {
            median,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        })
    }

    /// Largest sample divided by smallest; 1.00 means no observed spread.
    pub fn spread(self) -> f64 {
        self.max / self.min
    }
}

/// Summarize the per-round ratio between two measurements taken as a pair.
pub fn paired_ratio_stats(numerator: &[f64], denominator: &[f64]) -> Result<SampleStats, String> {
    if numerator.len() != denominator.len() {
        return Err(format!(
            "paired samples differ in length: {} versus {}",
            numerator.len(),
            denominator.len()
        ));
    }
    validate_samples(numerator).map_err(|error| format!("invalid numerator: {error}"))?;
    validate_samples(denominator).map_err(|error| format!("invalid denominator: {error}"))?;
    let ratios: Vec<f64> = numerator
        .iter()
        .zip(denominator)
        .map(|(a, b)| a / b)
        .collect();
    SampleStats::from_samples(&ratios)
}

/// Compare a benchmark arm with an independently computed finite reference.
///
/// A benchmark must fail closed here: a NaN makes ordinary `delta > tolerance`
/// comparisons false, while a missing write can otherwise become an impressive
/// latency. A zero tolerance means bit identity, including the sign of zero.
#[allow(dead_code)] // This shared module is compiled separately by non-SpMV examples.
pub fn verify_float_output(
    label: &str,
    got: &[f32],
    expected: &[f32],
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
            "{label}: tolerance must be finite and non-negative, got {tolerance}"
        ));
    }

    for (index, (&actual, &want)) in got.iter().zip(expected).enumerate() {
        if !actual.is_finite() || !want.is_finite() {
            return Err(format!(
                "{label}: non-finite value at output {index} ({actual:?} versus {want:?})"
            ));
        }
        if tolerance == 0.0 {
            if actual.to_bits() != want.to_bits() {
                return Err(format!(
                    "{label}: output {index} was {actual:?} (0x{:08X}), expected {want:?} (0x{:08X})",
                    actual.to_bits(),
                    want.to_bits()
                ));
            }
        } else {
            let delta = (actual - want).abs();
            if !delta.is_finite() || delta > tolerance {
                return Err(format!(
                    "{label}: output {index} differs by {delta}, above tolerance {tolerance} ({actual:?} versus {want:?})"
                ));
            }
        }
    }
    Ok(())
}

/// Largest absolute finite element, floored away from zero for error bounds.
#[allow(dead_code)] // This shared module is compiled separately by non-SpMV examples.
pub fn max_abs_finite(label: &str, values: &[f32]) -> Result<f32, String> {
    let mut max = f32::MIN_POSITIVE;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!("{label}[{index}] is non-finite: {value:?}"));
        }
        max = max.max(value.abs());
    }
    Ok(max)
}

/// Conservative largest `|weight * input|` used by the public SpMV bounds.
#[allow(dead_code)] // This shared module is compiled separately by non-SpMV examples.
pub fn max_abs_product(weights: &[f32], input: &[f32]) -> Result<f32, String> {
    let weights = max_abs_finite("weights", weights)?;
    let input = max_abs_finite("input", input)?;
    let product = weights * input;
    if !product.is_finite() {
        return Err(format!(
            "maximum |weight| * |input| overflowed f32 ({weights} * {input})"
        ));
    }
    Ok(product.max(f32::MIN_POSITIVE))
}

/// Independent scalar `y += A*x` oracle shared by the SpMV benchmarks.
#[allow(dead_code)] // This shared module is compiled separately by non-SpMV examples.
pub fn reference_spmv(
    csr: &sparsl::Csr,
    weights: &[f32],
    input: &[f32],
    initial: &[f32],
) -> Result<Vec<f32>, String> {
    if weights.len() != csr.nnz() {
        return Err(format!(
            "weight length {} differs from CSR nnz {}",
            weights.len(),
            csr.nnz()
        ));
    }
    if initial.len() != csr.nrows() {
        return Err(format!(
            "initial output length {} differs from row count {}",
            initial.len(),
            csr.nrows()
        ));
    }

    let mut output = initial.to_vec();
    for (row, value) in output.iter_mut().enumerate() {
        let start = csr.row_ptr[row] as usize;
        let end = csr.row_ptr[row + 1] as usize;
        // Accumulate the row from zero and fold the seed in once, rather than
        // accumulating into the seeded output. Both the CPU arm (`row_dot`,
        // then `*y += sum`) and every Metal SpMV kernel (`float sum = 0.0f`,
        // then `y[id] += sum`) associate it this way, and the preflight holds
        // CPU arms to *bit* identity. Seeding the accumulator instead
        // reassociates the row and lands 2-3 ULP away, which failed the
        // `crossover` and `spike_crossover` preflights outright and was
        // silently absorbed by `narrow_crossover`'s cross-backend tolerance --
        // spending that budget on an oracle mismatch rather than on the
        // substrate difference it is meant to bound.
        let mut sum = 0.0f32;
        let row_weights = weights.get(start..end).ok_or_else(|| {
            format!(
                "CSR row {row} addresses weight range {start}..{end}, weight length is {}",
                weights.len()
            )
        })?;
        let row_cols = csr.col.get(start..end).ok_or_else(|| {
            format!(
                "CSR row {row} addresses column range {start}..{end}, column length is {}",
                csr.col.len()
            )
        })?;
        for (offset, (&weight, &col)) in row_weights.iter().zip(row_cols).enumerate() {
            let edge = start + offset;
            let col = col as usize;
            let x = input.get(col).ok_or_else(|| {
                format!(
                    "CSR edge {edge} addresses column {col}, input length is {}",
                    input.len()
                )
            })?;
            sum += weight * x;
        }
        *value += sum;
    }
    Ok(output)
}

/// Longest row, rather than the misleading mean degree, for error bounds.
#[allow(dead_code)] // This shared module is compiled separately by non-SpMV examples.
pub fn max_row_nnz(csr: &sparsl::Csr) -> usize {
    csr.row_ptr
        .windows(2)
        .map(|bounds| (bounds[1] - bounds[0]) as usize)
        .max()
        .unwrap_or(0)
}

/// Time repeated `y = operation` samples implemented by an accumulating API.
///
/// The required clear is inside the timed loop, because a caller asking an
/// `y += ...` API for one product must pay for it too. Allocation stays outside.
#[allow(dead_code)] // This shared module is compiled separately by non-SpMV examples.
pub fn time_zeroed_output<E>(
    output: &mut [f32],
    iterations: usize,
    mut dispatch: impl FnMut(&mut [f32]) -> Result<(), E>,
) -> Result<f64, E> {
    assert!(iterations > 0, "benchmark iteration count must be positive");
    let start = Instant::now();
    for _ in 0..iterations {
        output.fill(0.0);
        dispatch(output)?;
    }
    let elapsed = start.elapsed();
    std::hint::black_box(&*output);
    Ok(milliseconds_per_iteration(elapsed, iterations))
}

/// Forward arm order on even rounds, reverse order on odd rounds.
#[allow(dead_code)] // This shared module is compiled separately by scan/batch examples.
pub fn balanced_arm_order(round: usize, arm_count: usize) -> impl Iterator<Item = usize> {
    let forward = round % 2 == 0;
    (0..arm_count).map(move |offset| {
        if forward {
            offset
        } else {
            arm_count - 1 - offset
        }
    })
}

pub fn milliseconds_per_iteration(elapsed: Duration, iterations: usize) -> f64 {
    assert!(iterations > 0, "benchmark iteration count must be positive");
    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

/// Print source identity and host-specific context needed to audit a result.
///
/// A dirty checkout is explicitly marked non-reproducible: its HEAD revision
/// alone does not identify the source that produced the number.
pub fn print_provenance(benchmark: &str, config: BenchConfig) {
    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().to_string())
        .unwrap_or_else(|_| "unknown".to_owned());
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let revision = git_output(&["rev-parse", "HEAD"]);
    let dirty = match Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
    {
        Ok(output) if output.status.success() => {
            if output.stdout.is_empty() {
                "false".to_owned()
            } else {
                "true".to_owned()
            }
        }
        _ => "unknown".to_owned(),
    };

    println!("benchmark provenance:");
    println!("  benchmark: {benchmark}");
    println!("  timestamp_unix_ms: {timestamp_ms}");
    println!("  sparsl_version: {}", env!("CARGO_PKG_VERSION"));
    println!("  git_revision: {revision}");
    println!("  git_dirty: {dirty}");
    let source_reproducible = match (revision.as_str(), dirty.as_str()) {
        ("unavailable" | "unknown", _) | (_, "unknown") => "unknown",
        (_, "false") => "true",
        (_, "true") => "false",
        _ => "unknown",
    };
    println!("  source_reproducible_from_revision: {source_reproducible}");
    println!(
        "  build: profile={profile}, arch={}, os={}, pointer_width={}, metal={}, cuda={}",
        env::consts::ARCH,
        env::consts::OS,
        usize::BITS,
        cfg!(feature = "metal"),
        cfg!(feature = "cuda")
    );
    println!(
        "  rustc_on_path: {}",
        command_output("rustc", &["--version"])
    );
    println!(
        "  cargo_on_path: {}",
        command_output("cargo", &["--version"])
    );
    println!("  host_uname: {}", command_output("uname", &["-a"]));
    println!("  host_load: {}", command_output("uptime", &[]));
    #[cfg(target_os = "macos")]
    {
        println!(
            "  macos_version: {}",
            command_output("sw_vers", &["-productVersion"])
        );
        println!(
            "  mac_model: {}",
            command_output("sysctl", &["-n", "hw.model"])
        );
        println!(
            "  cpu_brand: {}",
            command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
        );
    }
    println!(
        "  rayon_threads: {} (RAYON_NUM_THREADS={})",
        rayon::current_num_threads(),
        env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_owned())
    );
    println!(
        "  sampling: warmup_rounds={}, sample_rounds={}, iterations_per_sample={} (balanced forward/reverse order)",
        config.warmup_rounds, config.sample_rounds, config.iterations
    );
}

fn env_count(name: &str, default: usize, min: usize, max: usize) -> Result<usize, String> {
    match env::var(name) {
        Ok(raw) => parse_bounded_count(name, Some(&raw), default, min, max),
        Err(env::VarError::NotPresent) => parse_bounded_count(name, None, default, min, max),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} is not valid Unicode")),
    }
}

fn validate_samples(samples: &[f64]) -> Result<(), String> {
    if samples.is_empty() {
        return Err("cannot summarize an empty sample set".to_owned());
    }
    if let Some((index, value)) = samples
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value <= 0.0)
    {
        return Err(format!(
            "timing sample {index} must be finite and positive, got {value}"
        ));
    }
    Ok(())
}

fn require_even_sample_rounds(sample_rounds: usize) -> Result<(), String> {
    if sample_rounds % 2 == 0 {
        Ok(())
    } else {
        Err(format!(
            "SPARSL_BENCH_ROUNDS must be even so forward/reverse order is balanced, got {sample_rounds}"
        ))
    }
}

fn parse_bounded_count(
    name: &str,
    raw: Option<&str>,
    default: usize,
    min: usize,
    max: usize,
) -> Result<usize, String> {
    let value = match raw {
        Some(raw) => raw
            .trim()
            .parse::<usize>()
            .map_err(|_| format!("{name} must be an integer, got {raw:?}"))?,
        None => default,
    };
    if !(min..=max).contains(&value) {
        return Err(format!(
            "{name} must be in the inclusive range {min}..={max}, got {value}"
        ));
    }
    Ok(value)
}

fn command_output(program: &str, args: &[&str]) -> String {
    match Command::new(program).args(args).output() {
        Ok(output) if output.status.success() => normalize_command_output(&output.stdout),
        _ => "unavailable".to_owned(),
    }
}

fn git_output(args: &[&str]) -> String {
    match Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
    {
        Ok(output) if output.status.success() => normalize_command_output(&output.stdout),
        _ => "unavailable".to_owned(),
    }
}

fn normalize_command_output(stdout: &[u8]) -> String {
    let value = String::from_utf8_lossy(stdout);
    let one_line = value
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" | ");
    if one_line.is_empty() {
        "unknown".to_owned()
    } else {
        one_line
    }
}

#[cfg(test)]
mod tests {
    use super::{
        balanced_arm_order, max_abs_finite, max_abs_product, max_row_nnz, paired_ratio_stats,
        parse_bounded_count, reference_spmv, require_even_sample_rounds, time_zeroed_output,
        verify_float_output, SampleStats,
    };
    use sparsl::Csr;

    #[test]
    fn median_is_order_independent_for_odd_and_even_sets() {
        let odd = SampleStats::from_samples(&[9.0, 1.0, 4.0]).expect("odd samples");
        assert_eq!((odd.min, odd.median, odd.max), (1.0, 4.0, 9.0));
        assert_eq!(odd.spread(), 9.0);

        let even = SampleStats::from_samples(&[8.0, 2.0, 4.0, 6.0]).expect("even samples");
        assert_eq!((even.min, even.median, even.max), (2.0, 5.0, 8.0));
        assert_eq!(even.spread(), 4.0);

        let large = SampleStats::from_samples(&[f64::MAX, f64::MAX]).expect("large samples");
        assert_eq!(large.median, f64::MAX);
    }

    #[test]
    fn sample_summary_rejects_vacuous_or_poisoned_evidence() {
        assert!(SampleStats::from_samples(&[]).is_err());
        assert!(SampleStats::from_samples(&[0.0]).is_err());
        assert!(SampleStats::from_samples(&[f64::NAN]).is_err());
        assert!(SampleStats::from_samples(&[f64::INFINITY]).is_err());
    }

    #[test]
    fn paired_ratios_preserve_round_pairing() {
        let stats = paired_ratio_stats(&[10.0, 40.0, 30.0, 20.0], &[5.0, 10.0, 10.0, 5.0])
            .expect("paired ratios");
        assert_eq!((stats.min, stats.median, stats.max), (2.0, 3.5, 4.0));
        assert!(paired_ratio_stats(&[1.0], &[1.0, 2.0]).is_err());
        assert!(paired_ratio_stats(&[-1.0], &[-1.0]).is_err());
    }

    #[test]
    fn bounded_count_parser_applies_defaults_and_rejects_bad_values() {
        assert_eq!(parse_bounded_count("COUNT", None, 8, 4, 32), Ok(8));
        assert_eq!(parse_bounded_count("COUNT", Some(" 12 "), 8, 4, 32), Ok(12));
        assert!(parse_bounded_count("COUNT", Some("3"), 8, 4, 32).is_err());
        assert!(parse_bounded_count("COUNT", Some("33"), 8, 4, 32).is_err());
        assert!(parse_bounded_count("COUNT", Some("many"), 8, 4, 32).is_err());
        assert!(require_even_sample_rounds(8).is_ok());
        assert!(require_even_sample_rounds(7).is_err());
    }

    #[test]
    fn output_comparison_rejects_vacuous_or_unwritten_results() {
        verify_float_output("exact", &[1.0, -0.0], &[1.0, -0.0], 0.0)
            .expect("bit-identical finite output");
        verify_float_output("bounded", &[1.01], &[1.0], 0.02)
            .expect("finite output inside the bound");

        assert!(verify_float_output("length", &[], &[1.0], 0.0).is_err());
        assert!(verify_float_output("missing write", &[0.0], &[1.0], 0.0).is_err());
        assert!(verify_float_output("signed zero", &[0.0], &[-0.0], 0.0).is_err());
        assert!(verify_float_output("nan", &[f32::NAN], &[f32::NAN], 1.0).is_err());
        assert!(verify_float_output("infinite", &[f32::INFINITY], &[f32::INFINITY], 1.0).is_err());
        assert!(verify_float_output("bad bound", &[1.0], &[1.0], f32::INFINITY).is_err());
    }

    #[test]
    fn scale_helpers_reject_non_finite_or_overflowed_evidence() {
        assert_eq!(max_abs_finite("values", &[-2.0, 1.0]), Ok(2.0));
        assert_eq!(max_abs_product(&[-2.0], &[3.0]), Ok(6.0));
        assert!(max_abs_finite("values", &[f32::NAN]).is_err());
        assert!(max_abs_product(&[f32::MAX], &[2.0]).is_err());
    }

    #[test]
    fn scalar_oracle_rejects_bad_shapes_and_accumulates_every_edge() {
        let csr = Csr::from_adjacency(&[vec![0, 1], vec![1]]);
        assert_eq!(max_row_nnz(&csr), 2);
        assert_eq!(
            reference_spmv(&csr, &[2.0, 3.0, 4.0], &[5.0, 6.0], &[0.5, -0.5]),
            Ok(vec![28.5, 23.5])
        );
        assert!(reference_spmv(&csr, &[2.0], &[5.0, 6.0], &[0.0, 0.0]).is_err());
        assert!(reference_spmv(&csr, &[2.0, 3.0, 4.0], &[5.0], &[0.0, 0.0]).is_err());
        assert!(reference_spmv(&csr, &[2.0, 3.0, 4.0], &[5.0, 6.0], &[0.0]).is_err());
    }

    /// The oracle must associate a row exactly as the backends do.
    ///
    /// `verify_spmv_preflight` holds CPU arms to *bit* identity, so an oracle
    /// that reassociates is indistinguishable from a backend defect -- and on a
    /// GPU arm it quietly eats the cross-backend tolerance instead. The values
    /// above are small exact integers that round identically either way, so
    /// they cannot see this; these are chosen so the two orders genuinely
    /// differ, and the assertion below proves that before testing identity.
    #[test]
    fn scalar_oracle_associates_rows_exactly_as_the_cpu_backend_does() {
        const NNZ: usize = 64;
        let mut state = 0x5713_2026u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32) / 8_388_608.0 - 0.5
        };
        let csr = Csr::from_adjacency(&[(0..NNZ as u32).collect()]);
        let weights: Vec<f32> = (0..NNZ).map(|_| next()).collect();
        let input: Vec<f32> = (0..NNZ).map(|_| next()).collect();
        let initial = vec![0.25f32];

        // Guard: this case is only meaningful while the two associations differ.
        let mut seeded = initial[0];
        for (&w, &x) in weights.iter().zip(&input) {
            seeded += w * x;
        }
        let mut folded = 0.0f32;
        for (&w, &x) in weights.iter().zip(&input) {
            folded += w * x;
        }
        let folded = initial[0] + folded;
        assert_ne!(
            seeded.to_bits(),
            folded.to_bits(),
            "these operands no longer distinguish the two associations, so this \
             test would pass against an oracle that reassociates"
        );

        let expected = reference_spmv(&csr, &weights, &input, &initial).expect("oracle");
        let op = sparsl::Device::cpu_sequential()
            .prepare(&csr, NNZ, &weights)
            .expect("single-row operator");
        let mut got = initial.clone();
        op.spmv(&input, &mut got).expect("cpu sequential spmv");

        assert_eq!(
            expected[0].to_bits(),
            got[0].to_bits(),
            "oracle produced {:?} (0x{:08X}) where the CPU arm produced {:?} \
             (0x{:08X}); the preflight compares these bit for bit",
            expected[0],
            expected[0].to_bits(),
            got[0],
            got[0].to_bits()
        );
    }

    #[test]
    fn timed_samples_are_balanced_and_do_not_accumulate_previous_dispatches() {
        assert_eq!(balanced_arm_order(0, 3).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(balanced_arm_order(1, 3).collect::<Vec<_>>(), [2, 1, 0]);
        assert!(balanced_arm_order(0, 0).next().is_none());

        let csr = Csr::from_adjacency(&[vec![0], vec![1]]);
        let op = sparsl::Device::cpu_sequential()
            .prepare(&csr, 2, &[2.0, -3.0])
            .expect("identity-shaped operator");
        let mut output = vec![99.0, -99.0];
        let elapsed = time_zeroed_output(&mut output, 3, |output| op.spmv(&[4.0, 5.0], output))
            .expect("repeated products");
        assert!(elapsed.is_finite());
        assert_eq!(output, vec![8.0, -15.0]);
    }
}
