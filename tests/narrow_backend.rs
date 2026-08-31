//! Narrow weight storage — binary16 and bfloat16 — across backends.
//!
//! Two distinct claims, easy to conflate:
//!
//! * **The host encoders agree with Metal.** The host narrows and uploads raw
//!   `u16`; the kernels declare that same memory as `half` or `bfloat`. If the
//!   two spellings disagreed the kernel would silently read different weights,
//!   so this is checked rather than assumed.
//! * **The quantisation error is bounded by what the crate promises.**
//!   `tolerance_for_spmv_narrow` claims to bound a narrow operator against the
//!   answer the unquantised weights would have given.
//!
//! Every case runs against both formats. They differ only in their epsilon and
//! their range, so a test that covered one would leave the other's kernel and
//! encoder unexercised.

mod common;

use common::{max_abs, max_abs_term, random_csr, random_vec};
use sparsl::half::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, BF16_EPSILON,
    HALF_EPSILON, HALF_MAX,
};
use sparsl::{
    available_backends, tolerance_for_spmv, tolerance_for_spmv_narrow, Backend, Device, Rng,
    WeightPrecision,
};

/// The two narrow formats. Every test below runs against both.
const NARROW: [WeightPrecision; 2] = [WeightPrecision::F16, WeightPrecision::Bf16];

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect()
}

/// Round-trip through a narrow format, which is what `prepare_with` stores.
fn quantise(p: WeightPrecision, w: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|v| match p {
            WeightPrecision::F16 => f16_bits_to_f32(f32_to_f16_bits(*v)),
            WeightPrecision::Bf16 => bf16_bits_to_f32(f32_to_bf16_bits(*v)),
            WeightPrecision::F32 => *v,
        })
        .collect()
}

#[test]
fn the_host_encoders_agree_with_metals_half_and_bfloat() {
    // The load-bearing assumption of the whole path: host-encoded `u16` bits,
    // read back through the GPU's `half`, must be the values the host meant.
    //
    // Driven through the real SpMV rather than a bespoke probe kernel: an
    // identity operator with one weight per row makes `y[i]` exactly the
    // decoded weight, so any disagreement in the encoding shows up directly.
    let Ok(gpu) = Device::try_new(Backend::Metal) else {
        eprintln!("no Metal device; nothing to cross-check");
        return;
    };
    let n = 4096usize;
    let mut rng = Rng::new(0x11A1F);
    // Span the format: normals, subnormals, and values near its ceiling.
    let weights: Vec<f32> = (0..n)
        .map(|i| match i % 4 {
            0 => rng.next_f32() * 2.0 - 1.0,
            1 => (rng.next_f32() - 0.5) * 1e-4,
            2 => (rng.next_f32() - 0.5) * HALF_MAX,
            _ => (rng.next_f32() - 0.5) * 1e-7,
        })
        .collect();

    let adj: Vec<Vec<u32>> = (0..n).map(|i| vec![i as u32]).collect();
    let csr = sparsl::Csr::from_adjacency(&adj);

    for precision in NARROW {
        let op = gpu
            .prepare_with(&csr, n, &weights, precision)
            .expect("prepare_with");
        assert_eq!(
            op.weight_precision(),
            precision,
            "Metal operator should store {precision:?}"
        );

        let x = vec![1.0f32; n];
        let mut y = vec![0.0f32; n];
        op.spmv(&x, &mut y).expect("spmv");

        let want = quantise(precision, &weights);
        for i in 0..n {
            // `0.0 +` because the kernel accumulates into `y`, and IEEE addition
            // normalises -0.0 to +0.0. Comparing raw bits without it reports a
            // signed-zero difference the encoder did not cause.
            assert_eq!(
                y[i].to_bits(),
                (0.0f32 + want[i]).to_bits(),
                "weight {i}: host encoded {:?} -> {}, GPU read it as {}",
                weights[i],
                want[i],
                y[i]
            );
        }
    }
}

#[test]
fn narrow_storage_lands_within_the_derived_bound_of_the_exact_answer() {
    let mut rng = Rng::new(0xF16B);
    for &(nrows, ncols, deg) in &[(64usize, 64usize, 8usize), (256, 128, 24), (129, 63, 5)] {
        let csr = random_csr(nrows, ncols, deg, &mut rng);
        let weights = random_vec(csr.nnz(), 1.0, &mut rng);
        let x = random_vec(ncols, 1.0, &mut rng);

        // Reference: the answer the *unquantised* weights give, on CPU.
        let exact = Device::try_new(Backend::CpuSequential).expect("cpu");
        let ref_op = exact.prepare(&csr, ncols, &weights).expect("prepare");
        let mut want = vec![0.0f32; nrows];
        ref_op.spmv(&x, &mut want).expect("spmv");

        for precision in NARROW {
            let bound = tolerance_for_spmv_narrow(
                precision,
                ref_op.shape().max_row_nnz().max(1),
                max_abs_term(&weights, &x),
                max_abs(&want),
            );
            for device in devices() {
                let op = device
                    .prepare_with(&csr, ncols, &weights, precision)
                    .expect("prepare_with");
                let mut got = vec![0.0f32; nrows];
                op.spmv(&x, &mut got).expect("spmv");
                for r in 0..nrows {
                    assert!(
                        (got[r] - want[r]).abs() <= bound,
                        "{} {precision:?}: y[{r}] = {} against exact {} (bound {bound})",
                        op.label(),
                        got[r],
                        want[r]
                    );
                }
            }
        }
    }
}

#[test]
fn every_backend_agrees_with_every_other_on_the_same_quantised_weights() {
    // Both arms store the values `prepare_with` produced, so this comparison is
    // still an f32 one — the narrowing error is inside the operator, not between
    // the backends, and the f32 bound is what must hold.
    let mut rng = Rng::new(0xA6BEE);
    let (nrows, ncols, deg) = (192usize, 96usize, 12usize);
    let csr = random_csr(nrows, ncols, deg, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(ncols, 1.0, &mut rng);

    for precision in NARROW {
        let quantised = quantise(precision, &weights);
        let mut reference: Option<(String, Vec<f32>)> = None;
        for device in devices() {
            let op = device
                .prepare_with(&csr, ncols, &weights, precision)
                .expect("prepare_with");
            let mut got = vec![0.0f32; nrows];
            op.spmv(&x, &mut got).expect("spmv");

            let bound = tolerance_for_spmv(
                op.shape().max_row_nnz().max(1),
                max_abs_term(&quantised, &x),
                max_abs(&got),
            );
            match &reference {
                None => reference = Some((op.label().to_string(), got)),
                Some((label, want)) => {
                    for r in 0..nrows {
                        assert!(
                            (got[r] - want[r]).abs() <= bound,
                            "{} vs {label} ({precision:?}): y[{r}] = {} against {} (bound {bound})",
                            op.label(),
                            got[r],
                            want[r]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn set_weights_reaches_the_narrow_copy_too() {
    // Two representations exist on the GPU. If `set_weights` refreshed only the
    // wide one, the kernel would keep dispatching the weights the operator was
    // built with while the operator reported the new ones.
    let mut rng = Rng::new(0x5E7A);
    let (nrows, ncols) = (64usize, 48usize);
    let csr = random_csr(nrows, ncols, 6, &mut rng);
    let w0 = random_vec(csr.nnz(), 1.0, &mut rng);
    let w1 = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(ncols, 1.0, &mut rng);

    for precision in NARROW {
        for device in devices() {
            let mut op = device
                .prepare_with(&csr, ncols, &w0, precision)
                .expect("prepare_with");
            let mut before = vec![0.0f32; nrows];
            op.spmv(&x, &mut before).expect("spmv");

            op.set_weights(&w1).expect("set_weights");
            let mut after = vec![0.0f32; nrows];
            op.spmv(&x, &mut after).expect("spmv");

            let fresh = device
                .prepare_with(&csr, ncols, &w1, precision)
                .expect("prepare_with");
            let mut want = vec![0.0f32; nrows];
            fresh.spmv(&x, &mut want).expect("spmv");

            for r in 0..nrows {
                assert_eq!(
                    after[r].to_bits(),
                    want[r].to_bits(),
                    "{} {precision:?}: set_weights did not reach the narrow copy at row {r}",
                    op.label()
                );
            }
            assert!(
                before.iter().zip(&after).any(|(b, a)| b != a),
                "{} {precision:?}: new weights gave an identical result — the check proves nothing",
                op.label()
            );
        }
    }
}

#[test]
fn the_two_formats_trade_range_against_precision() {
    // Not interchangeable, and the numbers say which way. binary16 is 8x finer;
    // bfloat16 reaches 5 orders of magnitude further before overflowing.
    assert_eq!(BF16_EPSILON / HALF_EPSILON, 8.0);
    assert_eq!(HALF_EPSILON / f32::EPSILON, 8192.0);

    // A weight past binary16's ceiling survives bfloat16 unharmed.
    let big = HALF_MAX * 4.0;
    assert!(f16_bits_to_f32(f32_to_f16_bits(big)).is_infinite());
    assert!(bf16_bits_to_f32(f32_to_bf16_bits(big)).is_finite());

    // And the bound follows the epsilon, so a caller sees the trade priced.
    let f16 = tolerance_for_spmv_narrow(WeightPrecision::F16, 100, 1.0, 1.0);
    let bf16 = tolerance_for_spmv_narrow(WeightPrecision::Bf16, 100, 1.0, 1.0);
    assert!(bf16 > f16 * 7.0, "bf16 bound {bf16} should be ~8x {f16}");
    // f32 storage quantises nothing, so its bound is the accumulation alone.
    assert_eq!(
        tolerance_for_spmv_narrow(WeightPrecision::F32, 100, 1.0, 1.0),
        tolerance_for_spmv(100, 1.0, 1.0)
    );
}
