//! Binary16 weight storage, across backends.
//!
//! Two distinct claims, easy to conflate:
//!
//! * **The host encoder agrees with Metal.** `Device::prepare_f16` narrows on
//!   the host and uploads raw `u16`; the kernel declares that same memory as
//!   `half`. If the two spellings disagreed the kernel would silently read
//!   different weights, so this is checked rather than assumed.
//! * **The quantisation error is bounded by what the crate promises.**
//!   `tolerance_for_spmv_f16` claims to bound an f16-stored operator against
//!   the answer the unquantised weights would have given.

mod common;

use common::{max_abs, max_abs_term, random_csr, random_vec};
use sparsl::half::{f16_bits_to_f32, f32_to_f16_bits, HALF_EPSILON, HALF_MAX};
use sparsl::{
    available_backends, tolerance_for_spmv, tolerance_for_spmv_f16, Backend, Device, Rng,
};

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect()
}

/// Round-trip through binary16, which is what `prepare_f16` stores.
fn quantise(w: &[f32]) -> Vec<f32> {
    w.iter()
        .map(|v| f16_bits_to_f32(f32_to_f16_bits(*v)))
        .collect()
}

#[test]
fn the_host_encoder_agrees_with_metals_half() {
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
    let op = gpu.prepare_f16(&csr, n, &weights).expect("prepare_f16");
    assert!(op.weights_are_f16(), "Metal operator should store binary16");

    let x = vec![1.0f32; n];
    let mut y = vec![0.0f32; n];
    op.spmv(&x, &mut y).expect("spmv");

    let want = quantise(&weights);
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

#[test]
fn f16_storage_lands_within_the_derived_bound_of_the_exact_answer() {
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

        let bound = tolerance_for_spmv_f16(
            ref_op.shape().max_row_nnz().max(1),
            max_abs_term(&weights, &x),
            max_abs(&want),
        );

        for device in devices() {
            let op = device
                .prepare_f16(&csr, ncols, &weights)
                .expect("prepare_f16");
            let mut got = vec![0.0f32; nrows];
            op.spmv(&x, &mut got).expect("spmv");
            for r in 0..nrows {
                assert!(
                    (got[r] - want[r]).abs() <= bound,
                    "{}: y[{r}] = {} against exact {} (bound {bound})",
                    op.label(),
                    got[r],
                    want[r]
                );
            }
        }
    }
}

#[test]
fn every_backend_agrees_with_every_other_on_the_same_quantised_weights() {
    // Both arms store the values `prepare_f16` produced, so this comparison is
    // still an f32 one — the binary16 error is inside the operator, not between
    // the backends, and the f32 bound is what must hold.
    let mut rng = Rng::new(0xA6BEE);
    let (nrows, ncols, deg) = (192usize, 96usize, 12usize);
    let csr = random_csr(nrows, ncols, deg, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let x = random_vec(ncols, 1.0, &mut rng);
    let quantised = quantise(&weights);

    let mut reference: Option<(String, Vec<f32>)> = None;
    for device in devices() {
        let op = device
            .prepare_f16(&csr, ncols, &weights)
            .expect("prepare_f16");
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
                        "{} vs {label}: y[{r}] = {} against {} (bound {bound})",
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

    for device in devices() {
        let mut op = device.prepare_f16(&csr, ncols, &w0).expect("prepare_f16");
        let mut before = vec![0.0f32; nrows];
        op.spmv(&x, &mut before).expect("spmv");

        op.set_weights(&w1).expect("set_weights");
        let mut after = vec![0.0f32; nrows];
        op.spmv(&x, &mut after).expect("spmv");

        let fresh = device.prepare_f16(&csr, ncols, &w1).expect("prepare_f16");
        let mut want = vec![0.0f32; nrows];
        fresh.spmv(&x, &mut want).expect("spmv");

        for r in 0..nrows {
            assert_eq!(
                after[r].to_bits(),
                want[r].to_bits(),
                "{}: set_weights did not reach the narrow copy at row {r}",
                op.label()
            );
        }
        assert!(
            before.iter().zip(&after).any(|(b, a)| b != a),
            "{}: new weights gave an identical result — the check proves nothing",
            op.label()
        );
    }
}

#[test]
fn out_of_range_weights_overflow_rather_than_saturating() {
    // Saturating would turn an overflow into a plausible large number. The
    // caller should see the infinity.
    assert_eq!(f32_to_f16_bits(70_000.0), 0x7c00);
    assert!(f16_bits_to_f32(f32_to_f16_bits(70_000.0)).is_infinite());
    // And the documented epsilon is the ratio the tolerance is built on.
    assert_eq!(HALF_EPSILON / f32::EPSILON, 8192.0);
}
