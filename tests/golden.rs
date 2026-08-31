//! Pins the numeric output of the CPU reference.
//!
//! Every other suite here is *relative*: backends are compared against the CPU
//! reference, so a change that moves the reference moves everything with it and
//! nothing fails. The mutation campaign demonstrated this — summing each row in
//! reverse changed every number this crate produces and not one test noticed,
//! because reverse summation is a legal summation order and both CPU arms
//! adopted it together.
//!
//! For a crate whose stated property is bit-reproducibility, that is the gap
//! that matters most: a caller replaying a result by config hash needs the
//! reference itself to be nailed down, not merely self-consistent.
//!
//! So this fingerprints the actual bits. Any change to summation order,
//! iteration order, the RNG, the CSR layout, or the LIF update changes the
//! hash. If it fails, that is not automatically a bug — but it is always a
//! deliberate decision, and any downstream replay hash has just been invalidated.

mod common;

use common::*;
use sparsl::{Device, Rng};

fn fnv1a(hash: &mut u64, bits: u32) {
    *hash ^= bits as u64;
    *hash = hash.wrapping_mul(0x100_0000_01b3);
}

/// Fingerprint of the sequential CPU backend over a fixed workload.
fn reference_fingerprint() -> u64 {
    let device = Device::cpu_sequential();
    let params = default_params();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut rng = Rng::new(0x0060_1DE1);

    for &(nrows, ncols, max_deg) in SHAPES {
        let csr = random_csr(nrows, ncols, max_deg, &mut rng);
        let weights = random_vec(csr.nnz(), 1.0, &mut rng);
        let x = random_vec(ncols, 1.0, &mut rng);
        let op = device.prepare(&csr, ncols, &weights).expect("valid");

        let mut y = random_vec(nrows, 1.0, &mut rng);
        op.spmv(&x, &mut y).expect("spmv");
        for v in &y {
            fnv1a(&mut hash, v.to_bits());
        }

        let mut v = random_vec(nrows, 0.5, &mut rng);
        let mut theta: Vec<f32> = random_vec(nrows, 0.5, &mut rng)
            .iter()
            .map(|t| t.abs())
            .collect();
        let mut spikes = vec![false; nrows];
        op.fused_spmv_lif(&x, &mut v, &mut theta, &mut spikes, params)
            .expect("fused");
        for (a, b) in v.iter().zip(theta.iter()) {
            fnv1a(&mut hash, a.to_bits());
            fnv1a(&mut hash, b.to_bits());
        }
        for s in &spikes {
            fnv1a(&mut hash, *s as u32);
        }
    }
    hash
}

/// Recorded on an Apple M5 Pro, aarch64-apple-darwin, `--release`.
///
/// This is a *value*, not a platform property: it depends only on IEEE-754 f32
/// arithmetic performed in a fixed order, so it must hold on any target with
/// standard floats. A mismatch on a different machine means either the
/// arithmetic order changed or that target is doing something non-standard —
/// both worth knowing before trusting a replay.
const GOLDEN_REFERENCE_FINGERPRINT: u64 = 0xE7D4_DE54_FE3C_C803;

#[test]
fn cpu_reference_output_is_pinned() {
    let got = reference_fingerprint();
    assert_eq!(
        got, GOLDEN_REFERENCE_FINGERPRINT,
        "\nThe CPU reference produces different bits than when this value was recorded.\n\
         Computed: 0x{got:016X}\n\
         Expected: 0x{GOLDEN_REFERENCE_FINGERPRINT:016X}\n\n\
         Something changed the arithmetic: summation order, iteration order, the RNG \
         stream, the CSR layout, or the LIF update. If the change was intentional, update \
         this constant — and treat every downstream config-hash replay as invalidated."
    );
}

/// The fingerprint must not depend on which CPU arm computed it.
#[test]
fn fingerprint_is_stable_across_repeats() {
    assert_eq!(reference_fingerprint(), reference_fingerprint());
}
