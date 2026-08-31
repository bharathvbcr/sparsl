//! SIMD cell math (U02).
//!
//! Elementwise leak/integrate over SoA columns. The hot path is structured as
//! fixed-width lanes (`LANES = 8`) so LLVM can autovectorize; a scalar tail
//! handles the remainder. No `unsafe`, no extra ML deps.

use crate::time::Tick;

/// Lane width for the SIMD-structured leak/integrate kernel.
///
/// `8 × f32` is **not** a 256-bit vector on the host of record (Apple M5 Pro,
/// aarch64). NEON registers are 128-bit, so each `LANES` chunk lowers to two
/// `f32x4` vector ops rather than one. That is still the right width — the pair
/// gives LLVM a 2× unrolled body that hides `fdiv`/`fmla` latency — but the
/// "256-bit" reading of this constant is an x86/AVX assumption and does not
/// describe the generated code here.
///
/// Before changing this value, re-run `cargo bench -p binn-core
/// --bench simd_leak_integrate`; the correct width is an empirical question,
/// not a derivation from the register file.
pub const LANES: usize = 8;

/// One Euler step of the linear sub-threshold LIF dynamics
/// `τ dv/dt = −v + input`, i.e.
///
/// ```text
/// v ← v + (input − v) · (dt / τ)
/// ```
///
/// All slices must have the same length. `tau` entries must be finite and
/// non-zero. `dt` is the integer tick step, converted to `f32` for the update.
///
/// The implementation processes `LANES`-wide chunks (SIMD-shaped) and a
/// scalar remainder; results match [`scalar_leak_integrate`] within `1e-6`
/// on normal inputs.
pub fn simd_leak_integrate(v: &mut [f32], input: &[f32], tau: &[f32], dt: Tick) {
    let n = v.len();
    assert_eq!(input.len(), n, "input length must match v");
    assert_eq!(tau.len(), n, "tau length must match v");

    let dt = dt as f32;
    let mut i = 0;

    // SIMD-structured body: fixed lane groups for autovectorization.
    while i + LANES <= n {
        leak_integrate_lanes(
            (&mut v[i..i + LANES]).try_into().unwrap(),
            (&input[i..i + LANES]).try_into().unwrap(),
            (&tau[i..i + LANES]).try_into().unwrap(),
            dt,
        );
        i += LANES;
    }

    // Scalar tail.
    while i < n {
        leak_integrate_one(&mut v[i], input[i], tau[i], dt);
        i += 1;
    }
}

/// Scalar reference implementation of the same update as [`simd_leak_integrate`].
///
/// Provided for parity testing and as a readable specification of the dynamics.
#[inline]
pub fn scalar_leak_integrate(v: &mut [f32], input: &[f32], tau: &[f32], dt: Tick) {
    let n = v.len();
    assert_eq!(input.len(), n, "input length must match v");
    assert_eq!(tau.len(), n, "tau length must match v");
    let dt = dt as f32;
    for i in 0..n {
        leak_integrate_one(&mut v[i], input[i], tau[i], dt);
    }
}

#[inline(always)]
fn leak_integrate_one(v: &mut f32, input: f32, tau: f32, dt: f32) {
    let alpha = dt / tau;
    *v += (input - *v) * alpha;
}

/// One `LANES`-wide step. Written as an explicit lane loop so the intent is
/// SIMD-shaped even on targets where autovectorization is inactive.
#[inline(always)]
fn leak_integrate_lanes(v: &mut [f32; LANES], input: &[f32; LANES], tau: &[f32; LANES], dt: f32) {
    // Lane-parallel body (autovectorization target).
    let mut alpha = [0.0f32; LANES];
    let mut i = 0;
    while i < LANES {
        alpha[i] = dt / tau[i];
        i += 1;
    }
    i = 0;
    while i < LANES {
        v[i] += (input[i] - v[i]) * alpha[i];
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::{scalar_leak_integrate, simd_leak_integrate, LANES};
    use crate::rng::Rng;
    use crate::time::Tick;

    const ATOL: f32 = 1e-6;

    fn assert_close(a: &[f32], b: &[f32], atol: f32) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let err = (x - y).abs();
            assert!(
                err <= atol,
                "mismatch at {i}: simd={x} scalar={y} err={err} atol={atol}"
            );
        }
    }

    fn random_inputs(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Tick) {
        let mut rng = Rng::new(seed);
        let v: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let input: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        // tau in (0.5, 4.5] — safely away from zero.
        let tau: Vec<f32> = (0..n).map(|_| 0.5 + rng.next_f32() * 4.0).collect();
        let dt = (1 + rng.gen_index(4)) as Tick;
        (v, input, tau, dt)
    }

    #[test]
    fn simd_matches_scalar_random() {
        for &n in &[0, 1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 1024, 1024 + 3] {
            let (v0, input, tau, dt) = random_inputs(n, 0x51_4D_44_00 + n as u64);
            let mut v_simd = v0.clone();
            let mut v_scalar = v0;
            simd_leak_integrate(&mut v_simd, &input, &tau, dt);
            scalar_leak_integrate(&mut v_scalar, &input, &tau, dt);
            assert_close(&v_simd, &v_scalar, ATOL);
        }
    }

    #[test]
    fn simd_matches_scalar_many_seeds() {
        for seed in 0..32u64 {
            let n = 257;
            let (v0, input, tau, dt) = random_inputs(n, seed ^ 0xB177_C0DE);
            let mut v_simd = v0.clone();
            let mut v_scalar = v0;
            simd_leak_integrate(&mut v_simd, &input, &tau, dt);
            scalar_leak_integrate(&mut v_scalar, &input, &tau, dt);
            assert_close(&v_simd, &v_scalar, ATOL);
        }
    }

    #[test]
    fn lanes_constant_is_eight() {
        assert_eq!(LANES, 8);
    }

    #[test]
    #[should_panic(expected = "input length must match v")]
    fn rejects_input_len_mismatch() {
        let mut v = [0.0f32; 4];
        simd_leak_integrate(&mut v, &[0.0; 3], &[1.0; 4], 1);
    }
}
