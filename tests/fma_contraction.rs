//! Pins the rounding behaviour of the Metal elementwise kernel.
//!
//! `tolerance_for_elementwise` is not a guess at how far apart two substrates
//! might drift; it is sized for one specific, identified cause. Metal contracts
//! `v * decay + current` into a single `fma`, rounding once where the CPU
//! rounds twice. `CompileOptions::set_fast_math_enabled(false)` does not
//! prevent it, and the property is deprecated in recent Metal anyway.
//!
//! This test proves the cause is that and nothing else: every non-spiking GPU
//! membrane must match one of the two roundings *bit for bit*. A single value
//! matching neither would mean the gap has some other source, and the tolerance
//! elsewhere in the suite would be covering up a real defect rather than a
//! known rounding choice.
//!
//! It deliberately does not assert *which* rounding wins. Contraction is a
//! compiler decision that may legitimately change between Metal versions; what
//! must not change is that the answer is one of these two.
mod common;
use common::*;
use sparsl::{Backend, Device, Rng};

#[test]
fn metal_membrane_matches_one_of_the_two_roundings() {
    if !Backend::Metal.is_available() {
        return;
    }
    let device = Device::try_new(Backend::Metal).expect("metal");
    let params = default_params();
    let mut rng = Rng::new(0x0FA0_9403);
    let n = 4096;
    let v0 = random_vec(n, 1.0, &mut rng);
    let theta0: Vec<f32> = random_vec(n, 0.5, &mut rng).iter().map(|t| t.abs()).collect();
    let currents = random_vec(n, 1.0, &mut rng);

    let (mut v_gpu, mut th_gpu, mut sp_gpu) = (v0.clone(), theta0.clone(), vec![false; n]);
    device
        .lif_integrate(&mut v_gpu, &mut th_gpu, &currents, &mut sp_gpu, params)
        .expect("lif");

    let mut separate_matches = 0usize;
    let mut fused_matches = 0usize;
    let mut neither = 0usize;
    for i in 0..n {
        // Only compare the non-spiking cells: a spiking cell writes v_reset and
        // carries no information about how the membrane was rounded.
        if sp_gpu[i] {
            continue;
        }
        let separate = v0[i] * params.decay() + currents[i];
        let fused = v0[i].mul_add(params.decay(), currents[i]);
        let got = v_gpu[i];
        if got.to_bits() == separate.to_bits() {
            separate_matches += 1;
        } else if got.to_bits() == fused.to_bits() {
            fused_matches += 1;
        } else {
            neither += 1;
        }
    }
    println!(
        "non-spiking cells: separate-rounding matches = {separate_matches}, \
         fused (fma) matches = {fused_matches}, neither = {neither}"
    );
    assert_eq!(
        neither, 0,
        "{neither} GPU membranes matched neither `v*decay + current` nor \
         `fma(v, decay, current)`. The CPU/GPU gap has a source other than \
         multiply-add contraction, and `tolerance_for_elementwise` is sized for \
         the wrong thing."
    );
    assert!(
        fused_matches > 0,
        "no membrane matched the fused rounding: contraction appears to have \
         stopped happening. That is not a failure in itself, but \
         `tolerance_for_elementwise` and its documentation now describe a \
         behaviour this compiler no longer has."
    );
}
