//! Regression tests for the shared cross-backend float comparator.

mod common;

use sparsl::LifParams;

#[test]
fn assert_close_rejects_a_one_sided_nan() {
    let result = std::panic::catch_unwind(|| {
        common::assert_close(&[f32::NAN], &[0.0], 0.0, "one-sided NaN")
    });
    assert!(
        result.is_err(),
        "a one-sided NaN was silently accepted as a zero-error comparison"
    );
}

#[test]
fn assert_close_rejects_different_infinities() {
    let result = std::panic::catch_unwind(|| {
        common::assert_close(
            &[f32::INFINITY],
            &[f32::NEG_INFINITY],
            0.0,
            "opposite infinities",
        )
    });
    assert!(
        result.is_err(),
        "opposite infinities were silently accepted as a zero-error comparison"
    );
}

#[test]
fn assert_close_rejects_an_infinite_tolerance() {
    let result = std::panic::catch_unwind(|| {
        common::assert_close(&[1.0], &[2.0], f32::INFINITY, "infinite tolerance")
    });
    assert!(
        result.is_err(),
        "an infinite tolerance made a real mismatch vacuously pass"
    );
}

#[test]
fn assert_close_accepts_matching_non_finite_classes() {
    common::assert_close(
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        &[f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
        0.0,
        "matching non-finite values",
    );
}

#[test]
fn numeric_class_comparison_rejects_an_infinity_sign_flip() {
    let result = std::panic::catch_unwind(|| {
        common::assert_same_numeric_class(f32::NEG_INFINITY, f32::INFINITY, "sign-flipped infinity")
    });
    assert!(
        result.is_err(),
        "checking only finiteness and NaN-ness accepted +inf as -inf"
    );

    common::assert_same_numeric_class(f32::NAN, f32::NAN, "matching NaN");
    common::assert_same_numeric_class(1.0, 2.0, "finite classification only");
}

#[derive(Clone)]
struct LifInputs {
    v_got: Vec<f32>,
    theta_got: Vec<f32>,
    spikes_got: Vec<bool>,
    v_want: Vec<f32>,
    theta_want: Vec<f32>,
    spikes_want: Vec<bool>,
    v_pre: Vec<f32>,
    theta_pre: Vec<f32>,
    current_ref: Vec<f32>,
}

impl LifInputs {
    fn valid() -> Self {
        Self {
            v_got: vec![0.0],
            theta_got: vec![1.0],
            spikes_got: vec![false],
            v_want: vec![0.0],
            theta_want: vec![1.0],
            spikes_want: vec![false],
            v_pre: vec![0.0],
            theta_pre: vec![1.0],
            current_ref: vec![0.0],
        }
    }

    fn compare(&self) {
        common::compare_lif(
            &self.v_got,
            &self.theta_got,
            &self.spikes_got,
            &self.v_want,
            &self.theta_want,
            &self.spikes_want,
            &self.v_pre,
            &self.theta_pre,
            &self.current_ref,
            LifParams::new(0.9, 0.0, 0.1).expect("valid parameters"),
            0.0,
            "length guard",
        );
    }
}

type LifMutation = (&'static str, fn(&mut LifInputs));

#[test]
fn compare_lif_rejects_every_slice_length_mismatch() {
    let mutations: [LifMutation; 8] = [
        ("v_got", |x| x.v_got.push(0.0)),
        ("theta_got", |x| x.theta_got.push(1.0)),
        ("spikes_got", |x| x.spikes_got.push(false)),
        ("theta_want", |x| x.theta_want.push(1.0)),
        ("spikes_want", |x| x.spikes_want.push(false)),
        ("v_pre", |x| x.v_pre.push(0.0)),
        ("theta_pre", |x| x.theta_pre.push(1.0)),
        ("current_ref", |x| x.current_ref.push(0.0)),
    ];

    for (name, mutate) in mutations {
        let mut inputs = LifInputs::valid();
        mutate(&mut inputs);
        let result = std::panic::catch_unwind(|| inputs.compare());
        assert!(
            result.is_err(),
            "{name} had a trailing element that compare_lif silently ignored"
        );
    }
}
