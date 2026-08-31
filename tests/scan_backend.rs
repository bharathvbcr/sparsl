//! `Device::assoc_scan` across substrates.
//!
//! This is the one primitive where the crate's arms deliberately disagree in
//! the last few ulps: the CPU arms are bit-identical to a sequential fold, the
//! Metal arm reassociates so it can be parallel. The tests therefore assert two
//! different things — byte-identity *within* a backend, and a derived tolerance
//! *across* them — which is exactly the rule the crate states everywhere else.

mod common;

use sparsl::{assoc_scan_sequential, available_backends, Backend, Device, Rng, State};

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect()
}

/// Leak steps with bounded `a`, which is what the affine scan is for: a
/// membrane decay is in `(0, 1)`, so prefix products shrink rather than grow.
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

#[test]
fn each_backend_is_byte_identical_to_itself() {
    // The property the crate actually promises. It must hold on the GPU arm
    // too, where the *values* differ from the CPU arm.
    for device in devices() {
        for &n in &[1usize, 33, 1024, 4097] {
            let xs = leak_steps(n, 0x5CA1 + n as u64);
            let a = device.assoc_scan(&xs).expect("scan");
            let b = device.assoc_scan(&xs).expect("scan");
            for i in 0..n {
                assert_eq!(
                    (a[i].a.to_bits(), a[i].b.to_bits()),
                    (b[i].a.to_bits(), b[i].b.to_bits()),
                    "{}: repeated scan differs at {i} (n={n})",
                    device.label()
                );
            }
        }
    }
}

#[test]
fn the_cpu_arms_stay_bit_identical_to_the_sequential_fold() {
    // The CPU arms make a stronger promise than the GPU one, and it must not
    // quietly weaken now that a reassociating arm exists beside them.
    for backend in [Backend::CpuSequential, Backend::CpuParallel] {
        let Ok(device) = Device::try_new(backend) else {
            continue;
        };
        for &n in &[1usize, 255, 256, 257, 4096] {
            let xs = leak_steps(n, 0xB177 + n as u64);
            let got = device.assoc_scan(&xs).expect("scan");
            let want = assoc_scan_sequential(&xs, |a, b| a.combine(b));
            for i in 0..n {
                assert_eq!(
                    (got[i].a.to_bits(), got[i].b.to_bits()),
                    (want[i].a.to_bits(), want[i].b.to_bits()),
                    "{}: differs from the sequential fold at {i} (n={n})",
                    device.label()
                );
            }
        }
    }
}

#[test]
fn every_backend_agrees_with_an_f64_reference_within_a_derived_bound() {
    for device in devices() {
        for &n in &[1usize, 63, 1025, 8192] {
            let xs = leak_steps(n, 0xF64 + n as u64);
            let got = device.assoc_scan(&xs).expect("scan");

            // f64 sequential reference, plus the magnitudes the bound needs.
            let (mut ra, mut rb) = (1.0f64, 0.0f64);
            let mut max_b = 0.0f64;
            let mut want = Vec::with_capacity(n);
            for s in &xs {
                let (na, nb) = (s.a as f64, s.b as f64);
                let (ca, cb) = (na * ra, na * rb + nb);
                ra = ca;
                rb = cb;
                max_b = max_b.max(cb.abs());
                want.push((ca, cb));
            }

            // Each prefix is a chain of at most `n` multiply-adds. Both arms
            // reassociate differently; the bound is the usual recursive one,
            // scaled by the largest intermediate the chain reaches.
            let bound = 8.0 * f64::from(f32::EPSILON) * n as f64 * max_b.max(1.0);
            for i in 0..n {
                assert!(
                    (got[i].a as f64 - want[i].0).abs() <= bound,
                    "{}: a[{i}] = {} want {} (n={n}, bound {bound})",
                    device.label(),
                    got[i].a,
                    want[i].0
                );
                assert!(
                    (got[i].b as f64 - want[i].1).abs() <= bound,
                    "{}: b[{i}] = {} want {} (n={n}, bound {bound})",
                    device.label(),
                    got[i].b,
                    want[i].1
                );
            }
        }
    }
}

#[test]
fn the_scan_respects_the_monoid_identity() {
    // Prepending the identity must not change any prefix. On the GPU arm the
    // identity is also what pads the tail of a partial threadgroup, so this
    // doubles as a check that the padding contributes nothing.
    for device in devices() {
        let xs = leak_steps(300, 0x1DEA);
        let plain = device.assoc_scan(&xs).expect("scan");
        let mut padded = vec![State::identity()];
        padded.extend_from_slice(&xs);
        let with_id = device.assoc_scan(&padded).expect("scan");
        for i in 0..xs.len() {
            assert_eq!(
                (plain[i].a.to_bits(), plain[i].b.to_bits()),
                (with_id[i + 1].a.to_bits(), with_id[i + 1].b.to_bits()),
                "{}: a leading identity changed prefix {i}",
                device.label()
            );
        }
    }
}

#[test]
fn an_empty_scan_is_empty_on_every_backend() {
    for device in devices() {
        assert!(
            device.assoc_scan(&[]).expect("scan").is_empty(),
            "{}",
            device.label()
        );
    }
}

#[test]
fn a_scan_spanning_many_threadgroups_is_still_correct() {
    // Past one threadgroup the result depends on the block-offset pass, which
    // is a separate kernel. A test that only ran short inputs would never
    // execute it.
    for device in devices() {
        let n = 100_000usize;
        let xs = leak_steps(n, 0xB16);
        let got = device.assoc_scan(&xs).expect("scan");
        let (mut ra, mut rb) = (1.0f64, 0.0f64);
        let mut max_b = 0.0f64;
        for (i, s) in xs.iter().enumerate() {
            let (na, nb) = (s.a as f64, s.b as f64);
            let (ca, cb) = (na * ra, na * rb + nb);
            ra = ca;
            rb = cb;
            max_b = max_b.max(cb.abs());
            // `a` decays geometrically here and underflows to zero long before
            // the end; `b` is the component that stays informative.
            let bound = 8.0 * f64::from(f32::EPSILON) * (i + 1) as f64 * max_b.max(1.0);
            assert!(
                (got[i].b as f64 - cb).abs() <= bound,
                "{}: b[{i}] = {} want {cb} (bound {bound})",
                device.label(),
                got[i].b
            );
        }
    }
}

#[test]
fn block_offsets_are_exact_on_a_counting_scan() {
    // The tolerance tests above cannot see an off-by-one in the block-offset
    // pass, and a mutation proved it: making that prefix inclusive instead of
    // exclusive left all six of them green. The reason is that a leak chain
    // contracts — `a` decays geometrically and early error is forgotten — so a
    // bound derived as `n * eps * max` is enormous at n = 100000 and hides
    // almost anything.
    //
    // With `a = 1` there is no contraction and no rounding: every step is
    // `b -> b + 1`, so `prefix[i].b` is exactly `i + 1`, an integer f32
    // represents exactly below 2^24. Any offset applied once too often or not
    // at all shows up as an integer that is plainly wrong, on every backend,
    // with no tolerance to hide in.
    for device in devices() {
        for &n in &[1usize, 1023, 1024, 1025, 5000, 60_000] {
            let xs = vec![State { a: 1.0, b: 1.0 }; n];
            let got = device.assoc_scan(&xs).expect("scan");
            for (i, s) in got.iter().enumerate() {
                assert_eq!(
                    s.b,
                    (i + 1) as f32,
                    "{}: counting scan at {i} of {n} gave {} (a = {})",
                    device.label(),
                    s.b,
                    s.a
                );
                assert_eq!(s.a, 1.0, "{}: multiplier drifted at {i}", device.label());
            }
        }
    }
}

#[test]
fn block_offsets_are_exact_on_a_doubling_scan() {
    // Companion to the counting scan: there `a` is fixed at 1, so an error in
    // the multiplier could not show. Here `a = 2` and `b = 0`, making
    // `prefix[i].a` exactly `2^(i+1)` — again exact in f32 — so a misapplied
    // block offset is a whole power of two out.
    for device in devices() {
        // Kept under 127 so 2^(i+1) stays finite.
        let n = 100usize;
        let xs = vec![State { a: 2.0, b: 0.0 }; n];
        let got = device.assoc_scan(&xs).expect("scan");
        for (i, s) in got.iter().enumerate() {
            assert_eq!(
                s.a,
                (2.0f32).powi(i as i32 + 1),
                "{}: doubling scan at {i} gave {}",
                device.label(),
                s.a
            );
        }
    }
}
