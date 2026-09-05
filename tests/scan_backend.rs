//! `Device::assoc_scan` across substrates.
//!
//! This is the one primitive where the crate's arms deliberately disagree in
//! the last few ulps: the CPU arms are bit-identical to a sequential fold, the
//! Metal arm reassociates so it can be parallel. The tests therefore assert two
//! different things — byte-identity *within* a backend, and a derived tolerance
//! *across* them — which is exactly the rule the crate states everywhere else.

mod common;

use sparsl::{
    assoc_scan_sequential, available_backends, scan_magnitude_envelope, tolerance_for_scan,
    Backend, Device, Rng, State,
};

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
            let mut want = Vec::with_capacity(n);
            for s in &xs {
                let (na, nb) = (s.a as f64, s.b as f64);
                let (ca, cb) = (na * ra, na * rb + nb);
                ra = ca;
                rb = cb;
                want.push((ca, cb));
            }

            // The crate's own bound, not a copy of it. This used to be the
            // formula written out here; a caller comparing two backends could
            // not reach it, and a test carrying its own copy is free to drift
            // from what the library promises. `tolerance_for_scan` is now
            // public and this asserts against exactly what it returns.
            let bound = f64::from(tolerance_for_scan(n, scan_magnitude_envelope(&xs)));
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
fn scan_tolerance_accounts_for_multiplier_growth_when_b_is_zero() {
    // This is a valid public `State` chain, even though the leak helper usually
    // produces contractive multipliers. A max-|b| scale is zero here and used
    // to under-bound the observed f32 error by more than an order of magnitude.
    let xs = vec![State { a: 1.1, b: 0.0 }; 100];
    let got = assoc_scan_sequential(&xs, |a, b| a.combine(b));
    let reference_a = xs
        .iter()
        .fold(1.0f64, |acc, state| f64::from(state.a) * acc);
    let error = (f64::from(got.last().expect("non-empty").a) - reference_a).abs();

    let b_only_bound = f64::from(tolerance_for_scan(xs.len(), 0.0));
    assert!(
        error > b_only_bound,
        "fixture stopped demonstrating the old under-bound: error={error}, b-only={b_only_bound}"
    );

    let envelope = scan_magnitude_envelope(&xs);
    let bound = f64::from(tolerance_for_scan(xs.len(), envelope));
    assert!(
        error <= bound,
        "multiplier growth escaped the scan bound: error={error}, envelope={envelope}, bound={bound}"
    );
}

#[test]
fn the_scan_respects_the_monoid_identity() {
    // Prepending the identity must not change any prefix. What "not change"
    // means differs by arm, and this file's own rule says which is which:
    // byte-identity *within* a backend, a derived tolerance *across* an
    // association.
    //
    // A leading identity shifts every element one place, which on a blocked
    // parallel scan also shifts the data relative to the block boundaries.
    // Elements that shared a block now straddle two, and a straddling prefix is
    // combined as (block total) . (element) rather than by the balanced tree
    // used inside a block. That is a reassociation, and reassociation is the
    // whole reason the kernel is parallel.
    //
    // This assertion used to demand bit-identity from every arm, and passed --
    // but only because its fixture was 300 elements and a threadgroup is 1024,
    // so it never reached a block boundary. A direct check at 2000 elements
    // showed the GPU arm's first changed prefix at index 1023, exactly the
    // boundary. The guarantee was never "bit-exact everywhere", only
    // "bit-exact until the first block boundary", and the fixture was too
    // small to tell the difference. It is 2000 now, so the crossing is
    // actually exercised rather than assumed away.
    //
    // The CPU arms are sequential folds, for which the identity really is a
    // no-op at every index, so they are still held to the bit.
    for backend in available_backends() {
        let Ok(device) = Device::try_new(backend) else {
            continue;
        };
        // Spans several threadgroups, so a leading identity genuinely moves
        // elements across a block boundary on the GPU arm.
        let xs = leak_steps(2000, 0x1DEA);
        let plain = device.assoc_scan(&xs).expect("scan");
        let mut padded = vec![State::identity()];
        padded.extend_from_slice(&xs);
        let with_id = device.assoc_scan(&padded).expect("scan");
        let envelope = scan_magnitude_envelope(&padded);
        let mut reassociated = 0usize;

        for i in 0..xs.len() {
            let (want, got) = (plain[i], with_id[i + 1]);
            if backend.is_gpu() {
                let bound = tolerance_for_scan(i + 2, envelope);
                assert!(
                    (want.a - got.a).abs() <= bound && (want.b - got.b).abs() <= bound,
                    "{}: a leading identity moved prefix {i} further than the \
                     published scan bound {bound}: {want:?} versus {got:?}",
                    backend.label()
                );
                if (want.a.to_bits(), want.b.to_bits()) != (got.a.to_bits(), got.b.to_bits()) {
                    reassociated += 1;
                }
            } else {
                assert_eq!(
                    (want.a.to_bits(), want.b.to_bits()),
                    (got.a.to_bits(), got.b.to_bits()),
                    "{}: a leading identity changed prefix {i}; a sequential \
                     fold must treat the identity as a true no-op",
                    backend.label()
                );
            }
        }

        // Without this the tolerance branch above could be silently vacuous: a
        // fixture that never crosses a block boundary satisfies it bit-exactly
        // and would report a pass while testing nothing. That is precisely how
        // the 300-element version of this test read as a guarantee for so long.
        if backend.is_gpu() {
            assert!(
                reassociated > 0,
                "{}: no prefix was reassociated, so this fixture no longer \
                 crosses a block boundary and proves nothing",
                backend.label()
            );
        }
    }
}

#[test]
fn identity_padding_past_the_end_contributes_nothing() {
    // The half of the claim above that IS bit-exact on every arm, and the half
    // that was actually load-bearing: the GPU pads the tail of a partial
    // threadgroup with the identity, so tail identities must not perturb any
    // earlier prefix.
    //
    // Appending rather than prepending is the point. It leaves every element on
    // the side of the block boundary it was already on, so unlike a leading
    // identity this reassociates nothing and the result must match to the bit.
    // The appended lengths cross a threadgroup boundary (300 -> 1100) so the
    // padded run genuinely spans more groups than the original.
    for backend in available_backends() {
        let Ok(device) = Device::try_new(backend) else {
            continue;
        };
        let xs = leak_steps(300, 0x1DEA);
        let plain = device.assoc_scan(&xs).expect("scan");
        for extra in [1usize, 31, 724, 800] {
            let mut padded = xs.clone();
            padded.extend(std::iter::repeat_n(State::identity(), extra));
            let got = device.assoc_scan(&padded).expect("scan");
            for i in 0..xs.len() {
                assert_eq!(
                    (plain[i].a.to_bits(), plain[i].b.to_bits()),
                    (got[i].a.to_bits(), got[i].b.to_bits()),
                    "{}: {extra} trailing identities changed prefix {i}",
                    backend.label()
                );
            }
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
fn the_first_prefix_preserves_non_finite_and_signed_zero_bits() {
    // The algebraic identity `(1, 0)` is not an IEEE-754 identity when it is
    // evaluated: `infinity * 0` produces NaN and addition can erase a signed
    // zero. The first inclusive prefix needs no arithmetic at all, so every
    // backend must copy it bit-for-bit instead of combining it with an
    // explicitly materialised identity or a padded lane.
    let first = State {
        a: f32::INFINITY,
        b: -0.0,
    };
    for device in devices() {
        let got = device.assoc_scan(&[first]).expect("singleton scan");
        assert_eq!(got.len(), 1, "{}", device.label());
        assert_eq!(got[0].a.to_bits(), first.a.to_bits(), "{}", device.label());
        assert_eq!(got[0].b.to_bits(), first.b.to_bits(), "{}", device.label());
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
        let envelope = scan_magnitude_envelope(&xs);
        for (i, s) in xs.iter().enumerate() {
            let (na, nb) = (s.a as f64, s.b as f64);
            let (ca, cb) = (na * ra, na * rb + nb);
            ra = ca;
            rb = cb;
            // `a` decays geometrically here and underflows to zero long before
            // the end; `b` is the component that stays informative.
            let bound = f64::from(tolerance_for_scan(i + 1, envelope));
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

#[test]
fn multi_group_scan_preserves_both_affine_components_exactly() {
    // This deliberately spans more than four maximum-size Metal scan groups.
    // Unlike the long leak-chain test, its prefix multiplier never underflows:
    // the three-state cycle keeps `a` at exactly 1 or 2 while `b` grows through
    // exactly representable integers. The cycle length is three, whereas every
    // legal Metal group width is a power of two, so group boundaries also cut
    // different phases instead of repeatedly landing on a trivial identity.
    let n = 4 * 1024 + 33;
    let cycle = [
        State { a: 2.0, b: 1.0 },
        State { a: 0.5, b: 0.5 },
        State { a: 1.0, b: 1.0 },
    ];
    let xs: Vec<State> = (0..n).map(|i| cycle[i % cycle.len()]).collect();
    let want = assoc_scan_sequential(&xs, |a, b| a.combine(b));

    assert!(want
        .iter()
        .all(|state| state.a.is_finite() && state.a != 0.0));
    for device in devices() {
        let got = device.assoc_scan(&xs).expect("multi-group affine scan");
        for (i, (got, want)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                got.a.to_bits(),
                want.a.to_bits(),
                "{}: multi-group multiplier a[{i}] = {}, want {}",
                device.label(),
                got.a,
                want.a
            );
            assert_eq!(
                got.b.to_bits(),
                want.b.to_bits(),
                "{}: multi-group offset b[{i}] = {}, want {}",
                device.label(),
                got.b,
                want.b
            );
        }
    }
}

/// The Metal scan uploads `&[State]` straight to the device and reads the
/// result straight back, so `State`'s in-memory layout *is* the buffer format
/// the kernel indexes as `[a0, b0, a1, b1, …]`.
///
/// That is a load-bearing assumption with no runtime check on the hot path —
/// the upload copies `size_of_val(xs)` bytes and the kernel trusts them. If a
/// field were added to `State`, or `repr(C)` removed and the fields reordered,
/// every scan would read plausible-looking garbage rather than fail. The
/// compile-time assertion in `scan.rs` catches a size or alignment change; this
/// catches the ordering, which sizes alone cannot.
#[test]
fn state_layout_matches_flat_f32() {
    assert_eq!(std::mem::size_of::<State>(), 2 * std::mem::size_of::<f32>());
    assert_eq!(std::mem::align_of::<State>(), std::mem::align_of::<f32>());

    let states = leak_steps(64, 0xA57);
    let flat: Vec<f32> = states.iter().flat_map(|s| [s.a, s.b]).collect();

    // SAFETY: the two assertions above establish that `State` is exactly two
    // packed `f32`s with `f32` alignment, so a `&[State]` of length n is a
    // valid, initialised `&[f32]` of length 2n over the same bytes. This is the
    // identical reinterpretation the Metal upload performs.
    let viewed: &[f32] =
        unsafe { std::slice::from_raw_parts(states.as_ptr() as *const f32, states.len() * 2) };

    assert_eq!(
        viewed,
        &flat[..],
        "State no longer lays out as [a, b] pairs — the Metal scan's \
         zero-reformat/direct typed byte-copy upload would transpose every affine map"
    );
}
