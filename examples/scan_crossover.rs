//! Does the GPU scan actually beat the CPU one?
//!
//! `scan.rs` records that the chunked CPU scan buys 1.08x over a sequential
//! fold, because bit-identity forces phase 1 to be a complete sequential
//! left-fold and phase 2 then redoes that work. It also asserts that a
//! two-level tree scan "would deliver a real speedup". That kernel now exists,
//! so the assertion is measurable and this measures it.
//!
//! Obeys the same two rules as `crossover`, for the same reasons:
//!
//! * **Every arm is exercised before any arm is timed**, so the first-timed arm
//!   does not pay for the GPU clock ramp alone.
//! * **Each arm is timed twice in opposite orders**, and `spread` reports the
//!   ratio. A spread far from 1.00 means ordering still matters and the
//!   speedups should not be quoted.
//!
//! Run: `cargo run --release --features metal --example scan_crossover`

use std::time::Instant;

use sparsl::{assoc_scan, assoc_scan_sequential, available_backends, Device, Rng, State};

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

fn ms(f: &mut dyn FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let devices: Vec<Device> = available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect();

    for &n in &[1 << 16, 1 << 18, 1 << 20, 1 << 22] {
        let xs = leak_steps(n, 0x5CA1_0000 + n as u64);

        // Ramp: every arm runs before any is timed.
        for _ in 0..3 {
            std::hint::black_box(assoc_scan_sequential(&xs, State::combine));
            std::hint::black_box(assoc_scan(&xs, State::combine));
            for d in &devices {
                std::hint::black_box(d.assoc_scan(&xs).expect("scan"));
            }
        }

        let seq = || ms(&mut || drop(assoc_scan_sequential(&xs, State::combine)));
        let chunked = || ms(&mut || drop(assoc_scan(&xs, State::combine)));
        let dev = |d: &Device| ms(&mut || drop(d.assoc_scan(&xs).expect("scan")));

        // Pass 1, in order.
        let s1 = seq();
        let c1 = chunked();
        let dev1: Vec<f64> = devices.iter().map(dev).collect();

        // Pass 2, every arm in the opposite order; `spread` compares the two.
        let mut dev2: Vec<f64> = devices.iter().rev().map(dev).collect();
        dev2.reverse();
        let c2 = chunked();
        let s2 = seq();

        let best = |a: f64, b: f64| a.min(b);
        let spread = |a: f64, b: f64| a.max(b) / a.min(b).max(f64::MIN_POSITIVE);
        let seq_ms = best(s1, s2);

        println!("\nn = {n} ({:.1}M elements)", n as f64 / 1e6);
        println!(
            "  {:<22} {:>9.3} ms   {:>6}   spread {:.2}",
            "cpu sequential",
            seq_ms,
            "1.00x",
            spread(s1, s2)
        );
        println!(
            "  {:<22} {:>9.3} ms   {:>5.2}x   spread {:.2}",
            "cpu chunked (rayon)",
            best(c1, c2),
            seq_ms / best(c1, c2),
            spread(c1, c2)
        );
        for (i, d) in devices.iter().enumerate() {
            let t = best(dev1[i], dev2[i]);
            println!(
                "  {:<22} {:>9.3} ms   {:>5.2}x   spread {:.2}",
                d.label(),
                t,
                seq_ms / t,
                spread(dev1[i], dev2[i])
            );
        }
    }
}
