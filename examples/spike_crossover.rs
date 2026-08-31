//! What does a bitpacked spike vector buy?
//!
//! Narrowing the weights cut *streamed* traffic by 25% and bought a few
//! percent. This narrows the *gathered* operand 32-fold instead, and the
//! prediction is that it helps most exactly where narrow weights helped least:
//! at large `n`, where the f32 spike vector stops fitting in cache.
//!
//! Same rules as the other crossovers — every arm exercised before any is
//! timed, each timed in both orders, spread reported, and each timing averaged
//! over a run of dispatches because a single one is dominated by launch jitter.
//!
//! Run: `cargo run --release --features metal --example spike_crossover`

use std::time::Instant;

use sparsl::spikes::{pack_spikes, spikes_to_f32};
use sparsl::{Backend, Csr, Device, Rng};

fn main() {
    let Ok(gpu) = Device::try_new(Backend::Metal) else {
        eprintln!("no Metal device; nothing to measure");
        return;
    };

    println!("Metal SpMV: dense f32 spike vector against a bitpacked one");
    println!("the gathered operand shrinks 32x; the streamed arrays do not change\n");

    for &(n, deg) in &[(10_000usize, 500usize), (20_000, 1000), (50_000, 400)] {
        let mut rng = Rng::new(0x5B1C + n as u64);
        let adj: Vec<Vec<u32>> = (0..n)
            .map(|_| (0..deg).map(|_| rng.gen_index(n) as u32).collect())
            .collect();
        let csr = Csr::from_adjacency(&adj);
        let w: Vec<f32> = (0..csr.nnz()).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let op = gpu.prepare(&csr, n, &w).expect("prepare");

        let fired: Vec<bool> = (0..n).map(|_| rng.next_f32() < 0.05).collect();
        let packed = pack_spikes(&fired);
        let dense = spikes_to_f32(&packed, n);

        // Two closures capturing `y` mutably cannot coexist, so the timing
        // owns its own scratch and the arm is a flag.
        const REPS: usize = 20;
        let run = |packed_arm: bool| {
            let mut y = vec![0.0f32; n];
            let t = Instant::now();
            for _ in 0..REPS {
                if packed_arm {
                    op.spmv_spikes(&packed, &mut y).expect("spmv_spikes");
                } else {
                    op.spmv(&dense, &mut y).expect("spmv");
                }
            }
            t.elapsed().as_secs_f64() * 1000.0 / REPS as f64
        };
        // Ramp both arms before timing either.
        for _ in 0..3 {
            run(false);
            run(true);
        }
        let (d1, p1) = (run(false), run(true));
        // Reversed.
        let (p2, d2) = (run(true), run(false));

        let best = |a: f64, b: f64| a.min(b);
        let spread = |a: f64, b: f64| a.max(b) / a.min(b).max(f64::MIN_POSITIVE);
        let (dt, pt) = (best(d1, d2), best(p1, p2));

        println!(
            "n = {n:>6}  x is {:>6.1} KB dense / {:>5.1} KB packed   dense {dt:>6.3} ms   packed {pt:>6.3} ms   {:.2}x   spreads {:.2}/{:.2}",
            (n * 4) as f64 / 1024.0,
            (n as f64 / 8.0) / 1024.0,
            dt / pt,
            spread(d1, d2),
            spread(p1, p2)
        );
    }
}
