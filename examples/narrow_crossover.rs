//! What does narrow weight storage actually buy?
//!
//! The README predicted "a straight 2x", on the grounds that SpMV is
//! bandwidth-bound and f16 halves the weights. That reasoning skips a term: the
//! kernel streams `col_ind` (4 bytes) *and* `values` (4 bytes) per non-zero, so
//! narrowing only the values takes traffic from 8 bytes to 6. The ceiling is
//! 8/6 = 1.33x, not 2x — and that is a ceiling, not a prediction.
//!
//! Same two rules as `crossover`, for the same reasons: every arm is exercised
//! before any is timed, and each is timed twice in opposite orders with the
//! spread reported.
//!
//! Run: `cargo run --release --features metal --example f16_crossover`

use std::time::Instant;

use sparsl::{Backend, Csr, Device, Rng};

fn random_csr(nrows: usize, ncols: usize, deg: usize, rng: &mut Rng) -> Csr {
    let adj: Vec<Vec<u32>> = (0..nrows)
        .map(|_| (0..deg).map(|_| rng.gen_index(ncols) as u32).collect())
        .collect();
    Csr::from_adjacency(&adj)
}

fn ms(f: &mut dyn FnMut()) -> f64 {
    let t = Instant::now();
    f();
    t.elapsed().as_secs_f64() * 1000.0
}

fn main() {
    let Ok(gpu) = Device::try_new(Backend::Metal) else {
        eprintln!("no Metal device; nothing to measure");
        return;
    };

    println!("Metal SpMV, f32 weights against binary16 weights");
    println!("traffic per non-zero: 8 bytes -> 6, so the ceiling is 1.33x\n");

    for &(n, deg) in &[(10_000usize, 500usize), (20_000, 1000), (50_000, 400)] {
        let mut rng = Rng::new(0xF16 + n as u64);
        let csr = random_csr(n, n, deg, &mut rng);
        let w: Vec<f32> = (0..csr.nnz()).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        let x: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();

        let wide = gpu.prepare(&csr, n, &w).expect("prepare");
        let f16 = gpu.prepare_f16(&csr, n, &w).expect("prepare_f16");
        let bf16 = gpu.prepare_bf16(&csr, n, &w).expect("prepare_bf16");
        assert_eq!(f16.weight_precision(), sparsl::WeightPrecision::F16);
        assert_eq!(bf16.weight_precision(), sparsl::WeightPrecision::Bf16);

        // A single dispatch is dominated by launch jitter — the first version
        // of this reported spreads up to 2.09, which by the rule above means
        // the numbers cannot be quoted. Averaging over a run of dispatches puts
        // the spread back under control.
        const REPS: usize = 20;
        let mut y = vec![0.0f32; n];
        let mut run = |op: &sparsl::SparseOp| {
            ms(&mut || {
                for _ in 0..REPS {
                    op.spmv(&x, &mut y).expect("spmv");
                }
            }) / REPS as f64
        };

        // Ramp every arm before timing any.
        for _ in 0..3 {
            run(&wide);
            run(&f16);
            run(&bf16);
        }

        let (a1, b1, c1) = (run(&wide), run(&f16), run(&bf16));
        // Reversed.
        let (c2, b2, a2) = (run(&bf16), run(&f16), run(&wide));

        let best = |a: f64, b: f64| a.min(b);
        let spread = |a: f64, b: f64| a.max(b) / a.min(b).max(f64::MIN_POSITIVE);
        let (wt, ft, bt) = (best(a1, a2), best(b1, b2), best(c1, c2));

        println!(
            "n = {n:>6}  nnz = {:>9}   f32 {wt:>6.3} ms   f16 {ft:>6.3} ms ({:.2}x)   bf16 {bt:>6.3} ms ({:.2}x)   spreads {:.2}/{:.2}/{:.2}",
            csr.nnz(),
            wt / ft,
            wt / bt,
            spread(a1, a2),
            spread(b1, b2),
            spread(c1, c2)
        );
    }
}
