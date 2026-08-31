//! Where does each backend start winning?
//!
//! Runs every available backend over the same CSR shapes and reports ms/iter
//! plus speedup against the sequential CPU reference. Arms come from
//! `available_backends()`, so this can never print a GPU column produced by CPU
//! code, and a backend that is not really there simply does not appear.
//!
//! Two measurement rules this obeys, both learned the hard way:
//!
//! * **Every arm is exercised before any arm is timed.** Without a ramp, the
//!   first-timed arm pays for the GPU clock ramp-up and later arms do not,
//!   which produced the physically impossible result that adding a host memcpy
//!   to each iteration made it faster.
//! * **Each arm is timed twice in opposite orders**, and `spread` reports the
//!   ratio between passes. A spread far from 1.00 means ordering still matters
//!   and the speedups should not be quoted.
//!
//! Run: `cargo run --release --features metal --example crossover`

use std::time::{Duration, Instant};

use sparsl::{available_backends, Backend, Csr, Device, Rng, SparseOp};

const SIZES: &[usize] = &[1_000, 5_000, 10_000, 20_000];
const DENSITY: f32 = 0.05;
const RAMP: usize = 50;

fn build(n: usize, rng: &mut Rng) -> (Csr, Vec<f32>, Vec<f32>) {
    let nnz_per_row = ((n as f32) * DENSITY) as usize;
    let mut adj: Vec<Vec<u32>> = vec![Vec::with_capacity(nnz_per_row); n];
    for (r, row) in adj.iter_mut().enumerate() {
        for i in 0..nnz_per_row {
            row.push(((r + i * 3) % n) as u32);
        }
    }
    let csr = Csr::from_adjacency(&adj);
    // Varied values, not constants: with a constant weight and a constant `x`
    // every row is a sum of identical terms, and any two reduction orders agree
    // bitwise. A parity check against that data proves nothing.
    let weights = (0..csr.nnz()).map(|_| rng.next_f32() - 0.5).collect();
    let x = (0..n).map(|_| rng.next_f32() - 0.5).collect();
    (csr, weights, x)
}

fn time_spmv(op: &SparseOp, x: &[f32], n: usize, iters: usize) -> f64 {
    let mut y = vec![0.0f32; n];
    let start = Instant::now();
    for _ in 0..iters {
        op.spmv(x, &mut y).expect("spmv");
    }
    ms(start.elapsed(), iters)
}

fn ms(total: Duration, iters: usize) -> f64 {
    total.as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let arms = available_backends();
    println!("backends: {}", arms
        .iter()
        .map(|b| b.label())
        .collect::<Vec<_>>()
        .join(", "));
    for backend in Backend::ALL {
        if let Some(reason) = backend.unavailable_reason() {
            println!("  unavailable — {}: {reason}", backend.label());
        }
    }
    if let Some(name) = Device::try_new(Backend::Metal).ok().and_then(|d| d.device_name()) {
        println!("  Metal device: {name}");
    }
    println!();

    print!("| N | nnz |");
    for arm in &arms {
        print!(" {} (ms) |", arm.label());
    }
    println!(" best speedup | upload (ms) | spread |");
    print!("|---:|---:|");
    for _ in &arms {
        print!("---:|");
    }
    println!("---:|---:|---:|");

    let mut rng = Rng::new(0x5713_2026);
    for &n in SIZES {
        let (csr, weights, x) = build(n, &mut rng);
        let nnz = csr.nnz();
        let iters = if nnz > 2_000_000 { 50 } else { 200 };

        let mut upload_ms = 0.0f64;
        let ops: Vec<(Backend, SparseOp)> = arms
            .iter()
            .map(|&b| {
                let device = Device::try_new(b).expect("available");
                let start = Instant::now();
                let op = device.prepare(&csr, n, &weights).expect("valid csr");
                if b.is_gpu() {
                    upload_ms = upload_ms.max(start.elapsed().as_secs_f64() * 1000.0);
                }
                (b, op)
            })
            .collect();

        for (_, op) in &ops {
            time_spmv(op, &x, n, RAMP);
        }
        let pass_a: Vec<f64> = ops
            .iter()
            .map(|(_, op)| time_spmv(op, &x, n, iters))
            .collect();
        let pass_b: Vec<f64> = ops
            .iter()
            .rev()
            .map(|(_, op)| time_spmv(op, &x, n, iters))
            .collect();

        let best: Vec<f64> = pass_a
            .iter()
            .zip(pass_b.iter().rev())
            .map(|(a, b)| a.min(*b))
            .collect();
        let spread = pass_a
            .iter()
            .zip(pass_b.iter().rev())
            .map(|(a, b)| a.max(*b) / a.min(*b))
            .fold(1.0f64, f64::max);

        let reference = best[0];
        let fastest = best.iter().copied().fold(f64::INFINITY, f64::min);

        print!("| {n} | {nnz} |");
        for t in &best {
            print!(" {t:.3} |");
        }
        println!(" {:.2}x | {upload_ms:.3} | {spread:.2} |", reference / fastest);
    }
}
