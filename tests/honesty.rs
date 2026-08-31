//! The invariant that gives every other number in this crate its meaning: a
//! backend handle cannot exist for a substrate that will not run, and a label
//! always names what actually executed.
//!
//! These are regression tests for a specific historical defect, not hygiene. In
//! the code sparsl was extracted from, a `use_gpu: bool` was never read by any
//! dispatch path: a "GPU" backend and a "CPU" backend ran byte-identical rayon
//! code, benchmarks reported ~1.00x speedups as genuine cross-substrate
//! results, and generated reports printed CPU timings under a GPU heading. If
//! any test in this file fails, some code path is handing out a handle that
//! lies about where work runs.

mod common;

use common::*;
use sparsl::{available_backends, Backend, Device, Rng};

#[test]
fn cuda_is_declared_but_never_available() {
    assert!(
        !Backend::Cuda.is_available(),
        "CUDA has no dispatch implementation; it must never report available"
    );
    let reason = Backend::Cuda
        .unavailable_reason()
        .expect("an unavailable backend must say why");
    assert!(
        reason.contains("not implemented"),
        "the reason must name the actual cause, got: {reason}"
    );
    let err = Device::try_new(Backend::Cuda)
        .expect_err("Device::try_new(Cuda) must fail rather than fall back to CPU");
    assert_eq!(err.requested, Backend::Cuda);
    assert!(!available_backends().contains(&Backend::Cuda));
}

#[test]
fn unavailable_backends_are_unconstructible() {
    for backend in Backend::ALL {
        match backend.unavailable_reason() {
            None => {
                let device = Device::try_new(backend)
                    .expect("a backend reporting available must open successfully");
                assert_eq!(
                    device.backend(),
                    backend,
                    "try_new returned a different substrate than requested"
                );
            }
            Some(reason) => {
                let err = Device::try_new(backend).expect_err(
                    "a backend reporting unavailable must not produce a working handle",
                );
                assert_eq!(err.requested, backend);
                assert_eq!(err.reason, reason, "availability and try_new must agree");
            }
        }
    }
}

#[test]
fn available_backends_are_available_and_distinctly_labelled() {
    let arms = available_backends();
    assert!(
        arms.contains(&Backend::CpuSequential) && arms.contains(&Backend::CpuParallel),
        "the CPU arms are unconditional"
    );
    for arm in &arms {
        assert!(arm.is_available(), "{arm} advertised but unavailable");
    }
    let mut labels: Vec<_> = arms.iter().map(|b| b.label()).collect();
    labels.sort_unstable();
    let before = labels.len();
    labels.dedup();
    assert_eq!(
        labels.len(),
        before,
        "two available backends share a label; a two-arm table built from this \
         could show the same substrate twice"
    );
}

#[test]
fn device_label_names_the_executing_substrate() {
    for backend in available_backends() {
        let device = Device::try_new(backend).expect("available");
        assert_eq!(device.label(), backend.label());
        assert_eq!(device.backend(), backend);
        if backend.is_gpu() {
            assert!(
                device.device_name().is_some(),
                "a GPU handle must be able to name its physical device"
            );
        }
    }
}

/// rayon parallelism here is a map, never a reduction: every output element is
/// computed by exactly one thread from a fixed input order. So the parallel CPU
/// arm must be *bit*-identical to the sequential one, not merely close. If this
/// ever fails, some kernel started reducing across threads and the crate's
/// determinism claim no longer holds.
#[test]
fn cpu_parallel_is_bit_identical_to_sequential() {
    let seq = Device::cpu_sequential();
    let par = Device::cpu_parallel();
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);

    for &(nrows, ncols, max_deg) in SHAPES {
        let csr = random_csr(nrows, ncols, max_deg, &mut rng);
        let weights = random_vec(csr.nnz(), 1.0, &mut rng);
        let x = random_vec(ncols, 1.0, &mut rng);
        let op_seq = seq.prepare(&csr, ncols, &weights).expect("valid csr");
        let op_par = par.prepare(&csr, ncols, &weights).expect("valid csr");
        let params = default_params();

        let mut y_seq = random_vec(nrows, 1.0, &mut rng);
        let mut y_par = y_seq.clone();
        op_seq.spmv(&x, &mut y_seq).expect("spmv");
        op_par.spmv(&x, &mut y_par).expect("spmv");
        assert_eq!(
            y_seq.to_bits_vec(),
            y_par.to_bits_vec(),
            "spmv differs between CPU arms at nrows={nrows}"
        );

        let v0 = random_vec(nrows, 1.0, &mut rng);
        let theta0 = random_vec(nrows, 1.0, &mut rng);
        let (mut v_s, mut th_s, mut sp_s) = (v0.clone(), theta0.clone(), vec![false; nrows]);
        let (mut v_p, mut th_p, mut sp_p) = (v0.clone(), theta0.clone(), vec![false; nrows]);
        op_seq
            .fused_spmv_lif(&x, &mut v_s, &mut th_s, &mut sp_s, params)
            .expect("fused");
        op_par
            .fused_spmv_lif(&x, &mut v_p, &mut th_p, &mut sp_p, params)
            .expect("fused");
        assert_eq!(
            v_s.to_bits_vec(),
            v_p.to_bits_vec(),
            "fused v (nrows={nrows})"
        );
        assert_eq!(th_s.to_bits_vec(), th_p.to_bits_vec(), "fused theta");
        assert_eq!(sp_s, sp_p, "fused spikes");

        let currents = random_vec(nrows, 1.0, &mut rng);
        let (mut v_s, mut th_s, mut sp_s) = (v0.clone(), theta0.clone(), vec![false; nrows]);
        let (mut v_p, mut th_p, mut sp_p) = (v0, theta0, vec![false; nrows]);
        seq.lif_integrate(&mut v_s, &mut th_s, &currents, &mut sp_s, params)
            .expect("lif");
        par.lif_integrate(&mut v_p, &mut th_p, &currents, &mut sp_p, params)
            .expect("lif");
        assert_eq!(v_s.to_bits_vec(), v_p.to_bits_vec(), "lif v");
        assert_eq!(th_s.to_bits_vec(), th_p.to_bits_vec(), "lif theta");
        assert_eq!(sp_s, sp_p, "lif spikes");
    }
}

/// Same seed, same fingerprint. Carried over from the crate this code came
/// from, where it was the top-level determinism gate.
#[test]
fn same_seed_yields_identical_state_fingerprint() {
    fn fingerprint(seed: u64) -> u64 {
        let mut rng = Rng::new(seed);
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for _ in 0..256 {
            hash ^= rng.next_u64();
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        let csr = random_csr(64, 64, 8, &mut rng);
        for &p in &csr.row_ptr {
            hash ^= p as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        for &c in &csr.col {
            hash ^= c as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        hash
    }
    assert_eq!(fingerprint(7), fingerprint(7));
    assert_ne!(fingerprint(7), fingerprint(8));
}

/// Bitwise comparison helper: `-0.0 == 0.0` and `NaN != NaN` under `f32` Eq,
/// neither of which is what "bit-identical" means.
trait ToBits {
    fn to_bits_vec(&self) -> Vec<u32>;
}

impl ToBits for [f32] {
    fn to_bits_vec(&self) -> Vec<u32> {
        self.iter().map(|v| v.to_bits()).collect()
    }
}
