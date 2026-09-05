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
use sparsl::spikes::pack_spikes;
use sparsl::{available_backends, Backend, Device, LifParams, Rng, State};

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
///
/// This is the bounded multi-operation campaign: every documented boundary
/// shape plus 128 pseudo-random shapes from a fixed seed passes through SpMV,
/// SpMM, transpose, packed spikes, fused and dense LIF, and scan. Malformed CSR
/// rejection stays in `tests/stress.rs`, where the exact error class is checked
/// before any operator can be constructed.
#[test]
fn cpu_parallel_is_bit_identical_to_sequential() {
    const RANDOM_CASES: usize = 128;
    const BATCHES: &[usize] = &[1, 2, 3, 7, 16, 17, 31, 32, 33];

    let seq = Device::cpu_sequential();
    let par = Device::cpu_parallel();
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let mut shapes = SHAPES.to_vec();
    // `SHAPES` feeds the public replay fingerprint, so keep this additional
    // zero-column boundary local rather than changing that pinned workload.
    shapes.push((8, 0, 0));
    for _ in 0..RANDOM_CASES {
        shapes.push((rng.gen_index(385), rng.gen_index(257), rng.gen_index(35)));
    }

    for (case, &(nrows, ncols, max_deg)) in shapes.iter().enumerate() {
        let csr = random_csr(nrows, ncols, max_deg, &mut rng);
        let weights = random_vec(csr.nnz(), 1.0, &mut rng);
        let x = random_vec(ncols, 1.0, &mut rng);
        let op_seq = seq
            .prepare_with_transpose(&csr, ncols, &weights)
            .expect("valid csr");
        let op_par = par
            .prepare_with_transpose(&csr, ncols, &weights)
            .expect("valid csr");
        let params = match case % 4 {
            0 => LifParams::new(0.0, -0.25, 0.0),
            1 => LifParams::new(0.5, 0.0, 0.1),
            2 => LifParams::new(0.9, -1.0, 0.25),
            _ => LifParams::new(1.1, 0.5, -0.1),
        }
        .expect("finite campaign parameters");
        let context = format!("case={case} nrows={nrows} ncols={ncols} max_degree={max_deg}");

        let mut y_seq = random_vec(nrows, 1.0, &mut rng);
        let mut y_par = y_seq.clone();
        op_seq.spmv(&x, &mut y_seq).expect("spmv");
        op_par.spmv(&x, &mut y_par).expect("spmv");
        assert_eq!(
            y_seq.to_bits_vec(),
            y_par.to_bits_vec(),
            "spmv differs between CPU arms: {context}"
        );

        let n_vec = BATCHES[case % BATCHES.len()];
        let batch_x = random_vec(ncols * n_vec, 1.0, &mut rng);
        let mut batch_seq = random_vec(nrows * n_vec, 1.0, &mut rng);
        let mut batch_par = batch_seq.clone();
        op_seq.spmm(&batch_x, n_vec, &mut batch_seq).expect("spmm");
        op_par.spmm(&batch_x, n_vec, &mut batch_par).expect("spmm");
        assert_eq!(
            batch_seq.to_bits_vec(),
            batch_par.to_bits_vec(),
            "spmm differs between CPU arms at n_vec={n_vec}: {context}"
        );

        let transpose_x = random_vec(nrows, 1.0, &mut rng);
        let mut transpose_seq = random_vec(ncols, 1.0, &mut rng);
        let mut transpose_par = transpose_seq.clone();
        op_seq
            .spmv_t(&transpose_x, &mut transpose_seq)
            .expect("spmv_t");
        op_par
            .spmv_t(&transpose_x, &mut transpose_par)
            .expect("spmv_t");
        assert_eq!(
            transpose_seq.to_bits_vec(),
            transpose_par.to_bits_vec(),
            "transpose SpMV differs between CPU arms: {context}"
        );

        let spike_values: Vec<bool> = (0..ncols).map(|_| rng.next_u32() & 0b11 == 0).collect();
        let packed = pack_spikes(&spike_values);
        let mut spike_seq = random_vec(nrows, 1.0, &mut rng);
        let mut spike_par = spike_seq.clone();
        op_seq
            .spmv_spikes(&packed, &mut spike_seq)
            .expect("spmv_spikes");
        op_par
            .spmv_spikes(&packed, &mut spike_par)
            .expect("spmv_spikes");
        assert_eq!(
            spike_seq.to_bits_vec(),
            spike_par.to_bits_vec(),
            "packed-spike SpMV differs between CPU arms: {context}"
        );

        let v0 = random_vec(nrows, 1.0, &mut rng);
        let theta0 = random_vec(nrows, 1.0, &mut rng);
        let (mut v_s, mut th_s, mut sp_s) = (v0.clone(), theta0.clone(), vec![true; nrows]);
        let (mut v_p, mut th_p, mut sp_p) = (v0.clone(), theta0.clone(), vec![true; nrows]);
        op_seq
            .fused_spmv_lif(&x, &mut v_s, &mut th_s, &mut sp_s, params)
            .expect("fused");
        op_par
            .fused_spmv_lif(&x, &mut v_p, &mut th_p, &mut sp_p, params)
            .expect("fused");
        assert_eq!(
            v_s.to_bits_vec(),
            v_p.to_bits_vec(),
            "fused v differs: {context}"
        );
        assert_eq!(
            th_s.to_bits_vec(),
            th_p.to_bits_vec(),
            "fused theta differs: {context}"
        );
        assert_eq!(sp_s, sp_p, "fused spikes differ: {context}");

        let currents = random_vec(nrows, 1.0, &mut rng);
        let (mut v_s, mut th_s, mut sp_s) = (v0.clone(), theta0.clone(), vec![true; nrows]);
        let (mut v_p, mut th_p, mut sp_p) = (v0, theta0, vec![true; nrows]);
        seq.lif_integrate(&mut v_s, &mut th_s, &currents, &mut sp_s, params)
            .expect("lif");
        par.lif_integrate(&mut v_p, &mut th_p, &currents, &mut sp_p, params)
            .expect("lif");
        assert_eq!(v_s.to_bits_vec(), v_p.to_bits_vec(), "lif v: {context}");
        assert_eq!(
            th_s.to_bits_vec(),
            th_p.to_bits_vec(),
            "lif theta: {context}"
        );
        assert_eq!(sp_s, sp_p, "lif spikes: {context}");

        let scan_steps: Vec<State> = (0..nrows)
            .map(|_| State {
                a: 0.5 + 0.49 * rng.next_f32(),
                b: rng.next_f32() * 2.0 - 1.0,
            })
            .collect();
        let scan_seq = seq.assoc_scan(&scan_steps).expect("sequential scan");
        let scan_par = par.assoc_scan(&scan_steps).expect("parallel scan");
        assert_eq!(
            state_bits(&scan_seq),
            state_bits(&scan_par),
            "scan differs between CPU arms: {context}"
        );
    }
}

fn state_bits(states: &[State]) -> Vec<(u32, u32)> {
    states
        .iter()
        .map(|state| (state.a.to_bits(), state.b.to_bits()))
        .collect()
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
