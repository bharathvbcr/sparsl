//! Wave 3 usefulness: resident step API and CPU-visible contracts.
//!
//! Metal-only probes (host-copy counters, hybrid selection) live in
//! `src/backend/metal.rs` unit tests because the Metal module is `pub(crate)`.

use sparsl::{Device, LifParams};

#[test]
fn cpu_resident_step_matches_slice_api() {
    let device = Device::cpu_parallel();
    let csr = sparsl::Csr::from_adjacency(&[vec![1, 2], vec![0], vec![0, 1]]);
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let op = device.prepare(&csr, 3, &weights).expect("prepare");
    let x = vec![0.5, 1.0, 1.5];
    let mut y_slice = vec![0.25, 0.0, -0.5];
    op.spmv(&x, &mut y_slice).expect("slice spmv");

    let mut y_res = vec![0.25, 0.0, -0.5];
    op.write_x(&x).expect("write_x");
    op.write_y(&y_res).expect("write_y");
    op.spmv_resident().expect("resident spmv");
    op.sync_y(&mut y_res).expect("sync_y");
    assert_eq!(y_slice, y_res);

    let params = LifParams::new(0.9, 0.0, 0.1).expect("params");
    let mut v = vec![0.1, 0.2, 0.3];
    let mut theta = vec![1.0, 1.0, 1.0];
    let mut spikes = vec![false; 3];
    let mut v2 = v.clone();
    let mut th2 = theta.clone();
    let mut sp2 = spikes.clone();
    op.fused_spmv_lif(&x, &mut v, &mut theta, &mut spikes, params)
        .expect("slice fused");
    op.write_x(&x).expect("write_x");
    op.write_lif_state(&v2, &th2).expect("write_lif");
    op.fused_spmv_lif_resident(params).expect("resident fused");
    op.sync_lif_state(&mut v2, &mut th2, &mut sp2)
        .expect("sync_lif");
    assert_eq!(v, v2);
    assert_eq!(theta, th2);
    assert_eq!(spikes, sp2);
}

#[test]
fn hub_fixture_parallel_spmv_stays_bit_identical() {
    let nrows = 512usize;
    let mut adj = vec![vec![0u32]; nrows];
    adj[0] = (0..4_000u32).collect();
    for (r, row) in adj.iter_mut().enumerate().skip(1) {
        *row = vec![0, 1, (r as u32) % 7];
        row.sort();
    }
    let csr = sparsl::Csr::from_adjacency(&adj);
    let weights: Vec<f32> = (0..csr.nnz()).map(|i| (i % 11) as f32 * 0.1).collect();
    let x: Vec<f32> = (0..csr.ncols()).map(|i| (i % 5) as f32).collect();
    let seq = Device::cpu_sequential();
    let par = Device::cpu_parallel();
    let op_s = seq.prepare(&csr, csr.ncols(), &weights).expect("seq");
    let op_p = par.prepare(&csr, csr.ncols(), &weights).expect("par");
    let mut y_s = vec![0.25f32; csr.nrows()];
    let mut y_p = y_s.clone();
    op_s.spmv(&x, &mut y_s).expect("seq spmv");
    op_p.spmv(&x, &mut y_p).expect("par spmv");
    assert_eq!(y_s, y_p);
}
