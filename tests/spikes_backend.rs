//! Bitpacked spike vectors, across backends.
//!
//! The claim under test is stronger than anywhere else in this suite. Narrow
//! weights are bounded by a tolerance because quantisation loses information;
//! bitpacking loses none. A spike is 0 or 1, both exact in `f32`, and both
//! paths decode the bit to a float and multiply — the same operations in the
//! same order. So the assertion here is **bit-for-bit equality** with the dense
//! path, on every backend, and a tolerance would be the wrong instrument.

mod common;

use common::{random_csr, random_vec};
use sparsl::spikes::{pack_spikes, packed_len, spikes_to_f32};
use sparsl::{available_backends, Backend, Device, Rng};

fn devices() -> Vec<Device> {
    available_backends()
        .into_iter()
        .filter_map(|b| Device::try_new(b).ok())
        .collect()
}

fn random_spikes(n: usize, density: f32, rng: &mut Rng) -> Vec<bool> {
    (0..n).map(|_| rng.next_f32() < density).collect()
}

#[test]
fn the_spike_path_is_bit_identical_to_the_dense_one() {
    let mut rng = Rng::new(0x5B17);
    // Sizes that straddle word boundaries, and densities from nearly silent to
    // nearly saturated — a vector of all-zeros and one of all-ones exercise
    // different halves of the decode.
    for &(nrows, ncols, deg) in &[(64usize, 31usize, 6usize), (128, 64, 12), (97, 257, 20)] {
        for &density in &[0.0f32, 0.05, 0.5, 0.95, 1.0] {
            let csr = random_csr(nrows, ncols, deg, &mut rng);
            let weights = random_vec(csr.nnz(), 1.0, &mut rng);
            let spikes = random_spikes(ncols, density, &mut rng);
            let packed = pack_spikes(&spikes);
            let dense = spikes_to_f32(&packed, ncols);

            for device in devices() {
                let op = device.prepare(&csr, ncols, &weights).expect("prepare");

                let mut via_dense = vec![0.0f32; nrows];
                op.spmv(&dense, &mut via_dense).expect("spmv");

                let mut via_packed = vec![0.0f32; nrows];
                op.spmv_spikes(&packed, &mut via_packed)
                    .expect("spmv_spikes");

                for r in 0..nrows {
                    assert_eq!(
                        via_packed[r].to_bits(),
                        via_dense[r].to_bits(),
                        "{} (ncols={ncols}, density={density}): row {r} \
                         packed {} against dense {}",
                        op.label(),
                        via_packed[r],
                        via_dense[r]
                    );
                }
            }
        }
    }
}

#[test]
fn it_accumulates_into_y_rather_than_overwriting() {
    let mut rng = Rng::new(0xACC5);
    let (nrows, ncols) = (48usize, 96usize);
    let csr = random_csr(nrows, ncols, 8, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let packed = pack_spikes(&random_spikes(ncols, 0.4, &mut rng));

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let mut once = vec![0.0f32; nrows];
        op.spmv_spikes(&packed, &mut once).expect("spmv_spikes");

        let seed = random_vec(nrows, 1.0, &mut rng);
        let mut acc = seed.clone();
        op.spmv_spikes(&packed, &mut acc).expect("spmv_spikes");
        for r in 0..nrows {
            assert!(
                (acc[r] - (seed[r] + once[r])).abs() <= 1e-4,
                "{}: overwrote instead of accumulating at row {r}",
                op.label()
            );
        }
    }
}

#[test]
fn a_silent_vector_adds_nothing() {
    // The degenerate case, and the one where a branch-on-bit implementation
    // would differ from the multiply: every product is `w * 0.0`.
    let mut rng = Rng::new(0x51E17);
    let (nrows, ncols) = (32usize, 64usize);
    let csr = random_csr(nrows, ncols, 5, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let silent = pack_spikes(&vec![false; ncols]);

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let seed = random_vec(nrows, 1.0, &mut rng);
        let mut y = seed.clone();
        op.spmv_spikes(&silent, &mut y).expect("spmv_spikes");
        for r in 0..nrows {
            assert_eq!(
                y[r].to_bits(),
                seed[r].to_bits(),
                "{}: a silent vector changed row {r}",
                op.label()
            );
        }
    }
}

#[test]
fn every_backend_agrees_with_every_other_exactly() {
    let mut rng = Rng::new(0xA6BEE2);
    let (nrows, ncols, deg) = (192usize, 96usize, 12usize);
    let csr = random_csr(nrows, ncols, deg, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);
    let packed = pack_spikes(&random_spikes(ncols, 0.3, &mut rng));

    // Note this demands *equality* across backends, which the dense SpMV
    // cannot promise — there it takes `tolerance_for_spmv`, because the CPU and
    // GPU reduce in different orders. It holds here for the same reason it
    // holds against the dense path: one thread per row on every arm, summing
    // the same products in index order.
    let mut reference: Option<(String, Vec<f32>)> = None;
    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let mut got = vec![0.0f32; nrows];
        op.spmv_spikes(&packed, &mut got).expect("spmv_spikes");
        match &reference {
            None => reference = Some((op.label().to_string(), got)),
            Some((label, want)) => {
                for r in 0..nrows {
                    assert_eq!(
                        got[r].to_bits(),
                        want[r].to_bits(),
                        "{} vs {label}: row {r}",
                        op.label()
                    );
                }
            }
        }
    }
}

#[test]
fn an_undersized_packed_vector_is_refused() {
    let mut rng = Rng::new(0x51400);
    let (nrows, ncols) = (16usize, 64usize);
    let csr = random_csr(nrows, ncols, 4, &mut rng);
    let weights = random_vec(csr.nnz(), 1.0, &mut rng);

    for device in devices() {
        let op = device.prepare(&csr, ncols, &weights).expect("prepare");
        let mut y = vec![0.0f32; nrows];
        // One word short. Reading it as silence would look like cells that
        // simply did not fire.
        let short = vec![0u32; packed_len(ncols) - 1];
        assert!(
            op.spmv_spikes(&short, &mut y).is_err(),
            "{}: a short packed vector must be refused",
            op.label()
        );
        // And the right length is accepted.
        let ok = vec![0u32; packed_len(ncols)];
        assert!(op.spmv_spikes(&ok, &mut y).is_ok());
    }
}

#[test]
fn the_lif_output_feeds_straight_in() {
    // The flow this exists for: fused_spmv_lif produces `&mut [bool]`, which
    // packs directly into the next layer's input.
    let Ok(device) = Device::try_new(Backend::CpuSequential) else {
        return;
    };
    let mut rng = Rng::new(0x11F);
    let n = 128usize;
    let csr = random_csr(n, n, 10, &mut rng);
    let weights = random_vec(csr.nnz(), 0.5, &mut rng);
    let op = device.prepare(&csr, n, &weights).expect("prepare");

    let x = random_vec(n, 1.0, &mut rng);
    let mut v = vec![0.0f32; n];
    let mut theta = vec![0.1f32; n];
    let mut fired = vec![false; n];
    let params = sparsl::LifParams::new(0.9, 0.0, 0.1).expect("params");
    op.fused_spmv_lif(&x, &mut v, &mut theta, &mut fired, params)
        .expect("fused");

    let packed = pack_spikes(&fired);
    let mut y = vec![0.0f32; n];
    op.spmv_spikes(&packed, &mut y).expect("spmv_spikes");

    let mut want = vec![0.0f32; n];
    op.spmv(&spikes_to_f32(&packed, n), &mut want)
        .expect("spmv");
    for r in 0..n {
        assert_eq!(y[r].to_bits(), want[r].to_bits(), "row {r}");
    }
}
