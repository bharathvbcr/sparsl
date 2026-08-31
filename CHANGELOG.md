# Changelog

All notable changes to `sparsl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **The Metal backend now uses `objc2-metal` instead of the gfx-rs `metal`
  crate.** That removes `block 0.1.6` and `objc 0.2` from the tree entirely.
  `block` is unmaintained and triggers the `static of uninhabited type`
  future-incompatibility lint, which becomes a hard error in a future Rust;
  bumping `metal` did not help, because every release in that line including
  0.33 pulls the same crate. `tessl` was already on objc2, so the two crates
  now share one binding stack.
- Thread-safety is now asserted narrowly rather than inherited. metal-rs marked
  its handles `Send + Sync` blanket-wide; objc2's `Retained` is deliberately
  neither, because Objective-C thread-safety is per-class. `MetalDevice` and
  `MetalSparse` carry `unsafe impl`s justified against what Apple documents,
  and `tests/stress.rs` exercises the case they exist for.
- `MTLMathMode::Safe` replaces the deprecated `setFastMathEnabled(false)`. A
  comment in the old code recorded that metal-rs 0.29 could not express this;
  objc2-metal can. Verified equivalent rather than assumed: an FNV hash over
  the bits of a 512-row SpMV plus eight fused LIF steps is identical under both
  settings on this host.

### Added

- **`SparseOp::spmm` — batched sparse matrix times dense matrix**, `Y += A·X`
  over `n_vec` vectors, on both CPU arms and Metal. Batching raises arithmetic
  intensity rather than parallelism: each `weights[i]` and `col[i]` is loaded
  once and reused across the batch. Measured 9.6× to 22.5× against the same
  number of separate `spmv` calls on the same Metal device.
- Operands are batch-minor (`x[c * n_vec + v]`), which is what lets adjacent
  GPU threads read and write adjacent addresses and share one `col[i]` stream.
- A batch of one is bit-identical to `spmv` on every backend, asserted on raw
  bit patterns rather than within a tolerance.
- `csr_spmm_kernel` in `spmv.metal`, and `examples/batch_crossover.rs`.
- **`SparseOp::spmv_t` — transposed sparse matrix-vector product**, `y += Aᵀ·x`,
  on both CPU arms and Metal. This is the direction a gradient travels; without
  it there is no backward pass through a sparse layer. Opt in with
  `Device::prepare_with_transpose`; an operator built by `Device::prepare` has
  no reverse index and returns `OpError::TransposeNotPrepared` rather than
  building one implicitly and doubling its own memory.
- `Csc::from_csr_rect` / `Csr::to_csc_rect`, taking an explicit column count.
- `SparseOp::has_transpose`.
- `csc_spmv_t_kernel` in `spmv.metal`.
- `[package.metadata.docs.rs]` targeting `aarch64-apple-darwin` with the `metal`
  feature on. A default x86_64-linux docs build renders the crate with
  `Backend::Metal` permanently unavailable and `backend::metal` absent, which
  documents the half of the crate that is not the interesting half.

### Fixed

- `Csc::from_csr` hardcoded `ncols = csr.nrows()` and **panicked** on any
  rectangular matrix — correct for the square recurrent cell graph it was
  written for, wrong for a sparse layer whose input and output widths differ,
  which is most of them. `from_csr_rect` takes the count and returns
  `CsrError::ColumnOutOfRange`; `from_csr` keeps the square convention and
  delegates. Found by the first non-square caller.
- Three rustdoc warnings from public documentation linking private items. One
  named `SparseOp::prepare`, which exists but is crate-private; the public entry
  point is `Device::prepare`.

## [0.1.0]

Initial extraction from BINN's `binn-core`.

### Added

- `Backend` / `Device` with a single availability gate: a handle is only
  constructible for a substrate that can execute, and `Device::label` reports
  what ran rather than what was asked for.
- CSR SpMV, LIF integrate, and a fused SpMV+LIF kernel on CPU (sequential and
  rayon) and Metal.
- Chunked associative scan over affine maps, bit-identical to a sequential
  left-fold.
- `tolerance_for_spmv` / `tolerance_for_elementwise`, derived bounds rather than
  tuned constants.
- Canary sentinel buffers around every Metal allocation, and a golden output
  fingerprint pinned across releases.

[Unreleased]: https://github.com/bharathvbcr/sparsl
[0.1.0]: https://github.com/bharathvbcr/sparsl
