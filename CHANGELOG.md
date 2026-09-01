# Changelog

All notable changes to `sparsl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`documentation` in `Cargo.toml`, and the published docs linked from the
  README.** The 0.1.1 docs work landed on a page no GitHub reader was pointed
  at. The README now carries crates.io, docs.rs, CI and license badges, a nav
  line to the API docs, the crate page, the changelog and `tessl`, and an
  `API docs` row recording that the page is built on `aarch64-apple-darwin`
  with `--features metal` — which is why `Backend::Metal` is documented there
  rather than `cfg`'d away.

### Fixed

- The README Status row said `0.1.0` while crates.io served `0.1.1`. Corrected,
  and linked to the crate page.

## [0.1.1] — 2026-09-01

Documentation only. No code or API changes; the compiled crate is identical to
0.1.0.

### Added

- **A quickstart on the docs.rs landing page, as a real doctest** rather than a
  `no_run` sketch — the CPU backend runs anywhere, so it executes on every check
  and passes with and without `metal`. It shows the try_new-then-fall-back
  shape deliberately, since that pattern is what the availability gate exists to
  make possible.
- A module map, and a feature table that says plainly what `cuda` is: a
  declaration of intent that provides nothing, with `Backend::Cuda` left
  unconstructible. Burying that in a feature name would repeat the defect this
  crate was extracted to prevent.
- Contributor notes on why the tolerance functions are bounded from above as
  well as below, why every kernel-written buffer carries a sentinel tail, and
  why `build.rs` names `spmv.metal` explicitly.
- Real module docs for `buffer`, `rng`, `sparse` and `time`, which had one line
  each.

## [0.1.0] — 2026-08-31

First published release. The crate has not been on crates.io before, so
everything in this file ships in it — the sections below were written while the
work was unreleased and are kept as-is rather than reflowed, because they record
why each piece landed.

### Added

- **Bitpacked spike vectors.** `SparseOp::spmv_spikes` takes 32 spikes per
  `u32`; `crate::spikes` packs and unpacks them, and `fused_spmv_lif` output
  feeds straight in. The gathered operand shrinks 32x, and unlike narrow
  weights the win *grows* with `n` — 0.90-1.07x at 10,000 cells where the f32
  vector already fits in cache, **1.30-1.48x at 50,000** where it does not.
  That also explains the narrow-weight result: halving the weights moved
  streamed traffic, and the gather it left alone is where the cost was.
- The spike path is **exact**, so there is no tolerance for it. A spike is 0 or
  1, both exact in f32, and both paths decode the bit and multiply — so it is
  bit-identical to the dense one, and bit-identical *across backends*, which
  the dense SpMV is not. Metal's `fma` contraction cannot bite when the
  multiplier is exactly 0 or 1, because the product has no intermediate
  rounding to skip.

- **bfloat16 weight storage**, alongside binary16. `Device::prepare_bf16`, or
  `Device::prepare_with` when the format is a variable. The two are
  indistinguishable in speed — both store 2 bytes — so the choice is numerical:
  binary16 is 8x finer, bfloat16 reaches 3.39e38 instead of stopping at 65504.
- `WeightPrecision`, replacing the boolean `prepare` threaded through. A second
  format made the boolean wrong: two flags would have admitted a state meaning
  "both binary16 and bfloat16", which no operator can be in.
- `tolerance_for_spmv_narrow`, the one derivation both narrow bounds delegate
  to, parameterised by the format's epsilon. A third narrow type would add a
  `WeightPrecision` variant and no new formula.
- **IEEE binary16 weight storage.** `Device::prepare_f16` narrows the weights;
  `Backend::Metal` then streams 2 bytes per non-zero instead of 4.
  `SparseOp::weights_are_f16` reports the storage an operator actually has.
- `crate::half`: binary16 encode/decode as raw `u16`, because Rust's `f16` is
  unstable and this crate's MSRV is 1.82. Verified exhaustively — all 65536
  binary16 values round-trip — and the host encoder is cross-checked against
  Metal's own `half` through the real SpMV.
- `tolerance_for_spmv_f16`, the derivation the README named as the blocker for
  narrow types. It is `tolerance_for_spmv` plus a quantisation term rather than
  a separate formula, so the two cannot drift; the quantisation term dominates
  by ~8192x, which is `HALF_EPSILON / f32::EPSILON`.

### Changed

- `tolerance_for_scan` is public, and `tests/scan_backend.rs` asserts against it
  rather than against a private copy of the same formula. Every other
  cross-backend operation already exported its bound; the scan's lived only in
  the test, so a caller comparing two backends could not reach it.
- The prefix scan's performance is measured rather than predicted.
  `Backend::Metal` is **slower than the sequential CPU fold at every size** —
  0.13x at 0.1M rising to 0.45x at 4.2M. `scan.rs` previously asserted that a
  two-level tree scan "would deliver a real speedup"; that claim is now
  replaced by the table it was wrong about. `examples/scan_crossover.rs`
  reproduces it.

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

- **`Device::assoc_scan` — the affine-map scan on Metal as well as CPU.** A
  two-level Hillis-Steele scan in three dispatches. It reassociates, so it is
  not bit-identical to the CPU arms; that is this crate's stated rule
  (reproducibility within a backend, never across) rather than an exception,
  and the method's docs say so. Two runs on one device agree byte for byte.
- `build.rs` declaring `src/kernels/spmv.metal` an explicit build input. A
  mutation run edited that file, rebuilt nothing, and reported the mutant had
  survived — a check that never ran, looking exactly like one that ran and
  passed.
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

### Initial extraction from BINN's `binn-core`

#### Added

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

[0.1.0]: https://github.com/bharathvbcr/sparsl/releases/tag/v0.1.0
