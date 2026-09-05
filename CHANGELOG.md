# Changelog

All notable changes to `sparsl` are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this crate follows
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Metal line-parallel kernels come in three lane widths from one template.**
  The one-thread-per-row and one-simdgroup-per-row kernel pairs (SpMV in f32,
  binary16 and bfloat16, packed spikes, the transposed product, batched SpMM
  and the fused LIF step) are replaced by a single `LPR`-lane template
  instantiated at 1, 8 and 32 lanes per line, folded with an xor
  `simd_shuffle_xor` butterfly. `RowKernel` gained a `Vec8` tier: an operator
  takes one lane below a mean of 12 stored entries per line, eight up to 64,
  and a whole simdgroup above, and every kernel it dispatches reads that one
  decision, so the fused LIF step now reduces a row exactly as the operator's
  SpMV does instead of always using a simdgroup. All tiers are dispatched with
  uniform threadgroups sized as whole simdgroups. Measured on an M5 Pro at 8M
  non-zeros with the line length swept and twenty dispatches per command
  buffer (paired, order-balanced, per-call medians): the one-lane tier is
  never slower than the previous scalar kernel and up to 1.5× faster
  mid-range; eight lanes overtake it at 12 entries per line and hold
  1.3–1.95× to 64; the simdgroup tier ties at 64 and leads from 128
  (1.89× against 1.55×). The previous `simd_sum` kernel measured within 2% of
  the 32-lane butterfly everywhere. The full table is on `RowKernel` and in
  the README. Cross-backend comparisons take `tolerance_for_spmv` as they
  always did; the randomized packed-spike case in the differential soak
  demanded bit-identity with the CPU reference, which only ever held because
  its generated shapes all landed on the scalar tier, and now checks the
  documented contract instead — bit-identical to the same backend's dense
  product, within tolerance of the reference.

- **Metal completion waits on a shared event instead of a doubling-sleep
  poll.** Every command buffer now signals a monotonic `MTLSharedEvent` value
  after its last encoder, and the host spins on the status for 100 µs, then
  blocks in `waitUntilSignaledValue:timeoutMS:` for the rest of the 60-second
  deadline and re-checks the status when the event fires. The old poll slept
  10, 20, 40 … 1000 µs between status samples, so a kernel finishing at 1.4 ms
  was not observed until 2.4 ms. A paired, order-balanced in-process A/B on an
  M5 Pro (16 rounds, per-call medians) measured the poll at 1.02–1.28× the
  event wait's median `spmv` call time from 64K to 20M non-zeros; a hybrid
  that kept a few short sleeps before the event wait was slower than the pure
  event wait everywhere. Each operator owns its own event and counter, held
  with the scratch its mutex already serialises, so the signalled values are
  monotonic in execution order by construction; `assoc_scan` and the dense
  `lif_integrate` share one device-level timeline. The timeout, quarantine and
  retained-ownership semantics are unchanged, and the bounded-wait regression
  now drives the deadline through a real, never-signalled event. Requires the
  `MTLEvent` feature of `objc2-metal`, which the crate already depended on.

- The randomized CPU/Metal differential soak now honors a bounded,
  strictly parsed `PROPTEST_CASES` (`1..=1_000_000`), rejects zero or malformed
  values instead of silently running a vacuous/default campaign, and persists
  minimized failures in the existing tracked
  `tests/differential.proptest-regressions` corpus. Each generated case now
  selects one of nine operation families: SpMV, SpMM with scratch resize,
  transpose, packed spikes, fused or standalone LIF, affine scan, binary16
  SpMV, or bfloat16 SpMV. A deterministic sentinel executes every selector,
  while the original SpMV strategy remains a fixed 48-case replay lane so its
  historical corpus seeds keep their meaning and the configured 50,000-case
  soak remains 50,000 total operations rather than multiplying by every arm.
  Integration tests previously used proptest's source-parallel default, which
  cannot find Cargo's `src/lib.rs` from `tests/` and therefore warned while
  discarding the replay seed. Its dev dependency is pinned to `proptest` 1.6.0,
  the last rand-0.8 line, with unused fork/timeout features disabled. Proptest 1.7+
  resolves a rand-0.9 WASI branch whose current edition-2024 manifest cannot be
  parsed by Cargo 1.82 even when that target is not built; newer proptest itself
  also raises the compiler floor. CI now checks both locked metadata resolution
  and all targets with the declared MSRV.

- All five performance examples (`crossover`, `narrow_crossover`,
  `spike_crossover`, `batch_crossover`, and `scan_crossover`) now use bounded
  warmup plus eight paired, order-balanced sample rounds by default instead of
  choosing an optimistic minimum from two attempts. They report medians and
  max/min spread for each arm and the paired speedup, reset accumulating outputs
  before every timed call, black-box relevant outputs, and print
  source/build/host/device/sampling provenance. Every harness runs an untimed
  correctness preflight before sampling: scalar SpMV uses an independent scalar
  oracle (exact on CPU and within the public bound on GPU), narrow weights use
  their format-specific public bounds, packed spikes and batched SpMM must match
  their dense/repeated forms bit-for-bit, and each scan arm must match the
  sequential reference exactly on CPU or within the public derived bound on
  GPU. The scan harness additionally runs a full-length, exactly representable
  counting scan, so the conservative floating-point bound cannot hide a missing
  tail write. A missing write can no longer be reported as an optimisation. The
  existing commands are unchanged; bounded overrides are available through
  `SPARSL_BENCH_WARMUP_ROUNDS`, `SPARSL_BENCH_ROUNDS`, and
  `SPARSL_BENCH_ITERS`. The scan harness also accepts a bounded custom sweep
  through `SPARSL_BENCH_SCAN_SIZES`; `SPARSL_BENCH_CPU_ONLY=1` bypasses device
  discovery for serialized CPU/GPU measurement campaigns.

- **The Metal prefix scan no longer reformats on the host; retained historical
  measurements recorded a 1.28–1.50x improvement.** `Device::assoc_scan`
  converted `&[State]` to
  `Vec<(f32, f32)>`, `MetalDevice::assoc_scan` flattened that to a `Vec<f32>`
  for the upload, and both ran again in reverse on the way back — four full
  passes over the data inside the timed region, which the CPU arms never paid.
  Measured alone at n = 4.2M they cost 2.64 ms of a 13.5 ms call.

  `State` is now `#[repr(C)]`, so a `&[State]` already *is* the
  `[a0, b0, a1, b1, …]` buffer the kernel indexes: the upload reads the caller's
  slice and the readback fills the output vector directly. The 4.2M case went
  from 13.53 ms to 8.99–10.55 ms. The README records the measured values and
  their historical-evidence status; Metal still lost to the sequential fold at
  every measured size.

  No public API changed: `MetalDevice` is `pub(crate)`, and
  `Device::assoc_scan` keeps its `&[State] -> Result<Vec<State>, OpError>`
  signature.

### Added

- `OpError::Execution` and a shared Metal completion/quarantine policy. Every
  submitted command now owns its encoded resources, and host output is read
  only after it reaches `Completed`; 60 seconds of observed non-terminal state
  after `commit()` returns triggers quarantine. Device-reported terminal
  failures retain the operation name and Metal error detail instead of
  returning plausible stale scratch as success. A non-terminal timeout keeps
  the command and its resources retained until process exit, quarantines Metal
  process-wide, and makes later device discovery, preparation and mutation fail
  closed. An atomic permit state makes quarantine publication non-blocking and
  refuses every later submission admission. Work admitted before publication
  may still return from its opaque `commit()` call afterward; Metal provides no
  cancellable primitive with which to promise stronger physical ordering.
  `SparsePlanError::Backend` reports the preparation side of that quarantine;
  watchdog-bounded regressions use an uncommitted command buffer to prove the
  completion loop is bounded, force every public Metal entry point through an
  isolated quarantine child with an execution marker, drive deadline-edge
  transitions through the production state machine, and hold a pre-admitted
  submission while proving quarantine publication cannot block. The 60-second
  support boundary begins after `commit()` returns; opaque Metal selector calls
  cannot be pre-empted in-process, and longer legitimate workloads must be
  chunked.

- The deterministic CPU equivalence campaign now sends every documented empty
  and thread-boundary shape plus 128 fixed-seed random CSR shapes through SpMV,
  SpMM, transpose SpMV, packed-spike SpMV, fused and standalone LIF, and affine
  scan. It varies batch widths and LIF parameters, starts mutable outputs from
  non-zero state, and requires bit-for-bit sequential/parallel agreement.

- **Two guards on `State`'s layout**, which the zero-reformat/direct typed
  byte-copy upload above makes load-bearing. A `const` assertion in `scan.rs`
  fails the build if `State` is
  no longer two packed `f32`s; `state_layout_matches_flat_f32` in
  `tests/scan_backend.rs` fails if the fields are *reordered*, which a size
  check cannot see. Verified by injecting a swapped field order and confirming
  both the new test and four existing scan tests go red.

- **`documentation` in `Cargo.toml`, and the published docs linked from the
  README.** The 0.1.1 docs work landed on a page no GitHub reader was pointed
  at. The README now carries crates.io, docs.rs, CI and license badges, a nav
  line to the API docs, the crate page, the changelog and `tessl`, and an
  `API docs` row recording that the page is built on `aarch64-apple-darwin`
  with `--features metal` — which is why `Backend::Metal` is documented there
  rather than `cfg`'d away.

- `scan_magnitude_envelope`, which derives the condition scale consumed by
  `tolerance_for_scan` across both affine fields, including multiplier growth
  and cancellation. The existing tolerance signature remains source-compatible.

### Fixed

- SpMM now dispatches a two-dimensional `(vector, row)` grid, removing the
  per-output device division/modulo. Both input and output linearization widen
  before multiplication, and grow-only output scratch places its canary at the
  logical endpoint of each call so shrink/regrow sequences cannot hide a tail
  write inside unused capacity. Host-side input/output element-count overflow
  now returns `OpError::SizeOverflow` with the exact multiplication that failed,
  before reading an input or mutating the caller's output, instead of fabricating
  contradictory slice-length fields.
- Metal scan dispatch now rejects a length whose uniform power-of-two padding
  would exceed the kernel's `u32` thread-ID range. The fused SpMV+LIF kernel now
  widens padded row geometry and CSR lane/stride arithmetic before operating,
  preventing valid near-`u32::MAX` offsets from wrapping in device code.
- Infallible sparse constructors now reject unrepresentable dimensions before
  arithmetic can wrap: `Csr::empty` and `Csc::empty` check pointer-table
  `n + 1`, while `Csr::from_adjacency` checks its `usize` nnz sum and every
  narrowing into a `u32` row offset. The signatures are unchanged; their
  documented panic contract now covers sizes the formats cannot represent.
- The README Status row said `0.1.0` while crates.io served `0.1.1`. Corrected,
  and linked to the crate page.
- `Csc::from_csr_rect` now validates CSR structure even when the CSR came from
  `from_parts_unchecked`; malformed row pointers previously passed through or
  panicked during conversion instead of returning their documented `CsrError`.
- The cross-backend test comparator now rejects one-sided NaNs, mismatched
  infinities, non-finite tolerances, and inconsistent LIF slice lengths instead
  of allowing unordered IEEE comparisons or trailing data to disappear from
  its checks.
- Public rustdoc no longer links to a private layout-assertion constant, so
  `RUSTDOCFLAGS='-D warnings' cargo doc --features metal --no-deps` is a valid
  release gate. The assertion is anonymous, avoiding an MSRV-only dead-code
  warning while preserving the compile-time layout proof.

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

- **bfloat16 resident quantisation and compact plain-SpMV encoding**, alongside
  binary16. `Device::prepare_bf16`, or `Device::prepare_with` when the format is
  a variable. The compact encodings are indistinguishable in speed — both use 2
  bytes per streamed SpMV weight — so the choice is numerical: binary16 is 8x
  finer, while bfloat16's largest finite value is 3.39e38 instead of 65504.
  Metal also retains the quantised values widened to f32 for the other weighted
  operations; the 2-byte figure is not the operator's total resident footprint.
- `WeightPrecision`, replacing the boolean `prepare` threaded through. It
  selects the quantisation applied when weights become resident; on Metal it
  also selects plain SpMV's compact buffer. A second format made the boolean
  wrong: two flags would have admitted a state meaning "both binary16 and
  bfloat16", which no operator can be in.
- `tolerance_for_spmv_narrow`, the one derivation both narrow bounds delegate
  to, parameterised by the format's epsilon. A third narrow type would add a
  `WeightPrecision` variant and no new formula.
- **IEEE binary16 weight quantisation.** `Device::prepare_f16` narrows the
  resident values; plain `Backend::Metal` SpMV then streams 2 bytes per
  non-zero instead of 4. `SparseOp::weight_precision` reports the compact
  execution representation that plain SpMV can use on that backend.
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
  replaced by the table it was wrong about. `examples/scan_crossover.rs` keeps
  that comparison executable with bounded current defaults.

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
  objc2-metal can. Apple defines Safe mode as disabling unsafe floating-point
  optimisations; it does not promise the CPU's exact operation sequence. On
  this host, an FNV hash over the bits of a 512-row SpMV plus eight fused LIF
  steps was identical under both settings, while the contraction regression
  confirms that Safe mode can still fuse a multiply-add.

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
