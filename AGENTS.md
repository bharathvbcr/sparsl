# AGENTS.md

Guidance for coding agents (and people) changing `sparsl`.

## Role & Architecture

`sparsl` is the **dedicated Sparse Computing extension of [ojas](https://github.com/bharathvbcr/ojas)**, serving as the sparse counterpart to [tessl](https://github.com/bharathvbcr/tessl) (dense Metal 4 acceleration).

It provides deterministic sparse matrix representations (CSR, CSC), sparse linear algebra (SpMV), chunked prefix scans over affine maps, and event-driven simulation. In neural graph execution, sparsl provides sparse gradient masking, dynamic sparse recompute maps, and event gating (`st-gate`).

## Invariants a change must keep

- **The Honesty Invariant:** `Device::label()` reports the substrate that **executed**, not the one that was requested. There is no silent fallback: asking for an unavailable backend returns `BackendUnavailable` with an explicit reason.
- **Fail-Closed Validation:** Every path to a kernel validates connectivity against matrix dimensions (`Device::prepare` checks that all column indices are in range `[0, cols)`). An invalid sparse plan returns `SparsePlanError` immediately.
- **Bit-Identical Determinism *within* a backend, a tolerance *across* backends.** Chunked parallel
  scans and SpMV accumulations must produce bit-identical results for the same seed on the **same**
  substrate: `assoc_scan` left-folds rather than reassociating precisely so a parallel scan matches
  a sequential one bit for bit, and nothing here autotunes, because a runtime-chosen kernel changes
  the reduction order and therefore the floats. It does **not** hold across substrates (corrected
  2026-11-01; the previous wording claimed bit-identity "across sequential, parallel, and GPU
  execution paths", which the code contradicts). CPU and GPU reduce in different orders and differ
  in the last few ulps — see `src/lib.rs:16-18` and the four deliberate error bounds
  `backend::tolerance_for_spmv`, `_f16`, `_bf16` and `_narrow` (`src/backend/mod.rs:563`, `:682`,
  `:698`, `:713`), which exist only because cross-backend equality is not bitwise. Compare a GPU
  result against a CPU one with the matching tolerance, never with `==`.
- **No Dense Framework Duplication:** Dense tensors, autograd tape, optimizer matrix states, and model parameters belong in `ojas`. `sparsl` focuses strictly on sparse and scan compute kernels.
- **Safe Rust where it can be, audited `unsafe` where it cannot.** This crate does **not** set
  `#![forbid(unsafe_code)]` — that attribute appears nowhere in `src/`, and `Cargo.toml` has no
  `[lints]` entry for `unsafe_code` (corrected 2026-11-01; the previous wording of this line claimed
  the attribute was in force, which was false and would have misled anyone reasoning about this
  crate's memory safety). New code stays safe and bounds-checked by default, and `unsafe` is
  confined to the three existing surfaces below. Adding a fourth needs a SAFETY comment stating the
  invariant and why safe Rust cannot express it.

  The real `unsafe` surface, as of 2026-11-05. The 2026-11-01 inventory named these
  same three surfaces; the bitpacked spike commit moved the sites and added three
  shared-buffer copies, so the line numbers below are the ones `git grep unsafe`
  finds on that commit. The new CPU spike path does not add a site: it splits the
  output with `split_at_mut`.

  1. **Parallel output writes in the CPU sparse and LIF kernels** — `src/backend/mod.rs`, five sites
     (`:2300`, `:2392`, `:2436`, `:2455`, `:2538`). Each takes `y.as_mut_ptr() as usize`, hands that
     address to a rayon `for_each`, and reconstructs a `&mut f32` (or, at `:2392`, a
     `slice::from_raw_parts_mut`) per index. The invariant every site rests on is that
     `parallel_line_partitions` hands out **disjoint** index ranges, so no two threads write one
     element; `:2538` extends that to three arrays (`v`, `theta`, `spikes`) at the same row index.
     This is what the "write gathers, not scatters" discipline looks like when the disjointness is
     proved by the partitioner rather than by the type system. Changing
     `parallel_line_partitions` is therefore a memory-safety change, not a scheduling tweak.
  2. **`Send`/`Sync` assertions over Metal objects** — `src/backend/metal.rs`, six `unsafe impl`s:
     `RetainedCommandBuffer` (`:125-126`), `MetalDevice` (`:685-686`), `MetalSparse` (`:701-702`).
     The file's own header (`:28`) notes these go beyond what Apple documents and rest on objc2's
     markers and refcounting.
  3. **objc2 / Metal FFI calls** — `src/backend/metal.rs` (`:1033`, `:1405`, `:1427`, `:3263`,
     `:3286`, `:3321`, `:3350`, `:3385`, `:3429`, `:3458`), including buffer creation,
     `setBuffer_offset_atIndex`, shared-buffer copies, and the segment and `u8` spike reads
     on the resident spike path.

  Note the separate hazard recorded at `src/backend/mod.rs:33-44`: the `backend::metal` module is
  `pub(crate)` precisely because `MetalDevice::prepare` trusts its `SparseShape` and the kernels
  index `x[col[i]]` **unchecked**. An audit reached an out-of-bounds device read from
  caller code that was itself safe. Reachability is the enforcement — `SparseOp` is the only route
  to a sparse kernel — so making that module public again would reopen the hole.

## Toolchain floor

`rust-version = "1.100"` in `Cargo.toml` is the single source of the floor; the CI `msrv` job reads it
from there. It was 1.82 (measured 2026-11-01) until the owner raised it on 2026-11-12, so any older note
that says 1.82 is history. `.cargo/config.toml` sets a 3-day dependency cooldown for resolution. `f16`
is still unstable on 1.100, so `src/half.rs` keeps binary16 as raw `u16` bits.

## Testing & Verification

```sh
# Core test suite
cargo test

# With native Metal backend on Apple Silicon
cargo test --features metal
```
