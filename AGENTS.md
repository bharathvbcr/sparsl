# AGENTS.md

Guidance for coding agents (and people) changing `sparsl`.

## Role & Architecture

`sparsl` is the **dedicated Sparse Computing extension of [ojas](https://github.com/bharathvbcr/ojas)**, serving as the sparse counterpart to [tessl](https://github.com/bharathvbcr/tessl) (dense Metal 4 acceleration).

It provides deterministic sparse matrix representations (CSR, CSC), sparse linear algebra (SpMV), chunked prefix scans over affine maps, and event-driven simulation. In neural graph execution, sparsl provides sparse gradient masking, dynamic sparse recompute maps, and event gating (`st-gate`).

## Invariants a change must keep

- **The Honesty Invariant:** `Device::label()` reports the substrate that **executed**, not the one that was requested. There is no silent fallback: asking for an unavailable backend returns `BackendUnavailable` with an explicit reason.
- **Fail-Closed Validation:** Every path to a kernel validates connectivity against matrix dimensions (`Device::prepare` checks that all column indices are in range `[0, cols)`). An invalid sparse plan returns `SparsePlanError` immediately.
- **Bit-Identical Determinism:** Chunked parallel scans and SpMV accumulations must produce bit-identical results across sequential, parallel, and GPU execution paths.
- **No Dense Framework Duplication:** Dense tensors, autograd tape, optimizer matrix states, and model parameters belong in `ojas`. `sparsl` focuses strictly on sparse and scan compute kernels.
- **Safe Rust:** `#![forbid(unsafe_code)]` in library code. All memory access and indexing must remain bounds-checked and memory-safe.

## Testing & Verification

```sh
# Core test suite
cargo test

# With native Metal backend on Apple Silicon
cargo test --features metal
```
