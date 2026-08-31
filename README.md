# sparsl

Deterministic sparse and scan compute kernels for event-driven simulation, with fail-closed CPU and GPU backends.

Extracted from the numeric core of a spiking-network research harness. The kernels themselves are small and unglamorous — CSR sparse matrix-vector multiply, a leaky integrate-and-fire membrane update, a chunked prefix scan over affine maps, structure-of-arrays column buffers, a seeded RNG. What the crate is about is the two properties those kernels are held to: **results you can reproduce bit for bit**, and **a backend handle that cannot lie about where it ran**.

| | |
| --- | --- |
| **Status** | `0.1.0` — Metal verified, CUDA declared but unavailable |
| **Tests** | 56 passing; suite verified by a 20-case mutation campaign |
| **Platform** | Any CPU; Metal on macOS behind `--features metal` |
| **License** | MIT OR Apache-2.0 |

---

## 🗂️ Architecture

Every path to a kernel runs through two gates. A `Device` exists only for a substrate that can execute, and a `SparseOp` exists only for connectivity validated against its column count. Neither an unavailable backend nor an unchecked sparse matrix is reachable from the public API — both are error values, not runtime surprises.

```mermaid
flowchart LR
    accTitle: The Two Gates Before Any Kernel
    accDescr: A caller passes an availability gate that refuses unavailable backends and a validation gate that rejects out-of-range column indices before reaching any of the substrates that can execute.

    caller([👤 Caller])
    try_new[Device::try_new]
    availability{Backend available?}
    unavailable[❌ BackendUnavailable<br/>CUDA always lands here]
    prepare[Device::prepare]
    validate{Every col index in range?}
    rejected[❌ SparsePlanError]
    sparse_op([✅ SparseOp])

    subgraph substrates ["⚡ Substrates that can execute"]
        cpu_sequential[CPU sequential]
        cpu_parallel[CPU parallel]
        metal_gpu[☁️ Metal GPU]
    end

    caller --> try_new --> availability
    availability -->|no, with a reason| unavailable
    availability -->|yes| prepare
    prepare --> validate
    validate -->|no| rejected
    validate -->|yes| sparse_op
    sparse_op --> substrates

    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef primary fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class unavailable,rejected danger
    class availability,validate primary
    class sparse_op success
```

| Module | What it holds |
| --- | --- |
| `backend` | `Device`, `SparseOp`, availability, validation, CPU kernels |
| `backend::metal` | Metal device, pipelines, resident buffers, dispatch |
| `backend::cuda` | Why CUDA is declared and never available, and how to land it |
| `sparse` | `Csr` and its `Csc` reverse index |
| `scan` | Chunked prefix scan over affine maps, bit-exact against sequential |
| `simd` | Lane-shaped leak/integrate, no intrinsics, no `unsafe` |
| `buffer`, `rng`, `time` | SoA columns, seeded ChaCha, tick type |

---

## 🛡️ The honesty invariant

`Device::label()` reports the substrate that **executed**, not the one that was requested. There is no fallback: asking for an unavailable backend returns `BackendUnavailable` with a reason.

This is a regression guard, not decoration. The code this crate was extracted from carried a `use_gpu: bool` that no dispatch path ever read. A "GPU" handle and a "CPU" handle ran byte-identical `rayon` code, benchmarks reported roughly 1.00x speedups as genuine cross-substrate results, and generated reports printed CPU timings under a GPU heading.

```mermaid
flowchart TB
    accTitle: Backend Availability Gate
    accDescr: try_new refuses any backend that reports itself unavailable and then verifies that the substrate actually opened is the one requested, so a handle can never carry a label it does not run on.

    request([📥 try_new for a backend])
    check_reason{Reports unavailable?}
    refuse[❌ Return BackendUnavailable]
    open_substrate[⚙️ Open the substrate]
    label_matches{Inner matches the label?}
    mismatch[❌ Refuse the mismatch]
    handle([✅ Handle labelled with what runs])

    request --> check_reason
    check_reason -->|yes, with a reason| refuse
    check_reason -->|no| open_substrate
    open_substrate --> label_matches
    label_matches -->|no| mismatch
    label_matches -->|yes| handle

    classDef danger fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d
    classDef success fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class refuse,mismatch danger
    class handle success
```

One gate serves every backend, and that is deliberate. An earlier version gave each backend its own error arm; a mutation replacing one arm's `Err` with a CPU fallback passed the entire test suite, because on a machine where Metal works that arm is unreachable and no test on such a machine can execute it. Untestable code is not made safe by more tests, so the branch was removed rather than covered.

### Backend status

| Backend | Availability | Notes |
| --- | --- | --- |
| `CpuSequential` | Always | The determinism reference |
| `CpuParallel` | Always | Asserted **bit-identical** to sequential, not merely close |
| `Metal` | `--features metal`, macOS, real device | Implemented and verified against the reference |
| `Cuda` | **Never** | No dispatch written; none could be verified on Apple silicon |

`Backend::Cuda` fails closed with a reason and is omitted from `available_backends()`. `src/backend/cuda.rs` carries the steps to land it; the last one is that `tests/differential.rs` fuzzes any backend the moment it reports available, so that flag must not be flipped before the suite passes.

---

## ⚙️ How a dispatch works

Connectivity and weights are uploaded once. Only the input vector and the membrane state cross the boundary per call.

```mermaid
sequenceDiagram
    accTitle: Prepare Once Then Dispatch Many
    accDescr: Validation and the upload of connectivity and weights happen a single time at prepare, while each call transfers only the input vector and membrane state and then checks the sentinel tails.

    participant caller as 👤 Caller
    participant op as ⚙️ SparseOp
    participant gpu as ☁️ Metal device

    Note over caller,gpu: prepare — once per operator
    caller->>op: prepare(csr, ncols, weights)
    op->>op: Check every col index is in range
    op->>gpu: Upload row_ptr, col, values
    op->>gpu: Allocate scratch with sentinel tails

    Note over caller,gpu: spmv — every call
    caller->>op: spmv(x, y)
    op->>gpu: Copy x and incoming y
    op->>gpu: Dispatch csr_spmv_kernel
    gpu-->>op: Completion
    op->>op: Assert sentinel tails intact
    op-->>caller: Updated y
```

Two decisions in that diagram were forced by measurement rather than taste.

**Weights belong to the operator.** As a per-call argument they forced an 80 MB host-to-device copy before every dispatch on a 20M-non-zero operator, and Metal lost to `rayon` at every size on that copy alone. Moving them onto the operator took the 20,000-row case from 3.03 ms to 1.01 ms.

**Column indices are validated once, then trusted.** The GPU kernels index `x[col[i]]` with no range check, because a bounds check per non-zero costs more than the multiply it guards. That is sound only because `prepare` proves every stored index is in range before a byte is uploaded, and preparing is the only route to a sparse kernel. An out-of-range `Csr` is a rejected `SparsePlanError`, never an out-of-bounds read of device memory.

---

## 🔬 What the kernels compute

Each tick a cell decays its membrane, adds synaptic current, and either stays below threshold or fires — resetting the membrane and raising its own threshold.

```mermaid
stateDiagram-v2
    accTitle: LIF Cell Step Semantics
    accDescr: A cell integrates decayed membrane plus synaptic current each tick, then either remains subthreshold or fires and resets while raising its adaptive threshold.

    [*] --> subthreshold
    subthreshold --> integrated: v = v * decay + current
    integrated --> subthreshold: v < theta
    integrated --> fired: v >= theta
    fired --> subthreshold: v = v_reset, theta += delta_theta

    note right of integrated
        This comparison is where cross-backend rounding becomes visible. A membrane within an ulp of theta fires on one substrate and not the other, turning a rounding difference into a boolean one.
    end note
```

That note is the reason the differential suite treats spikes inside a tolerance band around threshold as legitimately ambiguous, and demands exact agreement everywhere else.

### The scan, on both substrates

`Device::assoc_scan` runs the affine-map scan on whichever substrate the handle names — and this is the one primitive where the arms deliberately disagree.

The CPU arms are bit-identical to a sequential left-fold. They buy that by making phase 1 a *complete* sequential fold, roughly `2n` combines to replace `n`, measured at 1.08x. The Metal arm is a two-level Hillis-Steele scan that reassociates, so it is genuinely parallel and is **not** bit-identical to the CPU arms.

That is the rule this crate already states, not an exception to it: reproducibility holds *within* a backend and never across one. Two runs on the same device agree byte for byte. If you need output bit-identical to the sequential fold, ask for a CPU backend — a GPU one cannot give it, and `Device::assoc_scan` says so in its docs rather than by silently differing.

Correctness rests on two exact tests rather than a tolerance. With `a = 1, b = 1` every prefix is exactly `i + 1`; with `a = 2, b = 0` every prefix is exactly `2^(i+1)`. Both are integers f32 represents exactly, so a misapplied block offset is plainly wrong with nowhere to hide — which matters, because a tolerance derived as `n · eps · max` is around 10% at n = 100000 and absorbs almost anything.

### The batched product

`SparseOp::spmm` computes `Y += A · X` for `n_vec` dense vectors in one dispatch.

A single-vector SpMV performs one multiply-add per index it loads, which is not enough arithmetic to cover the load — that is why the GPU arm needs a large problem before it overtakes rayon at all. Batching reuses each `weights[i]` and each `col[i]` across every vector, which is a change in arithmetic intensity rather than in parallelism.

`x` and `y` are **batch-minor**: `x[c * n_vec + v]` is column `c` of vector `v`. That is the opposite of storing each vector contiguously, and it is the entire point — adjacent GPU threads then differ only in `v`, so they read adjacent addresses, write adjacent addresses, and share the same `col[i]` sequence as a broadcast. Batch-major storage turns all three into scattered access.

A batch of one is **bit-identical** to `spmv` on every backend, not merely within tolerance. Both dispatch to the same scalar path, and the test asserts equality of raw bit patterns — which fails on any reassociation or stray fused multiply-add that a tolerance would absorb.

| n | nnz | n_vec | repeated `spmv` (ms) | `spmm` (ms) | same-backend speedup |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 50K | 8 | 2.360 | 0.247 | **9.6×** |
| 1,000 | 50K | 32 | 8.009 | 0.355 | **22.5×** |
| 5,000 | 1.25M | 8 | 3.013 | 0.358 | **8.4×** |
| 5,000 | 1.25M | 32 | 10.926 | 0.945 | **11.6×** |
| 10,000 | 5M | 32 | 19.106 | 1.464 | **13.1×** |

Metal GPU, `cargo run --release --features metal --example batch_crossover`. These compare `spmm` against `n_vec` separate `spmv` calls **on the same backend**, so machine load affects both arms alike. Cross-backend numbers are deliberately absent: they were taken at load average 49, where the rayon arm is contending for cores and Metal is not, and a crossover measured under that says more about the machine than the kernels.

### The transposed product

`SparseOp::spmv_t` computes `y += Aᵀ · x` — the direction a gradient travels. Given `dy` over a sparse layer's outputs it produces `dx` over its inputs, which is what a learning rule needs and what a forward-only SpMV cannot give.

It walks a CSC reverse index rather than materialising `Aᵀ`. Each CSC entry names the CSR row it came from and the slot its value occupies in the *forward* weight table, so one value table serves both directions and `set_weights` updates them together — there is no second copy to fall out of step.

```mermaid
flowchart LR
    accTitle: Forward and Transposed Products Share One Weight Table
    accDescr: The CSR index drives the forward product and the CSC reverse index drives the transposed one, but both read the same values array, so a weight update reaches both directions at once.

    w["📊 values[nnz]<br/>(one table)"]
    csr["➡️ CSR row_ptr / col"]
    csc["⬅️ CSC col_ptr / row / edge_idx"]
    fwd["y[nrows] += A · x[ncols]"]
    bwd["y[ncols] += Aᵀ · x[nrows]"]

    csr --> fwd
    csc --> bwd
    w --> fwd
    w --> bwd

    classDef shared fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef idx fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef out fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d

    class w shared
    class csr,csc idx
    class fwd,bwd out
```

The reverse index costs as much memory as the forward one, so it is opt-in: build the operator with `Device::prepare_with_transpose`. An operator from `Device::prepare` returns `OpError::TransposeNotPrepared` rather than building one implicitly, because a method that silently doubles an operator's footprint the first time it is called is worse than one that says it cannot.

Correctness is gated on the inner-product identity `⟨A·x, y⟩ == ⟨x, Aᵀ·y⟩`, which is the definition of the transpose. A wrong-but-plausible implementation — indices swapped, the weight table read by CSC position instead of `edge_idx` — still looks like a sparse product and still fails that identity. Both mutations were injected and both were caught.

---

## 🎯 Reproducibility, and where it stops

Same seed, same backend, same bits. The parallel scan left-folds instead of reassociating specifically so it matches a sequential scan bit for bit, and the two CPU arms are asserted bit-identical. Nothing here autotunes: a kernel picked by runtime benchmark differs per machine, which changes the reduction order, which changes the floats.

Across backends it does not hold. The crate names the three causes rather than implying they do not exist.

| Cause | Effect | Covered by |
| --- | --- | --- |
| Reduction order | GPU and CPU row sums differ in the last ulps | `tolerance_for_nnz_per_row`, the `n · eps · max|term|` bound with margin |
| Multiply-add contraction | Metal fuses `v * decay + current`, rounding once where the CPU rounds twice | `tolerance_for_elementwise`, pinned by `tests/fma_contraction.rs` |
| Threshold proximity | Either of the above can flip a spike, not merely perturb a float | Spike flips permitted only inside the tolerance band |

`CompileOptions::set_fast_math_enabled(false)` does **not** prevent the contraction — that was measured, not assumed. `tests/fma_contraction.rs` requires every non-spiking GPU membrane to match one of the two roundings bit for bit, so the tolerance is sized for an identified cause rather than for an unexplained gap.

`tests/golden.rs` pins the reference's actual output bits. Any change to summation order, iteration order, the RNG, or the LIF update moves the fingerprint — which is exactly the signal that a downstream replay hash has become invalid.

---

## 📦 Usage

```toml
[dependencies]
sparsl = { version = "0.1", features = ["metal"] }
```

```rust
use sparsl::{Backend, Csr, Device, LifParams};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let csr = Csr::from_adjacency(&[vec![1, 2], vec![0], vec![0, 1]]);
    let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Prefer the GPU, but never silently pretend to have one.
    let device = Device::try_new(Backend::Metal).unwrap_or_else(|why| {
        eprintln!("falling back to CPU: {why}");
        Device::cpu_parallel()
    });

    // Validates the CSR against ncols and uploads it once.
    let mut op = device.prepare(&csr, 3, &weights)?;

    let x = vec![0.5, 1.0, 1.5];
    let mut y = vec![0.0; 3];
    op.spmv(&x, &mut y)?;

    let params = LifParams::new(0.9, 0.0, 0.1)?;
    let (mut v, mut theta, mut spikes) = (vec![0.0; 3], vec![1.0; 3], vec![false; 3]);
    op.fused_spmv_lif(&x, &mut v, &mut theta, &mut spikes, params)?;

    // Connectivity is fixed; values move on the timescale of learning.
    op.set_weights(&[1.0, 2.0, 3.0, 4.0, 6.0])?;

    println!("ran on {}", op.label());
    Ok(())
}
```

This snippet is `examples/readme.rs`, built on every check so it cannot drift from the crate. Run it without `--features metal` and it prints the invariant working:

```text
falling back to CPU: backend `Metal GPU` is unavailable: sparsl was built without the `metal` cargo feature
ran on CPU parallel (rayon)
```

---

## 📊 Performance

`cargo run --release --features metal --example crossover` — Apple M5 Pro, CSR at 5% density, milliseconds per SpMV. Fastest arm per row in bold.

| N | nnz | CPU sequential | CPU parallel | Metal |
| ---: | ---: | ---: | ---: | ---: |
| 1,000 | 50K | **0.022** | 0.109 | 0.187 |
| 5,000 | 1.25M | 0.718 | 0.278 | **0.239** |
| 10,000 | 5M | 3.048 | **0.388** | 0.491 |
| 20,000 | 20M | 12.496 | 1.239 | **1.007** |

Read this cautiously. The example exercises every arm before timing any of them — without that ramp the first-timed arm pays for the GPU clock ramp-up and later arms do not, which once produced the physically impossible result that adding a host memcpy per iteration made it faster. It then times each arm twice in opposite orders and reports the spread between passes; the spread here was 1.24 to 1.66, so the `rayon` and Metal ordering at 10,000 rows is inside the noise.

The honest summary: `rayon` wins the middle of the range, Metal pulls ahead at the top by a margin that is not large, and below roughly 5,000 non-zeros per row the sequential arm beats both because neither parallel substrate earns its dispatch overhead.

### The prefix scan is slower on Metal, and that is the finding

`cargo run --release --features metal --example scan_crossover` — same host, milliseconds for a full prefix scan over affine maps.

| n | CPU sequential | Metal | ratio |
| ---: | ---: | ---: | ---: |
| 0.1M | **0.09** | 0.72 | 0.13x |
| 0.3M | **0.39** | 1.56 | 0.26x |
| 1.0M | **1.61** | 4.38 | 0.37x |
| 4.2M | **6.47** | 14.38 | 0.45x |

`Backend::Metal` loses at every size. `scan.rs` used to predict the opposite — that a two-level tree scan "would deliver a real speedup" — and that prediction is simply wrong here. Composing two affine maps is three flops over sixteen bytes moved, so the operation is memory-bound, and the three-phase tree makes roughly five passes over memory where a sequential fold makes one. Bandwidth does not rescue an algorithm that spends it on extra passes.

The gap does narrow with `n` (0.13x to 0.45x), so the GPU is amortising fixed costs — it just does not reach parity anywhere in this range.

The kernel ships anyway, for one reason: `Backend::Metal` should be able to run every operation this crate offers rather than silently falling back to CPU under a GPU label, which is precisely what [`Backend::Cuda`](#backend-status) refuses to do. It is documented as slow at its call site so nobody reaches for it expecting a win.

---

## 🧪 Testing

```bash
cargo test --features metal --release
```

| Suite | Tests | What it holds down |
| --- | ---: | --- |
| Unit (in-crate) | 28 | Scan bit-exactness, CSR/CSC invariants, RNG golden stream, tolerance bounds |
| `honesty` | 6 | Unconstructible unavailable backends, distinct labels, CPU arms bit-identical |
| `differential` | 6 | Every available backend against the CPU reference, boundary shapes plus a proptest fuzz |
| `stress` | 13 | Malformed CSR, out-of-range columns, non-finite data, subnormals, contention, soak |
| `golden` | 2 | The reference's own output bits |
| `fma_contraction` | 1 | That the CPU/GPU gap has exactly one identified cause |

The shape sweep straddles every boundary the backends care about — 31, 32, 33 for the SIMD width and 255, 256, 257 for the threadgroup — plus the degenerate zero-row, zero-edge and single-row cases that a sweep of reasonable sizes never reaches.

### Mutation campaign

All thirteen stress tests passed on their first run, which is precisely when a suite deserves suspicion. Twenty deliberate defects were injected. **Four survived, and each exposed a real gap that was then closed:**

| Surviving mutation | What it exposed | Fix |
| --- | --- | --- |
| Delete the fused kernel's row bounds guard | Out-of-bounds device writes landed in page padding and were invisible | Sentinel tails on every kernel-written buffer, checked after each dispatch |
| Sum each row in reverse | Every check is relative to the reference, so moving the reference moves everything and nothing fails | `tests/golden.rs` pins the reference's bits |
| `try_new(Metal)` falls back to CPU | The error branch is unreachable on a machine that has Metal | One availability gate for all backends, plus a directly tested consistency check |
| Widen a tolerance to `f32::MAX` | A tolerance can fail upward and make an entire suite vacuous | Upper as well as lower bounds asserted on both tolerance formulas |

Re-verification after the fixes: every former survivor is now caught.

---

## 🧭 Known gaps

Recorded rather than implied. Everything the crate *contains* is wired, tested and documented; these are capabilities it does not have yet. **SpMM shipped** — see [The batched product](#the-batched-product). The **`block 0.1.6`** entry is gone too: the Metal backend now uses `objc2-metal`, which does not depend on it, so the future-incompatibility lint that would have become a hard error no longer applies.

| Gap | Why it matters | Why not yet |
|---|---|---|
| **f16 / bf16 / bitpacked spikes** | SpMV is bandwidth-bound, so f16 is a straight 2x. A spike is one *bit*, not 32. | Every tolerance function here is derived for f32; narrower types need their bounds re-derived, not rescaled. |
| **CUDA** | `Backend::Cuda` is declared and permanently unavailable. | Deliberate. See `src/backend/cuda.rs`: it refuses rather than silently falling back to CPU under a GPU label. |

---

## 🔗 Relationship to tessl

[`tessl`](https://github.com/bharathvbcr/tessl) is the dense counterpart — a Metal 4 GEMM and encode runtime built on `objc2-metal`, `MTL4` argument tables and TensorOps. `sparsl` is sparse and CPU-first. Both now build on `objc2-metal`: the gfx-rs `metal` crate pulled `block 0.1.6` at every version, which is unmaintained and trips a future-incompatibility lint, so the two crates share one binding stack.

They are deliberately separate crates. `tessl`'s runtime, dispatch and tensor modules form a general Metal 4 compute runtime that `sparsl` could eventually sit on, but folding sparse SpMV and LIF kernels into a GEMM crate would blur what either one is. If `sparsl` moves to Metal 4, it should depend on that runtime rather than merge into it.


---

## 📄 License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate shall be dual-licensed as above, without any additional terms or conditions.
