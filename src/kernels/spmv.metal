#include <metal_stdlib>
using namespace metal;

// Every line-parallel kernel here -- SpMV at three lane widths, the spike and
// transposed products, batched SpMM and the fused LIF step -- is dispatched
// with UNIFORM threadgroups. Each derives the line it owns from
// `threads_per_threadgroup`, which non-uniform `dispatchThreads` shrinks for
// the tail group, silently shifting every line index inside it. The padding
// that uniform dispatch adds is bounded by the `line >= n_lines` guard each
// kernel opens with, and the sentinel tail on every writable buffer
// (`CANARY_ELEMS` in `backend/metal.rs`) is what proves the guard is still
// there. The element-wise kernels keep their `id >= n` guards for the same
// reason: a future caller that changes dispatch style cannot turn a
// scheduling detail into an out-of-bounds device write.
//
// Column indices are NOT range-checked here. `SparseOp::prepare` validates
// `col[i] < ncols` for every stored entry before a single byte is uploaded, so
// `x[col_ind[i]]` is in bounds by construction. That check is a precondition of
// these kernels, not an optimisation: an unvalidated CSR reaching this code
// would read arbitrary device memory.

/// LIF membrane decay, threshold, spike, reset and adaptive threshold bump.
kernel void lif_integrate_kernel(
    device float*       v           [[buffer(0)]],
    device float*       theta       [[buffer(1)]],
    device const float* currents    [[buffer(2)]],
    device uchar*       spikes      [[buffer(3)]],
    constant float&     decay       [[buffer(4)]],
    constant float&     v_reset     [[buffer(5)]],
    constant float&     delta_theta [[buffer(6)]],
    constant uint&      n_cells     [[buffer(7)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_cells) { return; }
    float voltage = v[id] * decay + currents[id];
    float th = theta[id];
    if (voltage >= th) {
        spikes[id] = 1;
        v[id]      = v_reset;
        theta[id]  = th + delta_theta;
    } else {
        spikes[id] = 0;
        v[id]      = voltage;
    }
}

// =============================================================================
// Line-parallel CSR kernels: `LPR` lanes per line, `LPR` in {1, 8, 32}.
//
// One thread per row is the natural way to write CSR SpMV and the wrong way to
// feed this memory system: thread `r` walks `values[row_ptr[r] .. row_ptr[r+1])`,
// so the 32 lanes of a simdgroup read 32 addresses one whole row apart and the
// coalescer can merge none of it. Giving a row `LPR` adjacent lanes inverts the
// pattern -- lane `l` reads `values[row_start + l]`, then `+ LPR` -- so one
// issued load covers `LPR` adjacent entries, at the cost of `LPR - k` idle
// lanes on a row of `k < LPR` entries. That trade is decided per operator from
// its mean line length (`RowKernel::for_shape` in `backend/metal.rs`): one
// lane below 12 entries, eight lanes up to 64, a whole simdgroup above.
//
// One template serves all three widths so they cannot drift: the lane
// partition, the per-lane sequential order and the butterfly reduction below
// are the same code at every width, and `LPR == 1` degenerates to the plain
// sequential loop with an empty reduction. The spike, transposed, batched and
// fused kernels are instantiated from the same helpers.
//
// Reduction order and bit-identity
// --------------------------------
// `fold_lanes` is an xor butterfly over `simd_shuffle_xor`, so it reassociates
// a line relative to the sequential fold the CPU arms perform. That is a
// cross-*backend* difference, and the crate's rule is reproducibility within a
// backend: two runs of one kernel on one device agree byte for byte, and the
// GPU is compared to the CPU under `tolerance_for_spmv`. A butterfly's error
// grows as log2(nnz) rather than nnz, so it sits inside a bound derived for
// the sequential order.
//
// What may NOT drift is the identity *between the kernels of one width*: the
// spike path is documented and tested as bit-identical to the dense one, and
// every SpMM column to the single-vector product on the same operator. They
// therefore share the loop below rather than each carrying a copy -- identical
// striding, identical fold -- and the spike variant keeps its `* float(bit)`
// multiply instead of branching, which would skip the `+= 0.0` and change the
// sign of a zero row besides diverging the lanes.
// =============================================================================

/// Lanes in a simdgroup.
///
/// A contract with the host, not a tunable: `simd_threadgroup` sizes every
/// threadgroup as a whole multiple of this, so an `LPR`-lane team never
/// straddles two simdgroups and `threads_per_threadgroup / LPR` is the exact
/// number of lines a group covers.
constant uint SIMD_LANES = 32u;

/// The line this thread's `LPR`-lane team owns under a UNIFORM dispatch.
///
/// Widened to `ulong` before the multiply: `n_lines` and every CSR offset fit
/// in `uint` individually, but the padded final group can push the product
/// past `uint` before the `>= n_lines` guard rejects it.
template <uint LPR>
inline ulong line_for(uint group_id, uint threads_per_group, uint tid) {
    return (ulong)group_id * (threads_per_group / LPR) + (tid / LPR);
}

/// Sum `acc` across the `LPR` lanes of a team; every lane receives the total.
///
/// An xor butterfly stays inside an aligned group of `LPR` lanes, so teams in
/// one simdgroup never mix. All lanes of a team share one line and therefore
/// take the same side of the `>= n_lines` guard, so no shuffle ever reads an
/// inactive lane. At `LPR == 1` the loop body never runs.
template <uint LPR>
inline float fold_lanes(float acc) {
    for (uint offset = LPR / 2u; offset > 0u; offset >>= 1u) {
        acc += simd_shuffle_xor(acc, offset);
    }
    return acc;
}

/// One line of `A * x`, strided across the team and folded.
///
/// Templated on the stored value type so f32, binary16 and bfloat16 share this
/// loop instead of keeping copies that drift. Every variant widens to `float`
/// on load and accumulates at `float`: binary16 has an 11-bit significand and
/// overflows at 65504, so a long row accumulated at that width would admit a
/// relative error near 24% and could overflow outright. Narrow storage, wide
/// arithmetic -- see `tolerance_for_spmv_f16` for the bound this earns.
///
/// The index is widened to `ulong` before the lane or the stride is added: a
/// valid line may begin within `LPR` entries of `UINT_MAX`, and adding either
/// in `uint` would wrap to the start of the values buffer instead of ending
/// the loop.
template <uint LPR, typename V>
inline float line_dot(
    device const uint*  col_ind,
    device const V*     values,
    device const float* x,
    uint line_start,
    uint line_end,
    uint lane
) {
    float acc = 0.0f;
    for (ulong i = (ulong)line_start + lane; i < (ulong)line_end; i += (ulong)LPR) {
        acc += float(values[i]) * x[col_ind[i]];
    }
    return fold_lanes<LPR>(acc);
}

/// One line of `A * s` for a bitpacked spike vector (32 spikes per word,
/// least-significant bit first), striding exactly as `line_dot`.
///
/// `float(bit)` is exactly 0.0 or 1.0, so `values[i] * float(bit)` performs the
/// same multiply-add the dense loop does, in the same lane order -- the two
/// agree bit for bit. The gain is not in this arithmetic: `s` is 32x smaller
/// than the f32 vector it replaces, and `s[col_ind[i]]` is a random read whose
/// cost is set by whether the vector fits in cache.
template <uint LPR>
inline float line_dot_spikes(
    device const uint*  col_ind,
    device const float* values,
    device const uint*  spikes,
    uint line_start,
    uint line_end,
    uint lane
) {
    float acc = 0.0f;
    for (ulong i = (ulong)line_start + lane; i < (ulong)line_end; i += (ulong)LPR) {
        uint c = col_ind[i];
        // In range by construction: `SparseOp::prepare` validated every column
        // against `ncols` before upload, so `c >> 5` is inside the packed vector.
        uint bit = (spikes[c >> 5u] >> (c & 31u)) & 1u;
        acc += values[i] * float(bit);
    }
    return fold_lanes<LPR>(acc);
}

/// One output column of `A^T * x`, walking the CSC reverse index.
///
/// For output column `c`, every stored entry `k` in `[col_ptr[c], col_ptr[c+1])`
/// names the CSR row it came from (`row_ind[k]`) and the slot its value
/// occupies in the forward `values` array (`edge_idx[k]`). One value table
/// serves both directions, so a weight update is visible to both without a
/// second upload. The gather stays a gather -- `values[edge_idx[k]]` is
/// indirect by construction -- so wider teams coalesce `edge_idx` and
/// `row_ind` but not `values`, and the win is smaller than the forward one.
///
/// `row_ind[k]` and `edge_idx[k]` are unchecked for the reason `col_ind` is:
/// `Csc::from_csr` derives them from a CSR `SparseOp::prepare` has validated.
template <uint LPR>
inline float line_dot_t(
    device const uint*  row_ind,
    device const uint*  edge_idx,
    device const float* values,
    device const float* x,
    uint col_start,
    uint col_end,
    uint lane
) {
    float acc = 0.0f;
    for (ulong k = (ulong)col_start + lane; k < (ulong)col_end; k += (ulong)LPR) {
        acc += values[edge_idx[k]] * x[row_ind[k]];
    }
    return fold_lanes<LPR>(acc);
}

/// `y += A * x`, `LPR` lanes per row.
#define SPMV_KERNEL(NAME, LPR, VALUE_T)                                        \
kernel void NAME(                                                              \
    device const uint*    row_ptr [[buffer(0)]],                               \
    device const uint*    col_ind [[buffer(1)]],                               \
    device const VALUE_T* values  [[buffer(2)]],                               \
    device const float*   x       [[buffer(3)]],                               \
    device float*         y       [[buffer(4)]],                               \
    constant uint&        n_rows  [[buffer(5)]],                               \
    uint tid               [[thread_position_in_threadgroup]],                 \
    uint group_id          [[threadgroup_position_in_grid]],                   \
    uint threads_per_group [[threads_per_threadgroup]]                         \
) {                                                                            \
    const ulong row = line_for<LPR>(group_id, threads_per_group, tid);         \
    if (row >= (ulong)n_rows) { return; }                                      \
    const uint lane = tid % LPR;                                               \
    const float total = line_dot<LPR, VALUE_T>(                                \
        col_ind, values, x, row_ptr[row], row_ptr[row + 1], lane);             \
    if (lane == 0u) { y[row] += total; }                                       \
}

/// `y += A * s` for a bitpacked spike vector, `LPR` lanes per row.
#define SPMV_SPIKES_KERNEL(NAME, LPR)                                          \
kernel void NAME(                                                              \
    device const uint*  row_ptr [[buffer(0)]],                                 \
    device const uint*  col_ind [[buffer(1)]],                                 \
    device const float* values  [[buffer(2)]],                                 \
    device const uint*  spikes  [[buffer(3)]],                                 \
    device float*       y       [[buffer(4)]],                                 \
    constant uint&      n_rows  [[buffer(5)]],                                 \
    uint tid               [[thread_position_in_threadgroup]],                 \
    uint group_id          [[threadgroup_position_in_grid]],                   \
    uint threads_per_group [[threads_per_threadgroup]]                         \
) {                                                                            \
    const ulong row = line_for<LPR>(group_id, threads_per_group, tid);         \
    if (row >= (ulong)n_rows) { return; }                                      \
    const uint lane = tid % LPR;                                               \
    const float total = line_dot_spikes<LPR>(                                  \
        col_ind, values, spikes, row_ptr[row], row_ptr[row + 1], lane);        \
    if (lane == 0u) { y[row] += total; }                                       \
}

/// `y += A^T * x`, `LPR` lanes per output column.
#define SPMV_T_KERNEL(NAME, LPR)                                               \
kernel void NAME(                                                              \
    device const uint*  col_ptr  [[buffer(0)]],                                \
    device const uint*  row_ind  [[buffer(1)]],                                \
    device const uint*  edge_idx [[buffer(2)]],                                \
    device const float* values   [[buffer(3)]],                                \
    device const float* x        [[buffer(4)]],                                \
    device float*       y        [[buffer(5)]],                                \
    constant uint&      n_cols   [[buffer(6)]],                                \
    uint tid               [[thread_position_in_threadgroup]],                 \
    uint group_id          [[threadgroup_position_in_grid]],                   \
    uint threads_per_group [[threads_per_threadgroup]]                         \
) {                                                                            \
    const ulong col = line_for<LPR>(group_id, threads_per_group, tid);         \
    if (col >= (ulong)n_cols) { return; }                                      \
    const uint lane = tid % LPR;                                               \
    const float total = line_dot_t<LPR>(                                       \
        row_ind, edge_idx, values, x, col_ptr[col], col_ptr[col + 1], lane);   \
    if (lane == 0u) { y[col] += total; }                                       \
}

/// Fused SpMV + LIF: the team reduces the row's current and its lead lane
/// commits the membrane update, without materialising the current vector.
#define FUSED_LIF_KERNEL(NAME, LPR)                                            \
kernel void NAME(                                                              \
    device const uint*  row_ptr     [[buffer(0)]],                             \
    device const uint*  col_ind     [[buffer(1)]],                             \
    device const float* values      [[buffer(2)]],                             \
    device const float* x           [[buffer(3)]],                             \
    device float*       v           [[buffer(4)]],                             \
    device float*       theta       [[buffer(5)]],                             \
    device uchar*       spikes      [[buffer(6)]],                             \
    constant float&     decay       [[buffer(7)]],                             \
    constant float&     v_reset     [[buffer(8)]],                             \
    constant float&     delta_theta [[buffer(9)]],                             \
    constant uint&      n_rows      [[buffer(10)]],                            \
    uint tid               [[thread_position_in_threadgroup]],                 \
    uint group_id          [[threadgroup_position_in_grid]],                   \
    uint threads_per_group [[threads_per_threadgroup]]                         \
) {                                                                            \
    const ulong row = line_for<LPR>(group_id, threads_per_group, tid);         \
    if (row >= (ulong)n_rows) { return; }                                      \
    const uint lane = tid % LPR;                                               \
    const float total_current = line_dot<LPR, float>(                          \
        col_ind, values, x, row_ptr[row], row_ptr[row + 1], lane);             \
    if (lane == 0u) {                                                          \
        float voltage = v[row] * decay + total_current;                        \
        float th = theta[row];                                                 \
        if (voltage >= th) {                                                   \
            spikes[row] = 1;                                                   \
            v[row]      = v_reset;                                             \
            theta[row]  = th + delta_theta;                                    \
        } else {                                                               \
            spikes[row] = 0;                                                   \
            v[row]      = voltage;                                             \
        }                                                                      \
    }                                                                          \
}

/// Vectors a lane keeps in registers per pass of the team SpMM kernels.
///
/// The row's indices and weights are re-read once per tile, so a wider tile
/// amortises that traffic; the tile also costs one live register per lane per
/// vector. Eight keeps the accumulator in registers while reading `col_ind`
/// and `values` at most `ceil(n_vec / 8)` times.
constant uint SIMD_SPMM_TILE = 8u;

/// `Y += A * X` for a batch of `n_vec` dense vectors, `LPR` lanes per row.
///
/// The team counterpart of `csr_spmm_kernel`, and it exists for a correctness
/// reason before a performance one. `SparseOp::spmm` is documented and tested
/// as producing, for every column, exactly what `SparseOp::spmv` produces for
/// that column *on the same backend* -- bit for bit, not within a tolerance.
/// That holds only while the two kernels reduce a row the same way, so an
/// operator that selected an `LPR`-lane SpMV must batch with the `LPR`-lane
/// SpMM. Pairing them by the operator's single `RowKernel` choice keeps the
/// guarantee structural rather than a coincidence of the shapes a test uses.
///
/// The lane partition, the traversal order within a lane and the fold are
/// therefore identical to `line_dot`; only the `x` stride differs, and at
/// `n_vec == 1` it degenerates to exactly that function.
///
/// `X` is `[ncols][n_vec]` and `Y` is `[nrows][n_vec]`, the batch-minor layout
/// `csr_spmm_kernel` documents. Adjacent tile entries `t` are adjacent in
/// memory, so a lane's tile read is contiguous.
///
/// The `v0 + t < n_vec` test inside is a BOUNDS GUARD on `x`, not a nicety
/// about the accumulate: `base` is the last column's tile origin on the final
/// pass, so an unguarded read would run up to `SIMD_SPMM_TILE - 1` past the
/// batch operand. Nothing downstream would notice -- the discarded `acc[t]`
/// never reaches memory and Metal rounds allocations to a page -- so no
/// black-box test can catch its removal. It is preserved by reading. The tile
/// loop keeps a constant trip count so it unrolls into registers; the partial
/// tail substitutes an exact zero rather than shortening the loop.
#define SPMM_KERNEL(NAME, LPR)                                                 \
kernel void NAME(                                                              \
    device const uint*  row_ptr [[buffer(0)]],                                 \
    device const uint*  col_ind [[buffer(1)]],                                 \
    device const float* values  [[buffer(2)]],                                 \
    device const float* x       [[buffer(3)]],                                 \
    device float*       y       [[buffer(4)]],                                 \
    constant uint&      n_rows  [[buffer(5)]],                                 \
    constant uint&      n_vec   [[buffer(6)]],                                 \
    uint tid               [[thread_position_in_threadgroup]],                 \
    uint group_id          [[threadgroup_position_in_grid]],                   \
    uint threads_per_group [[threads_per_threadgroup]]                         \
) {                                                                            \
    const ulong row = line_for<LPR>(group_id, threads_per_group, tid);         \
    if (row >= (ulong)n_rows) { return; }                                      \
    const uint lane = tid % LPR;                                               \
    const uint row_start = row_ptr[row];                                       \
    const uint row_end   = row_ptr[row + 1];                                   \
    /* `ulong` so `v0 += TILE` cannot wrap for an `n_vec` near `UINT_MAX`. */  \
    for (ulong v0 = 0ul; v0 < (ulong)n_vec; v0 += (ulong)SIMD_SPMM_TILE) {     \
        float acc[SIMD_SPMM_TILE];                                             \
        for (uint t = 0u; t < SIMD_SPMM_TILE; ++t) { acc[t] = 0.0f; }          \
        for (ulong i = (ulong)row_start + lane; i < (ulong)row_end;            \
             i += (ulong)LPR) {                                                \
            const float w = values[i];                                         \
            const ulong base = (ulong)col_ind[i] * n_vec + v0;                 \
            for (uint t = 0u; t < SIMD_SPMM_TILE; ++t) {                       \
                const float xv = (v0 + t < (ulong)n_vec) ? x[base + t] : 0.0f; \
                acc[t] += w * xv;                                              \
            }                                                                  \
        }                                                                      \
        for (uint t = 0u; t < SIMD_SPMM_TILE; ++t) {                           \
            /* Every lane of the team must reach the fold, so it stays      */ \
            /* outside the write guard.                                     */ \
            const float total = fold_lanes<LPR>(acc[t]);                       \
            if (lane == 0u && v0 + t < (ulong)n_vec) {                         \
                y[row * (ulong)n_vec + v0 + t] += total;                       \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// One lane per line: the sequential loop with an empty fold.
SPMV_KERNEL(csr_spmv_l1_kernel,       1u, float)
SPMV_KERNEL(csr_spmv_f16_l1_kernel,   1u, half)
SPMV_KERNEL(csr_spmv_bf16_l1_kernel,  1u, bfloat)
SPMV_SPIKES_KERNEL(csr_spmv_spikes_l1_kernel, 1u)
SPMV_T_KERNEL(csc_spmv_t_l1_kernel, 1u)
FUSED_LIF_KERNEL(fused_spmv_lif_l1_kernel, 1u)

// Eight lanes per line: four teams per simdgroup.
SPMV_KERNEL(csr_spmv_l8_kernel,       8u, float)
SPMV_KERNEL(csr_spmv_f16_l8_kernel,   8u, half)
SPMV_KERNEL(csr_spmv_bf16_l8_kernel,  8u, bfloat)
SPMV_SPIKES_KERNEL(csr_spmv_spikes_l8_kernel, 8u)
SPMV_T_KERNEL(csc_spmv_t_l8_kernel, 8u)
FUSED_LIF_KERNEL(fused_spmv_lif_l8_kernel, 8u)
SPMM_KERNEL(csr_spmm_l8_kernel, 8u)

// A whole simdgroup per line.
SPMV_KERNEL(csr_spmv_l32_kernel,      SIMD_LANES, float)
SPMV_KERNEL(csr_spmv_f16_l32_kernel,  SIMD_LANES, half)
SPMV_KERNEL(csr_spmv_bf16_l32_kernel, SIMD_LANES, bfloat)
SPMV_SPIKES_KERNEL(csr_spmv_spikes_l32_kernel, SIMD_LANES)
SPMV_T_KERNEL(csc_spmv_t_l32_kernel, SIMD_LANES)
FUSED_LIF_KERNEL(fused_spmv_lif_l32_kernel, SIMD_LANES)
SPMM_KERNEL(csr_spmm_l32_kernel, SIMD_LANES)

#undef SPMV_KERNEL
#undef SPMV_SPIKES_KERNEL
#undef SPMV_T_KERNEL
#undef FUSED_LIF_KERNEL
#undef SPMM_KERNEL

/// Y += A * X for a batch of `n_vec` dense vectors, one thread per output
/// element: the one-lane tier's SpMM.
///
/// # Layout: the batch index moves fastest
///
/// `X` is `[ncols][n_vec]` and `Y` is `[nrows][n_vec]` — all `n_vec` values for
/// a column sit adjacent, not all columns for a vector. That is the opposite of
/// how a batch is usually written, and it is the whole reason this kernel is
/// worth having.
///
/// The grid is two-dimensional: `id.x` is the vector and `id.y` is the row.
/// Adjacent x-lanes differ in `v` and share `row`, so:
///
///   - they read `X[col[i] * n_vec + v]` at adjacent addresses — one coalesced
///     transaction instead of `n_vec` scattered ones;
///   - they write `Y[row * n_vec + v]` at adjacent addresses, likewise;
///   - they walk the *same* `col[i]` sequence, so the index loads are a
///     broadcast rather than divergent traffic.
///
/// Stored batch-major (`X` as `[n_vec][ncols]`) every one of those becomes a
/// scattered access, because adjacent threads would then hold different rows
/// and therefore different `col[i]`.
///
/// Dispatched non-uniformly (`dispatchThreads`): unlike the team kernels it
/// takes its position straight from the grid, so the tail group is exact and
/// the `row >= n_rows || v >= n_vec` guard is the belt to that suspender.
///
/// Column indices are unchecked here for the same reason as `line_dot`:
/// `SparseOp::prepare` validated them before upload.
kernel void csr_spmm_kernel(
    device const uint*  row_ptr [[buffer(0)]],
    device const uint*  col_ind [[buffer(1)]],
    device const float* values  [[buffer(2)]],
    device const float* x       [[buffer(3)]],
    device float*       y       [[buffer(4)]],
    constant uint&      n_rows  [[buffer(5)]],
    constant uint&      n_vec   [[buffer(6)]],
    uint2 id [[thread_position_in_grid]]
) {
    const uint v = id.x;
    const uint row = id.y;
    if (row >= n_rows || v >= n_vec) { return; }
    const ulong out_idx = (ulong)row * n_vec + v;
    uint row_start = row_ptr[row];
    uint row_end   = row_ptr[row + 1];
    float sum = 0.0f;
    // Same traversal order as the one-lane `line_dot`, so a batch column is
    // bit-identical to a plain SpMV rather than merely close to it.
    for (uint i = row_start; i < row_end; ++i) {
        sum += values[i] * x[(ulong)col_ind[i] * n_vec + v];
    }
    y[out_idx] += sum;
}

// =============================================================================
// Associative scan over affine maps `v' = a*v + b`.
//
// The CPU `assoc_scan` is bit-identical to a sequential left-fold, which it buys
// by making phase 1 a *complete* sequential fold — about 2n combines to replace
// n. The hardened host sampler has not established a throughput crossover for
// that extra work. This is the other trade: a two-level Blelloch-style scan that
// is genuinely parallel and is *not* bit-identical to the sequential fold,
// because it reassociates.
//
// That is allowed by this crate's rule and not a weakening of it: reproducibility
// holds *within* a backend and never across one. Two runs of this kernel on the
// same device give byte-identical output; comparing it to the CPU arm is a
// cross-backend comparison and takes a tolerance, exactly as SpMV does.
// =============================================================================

/// `combine(x, y)` = apply x, then y: `v -> y.a*(x.a*v + x.b) + y.b`.
inline float2 scan_combine(float2 x, float2 y) {
    return float2(y.x * x.x, y.x * x.y + y.y);
}

constant float2 SCAN_IDENTITY = float2(1.0f, 0.0f);
constant uint SCAN_MAX_TG = 1024u;

/// Inclusive scan within each threadgroup; also writes each group's total.
///
/// Hillis-Steele: `log2(tptg)` rounds, every lane active. More total work than
/// a work-efficient Blelloch sweep, but half the barriers and no bank-conflict
/// padding, which wins at these widths.
///
/// A `simd_shuffle_up` rewrite of this kernel -- 5 shuffle rounds and two
/// barriers instead of 10 threadgroup round trips and twenty, with the scratch
/// down from 8 KB to 256 bytes -- was built and measured against it head to
/// head, both orders, at 0.26M / 1M / 4.2M / 16.7M elements. It was
/// indistinguishable from this at every size. The scan is bound by the ~3
/// passes it makes over device memory, not by barriers or occupancy, so the
/// cheaper reduction buys nothing and the simpler kernel stays.
kernel void scan_chunk(
    device const float2* xs      [[buffer(0)]],
    device float2*       out     [[buffer(1)]],
    device float2*       totals  [[buffer(2)]],
    constant uint&       n       [[buffer(3)]],
    uint  gid  [[thread_position_in_grid]],
    uint  lid  [[thread_position_in_threadgroup]],
    uint  grp  [[threadgroup_position_in_grid]],
    uint  tptg [[threads_per_threadgroup]]
) {
    threadgroup float2 scratch[SCAN_MAX_TG];
    scratch[lid] = (gid < n) ? xs[gid] : SCAN_IDENTITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint offset = 1u; offset < tptg; offset <<= 1) {
        float2 prev = (lid >= offset) ? scratch[lid - offset] : scratch[lid];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid >= offset) {
            scratch[lid] = scan_combine(prev, scratch[lid]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid < n) { out[gid] = scratch[lid]; }
    const uint group_start = grp * tptg;
    const uint valid = min(tptg, n - group_start);
    if (lid == valid - 1u) { totals[grp] = scratch[lid]; }
}

/// Exclusive scan of the per-group totals, in one threadgroup, on one lane.
///
/// Serial on purpose, and now on evidence. `n_groups` is `ceil(n / tptg)`, so
/// the work here is 1/1024 of the scan's -- 4096 combines for a 4.2M-element
/// scan, 16384 for a 16.7M one. A three-phase parallel block scan replacing
/// this (serial reduce per thread, scan the thread totals, re-walk applying
/// each thread's prefix) was measured against it at both those sizes and was
/// within noise, because those few thousand combines sit against three full
/// passes over tens of megabytes.
///
/// The loop is bounded by `n_groups`, which `scan_geometry` has already proven
/// fits `u32` alongside the padded grid, so it cannot run away.
kernel void scan_block_offsets(
    device float2*  totals   [[buffer(0)]],
    constant uint&  n_groups [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    // No threadgroup scratch. This kernel previously declared
    // `float2 scratch[SCAN_MAX_TG]` -- 8 KB -- and then discarded it with a
    // `(void)scratch`, so every dispatch reserved threadgroup memory that no
    // line of the kernel read or wrote.
    if (lid == 0u) {
        float2 running = totals[0];
        totals[0] = SCAN_IDENTITY;
        for (uint g = 1u; g < n_groups; ++g) {
            float2 total = totals[g];
            totals[g] = running;              // exclusive prefix
            running = scan_combine(running, total);
        }
    }
}

/// Fold each group's exclusive prefix into its elements.
kernel void scan_apply_offsets(
    device float2*       out    [[buffer(0)]],
    device const float2* totals [[buffer(1)]],
    constant uint&       n      [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grp [[threadgroup_position_in_grid]]
) {
    if (gid >= n) { return; }
    // Preserve the first group's IEEE-754 values exactly. Although `(1, 0)` is
    // the algebraic identity, evaluating it is not an identity for signed zero
    // or infinities (`inf * 0` is NaN).
    if (grp == 0u) { return; }
    out[gid] = scan_combine(totals[grp], out[gid]);
}
