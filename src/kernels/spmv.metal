#include <metal_stdlib>
using namespace metal;

// Kernels are dispatched with `dispatch_threads` (non-uniform threadgroups),
// so the grid is exactly `n` wide and no thread runs past the end. The explicit
// `id >= n` guards below are redundant under that dispatch and deliberately
// kept: they make every kernel safe under uniform `dispatch_thread_groups` too,
// so a future caller that changes dispatch style cannot turn a scheduling
// detail into an out-of-bounds device write.
//
// Column indices are NOT range-checked here. `SparseOp::prepare` validates
// `col[i] < ncols` for every stored entry before a single byte is uploaded, so
// `x[col_ind[i]]` is in bounds by construction. That check is a precondition of
// these kernels, not an optimisation: an unvalidated CSR reaching this code
// would read arbitrary device memory.

/// y += A * x, one thread per row.
kernel void csr_spmv_kernel(
    device const uint*  row_ptr [[buffer(0)]],
    device const uint*  col_ind [[buffer(1)]],
    device const float* values  [[buffer(2)]],
    device const float* x       [[buffer(3)]],
    device float*       y       [[buffer(4)]],
    constant uint&      n_rows  [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_rows) { return; }
    uint row_start = row_ptr[id];
    uint row_end   = row_ptr[id + 1];
    float sum = 0.0f;
    for (uint i = row_start; i < row_end; ++i) {
        sum += values[i] * x[col_ind[i]];
    }
    y[id] += sum;
}

/// y += A * x with binary16 weights, one thread per row.
///
/// The only difference from `csr_spmv_kernel` is the width of `values`: 2 bytes
/// per non-zero instead of 4. `col_ind` stays 4, so the streamed traffic per
/// non-zero goes 8 -> 6 bytes, a 25% cut and not the 2x that halving one array
/// suggests.
///
/// `half` widens to `float` on load and the accumulator is `float`. binary16
/// has an 11-bit significand and overflows at 65504; a 500-term row sum
/// accumulated at that width would admit a relative error near 24% and could
/// overflow outright. Narrow storage, wide arithmetic — see
/// `tolerance_for_spmv_f16` for the bound this earns.
kernel void csr_spmv_f16_kernel(
    device const uint*  row_ptr [[buffer(0)]],
    device const uint*  col_ind [[buffer(1)]],
    device const half*  values  [[buffer(2)]],
    device const float* x       [[buffer(3)]],
    device float*       y       [[buffer(4)]],
    constant uint&      n_rows  [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_rows) { return; }
    uint row_start = row_ptr[id];
    uint row_end   = row_ptr[id + 1];
    float sum = 0.0f;
    for (uint i = row_start; i < row_end; ++i) {
        sum += float(values[i]) * x[col_ind[i]];
    }
    y[id] += sum;
}

/// y += A * x with bfloat16 weights, one thread per row.
///
/// Identical to `csr_spmv_f16_kernel` but for the storage format. bfloat16 is
/// f32 with the low 16 significand bits dropped, so it keeps f32's exponent
/// range — no 65504 ceiling — and pays 8x the rounding error of binary16.
/// Which trade is right depends on the weights, so both are offered and
/// neither is a default.
kernel void csr_spmv_bf16_kernel(
    device const uint*   row_ptr [[buffer(0)]],
    device const uint*   col_ind [[buffer(1)]],
    device const bfloat* values  [[buffer(2)]],
    device const float*  x       [[buffer(3)]],
    device float*        y       [[buffer(4)]],
    constant uint&       n_rows  [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_rows) { return; }
    uint row_start = row_ptr[id];
    uint row_end   = row_ptr[id + 1];
    float sum = 0.0f;
    for (uint i = row_start; i < row_end; ++i) {
        sum += float(values[i]) * x[col_ind[i]];
    }
    y[id] += sum;
}

/// y += A * s, where `s` is a bitpacked spike vector: 32 spikes per word,
/// least-significant bit first.
///
/// The multiply is kept rather than branched away. `float(bit)` is exactly 0.0
/// or 1.0, so `values[i] * float(bit)` performs the same multiply-add the dense
/// kernel does, in the same order — the results are bit-identical, not merely
/// within a tolerance. Branching on the bit would skip the `+= 0.0` and change
/// the sign of a zero row, and would diverge across a simdgroup besides.
///
/// The gain is not in this arithmetic. It is that `s` is 32x smaller than the
/// f32 vector it replaces, and `s[col_ind[i]]` is a *random* read whose cost is
/// set by whether the vector fits in cache. At 50,000 cells that is 6.25 KB
/// against 200 KB.
kernel void csr_spmv_spikes_kernel(
    device const uint*  row_ptr [[buffer(0)]],
    device const uint*  col_ind [[buffer(1)]],
    device const float* values  [[buffer(2)]],
    device const uint*  spikes  [[buffer(3)]],
    device float*       y       [[buffer(4)]],
    constant uint&      n_rows  [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_rows) { return; }
    uint row_start = row_ptr[id];
    uint row_end   = row_ptr[id + 1];
    float sum = 0.0f;
    for (uint i = row_start; i < row_end; ++i) {
        uint c = col_ind[i];
        // `c` is in range by construction — `SparseOp::prepare` validated every
        // column against `ncols` before upload, exactly as for the dense
        // kernels — so `c >> 5` is inside the packed vector.
        uint bit = (spikes[c >> 5u] >> (c & 31u)) & 1u;
        sum += values[i] * float(bit);
    }
    y[id] += sum;
}

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

/// Fused SpMV + LIF: one SIMD-group per row, hardware reduction, lead lane
/// commits the membrane update.
///
/// Dispatched with UNIFORM threadgroups on purpose. This kernel derives its row
/// from `threads_per_threadgroup`, and under non-uniform dispatch that builtin
/// reports a smaller value for the tail group — which would silently shift
/// every row index in that group. `n_rows` bounds the result either way.
kernel void fused_spmv_lif_simdgroup_kernel(
    device const uint*  row_ptr     [[buffer(0)]],
    device const uint*  col_ind     [[buffer(1)]],
    device const float* values      [[buffer(2)]],
    device const float* x           [[buffer(3)]],
    device float*       v           [[buffer(4)]],
    device float*       theta       [[buffer(5)]],
    device uchar*       spikes      [[buffer(6)]],
    constant float&     decay       [[buffer(7)]],
    constant float&     v_reset     [[buffer(8)]],
    constant float&     delta_theta [[buffer(9)]],
    constant uint&      n_rows      [[buffer(10)]],
    uint thread_in_simdgroup  [[thread_index_in_simdgroup]],
    uint simdgroup_id         [[simdgroup_index_in_threadgroup]],
    uint group_id             [[threadgroup_position_in_grid]],
    uint threads_per_group    [[threads_per_threadgroup]]
) {
    uint row = group_id * (threads_per_group / 32) + simdgroup_id;
    if (row >= n_rows) { return; }

    uint row_start = row_ptr[row];
    uint row_end   = row_ptr[row + 1];

    float thread_sum = 0.0f;
    for (uint i = row_start + thread_in_simdgroup; i < row_end; i += 32) {
        thread_sum += values[i] * x[col_ind[i]];
    }
    float total_current = simd_sum(thread_sum);

    if (thread_in_simdgroup == 0) {
        float voltage = v[row] * decay + total_current;
        float th = theta[row];
        if (voltage >= th) {
            spikes[row] = 1;
            v[row]      = v_reset;
            theta[row]  = th + delta_theta;
        } else {
            spikes[row] = 0;
            v[row]      = voltage;
        }
    }
}

/// y += A^T * x, one thread per output column.
///
/// Walks the CSC reverse index rather than transposing the matrix: for output
/// column `c`, every stored entry `k` in `[col_ptr[c], col_ptr[c+1])` names the
/// CSR row it came from (`row[k]`) and the slot its value occupies in the
/// forward `values` array (`edge_idx[k]`). One value table serves both
/// directions, so a weight update is visible to both without a second upload.
///
/// `row[k]` and `edge_idx[k]` are unchecked here for the same reason `col_ind`
/// is above: `Csc::from_csr` derives them from a CSR that `SparseOp::prepare`
/// has already validated, so `row[k] < n_rows` and `edge_idx[k] < nnz` hold by
/// construction. An unvalidated CSC reaching this kernel would read arbitrary
/// device memory.
kernel void csc_spmv_t_kernel(
    device const uint*  col_ptr  [[buffer(0)]],
    device const uint*  row_ind  [[buffer(1)]],
    device const uint*  edge_idx [[buffer(2)]],
    device const float* values   [[buffer(3)]],
    device const float* x        [[buffer(4)]],
    device float*       y        [[buffer(5)]],
    constant uint&      n_cols   [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    if (id >= n_cols) { return; }
    uint col_start = col_ptr[id];
    uint col_end   = col_ptr[id + 1];
    float sum = 0.0f;
    for (uint k = col_start; k < col_end; ++k) {
        sum += values[edge_idx[k]] * x[row_ind[k]];
    }
    y[id] += sum;
}

/// Y += A * X for a batch of `n_vec` dense vectors, one thread per output
/// element.
///
/// # Layout: the batch index moves fastest
///
/// `X` is `[ncols][n_vec]` and `Y` is `[nrows][n_vec]` — all `n_vec` values for
/// a column sit adjacent, not all columns for a vector. That is the opposite of
/// how a batch is usually written, and it is the whole reason this kernel is
/// worth having.
///
/// Thread `id` covers `(row, v)` with `v = id % n_vec`. Adjacent threads differ in
/// `v` and share `row`, so:
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
/// Column indices are unchecked here for the same reason as `csr_spmv_kernel`:
/// `SparseOp::prepare` validated them before upload.
kernel void csr_spmm_kernel(
    device const uint*  row_ptr [[buffer(0)]],
    device const uint*  col_ind [[buffer(1)]],
    device const float* values  [[buffer(2)]],
    device const float* x       [[buffer(3)]],
    device float*       y       [[buffer(4)]],
    constant uint&      n_rows  [[buffer(5)]],
    constant uint&      n_vec   [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    const uint total = n_rows * n_vec;
    if (id >= total) { return; }
    const uint row = id / n_vec;
    const uint v   = id - row * n_vec;
    uint row_start = row_ptr[row];
    uint row_end   = row_ptr[row + 1];
    float sum = 0.0f;
    // Same traversal order as `csr_spmv_kernel`, so a batch of one is
    // bit-identical to a plain SpMV rather than merely close to it.
    for (uint i = row_start; i < row_end; ++i) {
        sum += values[i] * x[col_ind[i] * n_vec + v];
    }
    y[id] += sum;
}

// =============================================================================
// Associative scan over affine maps `v' = a*v + b`.
//
// The CPU `assoc_scan` is bit-identical to a sequential left-fold, which it buys
// by making phase 1 a *complete* sequential fold — about 2n combines to replace
// n, measured at 1.08x. This is the other trade: a two-level Blelloch-style
// scan that is genuinely parallel and is *not* bit-identical to the sequential
// fold, because it reassociates.
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
        float2 prev = (lid >= offset) ? scratch[lid - offset] : SCAN_IDENTITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        scratch[lid] = scan_combine(prev, scratch[lid]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (gid < n) { out[gid] = scratch[lid]; }
    if (lid == tptg - 1u) { totals[grp] = scratch[lid]; }
}

/// Exclusive scan of the per-group totals, in one threadgroup.
///
/// `n_groups` is `ceil(n / tptg)`, so for any input a single group can span it
/// while `tptg` stays at its cap — 1024 groups of 1024 is 2^20 elements, and
/// beyond that the strided loop below still covers it sequentially per lane.
kernel void scan_block_offsets(
    device float2*  totals   [[buffer(0)]],
    constant uint&  n_groups [[buffer(1)]],
    uint lid [[thread_position_in_threadgroup]]
) {
    threadgroup float2 scratch[SCAN_MAX_TG];
    // One lane walks the whole thing when there are more groups than lanes.
    // Groups are few by construction, so this is not the hot path.
    if (lid == 0u) {
        float2 running = SCAN_IDENTITY;
        for (uint g = 0u; g < n_groups; ++g) {
            float2 total = totals[g];
            totals[g] = running;              // exclusive prefix
            running = scan_combine(running, total);
        }
    }
    (void)scratch;
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
    // Group 0's prefix is the identity; combining anyway keeps every lane on
    // one path and costs two multiplies.
    out[gid] = scan_combine(totals[grp], out[gid]);
}
