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
