/// Softmax kernel using numerically stable parallel reduction
///
/// **Algorithm:**
/// The naive formula  softmax(x_i) = exp(x_i) / Σ exp(x_j)  overflows when
/// any x_i is large.  The numerically stable version subtracts the row max
/// before exponentiating, which keeps all exponents ≤ 0 (≤ exp(0) = 1).
///
/// Steps per row:
///   1. Parallel reduction → find max value m = max(x)
///   2. Subtract max      → shifted_i = x_i - m
///   3. Exponentiate      → e_i = exp(shifted_i)
///   4. Parallel reduction → compute sum S = Σ e_i
///   5. Normalize         → out_i = e_i / S
///
/// **Threadgroup strategy:**
/// One threadgroup handles one row of the input.
/// Up to 256 threads collaborate via threadgroup memory to compute
/// the row max and row sum using a parallel tree reduction.
///
/// **Dispatch shape:**
/// - threadsPerThreadgroup: (min(rowLen, 256), 1, 1)
/// - threadgroupsPerGrid:   (numRows, 1, 1)
///
/// **Buffers:**
///   [0] input  — float array, row-major [numRows × rowLen]
///   [1] output — float array, row-major [numRows × rowLen]
///   [2] dims   — uint2 { numRows, rowLen }

#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint2&     dims    [[buffer(2)]],   // { numRows, rowLen }
    uint  tgid [[threadgroup_position_in_grid]], // which row
    uint  lid  [[thread_position_in_threadgroup]],
    uint  tgSize [[threads_per_threadgroup]],
    threadgroup float* scratch  [[threadgroup(0)]] // tgSize floats
) {
    // ── Which row does this threadgroup own? ──────────────────────────────
    uint numRows = dims.x;
    uint rowLen  = dims.y;
    uint row     = tgid;

    if (row >= numRows) return;

    // Base pointer into input / output for this row
    device const float* in_row  = input  + row * rowLen;
    device float*       out_row = output + row * rowLen;

    // ── Step 1: Find row max via parallel reduction ───────────────────────
    // Each thread takes the max over its strided slice first.
    float thread_max = -INFINITY;
    for (uint i = lid; i < rowLen; i += tgSize) {
        thread_max = max(thread_max, in_row[i]);
    }

    // Write partial max into shared memory and synchronize
    scratch[lid] = thread_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over shared memory
    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] = max(scratch[lid], scratch[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = scratch[0];   // Broadcast max to all threads

    // ── Step 2 + 3: Subtract max, exponentiate; accumulate sum ───────────
    float thread_sum = 0.0;
    for (uint i = lid; i < rowLen; i += tgSize) {
        float e = exp(in_row[i] - row_max);   // shifted exponent
        out_row[i] = e;                        // store temporarily
        thread_sum += e;
    }

    // ── Step 4: Parallel sum reduction ───────────────────────────────────
    scratch[lid] = thread_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_sum = scratch[0];   // Broadcast sum

    // ── Step 5: Normalize ─────────────────────────────────────────────────
    // Add tiny epsilon in the denominator to avoid 0/0 for degenerate inputs.
    float inv_sum = 1.0 / (row_sum + 1e-7);
    for (uint i = lid; i < rowLen; i += tgSize) {
        out_row[i] *= inv_sum;
    }
}
