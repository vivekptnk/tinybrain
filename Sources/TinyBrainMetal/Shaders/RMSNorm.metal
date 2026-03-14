/// RMSNorm kernel (Root Mean Square Layer Normalization)
///
/// **What is RMSNorm?**
/// RMSNorm (Zhang & Sennrich, 2019) is a simplified alternative to LayerNorm
/// used by Llama, Mistral, Falcon, and most modern open-weight models.
/// It omits the mean-centering step, only normalizing by the RMS of activations.
///
/// **Formula:**
/// ```
///   RMS(x) = sqrt( (1/n) * Σ x_i²  +  ε )
///   out_i  = (x_i / RMS(x)) * weight_i
/// ```
///
/// **Why no mean subtraction?**
/// Empirically, removing the mean computation is cheaper and works just as
/// well in practice for large language models.  The learnable per-feature
/// weight (γ, sometimes called "scale") compensates for any offset bias.
///
/// **Threadgroup strategy:**
/// One threadgroup handles one token vector (one row of the input).
/// Threads collaborate to compute the row sum-of-squares using a parallel
/// tree reduction in threadgroup memory, then each thread normalizes and
/// scales its own elements independently.
///
/// **Dispatch shape:**
/// - threadsPerThreadgroup: (min(hiddenDim, 256), 1, 1)
/// - threadgroupsPerGrid:   (numTokens, 1, 1)
///
/// **Buffers:**
///   [0] input   — float [numTokens × hiddenDim]
///   [1] weight  — float [hiddenDim]   (per-feature scale γ)
///   [2] output  — float [numTokens × hiddenDim]
///   [3] dims    — uint2 { numTokens, hiddenDim }
///   [4] params  — float (epsilon, usually 1e-5 or 1e-6)

#include <metal_stdlib>
using namespace metal;

kernel void rmsnorm(
    device const float* input   [[buffer(0)]],
    device const float* weight  [[buffer(1)]],   // γ scale parameter
    device float*       output  [[buffer(2)]],
    constant uint2&     dims    [[buffer(3)]],   // { numTokens, hiddenDim }
    constant float&     eps     [[buffer(4)]],   // epsilon (e.g. 1e-5)
    uint  tgid   [[threadgroup_position_in_grid]],
    uint  lid    [[thread_position_in_threadgroup]],
    uint  tgSize [[threads_per_threadgroup]],
    threadgroup float* scratch  [[threadgroup(0)]]  // tgSize floats
) {
    uint numTokens = dims.x;
    uint hiddenDim = dims.y;
    uint token     = tgid;

    if (token >= numTokens) return;

    device const float* in_row  = input  + token * hiddenDim;
    device float*       out_row = output + token * hiddenDim;

    // ── Step 1: Compute partial sum-of-squares ─────────────────────────
    // Each thread accumulates over its strided slice.
    float thread_ss = 0.0;
    for (uint i = lid; i < hiddenDim; i += tgSize) {
        float x = in_row[i];
        thread_ss += x * x;
    }

    // Parallel reduction for total sum-of-squares
    scratch[lid] = thread_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            scratch[lid] += scratch[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Step 2: Compute RMS and its reciprocal ─────────────────────────
    // mean_sq = sum_sq / hiddenDim
    // rms     = sqrt(mean_sq + epsilon)
    // Scale inverted once so each thread just multiplies.
    float mean_sq  = scratch[0] / float(hiddenDim);
    float inv_rms  = rsqrt(mean_sq + eps);   // rsqrt is hardware-accelerated

    // ── Step 3: Normalize and scale with learned weight ────────────────
    for (uint i = lid; i < hiddenDim; i += tgSize) {
        out_row[i] = in_row[i] * inv_rms * weight[i];
    }
}
