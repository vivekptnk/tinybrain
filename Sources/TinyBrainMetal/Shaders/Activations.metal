/// Activation function kernels: SiLU and GELU
///
/// Both are element-wise operations — every thread independently handles
/// a chunk of the output.  No inter-thread communication needed.
///
/// ─────────────────────────────────────────────────────────────────────
/// SiLU  (Sigmoid Linear Unit / Swish)
/// ─────────────────────────────────────────────────────────────────────
///
/// Formula:  SiLU(x) = x * σ(x)   where σ(x) = 1 / (1 + exp(-x))
///
/// Properties:
///   - Smooth, non-monotonic (has a small dip below 0 for negative x)
///   - Self-gated: the input gates itself via the sigmoid
///   - Used in:  LLaMA FFN gate projection (x ⊙ SiLU(gate))
///              MobileNet-V3, EfficientNet
///
/// ─────────────────────────────────────────────────────────────────────
/// GELU  (Gaussian Error Linear Unit)
/// ─────────────────────────────────────────────────────────────────────
///
/// Exact formula (used here):
///   GELU(x) = x * Φ(x)     where Φ is the CDF of N(0,1)
///
/// Fast tanh approximation (Hendrycks & Gimpel):
///   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// Properties:
///   - Smooth, probabilistic gating
///   - Used in:  GPT-2, BERT, RoBERTa, ViT
///
/// ─────────────────────────────────────────────────────────────────────
/// Dispatch (both kernels):
///   Each thread handles one element.
///   threadsPerThreadgroup: 256
///   threadgroupsPerGrid:   ceil(n / 256)
/// ─────────────────────────────────────────────────────────────────────

#include <metal_stdlib>
using namespace metal;

/// SiLU (Swish) activation: out[i] = x[i] * sigmoid(x[i])
///
/// Buffers:
///   [0] input  — float [n]
///   [1] output — float [n]
///   [2] n      — uint (element count)
kernel void silu(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint&      n       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = input[gid];

    // σ(x) = 1 / (1 + exp(-x))
    // Numerically equivalent to using metal's built-in sigmoid, but written
    // explicitly here so the algorithm is obvious.
    float sigmoid_x = 1.0 / (1.0 + exp(-x));

    output[gid] = x * sigmoid_x;
}

/// GELU activation (tanh approximation)
///
/// Uses the fast polynomial approximation from Hendrycks & Gimpel (2016):
///   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
///
/// The constant √(2/π) ≈ 0.7978845608.
///
/// Buffers:
///   [0] input  — float [n]
///   [1] output — float [n]
///   [2] n      — uint (element count)
kernel void gelu(
    device const float* input   [[buffer(0)]],
    device float*       output  [[buffer(1)]],
    constant uint&      n       [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;

    float x = input[gid];

    // √(2/π) ≈ 0.7978845608
    const float kSqrt2OverPi = 0.7978845608;
    const float kCoeff        = 0.044715;

    float x3    = x * x * x;
    float inner = kSqrt2OverPi * (x + kCoeff * x3);
    output[gid] = 0.5 * x * (1.0 + tanh(inner));
}
