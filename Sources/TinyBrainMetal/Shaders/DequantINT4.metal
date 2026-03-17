/// INT4 Dequantization and Fused Matmul Kernel
///
/// **Phase 1:** Per-group INT4 quantization with packed byte storage
///
/// ## INT4 Packing Format
///
/// Two 4-bit signed integers are packed into each byte:
///   Byte layout: [high nibble: even element][low nibble: odd element]
///
///   Example:
///     element[0] = 5  → binary 0101 → stored in bits 7-4
///     element[1] = -3 → binary 1101 (4-bit two's complement) → stored in bits 3-0
///     packed byte  = 0101_1101
///
/// ## Per-Group Quantization
///
/// Every `groupSize` consecutive elements share one FP32 scale and one INT4 zero point.
/// Formula: float_value = (int4_value - zero_point) * scale
///
/// This balances accuracy (small groups = more scales = better precision)
/// against overhead (each group adds 5 bytes of metadata).

#include <metal_stdlib>
using namespace metal;

/// Helper: Extract a signed INT4 value from a packed byte
///
/// - byte: The packed byte containing two INT4 values
/// - isHigh: true for high nibble (even index), false for low nibble (odd index)
/// - Returns: Signed value in range [-8, 7]
inline int unpack_int4(char byte, bool isHigh) {
    // Extract the 4-bit nibble
    int nibble;
    if (isHigh) {
        // High nibble: shift right by 4, then mask
        nibble = (int(byte) >> 4) & 0x0F;
    } else {
        // Low nibble: just mask
        nibble = int(byte) & 0x0F;
    }

    // Sign-extend from 4-bit to 32-bit
    // If bit 3 is set (value >= 8), the number is negative in 4-bit two's complement
    if (nibble > 7) {
        nibble -= 16;
    }

    return nibble;
}

/// Simple INT4 dequantization kernel (for debugging/testing)
///
/// Converts packed INT4 → Float32 using per-group scales
///
/// Formula: float_value = (int4_value - zero_point) * scale[group]
kernel void dequantize_int4(
    device const char* quantized [[buffer(0)]],       // Packed INT4 input (2 values per byte)
    device const float* scales [[buffer(1)]],          // Per-group FP32 scales
    device const char* zeroPoints [[buffer(2)]],       // Per-group INT4 zero points
    device float* output [[buffer(3)]],                // Float32 output
    constant uint3& params [[buffer(4)]],              // [totalElements, groupSize, 0]
    uint gid [[thread_position_in_grid]]
) {
    uint totalElements = params.x;
    uint groupSize = params.y;

    if (gid >= totalElements) return;

    // Determine which group this element belongs to
    uint groupIdx = gid / groupSize;
    float scale = scales[groupIdx];
    int zp = int(zeroPoints[groupIdx]);

    // Unpack the INT4 value from its packed byte
    uint byteIdx = gid / 2;
    bool isHigh = (gid % 2 == 0);  // Even indices in high nibble
    int int4Val = unpack_int4(quantized[byteIdx], isHigh);

    // Dequantize: float = (int4 - zero_point) * scale
    output[gid] = float(int4Val - zp) * scale;
}

/// Fused INT4 dequant + matmul kernel
///
/// C[M,N] = A[M,K] x dequant(B_packed[K,N])
///
/// Where B is stored as packed INT4 with per-group scales.
/// Each pair of consecutive elements along the K dimension is packed
/// into one byte.
///
/// **Performance advantage:** Dequantization happens in registers
/// during the matmul computation — no intermediate Float32 buffer
/// is ever materialized in memory. This is critical for INT4 where
/// we save 87.5% memory vs Float32.
kernel void matmul_int4_dequant(
    device const float* A [[buffer(0)]],               // Float32 input [M, K]
    device const char* B_packed [[buffer(1)]],          // Packed INT4 weights [K*N/2 bytes]
    device const float* B_scales [[buffer(2)]],         // Per-group scales
    device const char* B_zeroPoints [[buffer(3)]],      // Per-group zero points
    device float* C [[buffer(4)]],                      // Float32 output [M, N]
    constant uint3& dims [[buffer(5)]],                 // [M, N, K]
    constant uint& groupSize [[buffer(6)]],             // Group size (e.g., 128)
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;

    if (row >= M || col >= N) return;

    float sum = 0.0;

    for (uint k = 0; k < K; k++) {
        // A value (already Float32)
        float a_val = A[row * K + k];

        // B value: unpack INT4 and dequantize on-the-fly
        //
        // B is stored in row-major order [K, N], so the linear index
        // for element (k, col) is: k * N + col
        uint linearIdx = k * N + col;
        uint byteIdx = linearIdx / 2;
        bool isHigh = (linearIdx % 2 == 0);

        int int4Val = unpack_int4(B_packed[byteIdx], isHigh);

        // Per-group scale: groups are over the flattened B tensor
        uint groupIdx = linearIdx / groupSize;
        float b_scale = B_scales[groupIdx];
        int b_zp = int(B_zeroPoints[groupIdx]);

        float b_val = float(int4Val - b_zp) * b_scale;

        sum += a_val * b_val;
    }

    C[row * N + col] = sum;
}
