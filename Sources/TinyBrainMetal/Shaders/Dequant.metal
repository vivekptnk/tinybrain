/// INT8 Dequantization and Fused Operations
///
/// **REVIEW HITLER FIX:** Real INT8 compute kernel, not Float32 conversion!
///
/// Implements:
/// - INT8 → Float32 dequantization
/// - Fused INT8 dequant + matmul (single kernel)
/// - Per-channel scale handling

#include <metal_stdlib>
using namespace metal;

/// Simple INT8 dequantization kernel
///
/// Converts INT8 → Float32 using per-channel scales
///
/// Formula: float_value = int8_value * scale[channel]
kernel void dequantize_int8(
    device const char* quantized [[buffer(0)]],      // INT8 input
    device const float* scales [[buffer(1)]],         // Per-channel scales
    device float* output [[buffer(2)]],               // Float32 output
    constant uint2& dims [[buffer(3)]],               // [numChannels, channelSize]
    uint2 gid [[thread_position_in_grid]]
) {
    uint numChannels = dims.x;
    uint channelSize = dims.y;
    
    uint row = gid.y;  // Channel index
    uint col = gid.x;  // Element within channel
    
    if (row >= numChannels || col >= channelSize) return;
    
    // Get scale for this channel
    float scale = scales[row];
    
    // Dequantize: int8 * scale
    uint idx = row * channelSize + col;
    char quantizedValue = quantized[idx];
    output[idx] = float(quantizedValue) * scale;
}

/// **REVIEW HITLER FIX:** Fused INT8 dequant + matmul kernel
///
/// This is THE solution - compute matmul directly from INT8 without materializing Float32!
///
/// C[M,N] = A[M,K] × dequant(B[K,N])
///
/// Where dequant(B) = B_int8 * scales (per-channel)
kernel void matmul_int8_dequant(
    device const float* A [[buffer(0)]],              // Float32 input [M, K]
    device const char* B_quantized [[buffer(1)]],     // INT8 weights [K, N]
    device const float* B_scales [[buffer(2)]],       // Per-channel scales [K]
    device float* C [[buffer(3)]],                    // Float32 output [M, N]
    constant uint3& dims [[buffer(4)]],               // [M, N, K]
    uint2 gid [[thread_position_in_grid]]
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    uint row = gid.y, col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // Compute dot product: A[row,:] · dequant(B[:,col])
    float sum = 0.0;
    
    for (uint k = 0; k < K; k++) {
        // A value (already Float32)
        float a_val = A[row * K + k];
        
        // B value (INT8, need to dequantize on-the-fly)
        char b_quantized = B_quantized[k * N + col];
        float b_scale = B_scales[k];  // Per-channel scale
        float b_val = float(b_quantized) * b_scale;
        
        // Accumulate
        sum += a_val * b_val;
    }
    
    C[row * N + col] = sum;
}

/// **OPTIMIZED:** Tiled fused INT8 dequant + matmul
///
/// Uses threadgroup memory for better cache locality
kernel void matmul_int8_dequant_tiled(
    device const float* A [[buffer(0)]],
    device const char* B_quantized [[buffer(1)]],
    device const float* B_scales [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint3& dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]]  // Dequantized on load
) {
    uint M = dims.x, N = dims.y, K = dims.z;
    const uint TILE_SIZE = 16;
    
    uint row = gid.y;
    uint col = gid.x;
    
    float sum = 0.0;
    
    // Tile over K dimension
    for (uint t = 0; t < K; t += TILE_SIZE) {
        // Load A tile (already Float32)
        if (row < M && (t + tid.x) < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[row * K + (t + tid.x)];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0;
        }
        
        // Load B tile and DEQUANTIZE during load!
        if ((t + tid.y) < K && col < N) {
            uint k_idx = t + tid.y;
            char b_quant = B_quantized[k_idx * N + col];
            float b_scale = B_scales[k_idx];
            tileB[tid.y * TILE_SIZE + tid.x] = float(b_quant) * b_scale;
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute using dequantized tile
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

