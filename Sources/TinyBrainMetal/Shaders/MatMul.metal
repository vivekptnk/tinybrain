#include <metal_stdlib>
using namespace metal;

/// Naive matrix multiplication kernel (educational implementation)
///
/// Computes: C = A × B where A is [M, K], B is [K, N], C is [M, N]
///
/// **How it works:**
/// Each GPU thread computes ONE element of the result matrix.
/// Thread at position (row, col) computes C[row, col] by:
/// - Taking dot product of A's row with B's column
/// - C[row, col] = Σ(A[row, k] × B[k, col]) for k = 0 to K-1
///
/// **Why "naive"?**
/// This implementation reads from global memory (slow!).
/// Each read takes ~100 GPU cycles.
/// We'll optimize with threadgroup memory in matmul_tiled.
///
/// **Parameters:**
/// - A: Input matrix A [M, K] (row-major)
/// - B: Input matrix B [K, N] (row-major)
/// - C: Output matrix C [M, N] (row-major)
/// - dims: uint3 containing (M, N, K)
/// - gid: 2D thread position (row, col) in output matrix
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // Extract dimensions
    uint M = dims.x;  // Rows in A
    uint N = dims.y;  // Cols in B
    uint K = dims.z;  // Cols in A = Rows in B
    
    // Which element am I computing?
    uint row = gid.y;  // My row in C
    uint col = gid.x;  // My column in C
    
    // Bounds check (some threads may be outside matrix)
    if (row >= M || col >= N) {
        return;
    }
    
    // Compute dot product: A[row, :] · B[:, col]
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        // A[row, k]: row-major index = row * K + k
        float a_elem = A[row * K + k];
        
        // B[k, col]: row-major index = k * N + col
        float b_elem = B[k * N + col];
        
        sum += a_elem * b_elem;
    }
    
    // Write result: C[row, col]
    C[row * N + col] = sum;
}

// Placeholder for tiled (optimized) implementation
// Will be written in Phase 4 (Task 10)
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    threadgroup float* tileA [[threadgroup(0)]],
    threadgroup float* tileB [[threadgroup(1)]]
) {
    // TODO: Implement tiled version with threadgroup memory
    // This will be 10-20× faster than naive!
}

