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

/// Tiled matrix multiplication kernel (OPTIMIZED!)
///
/// This is the FAST version using threadgroup (shared) memory.
///
/// **The Optimization:**
/// Instead of reading from slow global memory repeatedly, we:
/// 1. Load a TILE of data into fast threadgroup memory
/// 2. All threads in the group share this tile
/// 3. Compute using the fast shared memory
/// 4. Repeat for next tile
///
/// **Speed difference:**
/// - Global memory: ~100 cycles per read
/// - Threadgroup memory: ~5 cycles per read
/// - **20× faster memory access!**
///
/// **Tiling Strategy:**
/// For 1024×1024 matrix:
/// - Divide into 16×16 tiles (TILE_SIZE = 16)
/// - Each threadgroup processes one 16×16 output tile
/// - Loads 16×16 chunks from A and B into shared memory
/// - 64×64 threadgroups total for 1024×1024
///
/// **Threadgroup Memory Usage:**
/// - tileA: 16×16 floats = 1 KB
/// - tileB: 16×16 floats = 1 KB
/// - Total: 2 KB (well under 32 KB limit) ✅
kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint3& dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],           // Global thread ID
    uint2 tid [[thread_position_in_threadgroup]],    // Local thread ID within group
    threadgroup float* tileA [[threadgroup(0)]],     // Shared memory for A tile
    threadgroup float* tileB [[threadgroup(1)]]      // Shared memory for B tile
) {
    // Extract dimensions
    uint M = dims.x;  // Rows in A
    uint N = dims.y;  // Cols in B
    uint K = dims.z;  // Cols in A = Rows in B
    
    const uint TILE_SIZE = 16;
    
    // Which output element am I computing?
    uint row = gid.y;
    uint col = gid.x;
    
    // Accumulator for this thread's output element
    float sum = 0.0;
    
    // Loop over tiles along the K dimension
    for (uint t = 0; t < K; t += TILE_SIZE) {
        // Load tile of A into threadgroup memory
        // Each thread loads one element
        if (row < M && (t + tid.x) < K) {
            tileA[tid.y * TILE_SIZE + tid.x] = A[row * K + (t + tid.x)];
        } else {
            tileA[tid.y * TILE_SIZE + tid.x] = 0.0;  // Pad with zeros
        }
        
        // Load tile of B into threadgroup memory
        if ((t + tid.y) < K && col < N) {
            tileB[tid.y * TILE_SIZE + tid.x] = B[(t + tid.y) * N + col];
        } else {
            tileB[tid.y * TILE_SIZE + tid.x] = 0.0;  // Pad with zeros
        }
        
        // Wait for all threads in group to finish loading
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute using threadgroup memory (FAST!)
        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
        }
        
        // Wait before loading next tile
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Write final result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

