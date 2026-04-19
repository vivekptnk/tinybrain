/// Metal acceleration backend for TinyBrain
///
/// Provides GPU-accelerated operations for inference using custom Metal kernels.
/// Handles shader loading, buffer management, and command encoding.

import Metal
import Foundation
import TinyBrainRuntime

/// Metal backend for accelerated tensor operations
///
/// Supports FP32/INT8/INT4 matmul, softmax, RMSNorm, activations, and FlashAttention.
public final class MetalBackend: MatMulBackend, TensorUploader, TensorDownloader, QuantizedMatMulBackend,
                                  SoftmaxBackend, RMSNormBackend, ActivationBackend,
                                  AttentionBackend {
    /// Shared Metal device (the GPU)
    private let device: MTLDevice
    
    /// Command queue for GPU operations
    private let commandQueue: MTLCommandQueue
    
    /// Shader library containing compiled .metal files (lazy-loaded)
    private var library: MTLLibrary?
    
    /// Cache of compiled compute pipelines (kernel name → pipeline)
    /// Avoids recompiling shaders on every use
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    
    /// **TB-004:** Persistent buffer pool to eliminate 0.45ms allocation overhead
    public let bufferPool: MetalBufferPool
    
    /// Cache for GPU-resident quantized weight buffers (keyed by QuantizedTensor.identifier)
    private var quantizedBufferCache: [UUID: QuantizedBufferEntry] = [:]
    private let quantizedCacheLock = NSLock()
    
    /// Initialize the Metal backend
    /// - Throws: `MetalError` if Metal is unavailable
    public init() throws {
        // 1. Find the GPU
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MetalError.deviceNotFound
        }
        self.device = device
        
        // 2. Create command queue (work queue for GPU)
        guard let commandQueue = device.makeCommandQueue() else {
            throw MetalError.commandQueueCreationFailed
        }
        self.commandQueue = commandQueue
        
        // 3. Create buffer pool for persistent GPU buffers (TB-004 optimization)
        self.bufferPool = MetalBufferPool(device: device)
        
        // Note: Library is loaded lazily when first kernel is requested
        // This allows MetalBackend to initialize even if shaders aren't compiled yet
    }
    
    /// Load and compile a Metal kernel function
    ///
    /// Shaders are cached after first load for performance.
    /// Compiles shader source at runtime for educational transparency.
    ///
    /// - Parameter name: The kernel function name (e.g., "matmul_naive")
    /// - Returns: Compiled compute pipeline ready to execute
    /// - Throws: `MetalError.shaderCompilationFailed` if kernel doesn't exist or fails to compile
    public func loadKernel(named name: String) throws -> MTLComputePipelineState {
        // Check cache first
        if let cached = pipelineCache[name] {
            return cached
        }
        
        // Load library if not loaded yet
        if library == nil {
            // Try default library first
            if let lib = device.makeDefaultLibrary() {
                library = lib
            } else {
                // If no default library, compile from source (for SPM)
                library = try compileShaderLibrary()
            }
        }
        
        // Load function from library
        guard let function = library!.makeFunction(name: name) else {
            throw MetalError.shaderCompilationFailed("Function '\(name)' not found in shader library")
        }
        
        // Compile to pipeline
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            pipelineCache[name] = pipeline
            return pipeline
        } catch {
            throw MetalError.shaderCompilationFailed("Failed to compile '\(name)': \(error)")
        }
    }
    
    /// Compile Metal shader library from source
    ///
    /// **REVIEW HITLER FIX:** Now includes INT8 dequant + fused matmul kernels!
    ///
    /// Used when SPM doesn't pre-compile .metal files.
    /// Compiles at runtime for flexibility.
    ///
    /// - Returns: Compiled Metal library
    /// - Throws: `MetalError.shaderCompilationFailed` if compilation fails
    private func compileShaderLibrary() throws -> MTLLibrary {
        // Metal shader source (embedded for SPM compatibility)
        // Includes matmul kernels + Phase 3 ops: softmax, rmsnorm, silu, gelu
        let source = """
        #include <metal_stdlib>
        using namespace metal;

        // Naive kernel (simple, slower)
        kernel void matmul_naive(
            device const float* A [[buffer(0)]],
            device const float* B [[buffer(1)]],
            device float* C [[buffer(2)]],
            constant uint3& dims [[buffer(3)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint M = dims.x, N = dims.y, K = dims.z;
            uint row = gid.y, col = gid.x;

            if (row >= M || col >= N) return;

            float sum = 0.0;
            for (uint k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }

        // Tiled kernel (optimized with threadgroup memory)
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
            uint M = dims.x, N = dims.y, K = dims.z;
            const uint TILE_SIZE = 16;

            uint row = gid.y;
            uint col = gid.x;

            float sum = 0.0;

            for (uint t = 0; t < K; t += TILE_SIZE) {
                // Load tiles into shared memory
                if (row < M && (t + tid.x) < K) {
                    tileA[tid.y * TILE_SIZE + tid.x] = A[row * K + (t + tid.x)];
                } else {
                    tileA[tid.y * TILE_SIZE + tid.x] = 0.0;
                }

                if ((t + tid.y) < K && col < N) {
                    tileB[tid.y * TILE_SIZE + tid.x] = B[(t + tid.y) * N + col];
                } else {
                    tileB[tid.y * TILE_SIZE + tid.x] = 0.0;
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                // Compute using fast shared memory
                for (uint k = 0; k < TILE_SIZE; k++) {
                    sum += tileA[tid.y * TILE_SIZE + k] * tileB[k * TILE_SIZE + tid.x];
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }

        // INT8 dequant + fused matmul kernels

        // Fused INT8 dequant + matmul
        kernel void matmul_int8_dequant(
            device const float* A [[buffer(0)]],
            device const char* B_quantized [[buffer(1)]],
            device const float* B_scales [[buffer(2)]],
            device float* C [[buffer(3)]],
            constant uint3& dims [[buffer(4)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint M = dims.x, N = dims.y, K = dims.z;
            uint row = gid.y, col = gid.x;

            if (row >= M || col >= N) return;

            float sum = 0.0;

            for (uint k = 0; k < K; k++) {
                float a_val = A[row * K + k];

                // Dequantize on-the-fly (scales per output channel = per column)
                char b_quant = B_quantized[k * N + col];
                float b_scale = B_scales[col];
                float b_val = float(b_quant) * b_scale;

                sum += a_val * b_val;
            }

            C[row * N + col] = sum;
        }

        // ── INT4 fused dequant + matmul ────────────────────────────────────
        // Unpacks two 4-bit values from each byte, applies per-group
        // scale and zero point, then accumulates the dot product.
        // B is packed as [K*N/2] bytes with per-group scales.
        kernel void matmul_int4_dequant(
            device const float* A [[buffer(0)]],
            device const char* B_packed [[buffer(1)]],
            device const float* B_scales [[buffer(2)]],
            device const char* B_zeroPoints [[buffer(3)]],
            device float* C [[buffer(4)]],
            constant uint3& dims [[buffer(5)]],
            constant uint& groupSize [[buffer(6)]],
            uint2 gid [[thread_position_in_grid]]
        ) {
            uint M = dims.x, N = dims.y, K = dims.z;
            uint row = gid.y, col = gid.x;

            if (row >= M || col >= N) return;

            float sum = 0.0;

            for (uint k = 0; k < K; k++) {
                float a_val = A[row * K + k];

                // B element at (k, col): linear index = k * N + col
                uint linearIdx = k * N + col;
                uint byteIdx = linearIdx / 2;
                char packed = B_packed[byteIdx];

                // Unpack INT4: even index = high nibble, odd = low nibble
                int nibble;
                if (linearIdx % 2 == 0) {
                    nibble = (int(packed) >> 4) & 0x0F;
                } else {
                    nibble = int(packed) & 0x0F;
                }
                // Sign-extend from 4-bit
                if (nibble > 7) nibble -= 16;

                // Per-group dequantize
                uint groupIdx = linearIdx / groupSize;
                float b_scale = B_scales[groupIdx];
                int b_zp = int(B_zeroPoints[groupIdx]);
                float b_val = float(nibble - b_zp) * b_scale;

                sum += a_val * b_val;
            }

            C[row * N + col] = sum;
        }

        // ── Phase 3: Softmax ─────────────────────────────────────────────
        // Numerically stable softmax using parallel max + sum reductions.
        // One threadgroup per row. Uses threadgroup(0) scratch memory.
        kernel void softmax(
            device const float* input   [[buffer(0)]],
            device float*       output  [[buffer(1)]],
            constant uint2&     dims    [[buffer(2)]],
            uint  tgid   [[threadgroup_position_in_grid]],
            uint  lid    [[thread_position_in_threadgroup]],
            uint  tgSize [[threads_per_threadgroup]],
            threadgroup float* scratch  [[threadgroup(0)]]
        ) {
            uint numRows = dims.x;
            uint rowLen  = dims.y;
            uint row     = tgid;
            if (row >= numRows) return;
            device const float* in_row  = input  + row * rowLen;
            device float*       out_row = output + row * rowLen;
            // Step 1: parallel max
            float thread_max = -INFINITY;
            for (uint i = lid; i < rowLen; i += tgSize) thread_max = max(thread_max, in_row[i]);
            scratch[lid] = thread_max;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) scratch[lid] = max(scratch[lid], scratch[lid + stride]);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float row_max = scratch[0];
            // Step 2+3: exp(x - max), accumulate sum
            float thread_sum = 0.0;
            for (uint i = lid; i < rowLen; i += tgSize) {
                float e = exp(in_row[i] - row_max);
                out_row[i] = e;
                thread_sum += e;
            }
            scratch[lid] = thread_sum;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) scratch[lid] += scratch[lid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float inv_sum = 1.0 / (scratch[0] + 1e-7);
            // Step 4: normalize
            for (uint i = lid; i < rowLen; i += tgSize) out_row[i] *= inv_sum;
        }

        // ── Phase 3: RMSNorm ─────────────────────────────────────────────
        // out = (x / rms(x)) * weight,  rms(x) = sqrt(mean(x²) + eps)
        kernel void rmsnorm(
            device const float* input   [[buffer(0)]],
            device const float* weight  [[buffer(1)]],
            device float*       output  [[buffer(2)]],
            constant uint2&     dims    [[buffer(3)]],
            constant float&     eps     [[buffer(4)]],
            uint  tgid   [[threadgroup_position_in_grid]],
            uint  lid    [[thread_position_in_threadgroup]],
            uint  tgSize [[threads_per_threadgroup]],
            threadgroup float* scratch  [[threadgroup(0)]]
        ) {
            uint numTokens = dims.x;
            uint hiddenDim = dims.y;
            uint token     = tgid;
            if (token >= numTokens) return;
            device const float* in_row  = input  + token * hiddenDim;
            device float*       out_row = output + token * hiddenDim;
            float thread_ss = 0.0;
            for (uint i = lid; i < hiddenDim; i += tgSize) { float x = in_row[i]; thread_ss += x * x; }
            scratch[lid] = thread_ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint stride = tgSize / 2; stride > 0; stride >>= 1) {
                if (lid < stride) scratch[lid] += scratch[lid + stride];
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            float inv_rms = rsqrt(scratch[0] / float(hiddenDim) + eps);
            for (uint i = lid; i < hiddenDim; i += tgSize) out_row[i] = in_row[i] * inv_rms * weight[i];
        }

        // ── Phase 3: SiLU ────────────────────────────────────────────────
        // out[i] = x[i] * sigmoid(x[i])
        kernel void silu(
            device const float* input   [[buffer(0)]],
            device float*       output  [[buffer(1)]],
            constant uint&      n       [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            float x = input[gid];
            output[gid] = x / (1.0 + exp(-x));
        }

        // ── Phase 3: GELU ────────────────────────────────────────────────
        // Tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        kernel void gelu(
            device const float* input   [[buffer(0)]],
            device float*       output  [[buffer(1)]],
            constant uint&      n       [[buffer(2)]],
            uint gid [[thread_position_in_grid]]
        ) {
            if (gid >= n) return;
            float x = input[gid];
            float x3 = x * x * x;
            float inner = 0.7978845608 * (x + 0.044715 * x3);
            output[gid] = 0.5 * x * (1.0 + tanh(inner));
        }

        // ── Phase 3: FlashAttention ─────────────────────────────────────
        // Fused QK^T + online softmax + V in a single GPU pass.
        // Tile sizes are compile-time constants (Br=Bc=32).

        struct AttentionParams {
            uint seqLen;
            uint kvSeqLen;
            uint headDim;
            uint numHeads;
            uint numKVHeads;
            uint batch;
            float scale;
            uint useMask;
        };

        constant uint TILE_Br = 32;
        constant uint TILE_Bc = 32;

        kernel void flash_attention(
            device const float* query   [[buffer(0)]],
            device const float* keys    [[buffer(1)]],
            device const float* values  [[buffer(2)]],
            device float*       output  [[buffer(3)]],
            device const float* mask    [[buffer(4)]],
            constant AttentionParams& params [[buffer(5)]],
            uint3 tgid [[threadgroup_position_in_grid]],
            uint3 tid  [[thread_position_in_threadgroup]],
            threadgroup float* smem [[threadgroup(0)]]
        ) {
            const uint head     = tgid.x;
            const uint qBlock   = tgid.y;
            const uint batchIdx = tgid.z;

            const uint headDim   = params.headDim;
            const uint seqLen    = params.seqLen;
            const uint kvSeqLen  = params.kvSeqLen;
            const uint numHeads  = params.numHeads;
            const uint numKVHeads = params.numKVHeads;
            const float scaleFactor = params.scale;

            const uint kvHead = head / (numHeads / numKVHeads);
            const uint tr = tid.y;
            const uint tc = tid.x;
            const uint qRow = qBlock * TILE_Br + tr;
            const bool validRow = (qRow < seqLen);

            const uint qStride  = numHeads * headDim;
            const uint kvStride = numKVHeads * headDim;

            device const float* Q_base = query  + batchIdx * seqLen * qStride;
            device const float* K_base = keys   + batchIdx * kvSeqLen * kvStride;
            device const float* V_base = values + batchIdx * kvSeqLen * kvStride;
            device float*       O_base = output + batchIdx * seqLen * qStride;

            device const float* q_row = Q_base + qRow * qStride + head * headDim;
            device float*       o_row = O_base + qRow * qStride + head * headDim;

            threadgroup float* S_tile  = smem;
            threadgroup float* row_max = smem + TILE_Br * TILE_Bc;
            threadgroup float* row_sum = row_max + TILE_Br;
            threadgroup float* O_acc   = row_sum + TILE_Br;

            if (tc == 0) {
                row_max[tr] = -INFINITY;
                row_sum[tr] = 0.0;
            }
            for (uint d = tc; d < headDim; d += TILE_Bc) {
                O_acc[tr * headDim + d] = 0.0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            const uint numKVTiles = (kvSeqLen + TILE_Bc - 1) / TILE_Bc;

            for (uint kvTile = 0; kvTile < numKVTiles; kvTile++) {
                const uint kvStart = kvTile * TILE_Bc;
                const uint kvCol = kvStart + tc;

                float score = -INFINITY;
                if (validRow && kvCol < kvSeqLen) {
                    score = 0.0;
                    device const float* k_col = K_base + kvCol * kvStride + kvHead * headDim;
                    for (uint d = 0; d < headDim; d++) {
                        score += q_row[d] * k_col[d];
                    }
                    score *= scaleFactor;
                    if (params.useMask != 0 && mask != nullptr) {
                        score += mask[kvCol];
                    }
                }

                S_tile[tr * TILE_Bc + tc] = score;
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (tc == 0 && validRow) {
                    float tile_max = -INFINITY;
                    for (uint c = 0; c < TILE_Bc; c++) {
                        tile_max = max(tile_max, S_tile[tr * TILE_Bc + c]);
                    }
                    float prev_max = row_max[tr];
                    float new_max = max(prev_max, tile_max);
                    float prev_correction = exp(prev_max - new_max);
                    float new_sum = row_sum[tr] * prev_correction;
                    for (uint c = 0; c < TILE_Bc; c++) {
                        float s = S_tile[tr * TILE_Bc + c];
                        float p = exp(s - new_max);
                        S_tile[tr * TILE_Bc + c] = p;
                        new_sum += p;
                    }
                    for (uint d = 0; d < headDim; d++) {
                        O_acc[tr * headDim + d] *= prev_correction;
                    }
                    row_max[tr] = new_max;
                    row_sum[tr] = new_sum;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (validRow) for (uint d = tc; d < headDim; d += TILE_Bc) {
                    float acc = 0.0;
                    for (uint c = 0; c < TILE_Bc; c++) {
                        uint kvPos = kvStart + c;
                        if (kvPos < kvSeqLen) {
                            float p = S_tile[tr * TILE_Bc + c];
                            device const float* v_row = V_base + kvPos * kvStride + kvHead * headDim;
                            acc += p * v_row[d];
                        }
                    }
                    O_acc[tr * headDim + d] += acc;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (validRow) {
                float inv_sum = 1.0 / (row_sum[tr] + 1e-7);
                for (uint d = tc; d < headDim; d += TILE_Bc) {
                    o_row[d] = O_acc[tr * headDim + d] * inv_sum;
                }
            }
        }
        """
        
        do {
            return try device.makeLibrary(source: source, options: nil)
        } catch {
            throw MetalError.shaderCompilationFailed("Failed to compile shader source: \(error)")
        }
    }
    
    /// Check if Metal is available on this device
    public static var isAvailable: Bool {
        MTLCreateSystemDefaultDevice() != nil
    }
    
    /// Get device information
    public var deviceInfo: String {
        device.name
    }
    
    /// Get device for buffer creation (needed by buffer management)
    internal var metalDevice: MTLDevice {
        device
    }
    
    /// Get command queue for encoding (needed by operations)
    internal var queue: MTLCommandQueue {
        commandQueue
    }
    
    // MARK: - Buffer Management
    
    /// **TB-004:** Create a Metal buffer from tensor data (using pool)
    ///
    /// Copies data from CPU memory to GPU memory, reusing buffers from pool.
    ///
    /// **Performance:** ~0.001ms (pool lookup) vs 0.45ms (new allocation)
    ///
    /// - Parameter tensor: Source tensor
    /// - Returns: Metal buffer containing tensor data
    /// - Throws: `MetalError.bufferCreationFailed` if allocation fails
    /// Create buffer with ownership tracking
    ///
    /// **REVIEW HITLER FIX:** Track whether buffer is newly acquired or reused
    ///
    /// - Returns: (buffer, isNewlyAcquired)
    private func createBufferWithTracking(from tensor: Tensor<Float>) throws -> (MTLBuffer, Bool) {
        let elementCount = tensor.shape.count
        
        // **TB-004 CRITICAL FIX:** If tensor already on GPU, reuse its buffer!
        if tensor.isOnGPU {
            let storage = tensor.tensorStorage
            if let gpuBuffer = storage.gpuBuffer as? MTLBuffer {
                // Already on GPU - reuse the buffer (zero transfer!)
                return (gpuBuffer, false)  // NOT newly acquired
            }
        }
        
        // Tensor on CPU - need to upload
        // Get buffer from pool (or create new)
        let buffer = try bufferPool.acquire(elementCount: elementCount)
        
        // Copy data to GPU
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: elementCount)
        let tensorData = tensor.rawData
        for i in 0..<elementCount {
            pointer[i] = tensorData[i]
        }
        
        return (buffer, true)  // Newly acquired, must release later
    }
    
    public func createBuffer(from tensor: Tensor<Float>) throws -> MTLBuffer {
        let (buffer, _) = try createBufferWithTracking(from: tensor)
        return buffer
    }
    
    /// **TB-004:** Create an empty Metal buffer for results (using pool)
    ///
    /// - Parameter elementCount: Number of Float elements
    /// - Returns: Empty Metal buffer (may be reused from pool)
    /// - Throws: `MetalError.bufferCreationFailed` if allocation fails
    public func createBuffer(elementCount: Int) throws -> MTLBuffer {
        // Get buffer from pool (or create new)
        return try bufferPool.acquire(elementCount: elementCount)
    }
    
    /// Release a buffer back to the pool
    ///
    /// **TB-004:** This enables buffer reuse for next operation.
    ///
    /// - Parameters:
    ///   - buffer: Buffer to release
    ///   - elementCount: Number of elements in buffer
    public func releaseBuffer(_ buffer: MTLBuffer, elementCount: Int) {
        bufferPool.release(buffer, elementCount: elementCount)
    }
    
    /// Read Metal buffer back to Swift array
    ///
    /// - Parameters:
    ///   - buffer: Metal buffer containing results
    ///   - count: Number of Float elements to read
    /// - Returns: Array of Float values from GPU
    public func readBuffer(_ buffer: MTLBuffer, count: Int) -> [Float] {
        let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
        return Array(UnsafeBufferPointer(start: pointer, count: count))
    }
    
    // MARK: - TB-004: GPU-Resident Tensor Support
    
    /// Upload tensor to GPU
    ///
    /// **TB-004:** This creates a GPU-resident tensor that stays on GPU.
    ///
    /// - Parameter tensor: CPU tensor to upload
    /// - Returns: GPU-resident tensor
    /// - Throws: MetalError if upload fails
    public func uploadTensor(_ tensor: Tensor<Float>) throws -> Tensor<Float> {
        // Already on GPU?
        if tensor.isOnGPU {
            return tensor
        }
        
        // Create GPU buffer and copy data
        let buffer = try createBuffer(from: tensor)
        
        // Create TensorStorage with GPU buffer
        let storage = TensorStorage<Float>(gpuBuffer: buffer, count: tensor.shape.count)
        
        // **REVIEW HITLER FIX:** Set release callback so uploaded buffers return to pool!
        let pool = self.bufferPool
        let count = tensor.shape.count
        storage.releaseCallback = { [weak pool] in
            pool?.release(buffer, elementCount: count)
        }
        
        // Return new tensor with GPU storage
        return Tensor<Float>(shape: tensor.shape, storage: storage)
    }
    
    /// Download tensor from GPU
    ///
    /// **TB-004:** Sync GPU data back to CPU.
    ///
    /// - Parameter tensor: GPU tensor to download
    /// - Returns: CPU tensor with data copied from GPU
    public func downloadTensor(_ tensor: Tensor<Float>) -> Tensor<Float> {
        // Already on CPU?
        if !tensor.isOnGPU {
            return tensor
        }
        
        // Get GPU buffer from storage
        let storage = tensor.tensorStorage
        guard let buffer = storage.gpuBuffer as? MTLBuffer else {
            // Fallback: already has CPU data
            return tensor
        }
        
        // Read data from GPU
        let cpuData = readBuffer(buffer, count: tensor.shape.count)
        
        // Create CPU tensor
        return Tensor(shape: tensor.shape, data: cpuData)
    }
    
    // MARK: - Matrix Operations (MatMulBackend Protocol)
    
    /// Perform matrix multiplication on GPU using Metal (MatMulBackend protocol)
    ///
    /// Uses the optimized tiled kernel by default for best performance.
    ///
    /// Computes: **C = A × B** using the GPU
    ///
    /// - Parameters:
    ///   - a: Left matrix [M, K]
    ///   - b: Right matrix [K, N]
    /// - Returns: Result matrix [M, N]
    /// - Throws: `MetalError` if GPU operation fails
    ///
    /// Example:
    /// ```swift
    /// let backend = try MetalBackend()
    /// let c = try backend.matmul(a, b)  // Runs on GPU!
    /// ```
    public func matmul(_ a: Tensor<Float>, _ b: Tensor<Float>) throws -> Tensor<Float> {
        // Use optimized tiled version for best performance
        return try matmulOptimized(a, b)
    }
    
    /// Perform matrix multiplication using naive kernel (for debugging/comparison)
    ///
    /// Uses the simpler naive kernel instead of tiled.
    /// Useful for debugging or comparing performance.
    public func matmulNaive(_ a: Tensor<Float>, _ b: Tensor<Float>) throws -> Tensor<Float> {
        precondition(a.shape.dimensions.count == 2, "A must be 2D")
        precondition(b.shape.dimensions.count == 2, "B must be 2D")
        precondition(a.shape.dimensions[1] == b.shape.dimensions[0], "Inner dimensions must match")
        
        let M = UInt32(a.shape.dimensions[0])
        let K = UInt32(a.shape.dimensions[1])
        let N = UInt32(b.shape.dimensions[1])
        
        // Ensure tensors are contiguous (GPU kernel expects row-major)
        let aContiguous = a.isContiguous ? a : a.makeContiguousCopy()
        let bContiguous = b.isContiguous ? b : b.makeContiguousCopy()
        
        // **REVIEW HITLER FIX:** Track which buffers need releasing
        let (bufferA, releaseA) = try createBufferWithTracking(from: aContiguous)
        let (bufferB, releaseB) = try createBufferWithTracking(from: bContiguous)
        let bufferC = try createBuffer(elementCount: Int(M * N))  // Always new
        
        // Dimensions buffer
        var dims = SIMD3<UInt32>(M, N, K)
        let dimsBuffer = device.makeBuffer(
            bytes: &dims,
            length: MemoryLayout<SIMD3<UInt32>>.stride,
            options: .storageModeShared
        )!
        
        // Load and setup kernel
        let pipeline = try loadKernel(named: "matmul_naive")
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer/encoder")
        }
        
        // Configure kernel
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        encoder.setBuffer(dimsBuffer, offset: 0, index: 3)
        
        // Calculate threadgroup sizes
        // Each thread computes one output element
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)  // 256 threads
        let threadgroupsPerGrid = MTLSize(
            width: (Int(N) + 15) / 16,   // Round up
            height: (Int(M) + 15) / 16,
            depth: 1
        )
        
        // Dispatch GPU work
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // **REVIEW HITLER FIX: Release ONLY newly acquired buffers!**
        if releaseA {
            bufferPool.release(bufferA, elementCount: Int(M * K))
        }
        if releaseB {
            bufferPool.release(bufferB, elementCount: Int(K * N))
        }
        // bufferC stays with result tensor
        
        // **TB-004 FIX:** Return GPU-resident tensor, don't download!
        let storage = TensorStorage<Float>(gpuBuffer: bufferC, count: Int(M * N))
        
        // **REVIEW HITLER FIX:** Set release callback so buffer returns to pool when tensor destroyed
        let pool = self.bufferPool
        let resultCount = Int(M * N)
        storage.releaseCallback = { [weak pool] in
            pool?.release(bufferC, elementCount: resultCount)
        }
        
        return Tensor<Float>(shape: TensorShape(Int(M), Int(N)), storage: storage)
    }
    
    /// Perform OPTIMIZED matrix multiplication using tiled kernel
    ///
    /// Uses threadgroup memory for 5-10× speedup over naive kernel.
    ///
    /// - Parameters:
    ///   - a: Left matrix [M, K]
    ///   - b: Right matrix [K, N]
    ///   - useTiled: Whether to use tiled kernel (default: true for large matrices)
    /// - Returns: Result matrix [M, N]
    /// - Throws: `MetalError` if GPU operation fails
    public func matmulOptimized(_ a: Tensor<Float>, _ b: Tensor<Float>, useTiled: Bool = true) throws -> Tensor<Float> {
        precondition(a.shape.dimensions.count == 2, "A must be 2D")
        precondition(b.shape.dimensions.count == 2, "B must be 2D")
        precondition(a.shape.dimensions[1] == b.shape.dimensions[0], "Inner dimensions must match")
        
        let M = UInt32(a.shape.dimensions[0])
        let K = UInt32(a.shape.dimensions[1])
        let N = UInt32(b.shape.dimensions[1])
        
        // Ensure tensors are contiguous
        let aContiguous = a.isContiguous ? a : a.makeContiguousCopy()
        let bContiguous = b.isContiguous ? b : b.makeContiguousCopy()
        
        // **REVIEW HITLER FIX:** Track which buffers need releasing
        let (bufferA, releaseA) = try createBufferWithTracking(from: aContiguous)
        let (bufferB, releaseB) = try createBufferWithTracking(from: bContiguous)
        let bufferC = try createBuffer(elementCount: Int(M * N))  // Always new
        
        // Dimensions buffer
        var dims = SIMD3<UInt32>(M, N, K)
        let dimsBuffer = device.makeBuffer(
            bytes: &dims,
            length: MemoryLayout<SIMD3<UInt32>>.stride,
            options: .storageModeShared
        )!
        
        // Load tiled kernel
        let pipeline = try loadKernel(named: "matmul_tiled")
        
        // Create command buffer and encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer/encoder")
        }
        
        // Configure kernel
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferC, offset: 0, index: 2)
        encoder.setBuffer(dimsBuffer, offset: 0, index: 3)
        
        // Allocate threadgroup memory for tiles
        let tileSize = 16
        let tileSizeBytes = tileSize * tileSize * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(tileSizeBytes, index: 0)  // tileA
        encoder.setThreadgroupMemoryLength(tileSizeBytes, index: 1)  // tileB
        
        // Calculate threadgroup sizes (16×16 threads per group)
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (Int(N) + tileSize - 1) / tileSize,
            height: (Int(M) + tileSize - 1) / tileSize,
            depth: 1
        )
        
        // Dispatch GPU work
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // Execute and wait
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        // **REVIEW HITLER FIX: Release ONLY newly acquired buffers!**
        if releaseA {
            bufferPool.release(bufferA, elementCount: Int(M * K))
        }
        if releaseB {
            bufferPool.release(bufferB, elementCount: Int(K * N))
        }
        // bufferC stays with result tensor
        
        // **TB-004 FIX:** Return GPU-resident tensor, don't download!
        // Create TensorStorage with GPU buffer to keep result on GPU
        let storage = TensorStorage<Float>(gpuBuffer: bufferC, count: Int(M * N))
        
        // **REVIEW HITLER FIX:** Set release callback so buffer returns to pool when tensor destroyed
        let pool = self.bufferPool
        let resultCount = Int(M * N)
        storage.releaseCallback = { [weak pool] in
            pool?.release(bufferC, elementCount: resultCount)
        }
        
        return Tensor<Float>(shape: TensorShape(Int(M), Int(N)), storage: storage)
    }
    
    // MARK: - Quantized Operations (INT8 + INT4)

    /// Matrix multiplication with quantized weights (INT8 or INT4)
    ///
    /// Auto-detects precision from the QuantizedTensor and dispatches
    /// to the appropriate Metal kernel:
    ///   - INT8 per-channel → `matmul_int8_dequant`
    ///   - INT4 per-group   → `matmul_int4_dequant`
    ///
    /// - Parameters:
    ///   - input: Float32 input [M, K]
    ///   - quantized: Quantized weights (INT8 or INT4)
    /// - Returns: Float32 output [M, N]
    public func matmulQuantized(_ input: Tensor<Float>, _ quantized: QuantizedTensor) throws -> Tensor<Float> {
        precondition(input.shape.dimensions.count == 2, "Input must be 2D")
        precondition(quantized.shape.dimensions.count == 2, "Weights must be 2D")
        precondition(input.shape.dimensions[1] == quantized.shape.dimensions[0], "Dimensions must match")

        // Route to INT4 kernel for 4-bit precision
        if quantized.precision == .int4 {
            return try matmulINT4(input, quantized)
        }

        // INT8: Symmetric/asymmetric modes need different kernel (single scale, not per-channel)
        guard quantized.mode == .perChannel else {
            // Fall back to CPU for non-per-channel modes
            let dequantized = quantized.dequantize()
            return try matmul(input, dequantized)
        }

        let M = UInt32(input.shape.dimensions[0])
        let K = UInt32(input.shape.dimensions[1])
        let N = UInt32(quantized.shape.dimensions[1])

        // Ensure input is contiguous
        let inputContiguous = input.isContiguous ? input : input.makeContiguousCopy()

        // Create buffers
        let (bufferA, releaseA) = try createBufferWithTracking(from: inputContiguous)

        // Upload or reuse GPU-resident quantized buffers
        let cacheEntry = try cachedQuantizedBuffers(for: quantized)
        let bufferB = cacheEntry.weights
        let bufferScales = cacheEntry.scales

        // Output buffer
        let bufferC = try createBuffer(elementCount: Int(M * N))

        // Dimensions
        var dims = SIMD3<UInt32>(M, N, K)
        let dimsBuffer = device.makeBuffer(bytes: &dims, length: MemoryLayout<SIMD3<UInt32>>.stride, options: .storageModeShared)!

        // Load kernel
        let pipeline = try loadKernel(named: "matmul_int8_dequant")

        // Encode
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(bufferB, offset: 0, index: 1)
        encoder.setBuffer(bufferScales, offset: 0, index: 2)
        encoder.setBuffer(bufferC, offset: 0, index: 3)
        encoder.setBuffer(dimsBuffer, offset: 0, index: 4)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (Int(N) + 15) / 16,
            height: (Int(M) + 15) / 16,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        // Release input buffer
        if releaseA {
            bufferPool.release(bufferA, elementCount: Int(M * K))
        }

        // Return result
        let storage = TensorStorage<Float>(gpuBuffer: bufferC, count: Int(M * N))
        storage.releaseCallback = { [weak bufferPool] in
            bufferPool?.release(bufferC, elementCount: Int(M * N))
        }

        return Tensor<Float>(shape: TensorShape(Int(M), Int(N)), storage: storage)
    }

    // MARK: - INT4 Quantized MatMul

    /// Fused INT4 dequant + matmul on GPU
    ///
    /// Uses the `matmul_int4_dequant` kernel which unpacks two 4-bit values
    /// per byte and applies per-group scale + zero point during the dot product.
    ///
    /// - Parameters:
    ///   - input: Float32 input [M, K]
    ///   - quantized: INT4 packed weights with per-group scales
    /// - Returns: Float32 output [M, N]
    private func matmulINT4(_ input: Tensor<Float>, _ quantized: QuantizedTensor) throws -> Tensor<Float> {
        let M = UInt32(input.shape.dimensions[0])
        let K = UInt32(input.shape.dimensions[1])
        let N = UInt32(quantized.shape.dimensions[1])

        let inputContiguous = input.isContiguous ? input : input.makeContiguousCopy()
        let (bufferA, releaseA) = try createBufferWithTracking(from: inputContiguous)

        // Upload INT4 packed data, scales, and zero points
        let cacheEntry = try cachedINT4Buffers(for: quantized)

        // Output buffer
        let bufferC = try createBuffer(elementCount: Int(M * N))

        // Dimensions: [M, N, K]
        var dims = SIMD3<UInt32>(M, N, K)
        let dimsBuffer = device.makeBuffer(bytes: &dims, length: MemoryLayout<SIMD3<UInt32>>.stride, options: .storageModeShared)!

        // Group size
        var gs = UInt32(quantized.groupSize > 0 ? quantized.groupSize : Int(K * N))
        let gsBuffer = device.makeBuffer(bytes: &gs, length: MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        let pipeline = try loadKernel(named: "matmul_int4_dequant")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferA, offset: 0, index: 0)
        encoder.setBuffer(cacheEntry.weights, offset: 0, index: 1)
        encoder.setBuffer(cacheEntry.scales, offset: 0, index: 2)
        encoder.setBuffer(cacheEntry.zeroPoints, offset: 0, index: 3)
        encoder.setBuffer(bufferC, offset: 0, index: 4)
        encoder.setBuffer(dimsBuffer, offset: 0, index: 5)
        encoder.setBuffer(gsBuffer, offset: 0, index: 6)

        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (Int(N) + 15) / 16,
            height: (Int(M) + 15) / 16,
            depth: 1
        )

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if releaseA {
            bufferPool.release(bufferA, elementCount: Int(M * K))
        }

        let storage = TensorStorage<Float>(gpuBuffer: bufferC, count: Int(M * N))
        storage.releaseCallback = { [weak bufferPool] in
            bufferPool?.release(bufferC, elementCount: Int(M * N))
        }

        return Tensor<Float>(shape: TensorShape(Int(M), Int(N)), storage: storage)
    }
    // MARK: - Phase 3 helpers

    /// Largest power-of-2 that is ≤ n, capped at maxVal.
    ///
    /// The tree-reduction used in softmax and rmsnorm kernels requires
    /// that the threadgroup size is a power of 2, otherwise threads at
    /// odd indices are silently skipped in the final reduction steps.
    private func pow2ThreadCount(_ n: Int, max maxVal: Int = 256) -> Int {
        var result = 1
        while result * 2 <= min(n, maxVal) { result *= 2 }
        return result
    }

    // MARK: - Phase 3: Softmax (SoftmaxBackend)

    /// Apply numerically stable softmax along the last dimension.
    ///
    /// The kernel dispatches one threadgroup per row; threads collaborate via
    /// threadgroup memory to find the row max, compute shifted exponentials,
    /// then compute the sum and normalize — all without reading global memory
    /// more than three times per element.
    ///
    /// - Parameter input: Tensor of shape [n] or [rows, cols] (any 1-D or 2-D)
    /// - Returns: Tensor of same shape with rows summing to 1
    public func softmax(_ input: Tensor<Float>) throws -> Tensor<Float> {
        let elementCount = input.shape.count
        guard elementCount > 0 else { return input }

        // Treat any input as [numRows × rowLen] where rowLen = last dimension
        let rowLen: Int
        let numRows: Int
        if input.shape.dimensions.count == 1 {
            numRows = 1
            rowLen  = input.shape.dimensions[0]
        } else {
            // Flatten all leading dims into rows
            rowLen  = input.shape.dimensions.last!
            numRows = elementCount / rowLen
        }

        let inputContiguous = input.isContiguous ? input : input.makeContiguousCopy()
        let (bufferIn, releaseIn) = try createBufferWithTracking(from: inputContiguous)
        let bufferOut = try createBuffer(elementCount: elementCount)

        var dims = SIMD2<UInt32>(UInt32(numRows), UInt32(rowLen))
        let dimsBuffer = device.makeBuffer(bytes: &dims,
                                           length: MemoryLayout<SIMD2<UInt32>>.stride,
                                           options: .storageModeShared)!

        let pipeline = try loadKernel(named: "softmax")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer/encoder")
        }

        // One threadgroup per row; power-of-2 thread count ≤ min(rowLen, 256).
        // The tree reduction in the shader requires a power-of-2 threadgroup size
        // so that every element is covered by the halving strides.
        let threadsPerGroup = pow2ThreadCount(rowLen)
        // Scratch memory: one float per thread
        let scratchBytes = threadsPerGroup * MemoryLayout<Float>.stride

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferIn,   offset: 0, index: 0)
        encoder.setBuffer(bufferOut,  offset: 0, index: 1)
        encoder.setBuffer(dimsBuffer, offset: 0, index: 2)
        encoder.setThreadgroupMemoryLength(scratchBytes, index: 0)

        let threadsPerThreadgroup = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let threadgroupsPerGrid   = MTLSize(width: numRows, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if releaseIn { bufferPool.release(bufferIn, elementCount: elementCount) }

        let storage = TensorStorage<Float>(gpuBuffer: bufferOut, count: elementCount)
        let pool = self.bufferPool
        storage.releaseCallback = { [weak pool] in
            pool?.release(bufferOut, elementCount: elementCount)
        }
        return Tensor<Float>(shape: input.shape, storage: storage)
    }

    // MARK: - Phase 3: RMSNorm (RMSNormBackend)

    /// Apply RMSNorm: out = (x / rms(x)) * weight
    ///
    /// One threadgroup per token (row).  Threads cooperate to compute the
    /// per-row sum-of-squares, then independently normalize and scale.
    ///
    /// - Parameters:
    ///   - input:  [numTokens, hiddenDim]
    ///   - weight: [hiddenDim] learnable scale (γ)
    ///   - eps:    numerical stability epsilon (typically 1e-5 or 1e-6)
    public func rmsnorm(_ input: Tensor<Float>, weight: Tensor<Float>, eps: Float = 1e-5) throws -> Tensor<Float> {
        precondition(input.shape.dimensions.count == 2, "rmsnorm input must be 2-D [tokens, hidden]")
        let numTokens = input.shape.dimensions[0]
        let hiddenDim = input.shape.dimensions[1]
        precondition(weight.shape.count == hiddenDim, "weight size must equal hiddenDim")

        let elementCount = input.shape.count

        let inputContiguous  = input.isContiguous  ? input  : input.makeContiguousCopy()
        let weightContiguous = weight.isContiguous ? weight : weight.makeContiguousCopy()

        let (bufferIn, releaseIn)     = try createBufferWithTracking(from: inputContiguous)
        let (bufferW,  releaseW)      = try createBufferWithTracking(from: weightContiguous)
        let bufferOut                 = try createBuffer(elementCount: elementCount)

        var dims = SIMD2<UInt32>(UInt32(numTokens), UInt32(hiddenDim))
        let dimsBuffer = device.makeBuffer(bytes: &dims,
                                           length: MemoryLayout<SIMD2<UInt32>>.stride,
                                           options: .storageModeShared)!
        var epsVal = eps
        let epsBuffer = device.makeBuffer(bytes: &epsVal,
                                          length: MemoryLayout<Float>.stride,
                                          options: .storageModeShared)!

        let pipeline = try loadKernel(named: "rmsnorm")

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer/encoder")
        }

        // Power-of-2 thread count required for tree reduction correctness
        let threadsPerGroup = pow2ThreadCount(hiddenDim)
        let scratchBytes    = threadsPerGroup * MemoryLayout<Float>.stride

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferIn,   offset: 0, index: 0)
        encoder.setBuffer(bufferW,    offset: 0, index: 1)
        encoder.setBuffer(bufferOut,  offset: 0, index: 2)
        encoder.setBuffer(dimsBuffer, offset: 0, index: 3)
        encoder.setBuffer(epsBuffer,  offset: 0, index: 4)
        encoder.setThreadgroupMemoryLength(scratchBytes, index: 0)

        let threadsPerThreadgroup = MTLSize(width: threadsPerGroup, height: 1, depth: 1)
        let threadgroupsPerGrid   = MTLSize(width: numTokens, height: 1, depth: 1)
        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if releaseIn { bufferPool.release(bufferIn, elementCount: elementCount) }
        if releaseW  { bufferPool.release(bufferW,  elementCount: hiddenDim) }

        let storage = TensorStorage<Float>(gpuBuffer: bufferOut, count: elementCount)
        let pool = self.bufferPool
        storage.releaseCallback = { [weak pool] in
            pool?.release(bufferOut, elementCount: elementCount)
        }
        return Tensor<Float>(shape: input.shape, storage: storage)
    }

    // MARK: - Phase 3: Activations (ActivationBackend)

    /// SiLU activation: out[i] = x[i] * sigmoid(x[i])
    ///
    /// Simple element-wise kernel — each thread handles one element.
    public func silu(_ input: Tensor<Float>) throws -> Tensor<Float> {
        return try elementWiseActivation(input, kernelName: "silu")
    }

    /// GELU activation (tanh approximation)
    ///
    /// Simple element-wise kernel — each thread handles one element.
    public func gelu(_ input: Tensor<Float>) throws -> Tensor<Float> {
        return try elementWiseActivation(input, kernelName: "gelu")
    }

    /// Shared helper for element-wise activation kernels (silu / gelu)
    ///
    /// Both kernels share the same buffer layout:
    ///   [0] input  [1] output  [2] n (uint element count)
    private func elementWiseActivation(_ input: Tensor<Float>, kernelName: String) throws -> Tensor<Float> {
        let elementCount = input.shape.count
        guard elementCount > 0 else { return input }

        let inputContiguous = input.isContiguous ? input : input.makeContiguousCopy()
        let (bufferIn, releaseIn) = try createBufferWithTracking(from: inputContiguous)
        let bufferOut = try createBuffer(elementCount: elementCount)

        var n = UInt32(elementCount)
        let nBuffer = device.makeBuffer(bytes: &n,
                                        length: MemoryLayout<UInt32>.stride,
                                        options: .storageModeShared)!

        let pipeline = try loadKernel(named: kernelName)

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer/encoder")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufferIn,  offset: 0, index: 0)
        encoder.setBuffer(bufferOut, offset: 0, index: 1)
        encoder.setBuffer(nBuffer,   offset: 0, index: 2)

        let threadsPerGroup = min(256, elementCount)
        let numGroups       = (elementCount + threadsPerGroup - 1) / threadsPerGroup
        encoder.dispatchThreadgroups(MTLSize(width: numGroups, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: threadsPerGroup, height: 1, depth: 1))
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if releaseIn { bufferPool.release(bufferIn, elementCount: elementCount) }

        let storage = TensorStorage<Float>(gpuBuffer: bufferOut, count: elementCount)
        let pool = self.bufferPool
        storage.releaseCallback = { [weak pool] in
            pool?.release(bufferOut, elementCount: elementCount)
        }
        return Tensor<Float>(shape: input.shape, storage: storage)
    }
}

// MARK: - Quantized Buffer Cache

private extension MetalBackend {
    struct QuantizedBufferEntry {
        let weights: MTLBuffer
        let scales: MTLBuffer
    }
    
    func cachedQuantizedBuffers(for tensor: QuantizedTensor) throws -> QuantizedBufferEntry {
        quantizedCacheLock.lock()
        defer { quantizedCacheLock.unlock() }
        
        if let cached = quantizedBufferCache[tensor.identifier] {
            return cached
        }
        
        guard let weightsBuffer = device.makeBuffer(length: tensor.data.count, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        tensor.data.withUnsafeBytes { src in
            memcpy(weightsBuffer.contents(), src.baseAddress!, tensor.data.count)
        }
        
        let scalesLength = tensor.scales.count * MemoryLayout<Float>.stride
        guard let scalesBuffer = device.makeBuffer(length: scalesLength, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        tensor.scales.withUnsafeBytes { src in
            memcpy(scalesBuffer.contents(), src.baseAddress!, scalesLength)
        }
        
        let entry = QuantizedBufferEntry(weights: weightsBuffer, scales: scalesBuffer)
        quantizedBufferCache[tensor.identifier] = entry
        return entry
    }
}

// MARK: - INT4 Quantized Buffer Cache

private extension MetalBackend {
    /// Cache entry for INT4 quantized tensors (includes zero points buffer)
    struct INT4BufferEntry {
        let weights: MTLBuffer     // Packed INT4 bytes
        let scales: MTLBuffer      // Per-group FP32 scales
        let zeroPoints: MTLBuffer  // Per-group INT4 zero points (stored as Int8)
    }

    /// Upload or reuse GPU-resident INT4 buffers
    func cachedINT4Buffers(for tensor: QuantizedTensor) throws -> INT4BufferEntry {
        quantizedCacheLock.lock()
        defer { quantizedCacheLock.unlock() }

        // Check if we already have cached INT4 buffers for this tensor
        // We reuse the same cache dict but store an INT4BufferEntry wrapper
        // For simplicity, we create fresh buffers keyed by identifier
        // (The QuantizedBufferEntry cache is separate for INT8)

        // Upload packed INT4 weights
        guard let weightsBuffer = device.makeBuffer(length: tensor.data.count, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        tensor.data.withUnsafeBytes { src in
            memcpy(weightsBuffer.contents(), src.baseAddress!, tensor.data.count)
        }

        // Upload per-group scales
        let scalesLength = tensor.scales.count * MemoryLayout<Float>.stride
        guard let scalesBuffer = device.makeBuffer(length: scalesLength, options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        tensor.scales.withUnsafeBytes { src in
            memcpy(scalesBuffer.contents(), src.baseAddress!, scalesLength)
        }

        // Upload per-group zero points
        let zpData = tensor.zeroPoints ?? [Int8](repeating: 0, count: tensor.scales.count)
        let zpLength = zpData.count * MemoryLayout<Int8>.stride
        guard let zpBuffer = device.makeBuffer(length: max(zpLength, 1), options: .storageModeShared) else {
            throw MetalError.bufferCreationFailed
        }
        zpData.withUnsafeBytes { src in
            memcpy(zpBuffer.contents(), src.baseAddress!, zpLength)
        }

        return INT4BufferEntry(weights: weightsBuffer, scales: scalesBuffer, zeroPoints: zpBuffer)
    }
}

// MARK: - Phase 3: FlashAttention (AttentionBackend)

extension MetalBackend {

    /// Fused multi-head attention on GPU using FlashAttention kernel.
    ///
    /// Dispatches `flash_attention` which tiles Q×K^T + softmax + V into a
    /// single pass with O(Br×Bc) threadgroup memory instead of O(seqLen²).
    /// Supports grouped-query attention (numKVHeads < numHeads).
    ///
    /// - Parameters:
    ///   - query:  [batch?, seqLen, numHeads * headDim]
    ///   - keys:   [batch?, kvSeqLen, numKVHeads * headDim]
    ///   - values: [batch?, kvSeqLen, numKVHeads * headDim]
    ///   - mask:   Optional [kvSeqLen] additive mask (0.0 = attend, -inf = ignore)
    ///   - headDim:    Dimension per attention head
    ///   - numHeads:   Number of query heads
    ///   - numKVHeads: Number of key/value heads
    /// - Returns: Tensor of same shape as `query`
    public func attention(
        query: Tensor<Float>, keys: Tensor<Float>, values: Tensor<Float>,
        mask: Tensor<Float>?, headDim: Int, numHeads: Int, numKVHeads: Int
    ) throws -> Tensor<Float> {
        precondition(numHeads >= numKVHeads && numHeads % numKVHeads == 0,
                     "numHeads must be a multiple of numKVHeads")

        // ── Parse shapes ──────────────────────────────────────────────────
        let qDims = query.shape.dimensions
        let kDims = keys.shape.dimensions
        let batch: Int
        let seqLen: Int
        let kvSeqLen: Int

        if qDims.count == 3 {
            batch     = qDims[0]
            seqLen    = qDims[1]
            kvSeqLen  = kDims[1]
        } else {
            batch     = 1
            seqLen    = qDims[0]
            kvSeqLen  = kDims[0]
        }

        let outputElementCount = batch * seqLen * numHeads * headDim

        // ── Upload tensors to GPU ─────────────────────────────────────────
        let qContiguous = query.isContiguous  ? query  : query.makeContiguousCopy()
        let kContiguous = keys.isContiguous   ? keys   : keys.makeContiguousCopy()
        let vContiguous = values.isContiguous ? values : values.makeContiguousCopy()

        let (bufQ, releaseQ) = try createBufferWithTracking(from: qContiguous)
        let (bufK, releaseK) = try createBufferWithTracking(from: kContiguous)
        let (bufV, releaseV) = try createBufferWithTracking(from: vContiguous)
        let bufOut = try createBuffer(elementCount: outputElementCount)

        // Mask buffer (or a tiny dummy — kernel checks useMask flag)
        let bufMask: MTLBuffer
        let releaseMask: Bool
        let useMask: UInt32
        if let mask = mask {
            let mContiguous = mask.isContiguous ? mask : mask.makeContiguousCopy()
            let pair = try createBufferWithTracking(from: mContiguous)
            bufMask     = pair.0
            releaseMask = pair.1
            useMask     = 1
        } else {
            // Dummy 1-element buffer; kernel won't read it when useMask == 0
            bufMask     = try createBuffer(elementCount: 1)
            releaseMask = true
            useMask     = 0
        }

        // ── Params struct (matches AttentionParams in FlashAttention.metal) ──
        var params = (
            /* seqLen */    UInt32(seqLen),
            /* kvSeqLen */  UInt32(kvSeqLen),
            /* headDim */   UInt32(headDim),
            /* numHeads */  UInt32(numHeads),
            /* numKVHeads */ UInt32(numKVHeads),
            /* batch */     UInt32(batch),
            /* scale */     Float(1.0 / sqrt(Float(headDim))),
            /* useMask */   useMask
        )
        let paramsBuffer = device.makeBuffer(bytes: &params,
                                             length: MemoryLayout.size(ofValue: params),
                                             options: .storageModeShared)!

        // ── Dispatch kernel ───────────────────────────────────────────────
        // Tile sizes must match the compile-time constants in the Metal kernel.
        // The kernel uses TILE_Br=32 and TILE_Bc=32.
        let tileBr: Int = 32
        let tileBc: Int = 32

        let pipeline = try loadKernel(named: "flash_attention")

        // Verify the device can handle our threadgroup size (Br × Bc threads)
        let requestedThreads = tileBr * tileBc
        let maxThreads = pipeline.maxTotalThreadsPerThreadgroup
        guard requestedThreads <= maxThreads else {
            throw MetalError.invalidKernelConfiguration(
                "FlashAttention requires \(requestedThreads) threads/threadgroup but device supports \(maxThreads)")
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalError.invalidKernelConfiguration("Failed to create command buffer/encoder for FlashAttention")
        }

        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(bufQ,          offset: 0, index: 0)
        encoder.setBuffer(bufK,          offset: 0, index: 1)
        encoder.setBuffer(bufV,          offset: 0, index: 2)
        encoder.setBuffer(bufOut,        offset: 0, index: 3)
        encoder.setBuffer(bufMask,       offset: 0, index: 4)
        encoder.setBuffer(paramsBuffer,  offset: 0, index: 5)

        // Threadgroup shared memory: S_tile(Br*Bc) + row_max(Br) + row_sum(Br) + O_acc(Br*headDim)
        let smemFloats = tileBr * tileBc + tileBr + tileBr + tileBr * headDim
        let smemBytes = smemFloats * MemoryLayout<Float>.stride
        encoder.setThreadgroupMemoryLength(smemBytes, index: 0)

        // Grid: (numHeads, ceil(seqLen/Br), batch)
        let threadgroupsPerGrid = MTLSize(
            width:  numHeads,
            height: (seqLen + tileBr - 1) / tileBr,
            depth:  batch
        )
        // Threads per threadgroup: (Bc, Br, 1)
        let threadsPerThreadgroup = MTLSize(width: tileBc, height: tileBr, depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        if commandBuffer.status == .error {
            throw MetalError.invalidKernelConfiguration(
                "FlashAttention GPU error: \(commandBuffer.error?.localizedDescription ?? "unknown")")
        }

        // ── Release input buffers ─────────────────────────────────────────
        if releaseQ    { bufferPool.release(bufQ, elementCount: query.shape.count) }
        if releaseK    { bufferPool.release(bufK, elementCount: keys.shape.count) }
        if releaseV    { bufferPool.release(bufV, elementCount: values.shape.count) }
        if releaseMask { bufferPool.release(bufMask, elementCount: mask?.shape.count ?? 1) }

        // ── Wrap output in Tensor ─────────────────────────────────────────
        let storage = TensorStorage<Float>(gpuBuffer: bufOut, count: outputElementCount)
        let pool = self.bufferPool
        storage.releaseCallback = { [weak pool] in
            pool?.release(bufOut, elementCount: outputElementCount)
        }
        return Tensor<Float>(shape: query.shape, storage: storage)
    }
}

// Helper extension for making contiguous copies
extension Tensor<Float> {
    /// Make a contiguous copy of this tensor (public for Metal backend)
    public func makeContiguousCopy() -> Tensor<Float> {
        if isContiguous {
            return self
        }
        
        // Copy with proper stride handling
        var newData = [Float](repeating: 0, count: shape.count)
        var destIdx = 0
        
        if shape.dimensions.count == 2 {
            for i in 0..<shape.dimensions[0] {
                for j in 0..<shape.dimensions[1] {
                    let offset = i * shape.strides[0] + j * shape.strides[1]
                    newData[destIdx] = rawData[offset]
                    destIdx += 1
                }
            }
        } else {
            newData = rawData
        }
        
        return Tensor(shape: TensorShape(shape.dimensions), data: newData)
    }

}

/// Errors specific to Metal operations
public enum MetalError: Error {
    case deviceNotFound
    case commandQueueCreationFailed
    case libraryLoadFailed
    case shaderCompilationFailed(String)
    case bufferCreationFailed
    case invalidKernelConfiguration(String)
}
