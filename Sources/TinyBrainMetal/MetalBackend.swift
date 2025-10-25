/// Metal acceleration backend for TinyBrain
///
/// Provides GPU-accelerated operations for inference using custom Metal kernels.
/// Handles shader loading, buffer management, and command encoding.

import Metal
import Foundation
import TinyBrainRuntime

/// Metal backend for accelerated tensor operations
///
/// **REVIEW HITLER FIX:** Now includes INT8 quantized operations!
public final class MetalBackend: MatMulBackend, TensorUploader, TensorDownloader, QuantizedMatMulBackend {
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
        
        // **REVIEW HITLER FIX:** INT8 dequant + fused matmul kernels
        
        // Fused INT8 dequant + matmul (THE REAL FIX!)
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
                
                // Dequantize on-the-fly (no Float32 materialization!)
                char b_quant = B_quantized[k * N + col];
                float b_scale = B_scales[k];
                float b_val = float(b_quant) * b_scale;
                
                sum += a_val * b_val;
            }
            
            C[row * N + col] = sum;
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
    
    // MARK: - INT8 Quantized Operations
    
    /// **REVIEW HITLER FIX:** Fused INT8 dequant + matmul
    ///
    /// THE REAL SOLUTION: Compute directly from INT8 without Float32 materialization!
    ///
    /// - Parameters:
    ///   - input: Float32 input [M, K]
    ///   - quantized: INT8 weights
    /// - Returns: Float32 output [M, N]
    public func matmulQuantized(_ input: Tensor<Float>, _ quantized: QuantizedTensor) throws -> Tensor<Float> {
        precondition(input.shape.dimensions.count == 2, "Input must be 2D")
        precondition(quantized.shape.dimensions.count == 2, "Weights must be 2D")
        precondition(input.shape.dimensions[1] == quantized.shape.dimensions[0], "Dimensions must match")
        
        // **REVIEW HITLER FIX:** Only per-channel mode supported in current kernel!
        // Symmetric/asymmetric modes need different kernel (single scale, not per-channel)
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
