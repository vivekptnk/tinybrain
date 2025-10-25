/// Metal acceleration backend for TinyBrain
///
/// Provides GPU-accelerated operations for inference using custom Metal kernels.
/// Handles shader loading, buffer management, and command encoding.

import Metal
import Foundation
import TinyBrainRuntime

/// Metal backend for accelerated tensor operations
public final class MetalBackend {
    /// Shared Metal device (the GPU)
    private let device: MTLDevice
    
    /// Command queue for GPU operations
    private let commandQueue: MTLCommandQueue
    
    /// Shader library containing compiled .metal files (lazy-loaded)
    private var library: MTLLibrary?
    
    /// Cache of compiled compute pipelines (kernel name → pipeline)
    /// Avoids recompiling shaders on every use
    private var pipelineCache: [String: MTLComputePipelineState] = [:]
    
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
    
    /// Create a Metal buffer from tensor data
    ///
    /// Copies data from CPU memory to GPU memory.
    ///
    /// - Parameter tensor: Source tensor
    /// - Returns: Metal buffer containing tensor data
    /// - Throws: `MetalError.bufferCreationFailed` if allocation fails
    public func createBuffer(from tensor: Tensor) throws -> MTLBuffer {
        let byteCount = tensor.shape.count * MemoryLayout<Float>.stride
        
        guard let buffer = device.makeBuffer(
            bytes: tensor.rawData,
            length: byteCount,
            options: .storageModeShared  // CPU & GPU can access
        ) else {
            throw MetalError.bufferCreationFailed
        }
        
        return buffer
    }
    
    /// Create an empty Metal buffer for results
    ///
    /// - Parameter elementCount: Number of Float elements
    /// - Returns: Empty Metal buffer
    /// - Throws: `MetalError.bufferCreationFailed` if allocation fails
    public func createBuffer(elementCount: Int) throws -> MTLBuffer {
        let byteCount = elementCount * MemoryLayout<Float>.stride
        
        guard let buffer = device.makeBuffer(
            length: byteCount,
            options: .storageModeShared
        ) else {
            throw MetalError.bufferCreationFailed
        }
        
        return buffer
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
    
    // MARK: - Matrix Operations
    
    /// Perform matrix multiplication on GPU using Metal
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
    public func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor {
        precondition(a.shape.dimensions.count == 2, "A must be 2D")
        precondition(b.shape.dimensions.count == 2, "B must be 2D")
        precondition(a.shape.dimensions[1] == b.shape.dimensions[0], "Inner dimensions must match")
        
        let M = UInt32(a.shape.dimensions[0])
        let K = UInt32(a.shape.dimensions[1])
        let N = UInt32(b.shape.dimensions[1])
        
        // Ensure tensors are contiguous (GPU kernel expects row-major)
        let aContiguous = a.isContiguous ? a : a.makeContiguousCopy()
        let bContiguous = b.isContiguous ? b : b.makeContiguousCopy()
        
        // Create GPU buffers
        let bufferA = try createBuffer(from: aContiguous)
        let bufferB = try createBuffer(from: bContiguous)
        let bufferC = try createBuffer(elementCount: Int(M * N))
        
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
        
        // Read results back from GPU
        let resultData = readBuffer(bufferC, count: Int(M * N))
        
        return Tensor(shape: TensorShape(Int(M), Int(N)), data: resultData)
    }
}

// Helper extension for making contiguous copies
extension Tensor {
    /// Make a contiguous copy of this tensor (public for Metal backend)
    public func makeContiguousCopy() -> Tensor {
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

