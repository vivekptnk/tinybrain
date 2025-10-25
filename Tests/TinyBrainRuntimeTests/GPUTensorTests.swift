/// Tests for GPU-resident tensor functionality
///
/// Validates that tensors can be kept on GPU and operations chained
/// without unnecessary CPU↔GPU transfers.

import XCTest
@testable import TinyBrainRuntime
import TinyBrainMetal

final class GPUTensorTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        // Enable debug logging to see what's happening
        TinyBrainBackend.debugLogging = true
        
        // Initialize Metal backend before tests
        if TinyBrainBackend.metalBackend == nil {
            if MetalBackend.isAvailable {
                do {
                    TinyBrainBackend.metalBackend = try MetalBackend()
                    print("✅ Metal backend initialized successfully")
                } catch {
                    print("❌ Failed to initialize Metal: \(error)")
                }
            } else {
                print("⚠️ Metal not available on this device")
            }
        }
    }
    
    func testGPUTensorCreation() throws {
        // WHAT: Create GPU tensor from CPU tensor
        // WHY: Foundation of GPU-resident data - enables keeping data on GPU
        // HOW: Verify tensor is marked as GPU-resident
        
        // Skip if Metal not available
        try XCTSkipUnless(MetalBackend.isAvailable, "Metal not available on this device")
        
        let cpu = Tensor<Float>.zeros(shape: TensorShape(100, 100))
        let gpu = cpu.toGPU()
        
        XCTAssertTrue(gpu.isOnGPU, "Tensor should be marked as GPU-resident")
        XCTAssertEqual(gpu.shape, cpu.shape, "Shape should be preserved")
    }
    
    func testLazySynchronization() throws {
        // WHAT: Data only transfers when accessed from other side
        // WHY: Eliminate unnecessary CPU↔GPU copies that kill performance
        // HOW: Perform GPU-only operations, verify result stays on GPU
        
        try XCTSkipUnless(MetalBackend.isAvailable, "Metal not available on this device")
        
        let gpu = Tensor<Float>.zeros(shape: TensorShape(100, 100)).toGPU()
        let result = gpu.matmul(gpu)  // Stays on GPU
        
        XCTAssertTrue(result.isOnGPU, "Result should stay on GPU for GPU-only ops")
    }
    
    func testGPUChainedOperations() throws {
        // WHAT: Multiple GPU ops without CPU roundtrip
        // WHY: Batched operations eliminate transfer overhead = 3-8× speedup
        // HOW: Chain matmul → softmax → matmul, all on GPU
        
        try XCTSkipUnless(MetalBackend.isAvailable, "Metal not available on this device")
        
        let a = Tensor<Float>.random(shape: TensorShape(512, 512)).toGPU()
        
        // Chain operations: matmul → softmax → matmul
        let b = a.matmul(a)           // GPU
        let c = b.softmax()            // GPU
        let d = c.matmul(a)            // GPU
        
        XCTAssertTrue(d.isOnGPU, "Chained operations should stay on GPU")
        XCTAssertEqual(d.shape, TensorShape(512, 512))
    }
    
    func testGPUToCPUTransfer() throws {
        // WHAT: Explicit transfer from GPU to CPU
        // WHY: Eventually need results on CPU for further processing
        // HOW: toGPU() → operations → toCPU()
        
        try XCTSkipUnless(MetalBackend.isAvailable, "Metal not available on this device")
        
        let cpu = Tensor<Float>.filled(shape: TensorShape(10, 10), value: 5.0)
        let gpu = cpu.toGPU()
        let backToCPU = gpu.toCPU()
        
        XCTAssertFalse(backToCPU.isOnGPU, "Should be back on CPU")
        XCTAssertEqual(backToCPU.shape, cpu.shape)
        
        // Verify data integrity
        XCTAssertEqual(backToCPU[0, 0], 5.0, accuracy: 1e-5)
        XCTAssertEqual(backToCPU[9, 9], 5.0, accuracy: 1e-5)
    }
    
    func testCPUOnlyOperationsStillWork() {
        // WHAT: CPU-only code path still works
        // WHY: Fallback when Metal unavailable or explicitly requested
        // HOW: Use matmulCPU() for explicit CPU path
        
        let a = Tensor<Float>.random(shape: TensorShape(100, 100))
        let b = Tensor<Float>.random(shape: TensorShape(100, 100))
        let c = a.matmulCPU(b)  // Explicitly CPU
        
        XCTAssertFalse(c.isOnGPU, "matmulCPU should always return CPU tensor")
    }
}

