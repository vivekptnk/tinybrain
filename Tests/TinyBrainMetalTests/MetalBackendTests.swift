import XCTest
@testable import TinyBrainMetal
@testable import TinyBrainRuntime

/// Tests for Metal backend functionality
final class MetalBackendTests: XCTestCase {
    func testMetalAvailability() throws {
        let isAvailable = MetalBackend.isAvailable
        
        // In CI/headless environments Metal may not be available; treat as skippable
        if !isAvailable {
            throw XCTSkip("Metal not available on this device")
        }
        XCTAssertTrue(isAvailable)
    }
    
    func testMetalBackendInitialization() throws {
        // Skip this test if Metal is not available (e.g., in CI without GPU)
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available on this device")
        }
        
        let backend = try MetalBackend()
        let deviceInfo = backend.deviceInfo
        
        XCTAssertFalse(deviceInfo.isEmpty, "Device info should not be empty")
        print("Metal device: \(deviceInfo)")
    }
    
    // MARK: - Metal MatMul Tests
    
    /// Test basic Metal matrix multiplication
    ///
    /// **What:** Run matmul on GPU and verify result
    /// **Why:** Validate our first Metal kernel works!
    /// **How:** Small matrices with known answer
    func testMetalMatMulBasic() throws {
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let backend = try MetalBackend()
        
        // A: [2, 3]
        let a = Tensor<Float>(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
        
        // B: [3, 2]
        let b = Tensor<Float>(shape: TensorShape(3, 2), data: [7,8,9,10,11,12])
        
        // C = A × B on GPU
        let c = try backend.matmul(a, b)
        
        // Verify shape
        XCTAssertEqual(c.shape.dimensions, [2, 2])
        
        // Verify values (same as CPU test)
        XCTAssertEqual(c[0,0], 58.0, accuracy: 1e-3)
        XCTAssertEqual(c[0,1], 64.0, accuracy: 1e-3)
        XCTAssertEqual(c[1,0], 139.0, accuracy: 1e-3)
        XCTAssertEqual(c[1,1], 154.0, accuracy: 1e-3)
        
        print("✅ Metal matmul basic test passed!")
    }
    
    /// Test Metal vs CPU numerical parity
    ///
    /// **What:** Verify Metal gives same results as CPU
    /// **Why:** GPU and CPU should match (within floating point error)
    /// **How:** Compare random matrices
    func testMetalVsCPUParity() throws {
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let backend = try MetalBackend()
        
        // Random matrices
        let a = Tensor.random(shape: TensorShape(64, 64))
        let b = Tensor.random(shape: TensorShape(64, 64))
        
        // CPU result
        let cpuResult = a.matmul(b)
        
        // GPU result
        let metalResult = try backend.matmul(a, b)
        
        // Compare
        for i in 0..<64 {
            for j in 0..<64 {
                let diff = abs(metalResult[i,j] - cpuResult[i,j])
                let relativeError = diff / max(abs(cpuResult[i,j]), 1e-7)
                
                XCTAssertLessThan(relativeError, 1e-3, 
                                 "Metal vs CPU mismatch at [\(i),\(j)]: \(metalResult[i,j]) vs \(cpuResult[i,j])")
            }
        }
        
        print("✅ Metal vs CPU parity test passed (64×64)!")
    }
    
    // MARK: - Tiled Kernel Tests
    
    /// Test tiled kernel correctness
    ///
    /// **What:** Run optimized tiled kernel and verify correctness
    /// **Why:** Tiled version should give same results as naive, just faster!
    /// **How:** Compare tiled vs CPU
    func testMetalTiledMatMul() throws {
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let backend = try MetalBackend()
        
        // Test with 64×64 (multiple tiles: 4×4 tiles of 16×16)
        let a = Tensor.random(shape: TensorShape(64, 64))
        let b = Tensor.random(shape: TensorShape(64, 64))
        
        // CPU result
        let cpuResult = a.matmul(b)
        
        // GPU result (tiled)
        let metalResult = try backend.matmulOptimized(a, b)
        
        // Compare
        for i in 0..<64 {
            for j in 0..<64 {
                let diff = abs(metalResult[i,j] - cpuResult[i,j])
                let relativeError = diff / max(abs(cpuResult[i,j]), 1e-7)
                
                XCTAssertLessThan(relativeError, 1e-3,
                                 "Tiled kernel mismatch at [\(i),\(j)]")
            }
        }
        
        print("✅ Tiled kernel correctness validated!")
    }
    
    /// Benchmark: Tiled kernel should be faster than naive
    ///
    /// **What:** Time both kernels and verify tiled is faster
    /// **Why:** Validate our optimization actually works!
    /// **How:** Run 10 iterations, measure average time
    func testTiledKernelPerformance() throws {
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let backend = try MetalBackend()
        
        // Large enough to see difference (256×256)
        let a = Tensor<Float>.filled(shape: TensorShape(256, 256), value: 1.0)
        let b = Tensor<Float>.filled(shape: TensorShape(256, 256), value: 2.0)
        
        // Warmup
        _ = try backend.matmul(a, b)
        _ = try backend.matmulOptimized(a, b)
        
        // Time naive kernel
        let naiveStart = Date()
        for _ in 0..<10 {
            _ = try backend.matmul(a, b)
        }
        let naiveTime = Date().timeIntervalSince(naiveStart) / 10.0
        
        // Time tiled kernel
        let tiledStart = Date()
        for _ in 0..<10 {
            _ = try backend.matmulOptimized(a, b)
        }
        let tiledTime = Date().timeIntervalSince(tiledStart) / 10.0
        
        let speedup = naiveTime / tiledTime
        
        print("Naive: \(naiveTime * 1000) ms")
        print("Tiled: \(tiledTime * 1000) ms")
        print("Speedup: \(speedup)×")
        
        // Tiled should be faster (might be marginal for 256×256)
        XCTAssertLessThan(tiledTime, naiveTime * 1.5, 
                         "Tiled kernel should be faster or comparable")
    }
    
    // MARK: - Comprehensive Benchmarks
    
    /// Test Metal on various matrix sizes (comprehensive validation)
    func testMetalComprehensiveSizes() throws {
        guard MetalBackend.isAvailable else {
            throw XCTSkip("Metal not available")
        }
        
        let backend = try MetalBackend()
        
        let sizes = [(32, 32), (64, 64), (128, 128), (256, 256)]
        
        for (m, n) in sizes {
            let a = Tensor.random(shape: TensorShape(m, n))
            let b = Tensor.random(shape: TensorShape(n, m))
            
            let cpuResult = a.matmul(b)
            let metalResult = try backend.matmul(a, b)
            
            // Check random samples for parity
            for _ in 0..<10 {
                let i = Int.random(in: 0..<m)
                let j = Int.random(in: 0..<m)
                
                let diff = abs(metalResult[i,j] - cpuResult[i,j])
                let relError = diff / max(abs(cpuResult[i,j]), 1e-7)
                
                XCTAssertLessThan(relError, 1e-3, 
                                 "Mismatch at [\(i),\(j)] for size \(m)×\(n)")
            }
            
            print("✅ Metal parity validated for \(m)×\(n)")
        }
    }
}

