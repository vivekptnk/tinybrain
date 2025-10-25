import XCTest
@testable import TinyBrainMetal
@testable import TinyBrainRuntime

/// Tests for Metal backend functionality
final class MetalBackendTests: XCTestCase {
    func testMetalAvailability() {
        // Metal should be available on all Apple Silicon Macs
        // and simulators may not have it
        let isAvailable = MetalBackend.isAvailable
        
        #if targetEnvironment(simulator)
        // Simulators may not have Metal
        print("Running on simulator - Metal availability: \(isAvailable)")
        #else
        // Physical devices should have Metal
        XCTAssertTrue(isAvailable, "Metal should be available on physical Apple devices")
        #endif
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
        let a = Tensor(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
        
        // B: [3, 2]
        let b = Tensor(shape: TensorShape(3, 2), data: [7,8,9,10,11,12])
        
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
}

