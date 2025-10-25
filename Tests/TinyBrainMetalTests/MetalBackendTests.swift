import XCTest
@testable import TinyBrainMetal

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
}

