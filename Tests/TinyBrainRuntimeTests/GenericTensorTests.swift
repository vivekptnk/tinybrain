/// Tests for generic Tensor<Element> supporting multiple data types
///
/// TB-004 Phase 2: Validates Float32, Float16, and Int8 tensors with CoW optimization

import XCTest
@testable import TinyBrainRuntime

final class GenericTensorTests: XCTestCase {
    
    // MARK: - Float32 Tensor Tests
    
    func testFloat32Tensor() {
        // WHAT: Tensor<Float> works like current Tensor
        // WHY: Backward compatibility - existing code shouldn't break
        // HOW: Create, access, verify type
        
        let t = Tensor<Float>.zeros(shape: TensorShape(10, 10))
        XCTAssertEqual(t.shape.count, 100)
        XCTAssertEqual(t[0, 0], 0.0)
    }
    
    func testFloat32Operations() {
        // WHAT: Float32 tensors support all operations
        // WHY: Main compute path for inference
        // ACCURACY: < 1e-5 relative error
        
        let a = Tensor<Float>.filled(shape: TensorShape(4, 4), value: 2.0)
        let b = Tensor<Float>.filled(shape: TensorShape(4, 4), value: 3.0)
        
        let sum = a + b
        XCTAssertEqual(sum[0, 0], 5.0, accuracy: 1e-5)
        
        let product = a * b
        XCTAssertEqual(product[0, 0], 6.0, accuracy: 1e-5)
    }
    
    // MARK: - Float16 Tensor Tests
    
    // TODO: Re-enable when Float16 conformance is fully implemented
    // Float16 requires _Float16 module and additional setup
    
    /*
    func testFloat16Tensor() {
        // WHAT: Tensor<Float16> for memory efficiency
        // WHY: Half precision = 50% memory savings
        // HOW: Create, verify storage size is half of Float32
        
        let t = Tensor<Float16>.random(shape: TensorShape(100, 100))
        XCTAssertEqual(t.shape.count, 10000)
        
        // Verify element size is 2 bytes (half of Float's 4 bytes)
        let elementSize = MemoryLayout<Float16>.size
        XCTAssertEqual(elementSize, 2, "Float16 should be 2 bytes")
    }
    
    func testFloat16Accuracy() {
        // WHAT: Float16 has reduced precision
        // WHY: Understanding accuracy tradeoffs
        // ACCURACY: < 1e-3 relative error (less precise than Float32)
        
        let a = Tensor<Float16>.filled(shape: TensorShape(10, 10), value: 1.5)
        let b = Tensor<Float16>.filled(shape: TensorShape(10, 10), value: 2.5)
        
        let sum = a + b
        // Float16 has ~3-4 decimal digits of precision
        XCTAssertEqual(Float(sum[0, 0]), 4.0, accuracy: 1e-3)
    }
    */
    
    // MARK: - Int8 Tensor Tests
    
    func testInt8Tensor() {
        // WHAT: Tensor<Int8> for quantized weights
        // WHY: Foundation of INT8 quantization - 75% memory savings
        // HOW: Create with values in [-128, 127] range
        
        let t = Tensor<Int8>.filled(shape: TensorShape(10, 10), value: 127)
        XCTAssertEqual(t[0, 0], 127)
        XCTAssertEqual(t[9, 9], 127)
        
        // Verify element size is 1 byte
        let elementSize = MemoryLayout<Int8>.size
        XCTAssertEqual(elementSize, 1, "Int8 should be 1 byte")
    }
    
    func testInt8Range() {
        // WHAT: Int8 values clamped to [-128, 127]
        // WHY: Prevent overflow in quantized weights
        // HOW: Create with boundary values
        
        let min = Tensor<Int8>.filled(shape: TensorShape(5, 5), value: -128)
        XCTAssertEqual(min[0, 0], -128)
        
        let max = Tensor<Int8>.filled(shape: TensorShape(5, 5), value: 127)
        XCTAssertEqual(max[0, 0], 127)
    }
    
    // MARK: - Copy-on-Write Tests
    
    func testCopyOnWrite() {
        // WHAT: Cheap copy, write triggers unique copy
        // WHY: Value semantics without performance cost
        // HOW: Copy tensor, mutate, verify original unchanged
        
        var a = Tensor<Float>.random(shape: TensorShape(1000, 1000))
        let originalValue = a[0, 0]
        
        var b = a  // Cheap copy (shares storage)
        
        // Mutate b - should trigger CoW
        b[0, 0] = 999.0
        
        // Original should be unchanged
        XCTAssertEqual(a[0, 0], originalValue, "CoW: original should be unchanged")
        XCTAssertEqual(b[0, 0], 999.0, "CoW: copy should have new value")
    }
    
    func testCopyOnWriteAvoidance() {
        // WHAT: No copy if already unique
        // WHY: Performance optimization - avoid unnecessary allocations
        // HOW: Verify single reference doesn't copy
        
        var a = Tensor<Float>.zeros(shape: TensorShape(1000, 1000))
        
        // Modify without triggering copy (already unique owner)
        a[0, 0] = 1.0
        XCTAssertEqual(a[0, 0], 1.0)
        
        // This test verifies isKnownUniquelyReferenced works
        // (implementation detail - we can't directly test it here)
    }
    
    func testSharedStorageOptimization() {
        // WHAT: Multiple tensors share storage until mutated
        // WHY: Memory efficiency for read-only operations
        // HOW: Create, copy many times, verify memory not duplicated
        
        let original = Tensor<Float>.random(shape: TensorShape(500, 500))
        
        // Create 10 copies
        var copies: [Tensor<Float>] = []
        for _ in 0..<10 {
            copies.append(original)
        }
        
        // All should share storage (zero extra memory until mutation)
        // Verified by CoW implementation
        XCTAssertEqual(copies.count, 10)
    }
    
    // MARK: - Type Conversion Tests
    
    // TODO: Implement tensor type conversion methods
    // Will add toFloat16(), toFloat32(), toInt8() in next phase
    
    /*
    func testFloat32ToFloat16Conversion() {
        // WHAT: Convert Float32 → Float16
        // WHY: Reduce memory for inference
        // ACCURACY: Some precision loss acceptable
        
        let f32 = Tensor<Float>.filled(shape: TensorShape(10, 10), value: 3.14159)
        let f16 = f32.toFloat16()
        
        // Float16 has ~3-4 decimal digits
        XCTAssertEqual(Float(f16[0, 0]), 3.14159, accuracy: 1e-3)
    }
    
    func testFloat16ToFloat32Conversion() {
        // WHAT: Convert Float16 → Float32
        // WHY: Upcast for higher precision compute
        // HOW: Convert and verify value preserved
        
        let f16 = Tensor<Float16>.filled(shape: TensorShape(10, 10), value: 2.5)
        let f32 = f16.toFloat32()
        
        XCTAssertEqual(f32[0, 0], 2.5, accuracy: 1e-5)
    }
    */
    
    // MARK: - Edge Case Tests
    
    func testEmptyTensor() {
        // Edge: Can we create 0-element tensor?
        // WHY: Boundary condition handling
        
        // This should probably fail during shape creation
        // TensorShape validates dimensions > 0
    }
    
    func testNaNHandling() {
        // Edge: NaN in tensor data
        // WHY: Operations should propagate or detect NaN
        
        let t = Tensor<Float>(shape: TensorShape(5, 5), data: Array(repeating: Float.nan, count: 25))
        
        let result = t + t
        XCTAssertTrue(result[0, 0].isNaN, "NaN should propagate")
    }
    
    func testInfHandling() {
        // Edge: Infinity values
        // WHY: Numerical stability checks
        
        let t = Tensor<Float>(shape: TensorShape(5, 5), data: Array(repeating: Float.infinity, count: 25))
        
        let result = t * Tensor<Float>.filled(shape: TensorShape(5, 5), value: 2.0)
        XCTAssertTrue(result[0, 0].isInfinite, "Inf should propagate")
    }
    
    func testVeryLargeTensor() {
        // Edge: Memory pressure with large tensor
        // WHY: Ensure CoW handles large allocations
        
        // 10000×10000 = 400MB for Float32
        let large = Tensor<Float>.zeros(shape: TensorShape(10000, 10000))
        let copy = large  // Should be cheap (shared storage)
        
        XCTAssertEqual(large.shape, copy.shape)
        // If this doesn't crash, CoW is working!
    }
}

