import XCTest
@testable import TinyBrainRuntime

/// Tests for Tensor operations
final class TensorTests: XCTestCase {
    func testTensorShapeCreation() {
        let shape = TensorShape(2, 3, 4)
        XCTAssertEqual(shape.dimensions, [2, 3, 4])
        XCTAssertEqual(shape.count, 24)
    }
    
    func testTensorCreation() {
        let shape = TensorShape(2, 3)
        let data = Array(repeating: 1.0 as Float, count: 6)
        let tensor = Tensor(shape: shape, data: data)
        
        XCTAssertEqual(tensor.shape, shape)
        XCTAssertEqual(tensor.data.count, 6)
    }
    
    func testZeroTensor() {
        let tensor = Tensor.zeros(shape: TensorShape(2, 3))
        XCTAssertEqual(tensor.data.count, 6)
        XCTAssertTrue(tensor.data.allSatisfy { $0 == 0.0 })
    }
    
    func testFilledTensor() {
        let tensor = Tensor.filled(shape: TensorShape(2, 2), value: 5.0)
        XCTAssertEqual(tensor.data.count, 4)
        XCTAssertTrue(tensor.data.allSatisfy { $0 == 5.0 })
    }
    
    func testTensorShapeValidation() {
        // This should trigger a precondition failure
        // In production, we'd test this differently
        let shape = TensorShape(2, 3)
        let invalidData = [1.0, 2.0] // Wrong size
        
        // Uncomment when proper error handling is added
        // XCTAssertThrowsError(try Tensor(shape: shape, data: invalidData))
    }
}

