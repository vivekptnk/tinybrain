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
        let tensor = Tensor<Float>(shape: shape, data: data)
        
        XCTAssertEqual(tensor.shape, shape)
        XCTAssertEqual(tensor.data.count, 6)
    }
    
    func testZeroTensor() {
        let tensor = Tensor<Float>.zeros(shape: TensorShape(2, 3))
        XCTAssertEqual(tensor.data.count, 6)
        XCTAssertTrue(tensor.data.allSatisfy { $0 == 0.0 })
    }
    
    func testFilledTensor() {
        let tensor = Tensor<Float>.filled(shape: TensorShape(2, 2), value: 5.0)
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
    
    // MARK: - Stride and Memory Layout Tests (TB-003 TDD)
    
    /// Test that TensorShape calculates strides correctly for contiguous tensors
    ///
    /// **What:** Strides define how many elements to skip to move along each dimension
    /// **Why:** Needed for transpose/reshape without copying data
    /// **How:** Create shapes and verify stride calculation
    ///
    /// **Educational Note:**
    /// For row-major layout, strides are calculated from right to left:
    /// ```
    /// Shape: [2, 3, 4]
    /// stride[2] = 1                    (last dimension)
    /// stride[1] = 4                    (4 elements per row)
    /// stride[0] = 3 × 4 = 12          (12 elements per "sheet")
    /// ```
    func testStrideCalculationContiguous() {
        // 1D tensor
        let shape1D = TensorShape(5)
        XCTAssertEqual(shape1D.strides, [1], "1D stride should be [1]")
        
        // 2D tensor [2, 3]
        let shape2D = TensorShape(2, 3)
        XCTAssertEqual(shape2D.strides, [3, 1], "[2,3] strides should be [3,1]")
        
        // 3D tensor [2, 3, 4]
        let shape3D = TensorShape(2, 3, 4)
        XCTAssertEqual(shape3D.strides, [12, 4, 1], "[2,3,4] strides should be [12,4,1]")
        
        // 4D tensor [2, 3, 4, 5]
        let shape4D = TensorShape(2, 3, 4, 5)
        XCTAssertEqual(shape4D.strides, [60, 20, 5, 1], "[2,3,4,5] strides should be [60,20,5,1]")
    }
    
    /// Test that isContiguous correctly identifies contiguous vs strided tensors
    ///
    /// **What:** Contiguous = elements are packed sequentially in memory
    /// **Why:** Contiguous tensors are faster (better cache locality)
    /// **How:** Verify new tensors are contiguous, transposed are not
    func testIsContiguous() {
        let a = Tensor<Float>.zeros(shape: TensorShape(3, 4))
        XCTAssertTrue(a.isContiguous, "Newly created tensor should be contiguous")
        
        // After transpose, stride order changes -> not contiguous
        let b = a.transpose()
        XCTAssertFalse(b.isContiguous, "Transposed tensor should not be contiguous")
    }
    
    /// Test offset calculation with custom strides
    ///
    /// **What:** Verify we can calculate correct memory offset with non-standard strides
    /// **Why:** Transpose changes strides but not data - need correct indexing
    /// **How:** Create tensor with known strides, verify subscript access
    func testOffsetCalculationWithStrides() {
        // Create [2, 3]: [[1,2,3], [4,5,6]]
        let a = Tensor<Float>(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
        
        // Transpose to [3, 2]: [[1,4], [2,5], [3,6]]
        let b = a.transpose()
        
        // Verify transposed access
        XCTAssertEqual(b.shape.dimensions, [3, 2], "Transposed shape")
        XCTAssertEqual(b[0,0], 1.0, accuracy: 1e-5, "b[0,0] should access a[0,0]")
        XCTAssertEqual(b[0,1], 4.0, accuracy: 1e-5, "b[0,1] should access a[1,0]")
        XCTAssertEqual(b[1,0], 2.0, accuracy: 1e-5, "b[1,0] should access a[0,1]")
        XCTAssertEqual(b[1,1], 5.0, accuracy: 1e-5, "b[1,1] should access a[1,1]")
        XCTAssertEqual(b[2,0], 3.0, accuracy: 1e-5, "b[2,0] should access a[0,2]")
        XCTAssertEqual(b[2,1], 6.0, accuracy: 1e-5, "b[2,1] should access a[1,2]")
    }
    
    /// Test reshape on contiguous tensor (zero-copy)
    ///
    /// **What:** Reshape should create new view without copying when tensor is contiguous
    /// **Why:** Memory efficiency - don't copy gigabytes of data unnecessarily!
    /// **How:** Reshape and verify elements are accessible with new shape
    func testReshapeContiguous() {
        // Create 1D: [1,2,3,4,5,6]
        let a = Tensor<Float>(shape: TensorShape(6), data: [1,2,3,4,5,6])
        
        // Reshape to [2, 3]
        let b = a.reshape(TensorShape(2, 3))
        
        XCTAssertEqual(b.shape.dimensions, [2, 3])
        XCTAssertEqual(b[0,0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(b[0,1], 2.0, accuracy: 1e-5)
        XCTAssertEqual(b[0,2], 3.0, accuracy: 1e-5)
        XCTAssertEqual(b[1,0], 4.0, accuracy: 1e-5)
        XCTAssertEqual(b[1,1], 5.0, accuracy: 1e-5)
        XCTAssertEqual(b[1,2], 6.0, accuracy: 1e-5)
        
        // Reshape to [3, 2]
        let c = a.reshape(TensorShape(3, 2))
        XCTAssertEqual(c.shape.dimensions, [3, 2])
        XCTAssertEqual(c[0,0], 1.0, accuracy: 1e-5)
        XCTAssertEqual(c[2,1], 6.0, accuracy: 1e-5)
    }
    
    /// Test that transpose + matmul works (real transformer use case)
    ///
    /// **What:** Q × Kᵀ is how attention computes scores
    /// **Why:** Most important use of transpose in transformers!
    /// **How:** Create Q and K, transpose K, multiply
    func testTransposeMatMul() {
        // Q: [2, 3] - 2 queries, 3 dimensions each
        let q = Tensor<Float>(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
        
        // K: [2, 3] - 2 keys, 3 dimensions each
        let k = Tensor<Float>(shape: TensorShape(2, 3), data: [7,8,9,10,11,12])
        
        // Attention scores: Q × Kᵀ
        // Q: [2,3], Kᵀ: [3,2] → result: [2,2]
        let scores = q.matmul(k.transpose())
        
        XCTAssertEqual(scores.shape.dimensions, [2, 2])
        
        // Manual calculation:
        // Q: [[1,2,3], [4,5,6]]
        // K: [[7,8,9], [10,11,12]]
        // Kᵀ: [[7,10], [8,11], [9,12]]
        //
        // scores[0,0] = Q[0,:] · Kᵀ[:,0] = [1,2,3]·[7,8,9] = 1×7 + 2×8 + 3×9 = 50
        // scores[0,1] = Q[0,:] · Kᵀ[:,1] = [1,2,3]·[10,11,12] = 1×10 + 2×11 + 3×12 = 68
        // scores[1,0] = Q[1,:] · Kᵀ[:,0] = [4,5,6]·[7,8,9] = 4×7 + 5×8 + 6×9 = 122
        // scores[1,1] = Q[1,:] · Kᵀ[:,1] = [4,5,6]·[10,11,12] = 4×10 + 5×11 + 6×12 = 167
        
        XCTAssertEqual(scores[0,0], 50.0, accuracy: 1e-3)
        XCTAssertEqual(scores[0,1], 68.0, accuracy: 1e-3)
        XCTAssertEqual(scores[1,0], 122.0, accuracy: 1e-3)
        XCTAssertEqual(scores[1,1], 167.0, accuracy: 1e-3)  // Fixed: was 157
    }
    
    // MARK: - Subscript Access Tests (TDD)
    
    /// Test subscript read access for 2D tensors
    ///
    /// **What:** Verify we can read individual elements using `tensor[row, col]` syntax
    /// **Why:** Foundation for debugging and validation - need to inspect tensor contents
    /// **How:** Create a known tensor and read elements, comparing to expected values
    func testSubscriptRead2D() {
        // Create a 2×3 tensor: [[1,2,3], [4,5,6]]
        let tensor = Tensor<Float>(shape: TensorShape(2, 3), 
                           data: [1, 2, 3, 4, 5, 6])
        
        // Test row 0
        XCTAssertEqual(tensor[0, 0], 1.0, accuracy: 1e-5, "Row 0, Col 0 should be 1")
        XCTAssertEqual(tensor[0, 1], 2.0, accuracy: 1e-5, "Row 0, Col 1 should be 2")
        XCTAssertEqual(tensor[0, 2], 3.0, accuracy: 1e-5, "Row 0, Col 2 should be 3")
        
        // Test row 1
        XCTAssertEqual(tensor[1, 0], 4.0, accuracy: 1e-5, "Row 1, Col 0 should be 4")
        XCTAssertEqual(tensor[1, 1], 5.0, accuracy: 1e-5, "Row 1, Col 1 should be 5")
        XCTAssertEqual(tensor[1, 2], 6.0, accuracy: 1e-5, "Row 1, Col 2 should be 6")
    }
    
    /// Test subscript write access for 2D tensors
    ///
    /// **What:** Verify we can modify individual elements using `tensor[row, col] = value`
    /// **Why:** Needed for manual tensor construction and testing
    /// **How:** Create a zero tensor, write values, verify they were set correctly
    func testSubscriptWrite2D() {
        var tensor = Tensor<Float>.zeros(shape: TensorShape(2, 2))
        
        // Write values
        tensor[0, 0] = 10.0
        tensor[0, 1] = 20.0
        tensor[1, 0] = 30.0
        tensor[1, 1] = 40.0
        
        // Verify they were written
        XCTAssertEqual(tensor[0, 0], 10.0, accuracy: 1e-5)
        XCTAssertEqual(tensor[0, 1], 20.0, accuracy: 1e-5)
        XCTAssertEqual(tensor[1, 0], 30.0, accuracy: 1e-5)
        XCTAssertEqual(tensor[1, 1], 40.0, accuracy: 1e-5)
    }
    
    /// Test subscript with 1D tensor (vector)
    ///
    /// **What:** Verify subscript works for vectors (single index)
    /// **Why:** Not all tensors are 2D - need to support vectors too
    /// **How:** Create a 1D tensor and access elements with single index
    func testSubscriptRead1D() {
        let tensor = Tensor<Float>(shape: TensorShape(5), 
                           data: [10, 20, 30, 40, 50])
        
        XCTAssertEqual(tensor[0], 10.0, accuracy: 1e-5)
        XCTAssertEqual(tensor[2], 30.0, accuracy: 1e-5)
        XCTAssertEqual(tensor[4], 50.0, accuracy: 1e-5)
    }
    
    // MARK: - Matrix Multiplication Tests (TDD)
    
    /// Test basic 2×3 by 3×2 matrix multiplication
    ///
    /// **What:** Multiply two small matrices and verify the result
    /// **Why:** MatMul is the foundation of transformer attention and MLP layers.
    ///         In LLMs, 70% of compute time is matrix multiplication!
    /// **How:** Use known values that can be manually verified
    ///
    /// **Educational Note:**
    /// Matrix multiplication works by taking dot products of rows and columns:
    /// ```
    /// A[2,3] × B[3,2] = C[2,2]
    ///
    /// C[i,j] = Σₖ A[i,k] × B[k,j]
    /// ```
    func testMatMulBasic() {
        // A: [2, 3]
        let a = Tensor<Float>(shape: TensorShape(2, 3), 
                      data: [1, 2, 3,    // Row 0
                             4, 5, 6])   // Row 1
        
        // B: [3, 2]
        let b = Tensor<Float>(shape: TensorShape(3, 2), 
                      data: [7, 8,       // Row 0
                             9, 10,      // Row 1
                             11, 12])    // Row 2
        
        // C = A × B should be [2, 2]
        let c = a.matmul(b)
        
        // Verify shape
        XCTAssertEqual(c.shape, TensorShape(2, 2), "Result shape should be [2, 2]")
        
        // Manually calculated expected values:
        // C[0,0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
        // C[0,1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
        // C[1,0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
        // C[1,1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
        
        XCTAssertEqual(c[0, 0], 58.0, accuracy: 1e-5, "C[0,0] should be 58")
        XCTAssertEqual(c[0, 1], 64.0, accuracy: 1e-5, "C[0,1] should be 64")
        XCTAssertEqual(c[1, 0], 139.0, accuracy: 1e-5, "C[1,0] should be 139")
        XCTAssertEqual(c[1, 1], 154.0, accuracy: 1e-5, "C[1,1] should be 154")
    }
    
    /// Test matrix-vector multiplication
    ///
    /// **What:** Multiply a matrix by a vector (common in MLP layers)
    /// **Why:** Feed-forward networks in transformers do matrix-vector products
    /// **How:** [2,3] × [3,1] = [2,1]
    func testMatMulMatrixVector() {
        // A: [2, 3]
        let a = Tensor<Float>(shape: TensorShape(2, 3), 
                      data: [1, 2, 3,
                             4, 5, 6])
        
        // B: [3, 1] (column vector)
        let b = Tensor<Float>(shape: TensorShape(3, 1), 
                      data: [10, 20, 30])
        
        // C = A × B should be [2, 1]
        let c = a.matmul(b)
        
        XCTAssertEqual(c.shape, TensorShape(2, 1))
        
        // C[0,0] = 1×10 + 2×20 + 3×30 = 10 + 40 + 90 = 140
        // C[1,0] = 4×10 + 5×20 + 6×30 = 40 + 100 + 180 = 320
        XCTAssertEqual(c[0, 0], 140.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 0], 320.0, accuracy: 1e-5)
    }
    
    /// Test identity matrix multiplication
    ///
    /// **What:** A × I = A (identity property)
    /// **Why:** Validates matmul correctness - multiplying by identity shouldn't change anything
    /// **How:** Create a matrix and identity, multiply, verify result equals original
    func testMatMulIdentity() {
        // A: [3, 3]
        let a = Tensor<Float>(shape: TensorShape(3, 3), 
                      data: [1, 2, 3,
                             4, 5, 6,
                             7, 8, 9])
        
        // I: [3, 3] identity matrix
        let identity = Tensor<Float>.identity(size: 3)
        
        // A × I should equal A
        let c = a.matmul(identity)
        
        XCTAssertEqual(c.shape, a.shape)
        for i in 0..<3 {
            for j in 0..<3 {
                XCTAssertEqual(c[i, j], a[i, j], accuracy: 1e-5,
                              "A × I should equal A at [\(i),\(j)]")
            }
        }
    }
    
    /// Test large matrix multiplication (performance validation)
    ///
    /// **What:** Multiply two 128×128 matrices
    /// **Why:** Ensures Accelerate is actually being used (manual loops would be slow)
    /// **How:** Time the operation - should be < 10ms on Apple Silicon
    func testMatMulLarge() {
        let size = 128
        
        let a = Tensor<Float>.filled(shape: TensorShape(size, size), value: 1.0)
        let b = Tensor<Float>.filled(shape: TensorShape(size, size), value: 2.0)
        
        let start = Date()
        let c = a.matmul(b)
        let elapsed = Date().timeIntervalSince(start)
        
        // Verify result shape
        XCTAssertEqual(c.shape, TensorShape(size, size))
        
        // Each element should be: 1×2 × 128 times = 256
        XCTAssertEqual(c[0, 0], Float(size * 2), accuracy: 1e-3)
        
        // Performance check: Should be fast with Accelerate
        XCTAssertLessThan(elapsed, 0.01, "128×128 matmul should be < 10ms with Accelerate")
        
        print("✅ MatMul 128×128 completed in \(elapsed * 1000) ms")
    }
    
    // MARK: - Element-Wise Operations Tests (TDD)
    
    /// Test element-wise addition
    ///
    /// **What:** Add two tensors element-by-element: C[i] = A[i] + B[i]
    /// **Why:** Used in residual connections (the "skip connections" in transformers)
    ///         Every transformer layer has: output = LayerNorm(x + Attention(x))
    /// **How:** Create two tensors, add them, verify each element
    ///
    /// **Educational Note:**
    /// Element-wise addition is different from matrix multiplication!
    /// ```
    /// [1, 2, 3] + [10, 20, 30] = [11, 22, 33]
    /// ```
    /// Each element pairs with the element at the same position.
    func testElementWiseAddition() {
        let a = Tensor<Float>(shape: TensorShape(2, 3), 
                      data: [1, 2, 3, 4, 5, 6])
        let b = Tensor<Float>(shape: TensorShape(2, 3), 
                      data: [10, 20, 30, 40, 50, 60])
        
        let c = a + b
        
        // Verify shape unchanged
        XCTAssertEqual(c.shape, TensorShape(2, 3))
        
        // Verify each element is sum
        XCTAssertEqual(c[0, 0], 11.0, accuracy: 1e-5)
        XCTAssertEqual(c[0, 1], 22.0, accuracy: 1e-5)
        XCTAssertEqual(c[0, 2], 33.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 0], 44.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 1], 55.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 2], 66.0, accuracy: 1e-5)
    }
    
    /// Test element-wise multiplication
    ///
    /// **What:** Multiply two tensors element-by-element: C[i] = A[i] × B[i]
    /// **Why:** Used in attention masks and gating mechanisms
    ///         Attention applies a mask: masked_scores = scores × mask
    /// **How:** Create two tensors, multiply them, verify each element
    func testElementWiseMultiplication() {
        let a = Tensor<Float>(shape: TensorShape(3, 2), 
                      data: [2, 3, 4, 5, 6, 7])
        let b = Tensor<Float>(shape: TensorShape(3, 2), 
                      data: [10, 10, 10, 10, 10, 10])
        
        let c = a * b
        
        // Verify shape unchanged
        XCTAssertEqual(c.shape, TensorShape(3, 2))
        
        // Verify each element is product
        XCTAssertEqual(c[0, 0], 20.0, accuracy: 1e-5)
        XCTAssertEqual(c[0, 1], 30.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 0], 40.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 1], 50.0, accuracy: 1e-5)
        XCTAssertEqual(c[2, 0], 60.0, accuracy: 1e-5)
        XCTAssertEqual(c[2, 1], 70.0, accuracy: 1e-5)
    }
    
    /// Test scalar addition
    ///
    /// **What:** Add a scalar to every element: C[i] = A[i] + s
    /// **Why:** Used in bias addition and numerical adjustments
    /// **How:** Add 5.0 to every element of a tensor
    func testScalarAddition() {
        let a = Tensor<Float>(shape: TensorShape(2, 2), 
                      data: [1, 2, 3, 4])
        
        let c = a + 5.0
        
        XCTAssertEqual(c[0, 0], 6.0, accuracy: 1e-5)
        XCTAssertEqual(c[0, 1], 7.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 0], 8.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 1], 9.0, accuracy: 1e-5)
    }
    
    /// Test scalar multiplication
    ///
    /// **What:** Multiply every element by a scalar: C[i] = A[i] × s
    /// **Why:** Used for scaling (e.g., attention scores / sqrt(d_k))
    /// **How:** Multiply every element by 2.0
    func testScalarMultiplication() {
        let a = Tensor<Float>(shape: TensorShape(3, 1), 
                      data: [5, 10, 15])
        
        let c = a * 2.0
        
        XCTAssertEqual(c[0, 0], 10.0, accuracy: 1e-5)
        XCTAssertEqual(c[1, 0], 20.0, accuracy: 1e-5)
        XCTAssertEqual(c[2, 0], 30.0, accuracy: 1e-5)
    }
    
    // MARK: - Activation Function Tests (TDD)
    
    /// Test GELU activation function
    ///
    /// **What:** GELU (Gaussian Error Linear Unit) - smooth activation function
    /// **Why:** Used in GPT, BERT, and most modern transformers
    ///         Smoother than ReLU, allows small negative values
    /// **How:** Test known input/output pairs and verify properties
    ///
    /// **Educational Note:**
    /// GELU formula: `GELU(x) ≈ x × Φ(x)` where Φ is cumulative normal distribution
    ///
    /// Approximation: `GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))`
    ///
    /// **Properties:**
    /// - GELU(0) ≈ 0
    /// - GELU(x) > 0 for x > 0
    /// - GELU(x) < 0 for x < 0 (small negative values allowed)
    /// - Smooth (differentiable everywhere)
    func testGELU() {
        let input = Tensor<Float>(shape: TensorShape(5), 
                          data: [-2.0, -1.0, 0.0, 1.0, 2.0])
        
        let output = input.gelu()
        
        // Verify shape unchanged
        XCTAssertEqual(output.shape, input.shape)
        
        // GELU(0) should be approximately 0
        XCTAssertEqual(output[2], 0.0, accuracy: 1e-3, "GELU(0) ≈ 0")
        
        // GELU(1) ≈ 0.841
        XCTAssertEqual(output[3], 0.841, accuracy: 1e-2, "GELU(1) ≈ 0.841")
        
        // GELU is smooth - negative inputs give small negative outputs
        XCTAssertLessThan(output[1], 0.0, "GELU(-1) should be slightly negative")
        XCTAssertGreaterThan(output[1], -0.2, "GELU(-1) should be close to 0")
        
        // Positive inputs give positive outputs
        XCTAssertGreaterThan(output[4], 0.0, "GELU(2) should be positive")
    }
    
    /// Test ReLU activation function
    ///
    /// **What:** ReLU (Rectified Linear Unit) - simple thresholding: max(0, x)
    /// **Why:** Classic activation, still used in some transformers and MLPs
    ///         Very fast to compute
    /// **How:** Verify positive values pass through, negative become zero
    ///
    /// **Educational Note:**
    /// ReLU formula: `ReLU(x) = max(0, x)`
    /// ```
    /// ReLU(-5) = 0
    /// ReLU(-1) = 0
    /// ReLU(0) = 0
    /// ReLU(1) = 1
    /// ReLU(5) = 5
    /// ```
    func testReLU() {
        let input = Tensor<Float>(shape: TensorShape(6), 
                          data: [-2.0, -0.5, 0.0, 0.5, 1.0, 2.0])
        
        let output = input.relu()
        
        // Verify shape unchanged
        XCTAssertEqual(output.shape, input.shape)
        
        // Negative values become zero
        XCTAssertEqual(output[0], 0.0, accuracy: 1e-5, "ReLU(-2) = 0")
        XCTAssertEqual(output[1], 0.0, accuracy: 1e-5, "ReLU(-0.5) = 0")
        
        // Zero stays zero
        XCTAssertEqual(output[2], 0.0, accuracy: 1e-5, "ReLU(0) = 0")
        
        // Positive values pass through unchanged
        XCTAssertEqual(output[3], 0.5, accuracy: 1e-5, "ReLU(0.5) = 0.5")
        XCTAssertEqual(output[4], 1.0, accuracy: 1e-5, "ReLU(1) = 1")
        XCTAssertEqual(output[5], 2.0, accuracy: 1e-5, "ReLU(2) = 2")
    }
    
    // MARK: - Normalization Function Tests (TDD)
    
    /// Test Softmax normalization
    ///
    /// **What:** Convert numbers to probabilities that sum to 1.0
    /// **Why:** Used in attention mechanism to create probability distributions
    ///         "Which words should this word pay attention to?"
    /// **How:** Verify output sums to 1 and larger inputs get higher probabilities
    ///
    /// **Educational Note:**
    /// Softmax formula: `softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)`
    ///
    /// **Why subtract max?** Numerical stability!
    /// ```
    /// exp(1000) = ∞  (overflow!)
    /// exp(1000 - 1000) = exp(0) = 1  (safe!)
    /// ```
    func testSoftmax() {
        let input = Tensor<Float>(shape: TensorShape(5), 
                          data: [1.0, 2.0, 3.0, 4.0, 5.0])
        
        let output = input.softmax()
        
        // Verify shape unchanged
        XCTAssertEqual(output.shape, input.shape)
        
        // Critical property: All probabilities sum to 1.0
        let sum = (0..<5).map { output[$0] }.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5, "Softmax must sum to 1.0")
        
        // All values should be between 0 and 1 (probabilities)
        for i in 0..<5 {
            XCTAssertGreaterThanOrEqual(output[i], 0.0, "Probability must be >= 0")
            XCTAssertLessThanOrEqual(output[i], 1.0, "Probability must be <= 1")
        }
        
        // Larger input values should get higher probabilities
        XCTAssertGreaterThan(output[4], output[3], "softmax(5) > softmax(4)")
        XCTAssertGreaterThan(output[3], output[2], "softmax(4) > softmax(3)")
    }
    
    /// Test Softmax with large values (numerical stability)
    ///
    /// **What:** Test that softmax doesn't overflow with large numbers
    /// **Why:** Without max subtraction, exp(1000) = ∞
    /// **How:** Use large values and verify no NaN/Inf in output
    func testSoftmaxNumericalStability() {
        let input = Tensor<Float>(shape: TensorShape(3), 
                          data: [1000.0, 1001.0, 1002.0])
        
        let output = input.softmax()
        
        // Should not produce NaN or Inf
        for i in 0..<3 {
            XCTAssertFalse(output[i].isNaN, "Softmax should not produce NaN")
            XCTAssertFalse(output[i].isInfinite, "Softmax should not produce Inf")
        }
        
        // Should still sum to 1.0
        let sum = (0..<3).map { output[$0] }.reduce(0, +)
        XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
    }
    
    /// Test Softmax on 2D tensor (per-row normalization)
    ///
    /// **What:** Verify that each ROW sums to 1.0 independently
    /// **Why:** In attention, each token's attention weights must sum to 1.0
    ///         Attention scores shape: [seq_len, seq_len]
    ///         Each row = probability distribution for one token
    /// **How:** Create [3, 4] tensor, verify each of 3 rows sums to 1.0
    func testSoftmax2D() {
        // [3, 4] tensor - 3 rows, 4 columns
        let input = Tensor<Float>(shape: TensorShape(3, 4), 
                          data: [1, 2, 3, 4,      // Row 0
                                 5, 6, 7, 8,      // Row 1
                                 9, 10, 11, 12])  // Row 2
        
        let output = input.softmax()
        
        // Each row should sum to 1.0
        for row in 0..<3 {
            let rowSum = (0..<4).map { output[row, $0] }.reduce(0, +)
            XCTAssertEqual(rowSum, 1.0, accuracy: 1e-5, 
                          "Row \(row) should sum to 1.0")
        }
        
        // Within each row, larger values should get higher probabilities
        XCTAssertGreaterThan(output[0, 3], output[0, 0], 
                            "Row 0: softmax(4) > softmax(1)")
    }
    
    /// Test LayerNorm normalization
    ///
    /// **What:** Normalize to mean=0, variance=1
    /// **Why:** Stabilizes training in deep networks (every transformer layer uses this!)
    /// **How:** Verify output has mean≈0 and variance≈1
    ///
    /// **Educational Note:**
    /// LayerNorm formula: `LN(x) = (x - mean(x)) / sqrt(variance(x) + ε)`
    ///
    /// **Why normalize?**
    /// - Prevents exploding/vanishing activations in deep networks
    /// - Makes optimization easier
    /// - Critical for transformer stability
    func testLayerNorm() {
        let input = Tensor<Float>(shape: TensorShape(100), 
                          data: (0..<100).map { Float($0) })  // [0, 1, 2, ..., 99]
        
        let output = input.layerNorm()
        
        // Verify shape unchanged
        XCTAssertEqual(output.shape, input.shape)
        
        // Calculate mean of output
        let mean = output.data.reduce(0, +) / Float(output.data.count)
        XCTAssertEqual(mean, 0.0, accuracy: 1e-3, "LayerNorm output should have mean ≈ 0")
        
        // Calculate variance of output
        let variance = output.data.map { pow($0 - mean, 2) }.reduce(0, +) / Float(output.data.count)
        XCTAssertEqual(variance, 1.0, accuracy: 1e-2, "LayerNorm output should have variance ≈ 1")
    }
    
    /// Test LayerNorm on 2D tensor (per-row normalization)
    ///
    /// **What:** Verify that each ROW has mean≈0 and variance≈1 independently
    /// **Why:** In transformers, each token's embedding gets normalized separately
    ///         Shape [batch, hidden_dim]: normalize each batch item's hidden vector
    /// **How:** Create [2, 100] tensor, verify both rows have mean≈0, var≈1
    func testLayerNorm2D() {
        // [2, 100] tensor - 2 rows of 100 features each
        let data1 = (0..<100).map { Float($0) }           // Row 0: [0..99]
        let data2 = (0..<100).map { Float($0 * 2) }       // Row 1: [0, 2, 4, ..., 198]
        let input = Tensor<Float>(shape: TensorShape(2, 100), 
                          data: data1 + data2)
        
        let output = input.layerNorm()
        
        // Verify shape unchanged
        XCTAssertEqual(output.shape, TensorShape(2, 100))
        
        // Check each row independently
        for row in 0..<2 {
            // Extract row
            let rowData = (0..<100).map { output[row, $0] }
            
            // Calculate mean of this row
            let rowMean = rowData.reduce(0, +) / Float(rowData.count)
            XCTAssertEqual(rowMean, 0.0, accuracy: 1e-3, 
                          "Row \(row) should have mean ≈ 0")
            
            // Calculate variance of this row
            let rowVariance = rowData.map { pow($0 - rowMean, 2) }.reduce(0, +) / Float(rowData.count)
            XCTAssertEqual(rowVariance, 1.0, accuracy: 1e-2, 
                          "Row \(row) should have variance ≈ 1")
        }
    }
}

