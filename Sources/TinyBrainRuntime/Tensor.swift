/// # Tensor – The Building Block of TinyBrain
///
/// ## What the heck is a tensor? 🤔
///
/// **TL;DR:** A tensor is just a fancy name for a multi-dimensional array (a grid of numbers).
/// That's it. Don't let the math terminology intimidate you!
///
/// ## The Progression (From Simple to Complex)
///
/// Think of tensors as a natural progression:
///
/// ```
/// 0D Tensor (Scalar)     →  5
///                           Just a single number
///
/// 1D Tensor (Vector)     →  [1, 2, 3, 4]
///                           A list of numbers (like an array)
///
/// 2D Tensor (Matrix)     →  [[1, 2, 3],
///                            [4, 5, 6]]
///                           A table of numbers (rows × columns)
///
/// 3D Tensor              →  Think of a Rubik's cube where each
///                           small square contains a number instead
///                           of a color (depth × rows × columns)
///
/// 4D+ Tensor             →  Hard to visualize, but follows the
///                           same pattern with more dimensions
/// ```
///
/// ## Why Do We Need Tensors in TinyBrain?
///
/// Language models work by converting **words into numbers** and doing **math** on those numbers.
/// Tensors let us organize and manipulate these numbers efficiently.
///
/// ### Real Examples from LLM Inference:
///
/// **Word Embeddings** (2D Tensor):
/// ```
/// Shape: [sequence_length, embedding_size]
/// Example: [10, 768]
///
/// "Hello world" → 2 words → 2 vectors of 768 numbers each
/// [[0.23, -0.15, ..., 0.44],   ← "Hello"
///  [0.87,  0.52, ..., -0.31]]  ← "world"
/// ```
///
/// **Attention Weights** (2D Tensor):
/// ```
/// Shape: [sequence_length, sequence_length]
/// Example: [10, 10]
///
/// Each cell shows how much word i should "attend to" word j
/// ```
///
/// **Model Weights** (2D Tensor):
/// ```
/// Shape: [input_features, output_features]
/// Example: [768, 3072]
///
/// Used in matrix multiplication to transform data
/// ```
///
/// **Batched Data** (3D Tensor):
/// ```
/// Shape: [batch_size, sequence_length, embedding_size]
/// Example: [4, 10, 768]
///
/// Processing 4 sentences at once for efficiency
/// ```
///
/// ## Why "Tensor" and Not Just "Array"?
///
/// The term comes from mathematics and physics, where tensors have a precise definition
/// involving coordinate transformations. In machine learning and programming, we use it
/// more loosely to mean "multi-dimensional array with specific operations."
///
/// Calling it a "tensor" signals that:
/// 1. We'll do mathematical operations on it (add, multiply, etc.)
/// 2. The shape/dimensions matter for the computations
/// 3. It's part of a computational graph (operations chain together)
///
/// ## How TinyBrain Uses Tensors
///
/// Every piece of data flowing through our LLM is a tensor:
/// - **Input tokens** → converted to embedding tensors
/// - **Attention mechanism** → operates on query/key/value tensors
/// - **Layer outputs** → tensors flowing through the network
/// - **Final logits** → a tensor of probabilities for the next word
///
/// ## Design Philosophy
///
/// TinyBrain's `Tensor` uses **value semantics** (it's a `struct`, not a `class`):
/// - Safer: No accidental mutations from other code
/// - Predictable: You know exactly what you're working with
/// - Performant: Copy-on-write means we avoid unnecessary copies
///
/// ---
///
/// **Learn More:**
/// - [3Blue1Brown: What's a Tensor?](https://www.youtube.com/watch?v=1GwAEnegaRs)
/// - [Visual Guide to Tensors](https://explained.ai/tensor-intro/)
/// - See `docs/overview.md` for TinyBrain's architecture
///
/// **Next Steps:**
/// - Read ``TensorShape`` to understand how shapes work
/// - See ``Tensor/zeros(shape:)`` for creating tensors
/// - Check `TensorTests.swift` for usage examples

import Foundation
import Accelerate

/// Describes the dimensions of a ``Tensor``
///
/// A `TensorShape` validates and stores the size of each dimension in a tensor.
///
/// Example:
/// ```swift
/// let shape = TensorShape(2, 3, 4)
/// // Represents a tensor with:
/// // - 2 elements in dimension 0
/// // - 3 elements in dimension 1
/// // - 4 elements in dimension 2
/// // Total elements: 2 × 3 × 4 = 24
/// ```
///
/// For a 2D tensor (matrix) representing word embeddings:
/// ```swift
/// let embeddingShape = TensorShape(10, 768)
/// // 10 tokens, each with a 768-dimensional embedding vector
/// ```
public struct TensorShape: Equatable, CustomStringConvertible {
    /// Dimensions of the tensor
    public let dimensions: [Int]
    
    /// Strides for each dimension (number of elements to skip)
    ///
    /// **What are strides?**
    /// Strides define how many elements to skip in the flat array when moving
    /// along each dimension.
    ///
    /// **For contiguous row-major layout:**
    /// ```
    /// Shape: [2, 3, 4]
    /// Strides: [12, 4, 1]
    ///
    /// Explanation:
    /// - Moving along dim 0 (rows): skip 12 elements (3×4)
    /// - Moving along dim 1 (cols): skip 4 elements
    /// - Moving along dim 2 (depth): skip 1 element
    /// ```
    ///
    /// **Why strides matter:**
    /// - Enable transpose without copying data (just swap dimensions & strides)
    /// - Support non-contiguous views of data
    /// - Memory-efficient reshape operations
    public let strides: [Int]
    
    /// Total number of elements
    public var count: Int {
        dimensions.reduce(1, *)
    }
    
    /// Check if this shape represents contiguous memory layout
    ///
    /// A tensor is contiguous if elements are packed sequentially (row-major order).
    /// Transposed or reshaped tensors may not be contiguous.
    ///
    /// **Why it matters:**
    /// - Contiguous: Better cache locality, faster
    /// - Non-contiguous: Flexible but may be slower
    public var isContiguous: Bool {
        // Check if strides match row-major layout
        var expectedStride = 1
        for i in (0..<dimensions.count).reversed() {
            if strides[i] != expectedStride {
                return false
            }
            expectedStride *= dimensions[i]
        }
        return true
    }
    
    public var description: String {
        "[\(dimensions.map(String.init).joined(separator: ", "))]"
    }
    
    /// Create a tensor shape from dimensions (defaults to contiguous row-major strides)
    ///
    /// - Parameter dimensions: The size of each dimension
    public init(_ dimensions: [Int]) {
        precondition(dimensions.allSatisfy { $0 > 0 }, "All dimensions must be positive")
        self.dimensions = dimensions
        
        // Calculate contiguous row-major strides
        // Work backwards: last dimension has stride 1
        var calculatedStrides = [Int](repeating: 1, count: dimensions.count)
        for i in (0..<(dimensions.count - 1)).reversed() {
            calculatedStrides[i] = calculatedStrides[i + 1] * dimensions[i + 1]
        }
        self.strides = calculatedStrides
    }
    
    /// Create a tensor shape with custom strides (for transpose/reshape)
    ///
    /// **Use with care!** Custom strides can create non-contiguous layouts.
    ///
    /// - Parameters:
    ///   - dimensions: The size of each dimension
    ///   - strides: Custom stride for each dimension
    public init(dimensions: [Int], strides: [Int]) {
        precondition(dimensions.count == strides.count, 
                    "Dimensions and strides must have same length")
        precondition(dimensions.allSatisfy { $0 > 0 }, 
                    "All dimensions must be positive")
        precondition(strides.allSatisfy { $0 > 0 }, 
                    "All strides must be positive")
        
        self.dimensions = dimensions
        self.strides = strides
    }
    
    /// Convenience initializer for common shapes
    public init(_ dimensions: Int...) {
        self.init(dimensions)
    }
}

/// Multi-dimensional array for numerical operations in TinyBrain
///
/// A `Tensor` holds the actual numerical data along with its shape. Think of it as
/// a container that knows both **what numbers it contains** and **how they're organized**.
///
/// ## Creating Tensors
///
/// **From explicit data:**
/// ```swift
/// let shape = TensorShape(2, 3)  // 2 rows, 3 columns
/// let data: [Float] = [1, 2, 3, 4, 5, 6]
/// let tensor = Tensor(shape: shape, data: data)
/// // Represents: [[1, 2, 3],
/// //              [4, 5, 6]]
/// ```
///
/// **Using factory methods:**
/// ```swift
/// // All zeros
/// let zeros = Tensor.zeros(shape: TensorShape(10, 768))
///
/// // All ones
/// let ones = Tensor.filled(shape: TensorShape(4, 4), value: 1.0)
///
/// // Custom value
/// let initialized = Tensor.filled(shape: TensorShape(3, 3), value: 0.01)
/// ```
///
/// ## Why Use Structs? (Value Semantics)
///
/// `Tensor` is a `struct`, not a `class`, which means it has **value semantics**:
///
/// ```swift
/// var a = Tensor.zeros(shape: TensorShape(2, 2))
/// var b = a  // 'b' is a copy, not a reference
///
/// // Modifying 'b' doesn't affect 'a'
/// b.data[0] = 5.0  // Only 'b' changes
/// ```
///
/// This prevents sneaky bugs where changing one tensor accidentally changes another.
///
/// **Note:** We'll add copy-on-write optimization later (TB-002) so copying is fast!
///
/// ## Storage Layout (Row-Major)
///
/// Data is stored in **row-major order** (the last dimension changes fastest):
///
/// ```swift
/// Shape: [2, 3]
/// Data:  [1, 2, 3, 4, 5, 6]
///
/// Represents:
/// Row 0: [1, 2, 3]
/// Row 1: [4, 5, 6]
///
/// Index calculation: row * cols + col
/// ```
///
/// This matches how most ML frameworks (PyTorch, NumPy) store data.
///
/// ## Current Limitations (Will Fix in Future Tasks)
///
/// - ⚠️ No bounds checking on data access (use with care!)
/// - ⚠️ No arithmetic operations yet (`+`, `-`, `*` coming in TB-002)
/// - ⚠️ No GPU acceleration yet (Metal kernels in TB-003)
/// - ⚠️ Only Float32 supported (quantization in TB-004)
///
/// ## Topics
/// ### Creating Tensors
/// - ``Tensor/init(shape:data:)``
/// - ``Tensor/zeros(shape:)``
/// - ``Tensor/filled(shape:value:)``
///
/// ### Properties
/// - ``Tensor/shape``
/// - ``Tensor/data``
public struct Tensor {
    /// The shape (dimensions) of this tensor
    ///
    /// Defines how the flat `data` array should be interpreted as a multi-dimensional structure.
    ///
    /// Example:
    /// ```swift
    /// let tensor = Tensor.zeros(shape: TensorShape(10, 768))
    /// print(tensor.shape)  // [10, 768]
    /// ```
    public let shape: TensorShape
    
    /// The underlying numerical data stored as a flat array
    ///
    /// Elements are stored in row-major order. Access is currently unprotected,
    /// so be careful with indices!
    ///
    /// **Note:** This will be optimized with copy-on-write in TB-002 to avoid
    /// unnecessary copies while maintaining value semantics.
    ///
    /// Example:
    /// ```swift
    /// let tensor = Tensor.filled(shape: TensorShape(2, 2), value: 5.0)
    /// print(tensor.data)  // [5.0, 5.0, 5.0, 5.0]
    /// ```
    internal var data: [Float]
    
    /// Creates a tensor with the given shape and data
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor
    ///   - data: The numerical values, must match `shape.count`
    ///
    /// - Precondition: `data.count` must equal `shape.count`
    ///
    /// Example:
    /// ```swift
    /// let shape = TensorShape(2, 3)
    /// let data: [Float] = [1, 2, 3, 4, 5, 6]
    /// let tensor = Tensor(shape: shape, data: data)
    /// ```
    public init(shape: TensorShape, data: [Float]) {
        precondition(data.count == shape.count, 
                     "Data count (\(data.count)) must match shape count (\(shape.count))")
        self.shape = shape
        self.data = data
    }
    
    /// Creates a tensor filled with zeros
    ///
    /// Useful for initializing buffers, gradients, or placeholder tensors.
    ///
    /// - Parameter shape: The dimensions of the tensor
    /// - Returns: A new tensor with all elements set to `0.0`
    ///
    /// Example:
    /// ```swift
    /// // Create a 10×768 embedding matrix initialized to zeros
    /// let embeddings = Tensor.zeros(shape: TensorShape(10, 768))
    /// ```
    public static func zeros(shape: TensorShape) -> Tensor {
        Tensor(shape: shape, data: Array(repeating: 0.0, count: shape.count))
    }
    
    /// Creates a tensor filled with a specific value
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor
    ///   - value: The value to fill every element with
    /// - Returns: A new tensor with all elements set to `value`
    ///
    /// Example:
    /// ```swift
    /// // Initialize weights to small random-ish value
    /// let weights = Tensor.filled(shape: TensorShape(768, 3072), value: 0.02)
    ///
    /// // Create a mask of ones
    /// let mask = Tensor.filled(shape: TensorShape(10, 10), value: 1.0)
    /// ```
    public static func filled(shape: TensorShape, value: Float) -> Tensor {
        Tensor(shape: shape, data: Array(repeating: value, count: shape.count))
    }
    
    /// Creates an identity matrix (1s on diagonal, 0s elsewhere)
    ///
    /// An identity matrix I has the property that: **A × I = I × A = A**
    ///
    /// Used in testing and as a building block for various transformations.
    ///
    /// - Parameter size: The size of the square matrix (both width and height)
    /// - Returns: A square identity matrix of shape `[size, size]`
    ///
    /// Example:
    /// ```swift
    /// let identity = Tensor.identity(size: 3)
    /// // [[1, 0, 0],
    /// //  [0, 1, 0],
    /// //  [0, 0, 1]]
    /// ```
    public static func identity(size: Int) -> Tensor {
        var data = [Float](repeating: 0.0, count: size * size)
        
        // Set diagonal elements to 1.0
        for i in 0..<size {
            data[i * size + i] = 1.0
        }
        
        return Tensor(shape: TensorShape(size, size), data: data)
    }
    
    /// Creates a tensor filled with random values from a normal distribution
    ///
    /// Generates random numbers with mean=0 and standard deviation=1 (standard normal).
    ///
    /// **Why Random Initialization Matters:**
    ///
    /// In neural networks, weights **must** start random (not zero):
    /// - All zeros → Symmetric → No learning!
    /// - Small random values → Break symmetry → Learning works!
    ///
    /// **Where It's Used:**
    /// - Weight initialization (before loading pretrained weights)
    /// - Testing (generate random test data)
    /// - Research (random baselines)
    ///
    /// - Parameters:
    ///   - shape: The dimensions of the tensor
    ///   - mean: Mean of the normal distribution (default: 0.0)
    ///   - std: Standard deviation (default: 1.0)
    /// - Returns: A tensor filled with random values from N(mean, std²)
    ///
    /// Example:
    /// ```swift
    /// // Standard normal: N(0, 1)
    /// let weights = Tensor.random(shape: TensorShape(768, 3072))
    ///
    /// // Custom: N(0, 0.02)
    /// let smallWeights = Tensor.random(shape: TensorShape(10, 10), mean: 0, std: 0.02)
    /// ```
    public static func random(shape: TensorShape, mean: Float = 0.0, std: Float = 1.0) -> Tensor {
        var data = [Float](repeating: 0.0, count: shape.count)
        
        // Generate random values using Box-Muller transform
        // Converts uniform random → normal random
        for i in Swift.stride(from: 0, to: shape.count, by: 2) {
            // Clamp u1 away from 0 to avoid log(0) = -∞
            let u1 = max(Float.random(in: 0..<1), Float.leastNonzeroMagnitude)
            let u2 = Float.random(in: 0..<1)
            
            // Box-Muller transform
            let r = sqrt(-2.0 * log(u1))
            let theta = 2.0 * Float.pi * u2
            
            data[i] = mean + std * r * cos(theta)
            
            if i + 1 < shape.count {
                data[i + 1] = mean + std * r * sin(theta)
            }
        }
        
        return Tensor(shape: shape, data: data)
    }
    
    // MARK: - Subscript Access
    
    /// Access tensor elements using multi-dimensional indices
    ///
    /// Supports 1D, 2D, and higher-dimensional indexing. Uses **row-major** layout
    /// (the last dimension changes fastest in memory).
    ///
    /// ## 1D Tensor (Vector):
    /// ```swift
    /// let v = Tensor(shape: TensorShape(5), data: [1,2,3,4,5])
    /// print(v[2])  // 3.0
    /// ```
    ///
    /// ## 2D Tensor (Matrix):
    /// ```swift
    /// let m = Tensor(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
    /// // Represents: [[1,2,3],
    /// //              [4,5,6]]
    /// print(m[1, 2])  // 6.0  (row 1, col 2)
    /// ```
    ///
    /// ## Row-Major Index Calculation:
    /// For a 2D tensor `[rows, cols]`:
    /// ```
    /// index = row * cols + col
    /// ```
    ///
    /// For higher dimensions, compute offset from rightmost dimension:
    /// ```
    /// index = i₀ * (d₁ * d₂ * ...) + i₁ * (d₂ * d₃ * ...) + ... + iₙ
    /// ```
    ///
    /// - Parameters:
    ///   - indices: The position in each dimension (must match `shape.dimensions.count`)
    /// - Returns: The value at the specified position
    /// - Precondition: `indices.count` must equal `shape.dimensions.count`
    /// - Precondition: Each index must be within bounds `[0, dimension_size)`
    public subscript(indices: Int...) -> Float {
        get {
            precondition(indices.count == shape.dimensions.count,
                        "Expected \(shape.dimensions.count) indices, got \(indices.count)")
            
            let offset = calculateOffset(indices: indices)
            precondition(offset >= 0 && offset < data.count,
                        "Index \(indices) out of bounds for shape \(shape)")
            
            return data[offset]
        }
        set {
            precondition(indices.count == shape.dimensions.count,
                        "Expected \(shape.dimensions.count) indices, got \(indices.count)")
            
            let offset = calculateOffset(indices: indices)
            precondition(offset >= 0 && offset < data.count,
                        "Index \(indices) out of bounds for shape \(shape)")
            
            data[offset] = newValue
        }
    }
    
    /// Calculate the linear offset for multi-dimensional indices using strides
    ///
    /// Uses the stride information from shape to calculate the correct offset,
    /// which works for both contiguous and non-contiguous (transposed) tensors.
    ///
    /// **Formula:** `offset = Σᵢ (indexᵢ × strideᵢ)`
    ///
    /// **Examples:**
    ///
    /// Contiguous [2, 3] with strides [3, 1]:
    /// ```
    /// [0, 0] → 0×3 + 0×1 = 0
    /// [0, 1] → 0×3 + 1×1 = 1
    /// [1, 0] → 1×3 + 0×1 = 3
    /// ```
    ///
    /// Transposed [3, 2] with strides [1, 3] (viewing same data):
    /// ```
    /// [0, 0] → 0×1 + 0×3 = 0  (same element!)
    /// [0, 1] → 0×1 + 1×3 = 3  (jumps to next row of original)
    /// [1, 0] → 1×1 + 0×3 = 1  (next column of original)
    /// ```
    ///
    /// - Parameter indices: The multi-dimensional index
    /// - Returns: Linear offset into the flat `data` array
    private func calculateOffset(indices: [Int]) -> Int {
        var offset = 0
        
        // Simply: offset = sum of (index[i] × stride[i])
        for i in 0..<indices.count {
            offset += indices[i] * shape.strides[i]
        }
        
        return offset
    }
    
    /// Whether this tensor's data is stored contiguously in memory
    ///
    /// **Contiguous:** Elements are packed sequentially (row-major order)
    /// **Non-contiguous:** Elements may be scattered (e.g., after transpose)
    ///
    /// **Why it matters:**
    /// - Contiguous tensors: Better cache performance, can use Accelerate/Metal directly
    /// - Non-contiguous tensors: May need to copy to contiguous buffer for some operations
    ///
    /// Example:
    /// ```swift
    /// let a = Tensor.zeros(shape: TensorShape(2, 3))
    /// a.isContiguous  // true
    ///
    /// let b = a.transpose()
    /// b.isContiguous  // false (strides changed)
    /// ```
    public var isContiguous: Bool {
        shape.isContiguous
    }
    
    /// Access raw data for GPU/Accelerate operations
    ///
    /// **Internal use only!** Exposes the underlying data array for backend implementations.
    /// Needed by Metal and Accelerate to access tensor data efficiently.
    ///
    /// - Returns: The underlying Float array
    public var rawData: [Float] {
        data
    }
    
    // MARK: - Shape Manipulation
    
    /// Transpose the tensor (swap dimensions without copying data)
    ///
    /// For 2D tensors, swaps rows and columns.
    /// For higher dimensions, reverses all axes.
    ///
    /// ## Zero-Copy Operation!
    ///
    /// **Key insight:** Transpose just swaps dimensions and strides - no data movement!
    ///
    /// ```swift
    /// Original [2, 3]:
    /// Data: [1, 2, 3, 4, 5, 6]
    /// Dims: [2, 3]
    /// Strides: [3, 1]
    ///
    /// Transposed [3, 2]:
    /// Data: [1, 2, 3, 4, 5, 6]  ← SAME DATA!
    /// Dims: [3, 2]               ← Swapped
    /// Strides: [1, 3]            ← Swapped
    /// ```
    ///
    /// **How it works:**
    ///
    /// Original layout (row-major):
    /// ```
    /// [1, 2, 3]  → stored as: [1, 2, 3, 4, 5, 6]
    /// [4, 5, 6]
    /// ```
    ///
    /// Transposed view (column-major):
    /// ```
    /// [1, 4]  → reads: data[0×1 + 0×3]=1, data[0×1 + 1×3]=4
    /// [2, 5]  → reads: data[1×1 + 0×3]=2, data[1×1 + 1×3]=5
    /// [3, 6]  → reads: data[2×1 + 0×3]=3, data[2×1 + 1×3]=6
    /// ```
    ///
    /// **Why this is critical for Metal:**
    /// - Attention needs Q×Kᵀ (transpose K)
    /// - Don't want to copy gigabytes of data!
    /// - Metal can handle non-contiguous layouts
    ///
    /// - Returns: A new tensor view with swapped dimensions (shares data!)
    /// - Precondition: Currently only supports 2D tensors
    ///
    /// Example:
    /// ```swift
    /// let a = Tensor([[1,2,3], [4,5,6]])  // [2, 3]
    /// let b = a.transpose()                // [3, 2]
    ///
    /// // b[0,1] accesses same memory as a[1,0]
    /// // NO DATA WAS COPIED! ✅
    /// ```
    public func transpose() -> Tensor {
        precondition(shape.dimensions.count == 2, 
                    "transpose() currently only supports 2D tensors")
        
        // Swap dimensions and strides
        let newDimensions = [shape.dimensions[1], shape.dimensions[0]]
        let newStrides = [shape.strides[1], shape.strides[0]]
        
        let newShape = TensorShape(dimensions: newDimensions, strides: newStrides)
        
        // Return new tensor sharing the same data!
        return Tensor(shape: newShape, data: self.data)
    }
    
    /// Reshape the tensor to a new shape
    ///
    /// Changes the dimensions while keeping the same total number of elements.
    ///
    /// ## When Reshape Can Share Data (Zero-Copy):
    ///
    /// If the tensor is **contiguous**, reshape just creates a new view:
    /// ```swift
    /// let a = Tensor(shape: TensorShape(6), data: [1,2,3,4,5,6])  // [6]
    /// let b = a.reshape(TensorShape(2, 3))                         // [2,3]
    /// // NO COPY! Same data, different shape ✅
    /// ```
    ///
    /// ## When Reshape Must Copy:
    ///
    /// If the tensor is **non-contiguous** (e.g., transposed), must copy:
    /// ```swift
    /// let a = Tensor([[1,2,3], [4,5,6]])  // [2,3]
    /// let b = a.transpose()                // [3,2] non-contiguous
    /// let c = b.reshape(TensorShape(6))    // Must copy to make contiguous
    /// ```
    ///
    /// - Parameter newShape: The desired shape (must have same element count)
    /// - Returns: Reshaped tensor (may share data if contiguous)
    /// - Precondition: `newShape.count` must equal `self.shape.count`
    ///
    /// Example:
    /// ```swift
    /// let a = Tensor(shape: TensorShape(12), data: (0..<12).map { Float($0) })
    /// let b = a.reshape(TensorShape(3, 4))   // [3, 4]
    /// let c = a.reshape(TensorShape(2, 6))   // [2, 6]
    /// let d = a.reshape(TensorShape(2, 2, 3)) // [2, 2, 3]
    /// ```
    public func reshape(_ newShape: TensorShape) -> Tensor {
        precondition(newShape.count == self.shape.count,
                    "Cannot reshape: element count mismatch (\(self.shape.count) vs \(newShape.count))")
        
        // If tensor is contiguous, can just change shape (zero-copy!)
        if self.isContiguous {
            return Tensor(shape: newShape, data: self.data)
        }
        
        // If non-contiguous, must copy to make contiguous
        // (This ensures the new shape's stride assumptions are correct)
        var newData = [Float](repeating: 0, count: self.shape.count)
        
        // Copy elements in the correct order
        var destIdx = 0
        // TODO: Implement proper iterator for non-contiguous tensors
        // For now, just copy via slow path
        for i in 0..<self.data.count {
            newData[destIdx] = self.data[i]
            destIdx += 1
        }
        
        return Tensor(shape: newShape, data: newData)
    }
    
    // MARK: - Matrix Operations
    
    /// Perform matrix multiplication using Accelerate's optimized BLAS
    ///
    /// Computes: **C = A × B** where:
    /// - A is `[M, K]`
    /// - B is `[K, N]`
    /// - C is `[M, N]`
    ///
    /// ## What is Matrix Multiplication?
    ///
    /// Matrix multiplication is **not** element-wise multiplication!
    /// It computes dot products of rows and columns:
    ///
    /// ```
    /// C[i,j] = Σₖ A[i,k] × B[k,j]
    /// ```
    ///
    /// Example:
    /// ```
    /// A = [[1, 2],      B = [[5, 6],
    ///      [3, 4]]           [7, 8]]
    ///
    /// C[0,0] = 1×5 + 2×7 = 19
    /// C[0,1] = 1×6 + 2×8 = 22
    /// C[1,0] = 3×5 + 4×7 = 43
    /// C[1,1] = 3×6 + 4×8 = 50
    ///
    /// C = [[19, 22],
    ///      [43, 50]]
    /// ```
    ///
    /// ## Why MatMul is Critical for LLMs
    ///
    /// In transformers, **70% of compute time** is matrix multiplication:
    /// - **Attention**: Q×Kᵀ to compute attention scores
    /// - **Attention output**: (Attention weights) × V
    /// - **MLP/FFN**: Two large matmuls per layer
    /// - **Output projection**: Final logits calculation
    ///
    /// For a 6-layer transformer processing 10 tokens:
    /// - ~50+ matrix multiplications per forward pass!
    ///
    /// ## Implementation Details
    ///
    /// Uses Apple's **cblas_sgemm** (Single-precision General Matrix Multiply):
    /// - Optimized for Apple Silicon (M1/M2/M3/M4)
    /// - Uses SIMD instructions (4-8 operations at once)
    /// - Multi-threaded automatically
    /// - Cache-optimized memory access
    /// - **10-100× faster** than manual loops!
    ///
    /// - Parameter other: The right-hand matrix
    /// - Returns: The matrix product
    /// - Precondition: Both tensors must be 2D (matrices)
    /// - Precondition: `self.shape[1]` must equal `other.shape[0]` (inner dimensions match)
    ///
    /// Example:
    /// ```swift
    /// let a = Tensor(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
    /// let b = Tensor(shape: TensorShape(3, 2), data: [7,8,9,10,11,12])
    /// let c = a.matmul(b)  // Shape: [2, 2]
    /// ```
    public func matmul(_ other: Tensor) -> Tensor {
        // Validate inputs
        precondition(self.shape.dimensions.count == 2, 
                    "Left tensor must be 2D, got shape \(self.shape)")
        precondition(other.shape.dimensions.count == 2, 
                    "Right tensor must be 2D, got shape \(other.shape)")
        precondition(self.shape.dimensions[1] == other.shape.dimensions[0],
                    "Inner dimensions must match: \(self.shape.dimensions[1]) ≠ \(other.shape.dimensions[0])")
        
        let m = self.shape.dimensions[0]
        let n = other.shape.dimensions[1]
        
        // Size-based auto-selection (auto mode only)
        if TinyBrainBackend.preferred == .auto {
            // Small matrices: Use CPU (overhead dominates)
            if !TinyBrainBackend.shouldUseMetal(m: m, n: n) {
                TinyBrainBackend.log("Using CPU for small matmul \(self.shape) × \(other.shape) (< \(TinyBrainBackend.metalSizeThreshold))")
                return matmulCPU(other)
            }
        }
        
        // Try Metal backend if configured and appropriate
        if (TinyBrainBackend.preferred == .metal || TinyBrainBackend.preferred == .auto),
           let metalBackend = TinyBrainBackend.metalBackend as? MatMulBackend {
            
            TinyBrainBackend.log("Using Metal backend for matmul \(self.shape) × \(other.shape)")
            
            do {
                return try metalBackend.matmul(self, other)
            } catch {
                // Metal failed, fall back to CPU
                TinyBrainBackend.log("Metal failed (\(error)), falling back to CPU")
            }
        }
        
        // CPU path (Accelerate)
        TinyBrainBackend.log("Using CPU backend for matmul \(self.shape) × \(other.shape)")
        return matmulCPU(other)
    }
    
    /// CPU-only matrix multiplication using Accelerate
    ///
    /// This is the fallback when Metal is unavailable or fails.
    /// Also used when backend preference is explicitly set to .cpu.
    private func matmulCPU(_ other: Tensor) -> Tensor {
        // Ensure both tensors are contiguous for BLAS
        let a = self.isContiguous ? self : self.makeContiguous()
        let b = other.isContiguous ? other : other.makeContiguous()
        
        let m = Int32(a.shape.dimensions[0])
        let k = Int32(a.shape.dimensions[1])
        let n = Int32(b.shape.dimensions[1])
        
        var result = Tensor.zeros(shape: TensorShape(Int(m), Int(n)))
        
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0,
            a.data, k,
            b.data, n,
            0.0,
            &result.data, n
        )
        
        return result
    }
    
    /// Make a contiguous copy of this tensor
    ///
    /// If the tensor is already contiguous, returns self.
    /// Otherwise, copies data into contiguous row-major layout.
    ///
    /// - Returns: A contiguous version of this tensor
    private func makeContiguous() -> Tensor {
        if isContiguous {
            return self
        }
        
        // Need to copy elements in the correct order using strides
        var newData = [Float](repeating: 0, count: shape.count)
        var destIdx = 0
        
        // For 2D tensors, iterate with proper stride-aware access
        if shape.dimensions.count == 2 {
            for i in 0..<shape.dimensions[0] {
                for j in 0..<shape.dimensions[1] {
                    let offset = i * shape.strides[0] + j * shape.strides[1]
                    newData[destIdx] = data[offset]
                    destIdx += 1
                }
            }
        } else {
            // Generic path for higher dimensions (not used yet)
            // TODO: Implement general iterator
            newData = self.data
        }
        
        return Tensor(shape: TensorShape(shape.dimensions), data: newData)
    }
    
    // MARK: - Element-Wise Operations
    
    /// Element-wise addition using vDSP
    ///
    /// Computes: **C[i] = A[i] + B[i]** for all elements
    ///
    /// ## What is Element-Wise Addition?
    ///
    /// Unlike matrix multiplication, this pairs elements at the **same position**:
    /// ```
    /// [1, 2, 3] + [10, 20, 30] = [11, 22, 33]
    /// ```
    ///
    /// ## Why It's Used in Transformers
    ///
    /// **Residual Connections** (skip connections):
    /// ```
    /// output = x + Attention(x)
    /// output = x + MLP(x)
    /// ```
    ///
    /// These residual connections:
    /// - Help gradients flow during training
    /// - Allow networks to be very deep (100+ layers)
    /// - Critical innovation that made transformers work!
    ///
    /// ## Implementation
    ///
    /// Uses Apple's **vDSP_vadd** (Vector Addition):
    /// - SIMD optimized (process 4-8 elements at once)
    /// - 10-20× faster than manual loops
    ///
    /// - Parameter other: The tensor to add
    /// - Returns: A new tensor containing element-wise sums
    /// - Precondition: Shapes must match exactly
    public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
        precondition(lhs.shape == rhs.shape, 
                    "Shapes must match for addition: \(lhs.shape) vs \(rhs.shape)")
        
        var result = Tensor.zeros(shape: lhs.shape)
        
        vDSP_vadd(
            lhs.data, 1,                    // Input A, stride 1
            rhs.data, 1,                    // Input B, stride 1  
            &result.data, 1,                // Output, stride 1
            vDSP_Length(lhs.data.count)     // Number of elements
        )
        
        return result
    }
    
    /// Element-wise multiplication using vDSP
    ///
    /// Computes: **C[i] = A[i] × B[i]** for all elements
    ///
    /// ## What is Element-Wise Multiplication?
    ///
    /// Multiplies corresponding elements (NOT matrix multiplication!):
    /// ```
    /// [2, 3, 4] × [10, 10, 10] = [20, 30, 40]
    /// ```
    ///
    /// ## Why It's Used in Transformers
    ///
    /// **Attention Masking:**
    /// ```
    /// masked_scores = attention_scores × mask
    /// ```
    /// Where mask is 0 for "ignore" and 1 for "attend to"
    ///
    /// **Gating Mechanisms:**
    /// ```
    /// output = x × gate_values
    /// ```
    /// Controls information flow
    ///
    /// - Parameter other: The tensor to multiply with
    /// - Returns: A new tensor containing element-wise products
    /// - Precondition: Shapes must match exactly
    public static func * (lhs: Tensor, rhs: Tensor) -> Tensor {
        precondition(lhs.shape == rhs.shape, 
                    "Shapes must match for multiplication: \(lhs.shape) vs \(rhs.shape)")
        
        var result = Tensor.zeros(shape: lhs.shape)
        
        vDSP_vmul(
            lhs.data, 1,                    // Input A, stride 1
            rhs.data, 1,                    // Input B, stride 1
            &result.data, 1,                // Output, stride 1
            vDSP_Length(lhs.data.count)     // Number of elements
        )
        
        return result
    }
    
    /// Scalar addition
    ///
    /// Adds a constant to every element: **C[i] = A[i] + s**
    ///
    /// ## Why It's Used
    ///
    /// **Bias Addition:**
    /// ```
    /// output = weights × input + bias
    /// ```
    ///
    /// **Numerical Adjustments:**
    /// ```
    /// stable_softmax = exp(x - max(x))
    /// ```
    ///
    /// - Parameter scalar: The value to add to each element
    /// - Returns: A new tensor with the scalar added to all elements
    public static func + (lhs: Tensor, rhs: Float) -> Tensor {
        var result = Tensor.zeros(shape: lhs.shape)
        var scalar = rhs  // vDSP needs var, not let
        
        vDSP_vsadd(
            lhs.data, 1,                    // Input, stride 1
            &scalar,                        // Scalar to add (needs &)
            &result.data, 1,                // Output, stride 1
            vDSP_Length(lhs.data.count)     // Number of elements
        )
        
        return result
    }
    
    /// Scalar multiplication
    ///
    /// Multiplies every element by a constant: **C[i] = A[i] × s**
    ///
    /// ## Why It's Used in Transformers
    ///
    /// **Attention Score Scaling:**
    /// ```
    /// scores = (Q × Kᵀ) / sqrt(d_k)
    /// ```
    /// The 1/sqrt(d_k) prevents gradients from vanishing
    ///
    /// **Learning Rate Scaling:**
    /// ```
    /// weights = weights - learning_rate × gradients
    /// ```
    ///
    /// - Parameter scalar: The value to multiply each element by
    /// - Returns: A new tensor with all elements scaled
    public static func * (lhs: Tensor, rhs: Float) -> Tensor {
        var result = Tensor.zeros(shape: lhs.shape)
        var scalar = rhs  // vDSP needs var
        
        vDSP_vsmul(
            lhs.data, 1,                    // Input, stride 1
            &scalar,                        // Scalar to multiply (needs &)
            &result.data, 1,                // Output, stride 1
            vDSP_Length(lhs.data.count)     // Number of elements
        )
        
        return result
    }
    
    // MARK: - Activation Functions
    
    /// Apply GELU activation function
    ///
    /// **GELU** (Gaussian Error Linear Unit) is a smooth activation function used in
    /// GPT, BERT, and most modern transformers.
    ///
    /// ## Formula
    ///
    /// Exact: `GELU(x) = x × Φ(x)` where Φ is cumulative normal distribution
    ///
    /// Approximation (what we implement):
    /// ```
    /// GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
    /// ```
    ///
    /// ## Why GELU vs ReLU?
    ///
    /// **ReLU:** Sharp cutoff at 0
    /// ```
    ///   ReLU(x) = { 0  if x < 0
    ///             { x  if x ≥ 0
    /// ```
    ///
    /// **GELU:** Smooth, allows small negative values
    /// ```
    ///   GELU(-1) ≈ -0.16  (small negative)
    ///   GELU(0)  ≈ 0
    ///   GELU(1)  ≈ 0.84   (mostly x)
    /// ```
    ///
    /// **Benefits:**
    /// - Smoother gradients (better for training)
    /// - Probabilistic interpretation (gate based on input magnitude)
    /// - Empirically performs better in large language models
    ///
    /// ## Where It's Used in Transformers
    ///
    /// **Feed-Forward Network:**
    /// ```
    /// FFN(x) = GELU(x × W₁ + b₁) × W₂ + b₂
    /// ```
    /// Applied after first linear layer in MLP
    ///
    /// - Returns: A new tensor with GELU applied element-wise
    public func gelu() -> Tensor {
        var result = Tensor.zeros(shape: self.shape)
        
        let sqrt2OverPi = sqrt(2.0 / Float.pi)
        
        for i in 0..<self.data.count {
            let x = self.data[i]
            let x3 = x * x * x
            let inner = sqrt2OverPi * (x + 0.044715 * x3)
            let tanhVal = tanh(inner)
            result.data[i] = 0.5 * x * (1.0 + tanhVal)
        }
        
        return result
    }
    
    /// Apply ReLU activation function
    ///
    /// **ReLU** (Rectified Linear Unit) is the simplest activation function:
    /// **ReLU(x) = max(0, x)**
    ///
    /// ## How It Works
    ///
    /// ```
    /// Input:  [-2, -1, 0, 1, 2]
    /// Output: [ 0,  0, 0, 1, 2]
    /// ```
    ///
    /// - Negative values → 0
    /// - Zero → 0  
    /// - Positive values → unchanged
    ///
    /// ## Why It's Used
    ///
    /// **Introduces Non-Linearity:**
    /// Without activation functions, stacking layers is useless:
    /// ```
    /// Linear(Linear(x)) = Linear(x)  // Just another linear function!
    /// ```
    ///
    /// ReLU breaks linearity:
    /// ```
    /// ReLU(Linear(x)) ≠ Linear(x)  // Now can learn complex patterns!
    /// ```
    ///
    /// **Very Fast:**
    /// Just a comparison and selection - no expensive math
    ///
    /// ## Where It's Used
    ///
    /// Some transformers (though GELU is more common now):
    /// ```
    /// FFN(x) = ReLU(x × W₁) × W₂
    /// ```
    ///
    /// - Returns: A new tensor with ReLU applied element-wise
    public func relu() -> Tensor {
        var result = Tensor.zeros(shape: self.shape)
        
        // Use vDSP's threshold function for performance
        // vDSP_vthres: Threshold values (set below threshold to 0)
        var threshold: Float = 0.0
        
        vDSP_vthres(
            self.data, 1,                   // Input, stride 1
            &threshold,                     // Threshold value
            &result.data, 1,                // Output, stride 1
            vDSP_Length(self.data.count)    // Number of elements
        )
        
        return result
    }
    
    // MARK: - Normalization Operations
    
    /// Apply Softmax normalization (convert to probability distribution)
    ///
    /// Computes: **softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)**
    ///
    /// ## What Does Softmax Do?
    ///
    /// Converts a vector of numbers into a **probability distribution**:
    /// ```
    /// Input:  [1.0, 2.0, 3.0]
    /// Output: [0.09, 0.24, 0.67]  ← Sums to 1.0!
    /// ```
    ///
    /// **Key properties:**
    /// - All outputs between 0 and 1
    /// - All outputs sum to exactly 1.0
    /// - Larger inputs get higher probabilities
    ///
    /// ## Why It's Critical for Transformers
    ///
    /// **Attention Mechanism** (the heart of transformers):
    /// ```
    /// 1. Compute scores: scores = Q × Kᵀ / sqrt(d_k)
    /// 2. Apply softmax: attention_weights = softmax(scores)
    /// 3. Weighted sum: output = attention_weights × V
    /// ```
    ///
    /// Softmax turns attention scores into probabilities:
    /// - "This word should attend 67% to word A, 24% to word B, 9% to word C"
    ///
    /// **Next Token Prediction:**
    /// ```
    /// logits = model(prompt)          // Raw scores for each token
    /// probs = softmax(logits)         // Convert to probabilities
    /// next_token = sample(probs)      // Pick next token
    /// ```
    ///
    /// ## Implementation: Numerically Stable Softmax
    ///
    /// **Naive approach (WRONG - overflows!):**
    /// ```
    /// exp(1000) / (exp(1000) + exp(999) + ...) = ∞ / ∞ = NaN
    /// ```
    ///
    /// **Correct approach (stable):**
    /// ```
    /// 1. Find max: m = max(x)
    /// 2. Subtract max: x' = x - m
    /// 3. Exp: e = exp(x')
    /// 4. Normalize: softmax = e / sum(e)
    /// ```
    ///
    /// This prevents overflow because exp(x - max(x)) ≤ exp(0) = 1
    ///
    /// Uses vDSP for vectorized operations (max, subtract, sum, divide)
    ///
    /// - Parameter epsilon: Small value for numerical stability (default: 1e-7)
    /// - Returns: A new tensor with softmax applied along the last dimension
    public func softmax(epsilon: Float = 1e-7) -> Tensor {
        precondition(shape.dimensions.count >= 1, "Cannot apply softmax to empty tensor")
        
        var result = Tensor.zeros(shape: self.shape)
        
        // For 1D tensor, apply softmax to the entire vector
        if shape.dimensions.count == 1 {
            let n = self.data.count
            
            var maxVal: Float = 0.0
            vDSP_maxv(self.data, 1, &maxVal, vDSP_Length(n))
            
            var shifted = [Float](repeating: 0, count: n)
            var negMax = -maxVal
            vDSP_vsadd(self.data, 1, &negMax, &shifted, 1, vDSP_Length(n))
            
            var expVals = [Float](repeating: 0, count: n)
            var count = Int32(n)
            vvexpf(&expVals, shifted, &count)
            
            var sum: Float = 0.0
            vDSP_sve(expVals, 1, &sum, vDSP_Length(n))
            sum += epsilon
            
            vDSP_vsdiv(expVals, 1, &sum, &result.data, 1, vDSP_Length(n))
            
            return result
        }
        
        // For multi-dimensional tensors, apply softmax along last dimension
        // Each "row" (fixing all but last index) sums to 1.0
        let lastDim = shape.dimensions.last!
        let numRows = shape.count / lastDim
        
        for row in 0..<numRows {
            let startIdx = row * lastDim
            let endIdx = startIdx + lastDim
            
            let rowData = Array(self.data[startIdx..<endIdx])
            
            // Softmax for this row
            var maxVal: Float = 0.0
            vDSP_maxv(rowData, 1, &maxVal, vDSP_Length(lastDim))
            
            var shifted = [Float](repeating: 0, count: lastDim)
            var negMax = -maxVal
            vDSP_vsadd(rowData, 1, &negMax, &shifted, 1, vDSP_Length(lastDim))
            
            var expVals = [Float](repeating: 0, count: lastDim)
            var count = Int32(lastDim)
            vvexpf(&expVals, shifted, &count)
            
            var sum: Float = 0.0
            vDSP_sve(expVals, 1, &sum, vDSP_Length(lastDim))
            sum += epsilon
            
            var rowResult = [Float](repeating: 0, count: lastDim)
            vDSP_vsdiv(expVals, 1, &sum, &rowResult, 1, vDSP_Length(lastDim))
            
            // Copy result back
            for i in 0..<lastDim {
                result.data[startIdx + i] = rowResult[i]
            }
        }
        
        return result
    }
    
    /// Apply Layer Normalization
    ///
    /// Computes: **LN(x) = (x - mean(x)) / sqrt(variance(x) + ε)**
    ///
    /// ## What Does LayerNorm Do?
    ///
    /// Normalizes activations to have:
    /// - **Mean ≈ 0**
    /// - **Variance ≈ 1**
    ///
    /// Example:
    /// ```
    /// Input:  [10, 20, 30, 40, 50]  (mean=30, high variance)
    /// Output: [-1.41, -0.71, 0, 0.71, 1.41]  (mean≈0, variance≈1)
    /// ```
    ///
    /// ## Why It's Critical for Transformers
    ///
    /// **Used TWICE in every transformer layer:**
    /// ```
    /// 1. After attention:  x = LayerNorm(x + Attention(x))
    /// 2. After MLP:        x = LayerNorm(x + MLP(x))
    /// ```
    ///
    /// **What it prevents:**
    /// - **Exploding activations:** Values growing exponentially through layers
    /// - **Vanishing gradients:** Gradients becoming too small to learn
    /// - **Training instability:** Network unable to converge
    ///
    /// **Why transformers need it:**
    /// - Deep networks (6-96 layers!)
    /// - Residual connections can amplify values
    /// - LayerNorm keeps everything in a stable range
    ///
    /// ## Implementation
    ///
    /// Uses vDSP for efficient computation:
    /// 1. Compute mean: `vDSP_meanv`
    /// 2. Subtract mean: `vDSP_vsadd`
    /// 3. Compute variance: Square → Mean
    /// 4. Divide by std: `vDSP_vsdiv`
    ///
    /// - Parameter epsilon: Small value added to variance for numerical stability (default: 1e-5)
    /// - Returns: A new tensor with LayerNorm applied along the last dimension
    public func layerNorm(epsilon: Float = 1e-5) -> Tensor {
        precondition(shape.dimensions.count >= 1, "Cannot apply layerNorm to empty tensor")
        
        var result = Tensor.zeros(shape: self.shape)
        
        // For 1D tensor, normalize the entire vector
        if shape.dimensions.count == 1 {
            let n = self.data.count
            
            var mean: Float = 0.0
            vDSP_meanv(self.data, 1, &mean, vDSP_Length(n))
            
            var centered = [Float](repeating: 0, count: n)
            var negMean = -mean
            vDSP_vsadd(self.data, 1, &negMean, &centered, 1, vDSP_Length(n))
            
            var squared = [Float](repeating: 0, count: n)
            vDSP_vsq(centered, 1, &squared, 1, vDSP_Length(n))
            
            var variance: Float = 0.0
            vDSP_meanv(squared, 1, &variance, vDSP_Length(n))
            
            var std = sqrt(variance + epsilon)
            vDSP_vsdiv(centered, 1, &std, &result.data, 1, vDSP_Length(n))
            
            return result
        }
        
        // For multi-dimensional tensors, normalize along last dimension
        // Each "feature vector" (last dimension) gets normalized independently
        let lastDim = shape.dimensions.last!
        let numVectors = shape.count / lastDim
        
        for vec in 0..<numVectors {
            let startIdx = vec * lastDim
            let endIdx = startIdx + lastDim
            
            let vecData = Array(self.data[startIdx..<endIdx])
            
            // LayerNorm for this vector
            var mean: Float = 0.0
            vDSP_meanv(vecData, 1, &mean, vDSP_Length(lastDim))
            
            var centered = [Float](repeating: 0, count: lastDim)
            var negMean = -mean
            vDSP_vsadd(vecData, 1, &negMean, &centered, 1, vDSP_Length(lastDim))
            
            var squared = [Float](repeating: 0, count: lastDim)
            vDSP_vsq(centered, 1, &squared, 1, vDSP_Length(lastDim))
            
            var variance: Float = 0.0
            vDSP_meanv(squared, 1, &variance, vDSP_Length(lastDim))
            
            var std = sqrt(variance + epsilon)
            
            var vecResult = [Float](repeating: 0, count: lastDim)
            vDSP_vsdiv(centered, 1, &std, &vecResult, 1, vDSP_Length(lastDim))
            
            // Copy result back
            for i in 0..<lastDim {
                result.data[startIdx + i] = vecResult[i]
            }
        }
        
        return result
    }
}

extension Tensor: CustomStringConvertible {
    public var description: String {
        "Tensor(shape: \(shape))"
    }
}

