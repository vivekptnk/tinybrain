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
    
    /// Total number of elements
    public var count: Int {
        dimensions.reduce(1, *)
    }
    
    public var description: String {
        "[\(dimensions.map(String.init).joined(separator: ", "))]"
    }
    
    /// Create a tensor shape from dimensions
    public init(_ dimensions: [Int]) {
        precondition(dimensions.allSatisfy { $0 > 0 }, "All dimensions must be positive")
        self.dimensions = dimensions
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
}

extension Tensor: CustomStringConvertible {
    public var description: String {
        "Tensor(shape: \(shape))"
    }
}

