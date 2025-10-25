# BLAS & vDSP Tutorial for TinyBrain

**A Complete Guide to Apple's Accelerate Framework**

This document explains how to use BLAS (Basic Linear Algebra Subprograms) and vDSP (Digital Signal Processing) to implement fast tensor operations in Swift.

---

## 📚 Table of Contents

1. What is Accelerate?
2. Matrix Memory Layout (Row-Major vs Column-Major)
3. BLAS: Matrix Multiplication
4. vDSP: Vector Operations
5. Practical Examples for LLMs
6. Common Pitfalls & Debugging

---

## 1. What is Accelerate?

**Accelerate** is Apple's collection of highly optimized math libraries. It uses:
- **SIMD** instructions (process 4-16 numbers at once)
- **Multi-threading** (automatically parallelize)
- **Cache optimization** (smart memory access patterns)

**Result:** 10-100× faster than manual loops!

### Components We'll Use:

```
Accelerate
├── BLAS       → Matrix operations (matmul)
│   └── cblas_sgemm, cblas_dgemm
├── vDSP       → Vector operations (add, mul, exp)
│   └── vDSP_vadd, vDSP_vmul, vvexpf
└── BNNS       → Neural network ops (we'll skip this)
```

---

## 2. Matrix Memory Layout 🧠 CRITICAL TO UNDERSTAND

This is where most bugs happen! Pay close attention.

### The Concept

A 2D matrix `[[1,2,3], [4,5,6]]` can be stored in memory two ways:

#### Row-Major (Swift/NumPy/PyTorch default)

```
Matrix:
[1  2  3]
[4  5  6]

Memory: [1, 2, 3, 4, 5, 6]
         ↑        ↑
       Row 0    Row 1
```

**Rule:** Store rows consecutively.

**Index calculation:**
```
Element [row, col] is at: data[row * numCols + col]
```

#### Column-Major (Fortran/BLAS default)

```
Matrix:
[1  2  3]
[4  5  6]

Memory: [1, 4, 2, 5, 3, 6]
         ↑     ↑     ↑
       Col 0 Col 1 Col 2
```

**Rule:** Store columns consecutively.

**Index calculation:**
```
Element [row, col] is at: data[col * numRows + row]
```

### Why This Matters for BLAS

**BLAS was written in Fortran** (1970s) → **column-major by default**

**Swift/Python use row-major** → **we need to handle this!**

**Solutions:**
1. **Convert to column-major** (slow, requires copy)
2. **Use transpose flag** (smart!)
3. **Swap dimensions** (confusing but works)

---

## 3. BLAS: Matrix Multiplication 🎯

### The Function: cblas_sgemm

**"s" = Single precision (Float32)**  
**"gemm" = GEneral Matrix Multiply**

```c
cblas_sgemm(
    CblasRowMajor,         // Our memory layout
    CblasNoTrans,          // Don't transpose A
    CblasNoTrans,          // Don't transpose B
    M, N, K,               // Dimensions
    alpha,                 // Scaling factor
    A, lda,                // Matrix A
    B, ldb,                // Matrix B
    beta,                  // Scaling factor
    C, ldc                 // Result matrix C
)
```

### What It Does

Computes: **C = alpha × (A × B) + beta × C**

Most common case: **C = A × B**
- alpha = 1.0
- beta = 0.0 (ignore old C values)

### Parameters Explained

**Matrix dimensions:**
```
A: [M, K]
B: [K, N]
C: [M, N]  ← Result

M = rows in A
K = cols in A = rows in B
N = cols in B
```

**Leading dimension (ld):**
> "How many elements until the next row?"

For row-major contiguous arrays:
```
lda = K  (number of columns in A)
ldb = N  (number of columns in B)
ldc = N  (number of columns in C)
```

### Example: Multiply 2×3 by 3×2

```swift
import Accelerate

// A: [2, 3]
let A: [Float] = [1, 2, 3,  // Row 0
                  4, 5, 6]  // Row 1

// B: [3, 2]
let B: [Float] = [7, 8,   // Row 0
                  9, 10,  // Row 1
                  11, 12] // Row 2

// C: [2, 2] - result
var C = [Float](repeating: 0, count: 4)

let M = Int32(2)  // Rows in A
let K = Int32(3)  // Cols in A, Rows in B
let N = Int32(2)  // Cols in B

cblas_sgemm(
    CblasRowMajor,     // Row-major layout
    CblasNoTrans,      // A as-is
    CblasNoTrans,      // B as-is
    M, N, K,           // Dimensions
    1.0,               // alpha = 1
    A, K,              // A, leading dim = 3
    B, N,              // B, leading dim = 2
    0.0,               // beta = 0
    &C, N              // C, leading dim = 2
)

// Result C:
// [58, 64,    ← [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12]
//  139, 154]  ← [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]
```

### Manual Verification

```
C[0,0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58 ✅
C[0,1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64 ✅
C[1,0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139 ✅
C[1,1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154 ✅
```

---

## 4. vDSP: Vector Operations ⚡

vDSP works on **1D arrays** (vectors). Much simpler than BLAS!

### Addition: vDSP_vadd

**Add two vectors element-wise:**

```swift
import Accelerate

let A: [Float] = [1, 2, 3, 4]
let B: [Float] = [10, 20, 30, 40]
var C = [Float](repeating: 0, count: 4)

vDSP_vadd(
    A, 1,              // Input A, stride 1
    B, 1,              // Input B, stride 1
    &C, 1,             // Output C, stride 1
    vDSP_Length(4)     // Number of elements
)

// C = [11, 22, 33, 44]
```

**What's "stride"?**
- Stride = how many elements to skip between values
- Stride 1 = process every element
- Stride 2 = process every other element

**Example with stride 2:**
```swift
let A: [Float] = [1, 2, 3, 4, 5, 6]
//                 ↑     ↑     ↑
//               Use these (stride 2)

vDSP_vadd(A, 2, B, 2, &C, 2, vDSP_Length(3))
// Processes: A[0], A[2], A[4]
```

### Multiplication: vDSP_vmul

```swift
let A: [Float] = [2, 3, 4, 5]
let B: [Float] = [10, 10, 10, 10]
var C = [Float](repeating: 0, count: 4)

vDSP_vmul(A, 1, B, 1, &C, 1, vDSP_Length(4))

// C = [20, 30, 40, 50]
```

### Scalar Operations: vDSP_vsmul

**Multiply vector by scalar:**

```swift
let A: [Float] = [1, 2, 3, 4]
var scalar: Float = 10.0
var C = [Float](repeating: 0, count: 4)

vDSP_vsmul(
    A, 1,              // Input
    &scalar,           // Scalar value (must be var!)
    &C, 1,             // Output
    vDSP_Length(4)
)

// C = [10, 20, 30, 40]
```

### Exponential: vvexpf

**For softmax: exp(x)**

```swift
import Accelerate

var input: [Float] = [0, 1, 2, 3]
var output = [Float](repeating: 0, count: 4)
var count = Int32(4)

vvexpf(&output, input, &count)

// output = [e^0, e^1, e^2, e^3]
//        = [1.0, 2.718, 7.389, 20.086]
```

**Note:** `vvexpf` requires passing count as pointer!

---

## 5. Practical Examples for LLMs 🧠

### Example 1: Softmax (Real LLM Operation)

**What is softmax?**
Converts numbers into probabilities that sum to 1.

```
Input:  [2.0, 1.0, 0.1]
Output: [0.659, 0.242, 0.099]  ← These sum to 1.0
```

**Formula:**
```
softmax(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

**Problem:** `exp(large_number)` overflows!

**Solution:** Subtract max first (numerically stable)

```swift
import Accelerate

func softmax(_ input: [Float]) -> [Float] {
    let n = input.count
    
    // 1. Find max (for numerical stability)
    var maxVal: Float = 0
    vDSP_maxv(input, 1, &maxVal, vDSP_Length(n))
    
    // 2. Subtract max from all elements
    var shifted = [Float](repeating: 0, count: n)
    var negMax = -maxVal
    vDSP_vsadd(input, 1, &negMax, &shifted, 1, vDSP_Length(n))
    
    // 3. Compute exp
    var expVals = [Float](repeating: 0, count: n)
    var count = Int32(n)
    vvexpf(&expVals, shifted, &count)
    
    // 4. Sum all exp values
    var sum: Float = 0
    vDSP_sve(expVals, 1, &sum, vDSP_Length(n))
    
    // 5. Divide by sum
    var result = [Float](repeating: 0, count: n)
    vDSP_vsdiv(expVals, 1, &sum, &result, 1, vDSP_Length(n))
    
    return result
}

// Test:
let logits: [Float] = [2.0, 1.0, 0.1]
let probs = softmax(logits)
// [0.659, 0.242, 0.099]
```

**Why this is stable:**
```
Original: exp(1000) / (exp(1000) + exp(999) + ...)
          ↓ OVERFLOW! ∞

Shifted:  exp(0) / (exp(0) + exp(-1) + ...)
          ↓ No overflow! Works!
```

### Example 2: GELU Activation

**What is GELU?**
Smooth activation function used in GPT, BERT, etc.

```
GELU(x) ≈ 0.5 × x × (1 + tanh(√(2/π) × (x + 0.044715 × x³)))
```

**Implementation:**

```swift
func gelu(_ input: [Float]) -> [Float] {
    var output = [Float](repeating: 0, count: input.count)
    
    let sqrt2OverPi = sqrt(2.0 / Float.pi)
    
    for i in 0..<input.count {
        let x = input[i]
        let x3 = x * x * x
        let inner = sqrt2OverPi * (x + 0.044715 * x3)
        let tanhVal = tanh(inner)
        output[i] = 0.5 * x * (1.0 + tanhVal)
    }
    
    return output
}
```

**Can we use vDSP here?**
- Partially! vDSP has `vvtanhf` for tanh
- But: Polynomial part is easier manually
- Tradeoff: Readability vs speed

### Example 3: LayerNorm

**What is LayerNorm?**
Normalizes activations to have mean=0, variance=1.

```
LayerNorm(x) = (x - mean(x)) / sqrt(variance(x) + ε)
```

**Implementation with vDSP:**

```swift
func layerNorm(_ input: [Float], epsilon: Float = 1e-5) -> [Float] {
    let n = input.count
    
    // 1. Compute mean
    var mean: Float = 0
    vDSP_meanv(input, 1, &mean, vDSP_Length(n))
    
    // 2. Subtract mean (x - mean)
    var centered = [Float](repeating: 0, count: n)
    var negMean = -mean
    vDSP_vsadd(input, 1, &negMean, &centered, 1, vDSP_Length(n))
    
    // 3. Compute variance
    // variance = mean((x - mean)²)
    var squared = [Float](repeating: 0, count: n)
    vDSP_vsq(centered, 1, &squared, 1, vDSP_Length(n))
    
    var variance: Float = 0
    vDSP_meanv(squared, 1, &variance, vDSP_Length(n))
    
    // 4. Compute std = sqrt(variance + epsilon)
    let std = sqrt(variance + epsilon)
    
    // 5. Divide by std
    var result = [Float](repeating: 0, count: n)
    vDSP_vsdiv(centered, 1, &std, &result, 1, vDSP_Length(n))
    
    return result
}
```

---

## 6. Common Parameters Explained

### vDSP Functions Pattern

Most vDSP functions follow this pattern:

```swift
vDSP_<operation>(
    inputA, strideA,       // First input + stride
    inputB, strideB,       // Second input + stride (if binary op)
    &output, strideOutput, // Output + stride
    vDSP_Length(count)     // Number of elements to process
)
```

### BLAS Functions Pattern

```c
cblas_<precision><type><operation>(
    layout,           // CblasRowMajor or CblasColMajor
    ...operation-specific params...
)
```

**Precision:**
- `s` = Single (Float32)
- `d` = Double (Float64)
- `c` = Complex single
- `z` = Complex double

**Type:**
- `ge` = General (full matrix)
- `sy` = Symmetric
- `tr` = Triangular

**Operation:**
- `mm` = Matrix-Matrix multiply
- `mv` = Matrix-Vector multiply
- `axpy` = a×x + y

---

## 7. Debugging Tips 🐛

### Problem 1: Wrong Results

**Checklist:**
- ✅ Is `lda`, `ldb`, `ldc` correct? (= number of columns for row-major)
- ✅ Are dimensions in right order? (M, N, K not M, K, N)
- ✅ Is alpha = 1.0 and beta = 0.0? (unless you want scaling)
- ✅ Is array contiguous in memory?

### Problem 2: Crashes

**Common causes:**
- Buffer too small (check `count` parameter)
- Stride too large (accessing beyond array)
- Passing wrong type (Float vs Double)
- Forgetting `&` for output parameters

### Problem 3: Performance Not Improving

**Check:**
- Are you calling vDSP on tiny arrays? (< 100 elements → overhead dominates)
- Is array actually contiguous? (stride = 1)
- Are you in Debug build? (use Release: `-c release`)

---

## 8. Reference Chart

### Common vDSP Operations

| Operation | Function | Example |
|-----------|----------|---------|
| Add | `vDSP_vadd(A, 1, B, 1, &C, 1, n)` | C = A + B |
| Subtract | `vDSP_vsub(A, 1, B, 1, &C, 1, n)` | C = A - B |
| Multiply | `vDSP_vmul(A, 1, B, 1, &C, 1, n)` | C = A × B (element-wise) |
| Divide | `vDSP_vdiv(A, 1, B, 1, &C, 1, n)` | C = A / B |
| Scalar add | `vDSP_vsadd(A, 1, &s, &C, 1, n)` | C = A + s |
| Scalar mul | `vDSP_vsmul(A, 1, &s, &C, 1, n)` | C = A × s |
| Square | `vDSP_vsq(A, 1, &C, 1, n)` | C = A² |
| Sqrt | `vvsqrtf(&C, A, &n)` | C = √A |
| Exp | `vvexpf(&C, A, &n)` | C = exp(A) |
| Log | `vvlogf(&C, A, &n)` | C = log(A) |
| Max | `vDSP_maxv(A, 1, &max, n)` | Find maximum |
| Sum | `vDSP_sve(A, 1, &sum, n)` | Sum all elements |
| Mean | `vDSP_meanv(A, 1, &mean, n)` | Average |

### Common BLAS Operations

| Operation | Function | Dimensions |
|-----------|----------|------------|
| Matrix × Matrix | `cblas_sgemm` | [M,K] × [K,N] → [M,N] |
| Matrix × Vector | `cblas_sgemv` | [M,N] × [N] → [M] |
| Vector dot | `cblas_sdot` | [N] · [N] → scalar |

---

## 9. TinyBrain Tensor Wrapper

**Goal:** Hide the complexity, expose clean API

**User writes:**
```swift
let c = a.matmul(b)
```

**We implement:**
```swift
extension Tensor {
    func matmul(_ other: Tensor) -> Tensor {
        // Validate shapes
        precondition(self.shape.dimensions.count == 2)
        precondition(other.shape.dimensions.count == 2)
        precondition(self.shape[1] == other.shape[0])
        
        let m = Int32(self.shape[0])
        let k = Int32(self.shape[1])
        let n = Int32(other.shape[1])
        
        var result = Tensor.zeros(shape: TensorShape(Int(m), Int(n)))
        
        // Call BLAS
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            m, n, k,
            1.0,
            self.data, k,
            other.data, n,
            0.0,
            &result.data, n
        )
        
        return result
    }
}
```

**User doesn't see BLAS complexity!** ✨

---

## 10. Testing Strategy

### Test 1: Basic Correctness

```swift
func testMatMulBasic() {
    // A: [2, 3] = [[1,2,3], [4,5,6]]
    let a = Tensor(shape: TensorShape(2, 3), 
                   data: [1,2,3,4,5,6])
    
    // B: [3, 2] = [[7,8], [9,10], [11,12]]
    let b = Tensor(shape: TensorShape(3, 2), 
                   data: [7,8,9,10,11,12])
    
    let c = a.matmul(b)
    
    // Expected: [[58,64], [139,154]]
    XCTAssertEqual(c.shape, TensorShape(2, 2))
    XCTAssertEqual(c[0,0], 58, accuracy: 1e-5)
    XCTAssertEqual(c[0,1], 64, accuracy: 1e-5)
    XCTAssertEqual(c[1,0], 139, accuracy: 1e-5)
    XCTAssertEqual(c[1,1], 154, accuracy: 1e-5)
}
```

### Test 2: Edge Cases

```swift
func testMatMulMismatchedShapes() {
    let a = Tensor(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
    let b = Tensor(shape: TensorShape(2, 2), data: [7,8,9,10])
    
    // Should crash with precondition failure
    // (In real code, we'd throw an error instead)
    expectPreconditionFailure {
        _ = a.matmul(b)
    }
}
```

### Test 3: Numerical Accuracy

```swift
func testSoftmaxSumsToOne() {
    let input = Tensor(shape: TensorShape(5), 
                       data: [1.0, 2.0, 3.0, 4.0, 5.0])
    let output = input.softmax()
    
    let sum = output.data.reduce(0, +)
    XCTAssertEqual(sum, 1.0, accuracy: 1e-5)
}
```

---

## 11. Performance Expectations

### MatMul Performance (M4 Max)

| Size | Manual Loop | Accelerate | Speedup |
|------|-------------|------------|---------|
| 100×100 | 50 ms | 0.5 ms | **100×** |
| 1000×1000 | 5000 ms | 50 ms | **100×** |
| 2000×2000 | 40000 ms | 200 ms | **200×** |

### vDSP Performance

| Operation | Array Size | Manual | vDSP | Speedup |
|-----------|-----------|---------|------|---------|
| Add | 1M elements | 5 ms | 0.3 ms | **16×** |
| Mul | 1M elements | 5 ms | 0.3 ms | **16×** |
| Softmax | 50K elements | 20 ms | 2 ms | **10×** |

---

## 12. Quick Reference

**Import:**
```swift
import Accelerate
```

**Matrix multiply:**
```swift
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, 1.0, A, K, B, N, 0.0, &C, N)
```

**Vector add:**
```swift
vDSP_vadd(A, 1, B, 1, &C, 1, vDSP_Length(count))
```

**Softmax:**
```swift
1. Find max: vDSP_maxv
2. Subtract max: vDSP_vsadd  
3. Exp: vvexpf
4. Sum: vDSP_sve
5. Divide: vDSP_vsdiv
```

---

## ❓ Questions to Test Your Understanding

1. **Why do we subtract max before softmax?**
   <details>
   <summary>Answer</summary>
   To prevent overflow. exp(1000) = ∞, but exp(0) = 1.
   </details>

2. **What does "leading dimension" mean?**
   <details>
   <summary>Answer</summary>
   Number of elements between consecutive rows. For row-major, it's the number of columns.
   </details>

3. **Why pass `&` for output in vDSP?**
   <details>
   <summary>Answer</summary>
   vDSP is C API - needs pointer to write results. Swift's `&` creates an UnsafeMutablePointer.
   </details>

4. **What's the difference between `vDSP_vmul` and `cblas_sgemm`?**
   <details>
   <summary>Answer</summary>
   `vDSP_vmul` is element-wise multiply. `cblas_sgemm` is matrix multiplication (dot products).
   </details>

---

**Ready to start implementing TB-002?** Now you understand the tools we'll use! 🎓

