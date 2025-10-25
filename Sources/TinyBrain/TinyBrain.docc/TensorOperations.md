# Tensor Operations

Learn how to perform mathematical operations on tensors in TinyBrain.

## Overview

TinyBrain provides a complete set of tensor operations optimized for transformer inference using Apple's Accelerate framework. All operations are implemented with both **educational clarity** and **production performance**.

##  Operations Summary

| Category | Operations | Performance |
|----------|------------|-------------|
| **Matrix Ops** | matmul | 10-100× faster than manual |
| **Element-Wise** | +, *, scalar ops | 10-20× faster |
| **Activations** | GELU, ReLU | Optimized |
| **Normalization** | Softmax, LayerNorm | Numerically stable |

---

## Matrix Multiplication

The most critical operation in LLMs (70% of compute time).

### Usage

```swift
let a = Tensor(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
let b = Tensor(shape: TensorShape(3, 2), data: [7,8,9,10,11,12])

let c = a.matmul(b)  // [2, 2]
```

### Where It's Used

- **Attention**: Q×Kᵀ computes attention scores
- **Attention output**: (weights) × V
- **MLP layers**: Two matmuls per transformer layer
- **Output projection**: Generate final logits

### Performance

- 128×128: ~0.04ms (M4 Max)
- 512×512: ~2ms
- 2048×2048: ~100ms

**Implementation:** Uses `cblas_sgemm` from Accelerate

---

## Element-Wise Addition

Add tensors element-by-element.

### Usage

```swift
let a = Tensor(shape: TensorShape(2, 3), data: [1,2,3,4,5,6])
let b = Tensor(shape: TensorShape(2, 3), data: [10,20,30,40,50,60])

let c = a + b  // [11,22,33,44,55,66]

// Scalar addition
let d = a + 5.0  // [6,7,8,9,10,11]
```

### Where It's Used

**Residual Connections** (critical for deep networks):
```swift
output = x + attention(x)
output = x + mlp(x)
```

These "skip connections" allow gradients to flow through 100+ layers!

**Implementation:** Uses `vDSP_vadd` from Accelerate

---

## Element-Wise Multiplication

Multiply tensors element-by-element.

### Usage

```swift
let a = Tensor(shape: TensorShape(3), data: [2,3,4])
let b = Tensor(shape: TensorShape(3), data: [10,10,10])

let c = a * b  // [20,30,40]

// Scalar multiplication  
let d = a * 2.0  // [4,6,8]
```

### Where It's Used

**Attention Masking:**
```swift
masked_scores = attention_scores * mask
```

**Scaling:**
```swift
scaled_scores = scores * (1.0 / sqrt(d_k))
```

**Implementation:** Uses `vDSP_vmul` from Accelerate

---

## GELU Activation

Smooth activation function used in GPT, BERT.

### Usage

```swift
let x = Tensor(shape: TensorShape(5), data: [-2,-1,0,1,2])
let activated = x.gelu()
// Approximately: [-0.05, -0.16, 0, 0.84, 1.95]
```

### Properties

- Smooth (differentiable everywhere)
- Allows small negative values
- Better than ReLU for large models

### Where It's Used

**Feed-Forward Networks:**
```swift
FFN(x) = GELU(x × W₁ + b₁) × W₂ + b₂
```

Applied after the first linear layer in every MLP block.

---

## ReLU Activation

Simple thresholding: max(0, x)

### Usage

```swift
let x = Tensor(shape: TensorShape(4), data: [-2,-1,1,2])
let activated = x.relu()
// [0, 0, 1, 2]
```

### Properties

- Very fast
- Sharp cutoff at 0
- Classic activation function

**Implementation:** Uses `vDSP_vthres` from Accelerate

---

## Softmax Normalization

Convert numbers to probabilities.

### Usage

```swift
let logits = Tensor(shape: TensorShape(3), data: [1.0, 2.0, 3.0])
let probs = logits.softmax()
// [0.09, 0.24, 0.67] ← Sums to 1.0!
```

### Properties

- All outputs between 0 and 1
- Sums to exactly 1.0
- Larger inputs → higher probabilities

### Where It's Used

**Attention Mechanism:**
```swift
scores = Q × Kᵀ / sqrt(d_k)
attention_weights = softmax(scores)  // ← HERE!
output = attention_weights × V
```

**Token Sampling:**
```swift
logits = model(prompt)
probs = softmax(logits)  // ← HERE!
next_token = sample(probs)
```

### Numerical Stability

Subtracts max before exp to prevent overflow:
```swift
exp(1000) = ∞  // Overflow!
exp(1000 - 1000) = 1  // Safe!
```

---

## LayerNorm Normalization

Normalize to mean=0, variance=1.

### Usage

```swift
let x = Tensor(shape: TensorShape(100), data: ...)
let normalized = x.layerNorm()
// Output has mean≈0, variance≈1
```

### Where It's Used

**Every Transformer Layer (twice!):**
```swift
// After attention
x = layerNorm(x + attention(x))

// After MLP
x = layerNorm(x + mlp(x))
```

### Why It's Critical

- Prevents exploding activations
- Stabilizes deep networks
- Makes training possible

Without LayerNorm, transformers don't train!

---

## Performance Summary

All operations use Apple's Accelerate framework for optimal performance:

| Operation | Framework | Speedup vs Manual |
|-----------|-----------|-------------------|
| MatMul | BLAS (`cblas_sgemm`) | 100× |
| Add/Mul | vDSP (`vDSP_vadd/vmul`) | 20× |
| Softmax | vDSP (max, exp, sum, div) | 10× |
| LayerNorm | vDSP (mean, variance, div) | 10× |
| GELU | Custom (tanh approximation) | N/A |
| ReLU | vDSP (`vDSP_vthres`) | 5× |

**Total speedup for transformer inference: ~50-100× faster than naive Python!**

---

## Next Steps

- See ``Tensor`` for API reference
- Read `docs/BLAS-vDSP-Tutorial.md` for implementation details
- Check `TensorTests.swift` for usage examples
- Run benchmarks: `make bench`

