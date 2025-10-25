# TinyBrain Architecture Overview

**Version:** 1.0  
**Last Updated:** 2025-10-25  
**Status:** Living Document

---

## 1. Introduction

TinyBrain is a Swift-native runtime for running large language models (LLMs) entirely on-device on iOS and macOS. This document provides a comprehensive architectural overview of the system, its components, and design rationale.

### Goals

1. **Educational Transparency**: Make transformer inference understandable
2. **Practical Performance**: Achieve real-time inference on Apple Silicon
3. **Native Integration**: Deep iOS/macOS ecosystem compatibility
4. **Hackability**: Enable researchers and developers to experiment

### Non-Goals

- Training or fine-tuning models
- Supporting models > 7B parameters
- Cloud-based inference
- Cross-platform (non-Apple) support

---

## 2. System Architecture

### 2.1 Layer Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Application Layer (SwiftUI)                │
│  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐ │
│  │ TinyBrain Chat │  │ Metrics View │  │ Model Picker│ │
│  └────────────────┘  └──────────────┘  └─────────────┘ │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                  Runtime Layer                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │              ModelRunner                         │  │
│  │  ┌────────────┬─────────────┬─────────────────┐ │  │
│  │  │ Tokenizer  │   Sampler   │   KV-Cache      │ │  │
│  │  └────────────┴─────────────┴─────────────────┘ │  │
│  │  ┌───────────────────────────────────────────┐  │  │
│  │  │      Transformer Pipeline                 │  │  │
│  │  │  • Embeddings                             │  │  │
│  │  │  • Self-Attention (× N layers)            │  │  │
│  │  │  • Feed-Forward Networks                  │  │  │
│  │  │  • Layer Normalization                    │  │  │
│  │  └───────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────┬───────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────┐
│                 Backend Layer                           │
│  ┌──────────────────┐       ┌──────────────────────┐   │
│  │  Metal Backend   │       │   CPU Fallback       │   │
│  │  • MatMul        │       │   (Accelerate)       │   │
│  │  • Softmax       │       │   • BLAS ops         │   │
│  │  • LayerNorm     │       │   • vDSP functions   │   │
│  │  • Quant/Dequant │       │   • Compatibility    │   │
│  └──────────────────┘       └──────────────────────┘   │
│         ┌────────────────────────┐                      │
│         │  Core ML (Optional)    │                      │
│         │  • ANE Offload         │                      │
│         └────────────────────────┘                      │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Component Responsibilities

#### Application Layer
- **SwiftUI Views**: User interface and interaction
- **View Models**: State management via `@ObservableObject`
- **Metrics Display**: Real-time performance visualization

#### Runtime Layer
- **ModelRunner**: Orchestrates the inference pipeline
- **Tokenizer**: Text ↔ token ID conversion
- **Sampler**: Probabilistic next-token selection
- **KV-Cache**: Manages attention key-value pairs
- **Model Loader**: Memory-mapped weight loading

#### Backend Layer
- **Metal**: GPU-accelerated tensor operations
- **Accelerate**: CPU fallback for compatibility
- **Core ML**: Optional ANE acceleration

---

## 3. Data Structures

### 3.1 Tensor

**Purpose**: Multi-dimensional array abstraction for numerical operations

**Design**: Value semantics with copy-on-write optimization

```swift
public struct Tensor {
    public let shape: TensorShape
    internal var data: [Float]  // Will use COW buffer in production
}
```

**Rationale**: 
- Value semantics prevent accidental mutations
- Copy-on-write avoids unnecessary copies
- Simple interface for educational clarity

### 3.2 TensorShape

**Purpose**: Shape validation and dimensionality tracking

```swift
public struct TensorShape: Equatable {
    public let dimensions: [Int]
    public var count: Int { dimensions.reduce(1, *) }
}
```

### 3.3 Model Format (.tbf)

**TinyBrain Binary Format** contains:

```
┌─────────────────────────────────────┐
│ Header                              │
│  • Magic bytes: "TBFM"              │
│  • Version: UInt32                  │
│  • Model type: String               │
│  • Config: JSON metadata            │
├─────────────────────────────────────┤
│ Quantization Metadata               │
│  • Precision: INT8/INT4/FP16        │
│  • Scales per channel/group         │
│  • Zero points                      │
├─────────────────────────────────────┤
│ Weights (memory-mapped)             │
│  • Embeddings                       │
│  • Attention weights (Q/K/V/O)      │
│  • MLP weights (up/down projections)│
│  • Layer norm parameters            │
└─────────────────────────────────────┘
```

**Design Decisions**:
- Memory-mapped to avoid loading full model into RAM
- Versioned format for backward compatibility
- Separate quantization metadata for flexibility

---

## 4. Inference Pipeline

### 4.1 Token Generation Flow

```
1. Tokenization
   Input: "Explain gravity"
   Output: [101, 2054, 2003, 2054, ...]
   
2. Embedding Lookup
   Input: Token IDs
   Output: Tensor(seq_len, d_model)
   
3. Transformer Blocks (× N)
   For each layer:
     a. Self-Attention
        • Query/Key/Value projections
        • Scaled dot-product attention
        • KV-Cache update
     b. Layer Normalization
     c. Feed-Forward Network
        • Up projection
        • Activation (GELU/SiLU)
        • Down projection
     d. Residual connection
   
4. Final Layer Norm + LM Head
   Output: Logits(vocab_size)
   
5. Sampling
   Input: Logits
   Apply: temperature, top-k, top-p
   Output: Next token ID
   
6. Decode
   Input: Token ID
   Output: String fragment
   
7. Stream to UI
   AsyncSequence emission
```

### 4.2 KV-Cache Management

**Problem**: Recomputing attention for all previous tokens is wasteful

**Solution**: Paged KV-cache

```swift
struct KVCache {
    var keys: Tensor    // (num_layers, max_seq, d_model)
    var values: Tensor  // (num_layers, max_seq, d_model)
    var currentLength: Int
    
    mutating func append(layer: Int, key: Tensor, value: Tensor)
    func get(layer: Int, upTo: Int) -> (Tensor, Tensor)
}
```

**Memory Efficiency**:
- Pre-allocate for max context length (e.g., 2048)
- Reuse buffer across generation steps
- Page out cold entries for long contexts

---

## 5. Metal Acceleration

### 5.1 Kernel Catalog

| Kernel | Purpose | Optimization |
|--------|---------|--------------|
| `matmul` | Matrix multiplication | Tiled, shared memory |
| `softmax` | Attention normalization | Numerically stable |
| `layer_norm` | Layer normalization | Fused mean/variance |
| `rope` | Rotary position embeddings | In-place rotation |
| `dequant_int8` | INT8 → FP16 | Vectorized |
| `dequant_int4` | INT4 → FP16 | Bitpacking |

### 5.2 MatMul Optimization

**Challenge**: Largest compute bottleneck (60-70% of runtime)

**Strategy**:
1. **Tiling**: Fit into threadgroup memory (32 KB)
2. **Shared Memory**: Reduce global memory bandwidth
3. **Simdgroup Ops**: Use Apple GPU SIMD primitives

**Pseudocode**:
```metal
kernel void matmul_tiled(
    device const float* A,
    device const float* B,
    device float* C,
    threadgroup float* sharedA,
    threadgroup float* sharedB,
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    // Load tile into shared memory
    // Compute partial dot products
    // Write result to C
}
```

**Performance**: Target 90% of theoretical peak FLOPS

---

## 6. Quantization

### 6.1 INT8 Quantization

**Per-Channel Symmetric**:

```
quantized = round(float_value / scale)
float_value = quantized * scale
```

**Benefits**:
- 4× memory reduction vs FP32
- 2× vs FP16
- Negligible accuracy loss (< 1% perplexity increase)

**Implementation**:
- Store scales per output channel
- Dequantize on-the-fly in kernels
- Fuse dequantization with compute

### 6.2 INT4 Quantization (Phase 2)

**Per-Group Quantization**:

```
Group size: 128 elements
Scales: FP16[num_groups]
Zero-points: INT4[num_groups]
```

**Benefits**:
- 8× memory reduction vs FP32
- Enables larger models on-device

**Challenges**:
- Higher perplexity delta (5-10%)
- More complex dequantization kernel

---

## 7. Streaming Output

### 7.1 AsyncSequence Design

**Interface**:
```swift
public struct TokenStream: AsyncSequence {
    public typealias Element = String
    
    public func makeAsyncIterator() -> AsyncIterator
}

for try await token in model.generateStream(prompt: "...") {
    print(token, terminator: "")
}
```

**Benefits**:
- Responsive UI updates
- Backpressure handling
- Swift Concurrency native

### 7.2 Implementation

**Token Buffer**:
```swift
actor TokenBuffer {
    private var continuation: AsyncStream<String>.Continuation?
    
    func emit(_ token: String) async {
        continuation?.yield(token)
    }
    
    func finish() {
        continuation?.finish()
    }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Tensor Operations**:
- Shape validation
- Basic arithmetic
- Broadcasting rules

**Quantization**:
- Round-trip accuracy
- Scale/zero-point correctness

**Tokenization**:
- BPE encoding/decoding
- Special token handling

### 8.2 Integration Tests

**End-to-End**:
- Load model → generate → validate output
- KV-cache consistency
- Memory usage profiling

**Platform Coverage**:
- macOS (M1/M2/M3/M4)
- iOS Simulator
- iOS Device (physical)

---

## 9. Performance Targets

| Metric | Target | Device |
|--------|--------|--------|
| Latency (first token) | ≤ 200 ms | iPhone 15 Pro |
| Latency (subsequent) | ≤ 150 ms | iPhone 15 Pro |
| Throughput | ≥ 6 tokens/sec | iPhone 15 Pro |
| Energy | ≤ 1.5 J/token | iPhone 15 Pro |
| Memory (INT8) | ≤ 1 GB | TinyLlama 1.1B |
| Perplexity Δ | ≤ 15% | vs FP16 reference |

---

## 10. Future Directions

### 10.1 Phase 2 Features

- **INT4 Quantization**: 8× memory savings
- **FlashAttention**: Fused attention kernel
- **ANE Integration**: Core ML hybrid mode
- **Speculative Decoding**: Parallel token generation

### 10.2 Research Opportunities

- **Energy-Performance Tradeoffs**: Pareto frontier analysis
- **Quantization-Aware Training**: Custom TinyBrain models
- **Hybrid Execution**: Dynamic Metal/ANE scheduling

---

## 11. References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) – Vaswani et al.
- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)
- [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135)
- [Apple Metal Programming Guide](https://developer.apple.com/metal/)

---

## 12. Contributing

See `AGENTS.md` for agent-specific rules and `docs/tasks/` for implementation roadmap.

---

**Maintained by**: Vivek Pattanaik  
**License**: MIT

