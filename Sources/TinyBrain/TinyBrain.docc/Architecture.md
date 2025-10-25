# Architecture

Understanding TinyBrain's design and components.

## Overview

TinyBrain is architected as a layered system, separating concerns between the UI, runtime logic, and hardware acceleration. This design enables both educational clarity and practical performance.

## System Layers

### Application Layer

The SwiftUI-based demo apps and user-facing APIs:

- **TinyBrain Chat**: Reference SwiftUI chat application
- **Public API**: Simple `load()` and `generateStream()` interface
- **View Models**: Reactive state management with Combine

### Runtime Layer

Core inference logic and orchestration:

- **ModelRunner**: Coordinates the inference pipeline
- **Tokenizer**: Text ↔ token ID conversion (BPE/SentencePiece)
- **Sampler**: Next-token selection (top-k, top-p, temperature)
- **KV-Cache**: Paged memory management for attention states
- **Streamer**: AsyncSequence-based token generation

### Backend Layer

Hardware acceleration and numerical operations:

- **Metal Kernels**: GPU-accelerated MatMul, Softmax, LayerNorm
- **Quantization**: INT8/INT4 weight compression and dequantization
- **CPU Fallback**: Accelerate framework for non-GPU execution
- **Core ML (Optional)**: ANE offload for specific operations

## Data Flow

```
User Prompt
    ↓
Tokenizer (encode)
    ↓
Token IDs → [101, 2054, 2003, ...]
    ↓
Model Runner
    ├─→ Embedding Lookup
    ├─→ Transformer Blocks (× N)
    │   ├─→ Self-Attention (Metal)
    │   │   └─→ KV-Cache Update
    │   ├─→ Layer Norm
    │   └─→ MLP (Metal)
    └─→ Output Logits
        ↓
Sampler (top-k/top-p)
    ↓
Next Token ID
    ↓
Tokenizer (decode)
    ↓
Stream → AsyncSequence<String>
    ↓
SwiftUI View Update
```

## Module Boundaries

### TinyBrainRuntime

Core types and protocols:

- `Tensor`: Multi-dimensional array abstraction
- `TensorShape`: Shape validation and utilities
- `ModelConfig`: Architecture parameters
- `QuantizationMetadata`: Precision information

**Dependencies**: None (pure Swift)

### TinyBrainMetal

GPU acceleration:

- `MetalBackend`: Device and queue management
- `MatMulKernel`: Optimized matrix multiplication
- `AttentionKernel`: Fused attention operations
- `QuantKernel`: INT8/INT4 dequantization

**Dependencies**: `TinyBrainRuntime`, Metal framework

### TinyBrainTokenizer

Text processing:

- `Tokenizer` protocol
- `BPETokenizer`: Byte-pair encoding
- `SentencePieceTokenizer`: SentencePiece support

**Dependencies**: `TinyBrainRuntime`

### TinyBrainDemo

UI components:

- `ChatViewModel`: State management
- `MetricsOverlay`: Performance display
- `ModelPicker`: Model selection UI

**Dependencies**: All TinyBrain modules, SwiftUI

## Design Principles

### Value Semantics

Tensor operations use value semantics where practical:

```swift
var a = Tensor.zeros(shape: TensorShape(2, 3))
var b = a  // Copy-on-write
b.data[0] = 1.0  // 'a' remains unchanged
```

### Protocol-Oriented

Swappable implementations for backends and tokenizers:

```swift
protocol Tokenizer {
    func encode(_ text: String) -> [Int]
    func decode(_ tokens: [Int]) -> String
}

// BPE, SentencePiece, or custom implementations
```

### Swift Concurrency

Modern async/await throughout:

```swift
let model = try await TinyBrain.load("model.tbf")
for try await token in model.generateStream(prompt: "...") {
    // Process tokens as they're generated
}
```

### CPU Fallbacks

Every Metal operation has an Accelerate-based CPU fallback for testing and compatibility.

## Performance Considerations

- **Memory-mapped weights**: Avoid loading entire model into RAM
- **Paged KV-cache**: Reuse memory across generation steps
- **Fused kernels**: Combine operations to reduce bandwidth
- **Threadgroup tuning**: Optimize Metal dispatch sizes per-device

## Next Steps

- Read the <doc:GettingStarted> guide
- Explore the ``TinyBrain`` API
- See implementation tasks in `docs/tasks/`

