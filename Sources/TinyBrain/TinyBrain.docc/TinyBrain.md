# ``TinyBrain``

Swift-Native On-Device LLM Inference Kit

## Overview

TinyBrain is a Swift-native runtime for running large language models (LLMs) entirely on-device on iOS and macOS. It provides transparent, educational access to transformer inference while maintaining practical performance on Apple Silicon.

### Key Features

- **Swift-First Architecture**: Pure Swift + Metal, no C++ dependencies
- **Educational Transparency**: Well-documented, hackable implementation
- **Quantization Support**: INT8 and INT4 for memory efficiency
- **Metal Acceleration**: GPU-optimized kernels for Apple Silicon
- **Streaming Output**: AsyncSequence-based token generation
- **Native Integration**: Deep iOS/macOS SwiftUI support

## Topics

### Getting Started

- <doc:GettingStarted>
- <doc:Architecture>
- <doc:TensorOperations>
- ``TinyBrain/TinyBrain``

### Core Runtime

- ``TinyBrain/ModelRunner``
- ``TinyBrain/Tensor``
- ``TinyBrain/TensorShape``

### Tensor Operations

- ``TinyBrain/Tensor/matmul(_:)``
- ``TinyBrain/Tensor/gelu()``
- ``TinyBrain/Tensor/relu()``
- ``TinyBrain/Tensor/softmax(epsilon:)``
- ``TinyBrain/Tensor/layerNorm(epsilon:)``

### Backends

- ``TinyBrainMetal/MetalBackend``
- ``TinyBrainMetal/MetalError``

### Tokenization

- ``TinyBrainTokenizer/Tokenizer``
- ``TinyBrainTokenizer/BPETokenizer``

### Errors

- ``TinyBrain/ModelError``

