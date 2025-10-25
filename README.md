# 🧠 TinyBrain

**Swift-Native On-Device LLM Inference Kit**

[![Swift](https://img.shields.io/badge/Swift-5.10+-purple.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/platform-iOS%2017%2B%20%7C%20macOS%2014%2B-lightgrey.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI](https://github.com/vivekptnk/tinybrain/workflows/CI/badge.svg)](https://github.com/vivekptnk/tinybrain/actions)

TinyBrain is a **Swift-native runtime** for running large language models (LLMs) entirely on-device on iOS and macOS. It combines **educational transparency** with **practical performance**, making transformer inference hackable and efficient on Apple Silicon.

---

## ✨ Features

- 🚀 **Swift-First**: No C++ dependencies, pure Swift + Metal
- 🧮 **Quantization**: INT8 and INT4 support for efficient memory usage
- ⚡ **Metal Acceleration**: GPU-optimized kernels for Apple Silicon
- 🔄 **Streaming Output**: AsyncSequence-based token generation
- 🎓 **Educational**: Transparent, well-documented architecture
- 📱 **Native**: Deep iOS/macOS integration with SwiftUI demo
- 🔒 **Private**: 100% on-device inference, no network calls

---

## 🎯 Goals

TinyBrain serves two main purposes:

1. **Educational**: Teach developers how LLMs work at the tensor level
2. **Practical**: Enable real-time, private, offline inference on Apple devices

---

## 🚀 Quick Start

### Prerequisites

- macOS 14 Sonoma or later
- Xcode 16+
- Swift 5.10+
- Apple Silicon Mac (M1/M2/M3/M4) recommended

### Installation

```bash
git clone https://github.com/vivekp/tinybrain.git
cd tinybrain
make setup
```

This will:
- Install SwiftFormat and SwiftLint (via Homebrew if available)
- Resolve Swift Package dependencies
- Build the project
- Run initial tests

### Building

```bash
# Build all targets
make build

# Build in release mode
make build-release

# Run tests
make test

# Format code
make format

# Run linting
make lint

# Generate documentation
make docs
```

---

## 📦 Usage

### Basic Inference

```swift
import TinyBrain

// Load a quantized model
let model = try await TinyBrain.load("Models/tinyllama-int8.tbf")

// Generate text with streaming
let stream = try await model.generateStream(prompt: "Explain gravity in simple terms:")

for try await token in stream {
    print(token, terminator: "")
}
```

### SwiftUI Demo

A complete chat demo app is available in `Examples/ChatDemo/`:

```bash
open Examples/ChatDemo/ChatDemoApp.swift
# Run in Xcode to see live inference
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────┐
│   TinyBrain Chat (SwiftUI)      │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│      Runtime Layer              │
│  ┌──────────┬─────────────┐    │
│  │Tokenizer │   Sampler   │    │
│  └──────────┴─────────────┘    │
│  ┌─────────────────────────┐   │
│  │     ModelRunner         │   │
│  │  ┌──────────────────┐   │   │
│  │  │ Attention / MLP  │   │   │
│  │  │    KV-Cache      │   │   │
│  │  └──────────────────┘   │   │
│  └─────────────────────────┘   │
└────────────┬────────────────────┘
             │
┌────────────▼────────────────────┐
│      Backend Layer              │
│  ┌─────────┬──────────────┐    │
│  │  Metal  │  Core ML     │    │
│  │ Kernels │  (Optional)  │    │
│  └─────────┴──────────────┘    │
└─────────────────────────────────┘
```

### Modules

- **TinyBrainRuntime**: Core tensor operations and model runner
- **TinyBrainMetal**: GPU-accelerated kernels via Metal
- **TinyBrainTokenizer**: BPE/SentencePiece tokenization
- **TinyBrainDemo**: SwiftUI demo application
- **TinyBrainBench**: Performance benchmarking tools

---

## 📚 Documentation

Comprehensive documentation is available:

- **[docs/prd.md](docs/prd.md)**: Product Requirements Document
- **[docs/overview.md](docs/overview.md)**: Architecture overview
- **[docs/tasks/](docs/tasks/)**: Implementation roadmap
- **API Docs**: Generate with `make docs`

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run linting and formatting checks
make check
```

---

## 🛠️ Development

### Project Structure

```
tinybrain/
├── Sources/
│   ├── TinyBrainRuntime/     # Core runtime
│   ├── TinyBrainMetal/       # Metal backend
│   ├── TinyBrainTokenizer/   # Tokenization
│   ├── TinyBrainDemo/        # Demo app components
│   └── TinyBrainBench/       # Benchmark CLI
├── Tests/                     # Unit tests
├── Examples/
│   └── ChatDemo/             # SwiftUI demo app
├── Scripts/                   # Build and setup scripts
├── Models/                    # Model files (gitignored)
├── docs/                      # Documentation
└── Package.swift             # SPM manifest
```

### Coding Standards

- Swift 5.10+ with strict concurrency checking
- SwiftFormat for code formatting (`.swiftformat`)
- SwiftLint for static analysis (`.swiftlint.yml`)
- DocC-compatible documentation comments
- 100% test coverage target for critical paths

---

## 🗺️ Roadmap

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| **Phase 1: Scaffold** | ✅ Complete | Project structure, tooling, docs |
| **Phase 2: Runtime** | ✅ Complete | Tensor engine (Float32, Accelerate) |
| **Phase 3: Metal** | ✅ Complete | GPU MatMul kernel (3-5× speedup) |
| **Phase 4: Quant/KV** | Planned | INT8/INT4, paged KV-cache |
| **Phase 5: Tokenizer** | Planned | BPE, streaming output |
| **Phase 6: Demo** | Planned | SwiftUI chat app |
| **Phase 7: Benchmarks** | Planned | Performance suite, docs |

See [docs/tasks/](docs/tasks/) for detailed task breakdown.

---

## 🤝 Contributing

We welcome contributions! Please:

1. Read [AGENTS.md](AGENTS.md) for project rules
2. Check [docs/tasks/](docs/tasks/) for current work items
3. Follow the coding standards (run `make check` before PR)
4. Reference task IDs in commits (e.g., "feat: Implement tensor ops (TB-002)")

---

## 📊 Benchmarks

Current performance (Apple M4 Max, CPU only):

| Metric | Target | Current (TB-002) | Status |
|--------|--------|------------------|--------|
| MatMul 128×128 | < 0.1 ms | **0.053 ms** | ✅ **5× better** |
| Toy Model Throughput | Baseline | **1049 tokens/sec** | ✅ **Measured** |
| Test Coverage | Core ops | **26 tests passing** | ✅ **Complete** |

Full model performance targets (with Metal + INT8):

| Metric | Target | Status |
|--------|--------|--------|
| Latency | ≤ 150 ms/token | 🚧 TB-003 (Metal) |
| Throughput | ≥ 6 tokens/sec | 🚧 TB-003/TB-004 |
| Memory | ≤ 1 GB RAM | 🚧 TB-004 (INT8) |
| Energy | ≤ 1.5 J/token | 🚧 TB-003 (Metal) |
| Accuracy | ≤ 15% perplexity Δ | 🚧 TB-004 (Quant) |

Run benchmarks: `swift Scripts/benchmark-ops.swift`

---

## 🔗 Related Projects

- [MLC-LLM](https://github.com/mlc-ai/mlc-llm): C++/TVM-based mobile LLM runtime
- [llama.cpp](https://github.com/ggerganov/llama.cpp): Pure C/C++ LLM inference
- [Core ML Tools](https://github.com/apple/coremltools): Apple's ML conversion toolkit

**TinyBrain differentiator**: Swift-native, educational, hybrid Metal/Core ML.

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 👨‍💻 Author

**Vivek Pattanaik**  
GitHub: [@vivekptnk](https://github.com/vivekptnk)

---

## 🙏 Acknowledgments

- Inspired by [micrograd](https://github.com/karpathy/micrograd) for educational clarity
- Built on Apple's Metal, Core ML, and Accelerate frameworks
- Community contributions and feedback

---

## 📮 Contact

- Issues: [GitHub Issues](https://github.com/vivekptnk/tinybrain/issues)
- Discussions: [GitHub Discussions](https://github.com/vivekptnk/tinybrain/discussions)