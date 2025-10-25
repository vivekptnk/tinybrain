# 🧠 **TinyBrain – Swift-Native On-Device LLM Inference Kit**

**Product Requirements Document (PRD v2)**
**Author:** Vivek Pattanaik
**Date:** 2025-10-24
**Version:** 2.0
**Status:** Active Draft

---

## 1. 📜 Overview

**TinyBrain** is a **Swift-native runtime** for running large language models (LLMs) *entirely on-device* on iOS and macOS.
It aims to make transformer inference transparent, hackable, and efficient—bridging **machine learning** and **Apple-native engineering**.

TinyBrain serves two main goals:

1. **Educational:** teach developers and students how LLMs “think” at the tensor level.
2. **Practical:** enable real-time, private, offline inference for small to mid-size models on Apple Silicon.

---

## 2. 🌍 Competitive & Market Context

On-device LLMs are becoming central to AI adoption—offering privacy, low latency, and cost savings.
Current solutions (e.g., **MLC-LLM**, **llama.cpp**, **Core ML Tools**) either rely on heavy C++/compiler stacks or lack pedagogical clarity.

**TinyBrain fills this gap** by providing:

* A **Swift-first** framework (no C++)
* Educational readability akin to *micrograd*
* Modular architecture mixing **Metal** and **Core ML** backends
* Built-in performance benchmarking & visualization

**Target users**

* iOS/macOS engineers entering ML
* Startups building offline chat or agents
* Educators teaching transformer fundamentals

---

## 3. 🎯 Goals & Non-Goals

| Type             | Details                                                                                                                                                                      |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅ **Goals**      | Swift-native tensor runtime; quantization (INT8/INT4); paged KV-cache; streaming token generation; hybrid Metal/Core ML backend; SwiftUI demo; open-source educational docs. |
| 🚫 **Non-Goals** | Training/fine-tuning; models > 7 B parameters; cloud inference; closed-source SDK.                                                                                           |

---

## 4. 🧩 Key Features

| Feature             | Description                               | Priority |
| :------------------ | :---------------------------------------- | :------- |
| Swift Tensor Engine | Core tensor struct + ops                  | P0       |
| Metal Kernels       | MatMul, Softmax, LayerNorm, Quant/Dequant | P0       |
| Quantization        | INT8 (v1), INT4 (v2)                      | P0       |
| KV-Cache            | Paged, reusable memory buffer             | P1       |
| Streaming Output    | AsyncSequence / Combine                   | P1       |
| Tokenizer           | BPE / SentencePiece in Swift              | P0       |
| Sampler             | top-k / top-p / temperature               | P0       |
| Core ML Hybrid      | Optional ANE offload                      | P2       |
| Benchmark Harness   | Latency, tokens/sec, energy               | P1       |
| SwiftUI Demo        | “TinyBrain Chat” with live stream         | P0       |
| Converter Tool      | PyTorch → TinyBrain weights               | P2       |
| Docs                | Literate code + Markdown site             | P0       |

---

## 5. ⚙️ Technical Specifications

| Area         | Spec                                       |
| :----------- | :----------------------------------------- |
| Language     | Swift 5.10 + Metal 1.3                     |
| Devices      | iPhone 13+, iPad M1+, Mac M1–M4            |
| Frameworks   | Core ML, Metal, Accelerate                 |
| Quantization | per-channel INT8 (v1), per-group INT4 (v2) |
| Context Len  | 2048 tokens                                |
| Metrics      | tokens/sec, ms/token, energy J/token       |
| License      | MIT                                        |

---

## 6. 🧠 Model Compatibility Matrix

| Model            | Size  | Quantization | Works On         | Status   |
| :--------------- | :---- | :----------- | :--------------- | :------- |
| TinyLlama        | 1.1 B | INT8/INT4    | iPhone 15 Pro+   | MVP      |
| Gemma            | 2 B   | INT8         | iPad M1+/Mac M1+ | Phase 2  |
| Phi-2            | 2.7 B | INT8         | Mac M2+          | Phase 3  |
| Mistral          | 7 B   | INT4         | Mac M3+          | Stretch  |
| Apple Foundation | 3 B   | FP16         | Core ML demo     | Optional |

---

## 7. 🏗️ System Architecture

```
[ TinyBrain Chat (SwiftUI) ]
       ↓
[ Runtime Layer ]
   ├── Tokenizer
   ├── Sampler
   ├── Streamer
   └── ModelRunner
        ├── Attention / MLP / Norm
        └── KV-Cache
       ↓
[ Backends ]
   ├── Metal Kernels
   └── Core ML (Optional)
```

**Execution Flow**

1. Prompt → tokenize
2. Load quantized weights (mmap)
3. Metal kernels compute attention/MLP
4. Logits → sample next token
5. Stream tokens → UI

---

## 8. 💻 Integration / Public API

```swift
import TinyBrain

let model = try TinyBrain.load("tinyllama-int8.tbf")
let stream = try await model.generateStream(prompt: "Explain gravity.")

for try await token in stream {
    print(token, terminator: "")
}
```

---

## 9. 🔬 Evaluation Protocols

| Metric     | Description                | Tool         | Target   |
| :--------- | :------------------------- | :----------- | :------- |
| Latency    | ms/token (first 50 tokens) | Instruments  | ≤ 150 ms |
| Throughput | Tokens/sec                 | Custom Bench | ≥ 6 t/s  |
| Energy     | Joules/token               | MetricsKit   | ≤ 1.5 J  |
| Quality    | Perplexity Δ vs FP16       | Python Eval  | ≤ 15 %   |
| Memory     | Peak MB RAM                | Xcode Graph  | ≤ 1 GB   |

---

## 10. 🔒 Safety & Privacy

* All inference **on-device**
* No telemetry by default
* Optional opt-in analytics
* Warn against prompt injection
* Sandbox compliant; no private APIs

---

## 11. 📦 Distribution & Versioning

* **SPM** package (`TinyBrain`)
* Semantic ver (`0.x` exp, `1.0` stable)
* **CI/CD:** GitHub Actions for iOS/macOS
* Channels: `main` (nightly) / `release` (stable)

---

## 12. 📘 Documentation & Education

TinyBrain will include:

* Inline docstrings & code comments
* `docs/` Markdown site with diagrams
* Swift Playgrounds tutorials
* Blog series + YouTube visuals explaining transformers

---

## 13. 🧪 Testing & Benchmarks

| Test          | Goal                        | Tool               |
| :------------ | :-------------------------- | :----------------- |
| Unit          | Tensor ops / quant accuracy | XCTest             |
| Integration   | Prompt → output             | Simulator + Device |
| Perf          | Latency / energy            | MetricsKit         |
| Memory        | Peak usage                  | Instruments        |
| Compatibility | iOS 17+, macOS 14+          | GitHub CI          |

**Success Metrics**

* ≤ 150 ms/token (A17)
* ≤ 1 GB RAM
* ≤ 15 % accuracy loss

---

## 14. 🎨 Demo App – TinyBrain Chat

SwiftUI app showing live local inference.

**Features**

* Prompt input & streamed output
* Token speed + energy overlay
* Quantization toggle (INT8/INT4)
* Token probability graph

---

## 15. 🔧 Dev Environment

| Requirement | Min Version          |
| :---------- | :------------------- |
| macOS       | 14 Sonoma            |
| Xcode       | 16+                  |
| Swift       | 5.10+                |
| Python      | 3.11 (for converter) |

```bash
git clone https://github.com/vivekp/tinybrain.git
cd tinybrain && open TinyBrain.xcodeproj
```

---

## 16. 🧭 Roadmap & Phases

| Phase           | Months | Deliverables                  |
| :-------------- | :----- | :---------------------------- |
| Prototype (MVP) | 1–2    | Tensor engine + MatMul + demo |
| Quant + Cache   | 3–4    | INT8 + KV-cache + stream      |
| Bench + Docs    | 5–6    | Benchmark suite + release     |
| Hybrid V2       | 7–8    | INT4 + Core ML + leaderboard  |

---

## 17. 📊 Success Criteria

| Type                | Target         |
| :------------------ | :------------- |
| GitHub Stars        | 3 k in 6 mo    |
| Citations (arXiv)   | ≥ 5 in 12 mo   |
| Avg Latency         | < 150 ms/token |
| Community Forks     | 100+           |
| TestFlight Installs | 10 k +         |

---

## 18. 📦 Dependencies

| Library              | Purpose           | License    |
| :------------------- | :---------------- | :--------- |
| Swift ArgumentParser | CLI converter     | MIT        |
| Core ML / Metal      | Inference backend | Apple      |
| Accelerate           | Math ops          | BSD        |
| SentencePiece        | Tokenizer         | Apache 2.0 |
| PyTorch              | Export pipeline   | BSD        |

---

## 19. ⚠️ Risks & Mitigations

| Risk               | Impact | Mitigation                                    |
| :----------------- | :----- | :-------------------------------------------- |
| Low Perf           | Med    | Optimize Metal kernels / use Core ML fallback |
| SDK Changes        | Med    | CI on beta SDKs                               |
| Limited ANE Access | High   | Hybrid mode                                   |
| Memory Pressure    | Med    | Quant + mmap                                  |
| Model Licenses     | Low    | Use Apache/BSD models only                    |

---

## 20. 🧠 Research & Publication Tie-ins

Planned papers:

* *“TinyBrain: A Swift-Native LLM Runtime for Apple Devices.”*
* *“Energy–Performance Trade-offs in On-Device LLMs.”*
* *“Hybrid Metal/Core ML Attention for Edge AI.”*

All benchmark data will be reproducible; scripts released under MIT.

---

## 21. 🔄 Future Extensions

* FlashAttention kernel (optimized Metal)
* ANE-specific ops integration
* Distillation toolkit for mobile LLMs
* iOS Shortcut agent automation
* TinyBrain Playgrounds for education

---

## 22. 🧩 Competitive Positioning Summary

| Aspect          | TinyBrain     | MLC-LLM   | llama.cpp | Core ML         |
| :-------------- | :------------ | :-------- | :-------- | :-------------- |
| Language        | Swift + Metal | C++ + TVM | C/C++     | Objective-C API |
| Educational     | ✅             | ❌         | ❌         | ❌               |
| Hybrid Backend  | ✅             | ⚠️        | ❌         | ✅               |
| iOS Integration | Deep          | Medium    | Low       | High            |
| Benchmark Suite | ✅             | Partial   | ✅         | ❌               |

---

## 23. 📋 Notes for AI Agents (Cursor / Windsurf / Claude / Copilot)

* **Scaffold repo**: `/Sources/TinyBrain`, `/Examples/ChatDemo`, `/Tests`.
* Generate `Package.swift`, `README.md`, and `docs/overview.md`.
* Implement base tensor ops first.
* Follow semantic commit format (`feat/runtime`, `core/metal`, `ui/demo`).
* Autogenerate API docs via DocC.
* Add CI tests for latency benchmarks.

---

## 24. 📈 Success Narrative

> TinyBrain will become the reference educational and practical runtime for transformer inference on Apple hardware—doing for iOS AI what *micrograd* did for neural-net intuition.
> It will drive open-source contributions, research citations, and industry adoption across privacy-focused mobile AI apps.