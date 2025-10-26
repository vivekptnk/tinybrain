# AGENTS – TinyBrain Project Rules

## 1. Mission & Canonical Sources
- **Primary goal:** build a Swift-native, on-device LLM runtime that is transparent, educational, and fast on Apple Silicon (per `docs/prd.md`).
- **Truth hierarchy:** PRD → tasks in `docs/tasks/` → code comments/DocC → other docs. Resolve conflicts by updating upstream documents first.
- **Deliverable cadence:** follow the roadmap phases (Scaffold → Runtime → Metal → Quant/KV → Tokenizer/Streaming → Demo App → Benchmarks/Docs) unless the product owner reprioritizes.

## 2. Repository Expectations
- Maintain the structure defined in TB-001 (`Sources/`, `Examples/ChatDemo`, `Tests/`, `Scripts/`, `docs/`). Keep module boundaries clean (`TinyBrainRuntime`, `TinyBrainMetal`, `TinyBrainTokenizer`, etc.).
- All new binaries/models go under `Models/` with `.gitignore` entries; keep the repo lightweight.
- Use semantic commit prefixes noted in the PRD (e.g., `feat/runtime`, `core/metal`, `ui/demo`).

## 3. Swift Coding Best Practices
- Target Swift 5.10+, iOS 17, macOS 14. Prefer Swift Package Manager; avoid mixed Objective-C/C++ unless no Swift alternative exists.
- Design APIs with value semantics where practical (e.g., `struct Tensor`) but fall back to reference types for buffer managers to avoid copies.
- Use Swift Concurrency (`async/await`, `AsyncSequence`) for streaming and long-running tasks; isolate Metal/Accelerate work on dedicated executors/queues.
- Keep public APIs documented with DocC-compatible comments. Include preconditions/postconditions for tensor shapes and dtypes.
- Favor protocol-oriented design for pluggable components (tokenizers, samplers, backends) to preserve educational clarity.

## 4. Metal & Performance Rules
- Encapsulate Metal objects inside `TinyBrainMetal`—no leaking `MTLDevice/CommandQueue` beyond backend interfaces.
- Provide CPU fallbacks for every Metal op; tests should run headless on CI even without GPU acceleration.
- Benchmark critical kernels (MatMul, Softmax, LayerNorm, quant/dequant) and commit representative numbers under `benchmarks/`.
- Use shared buffer pools, avoid per-token allocations, and document tuning knobs (threadgroup sizes, tile dimensions).

## 5. Quantization, Memory, and KV-Cache
- Implement per-channel INT8 first; INT4 comes later via dedicated tasks. Keep metadata schemas versioned and documented.
- KV-cache must be paged and reusable across tokens/streams; expose telemetry hooks for cache hits/evictions.
- Always test quantized paths against FP16 references; define acceptable error tolerances (≤1% perplexity delta or <1e-3 relative error depending on metric).

## 6. Test-Driven Development (TDD) Methodology
- **Write tests FIRST, then implement:** For all core functionality (tensor ops, tokenization, sampling), write failing tests before writing implementation code.
- **Red-Green-Refactor cycle:**
  1. **RED**: Write a failing test that defines desired behavior
  2. **GREEN**: Write minimal code to make the test pass
  3. **REFACTOR**: Improve code quality while keeping tests green
- **Tests as documentation:** Each test should be readable and educational, explaining WHAT the operation does, WHY it matters, and HOW to verify correctness.
- **Numerical accuracy requirements:**
  - Tensor operations: relative error < 1e-5 for Float32, < 1e-3 for Float16
  - Quantized operations: perplexity delta ≤ 1% vs FP16 reference
  - Include edge cases: empty tensors, mismatched shapes, numerical stability (NaN, Inf)
- **Test naming convention:** `test<Operation><Scenario>` (e.g., `testMatMulBasic`, `testMatMulBroadcast`, `testSoftmaxNumericalStability`)

## 7. Testing & Tooling
- `swift test` and linting (SwiftFormat/SwiftLint or compiler plugins) must pass before opening PRs.
- Add targeted unit tests for tensor math, tokenizer parity, sampler randomness, and streaming latencies. Include snapshot/UI tests for SwiftUI components.
- Benchmark harness (`tinybrain-bench`) should support scripted scenarios and run smoke tests in CI (reduced workloads) while full runs are manual/weekly.
- Never merge failing CI; if CI lacks required device features, gate tests with availability checks and document the manual validation plan.

## 8. Development Workflows (TB-007)
- **SPM Command Line:** Use `swift build`, `swift test`, and `make` commands for CI/CD, linting, testing, and library development.
- **Xcode IDE:** Open `Package.swift` directly in Xcode (modern approach). Do NOT use deprecated `swift package generate-xcodeproj`.
- **ChatDemo App:** Run in Xcode with sandbox disabled (Edit Scheme → Run → Options → Uncheck "Use the sandbox") to fix TextField input on macOS 15.x Tahoe. This is a known SPM executable limitation.
- **Info.plist:** Included in `Examples/ChatDemo/Info.plist` for future app bundle support and proper CFBundleIdentifier configuration.
- **iOS Development:** TextField works fine on iOS; macOS-specific workaround only needed for SPM executables on macOS 15.x.

## 9. Documentation & Developer Experience
- Keep `docs/overview.md`, DocC articles, and `README.md` current with architectural diagrams and setup instructions.
- Every feature task must update docs: architecture narratives, troubleshooting, and API examples.
- Surface telemetry/UX copy ("TinyBrain Chat", "Energy Overlay") consistently across UI and docs.

## 10. Collaboration Protocol
- Reference TB task IDs in issues/PRs (e.g., "Implements TB-003").
- Discuss major API or architecture changes in design docs before coding; store them under `docs/rfcs/`.
- When blocked, document the issue in the task file and propose mitigation paths aligned with PRD risk table.
- Keep responses concise and action-focused; summarize deviations from the plan in AGENTS.md if they become permanent rules.
- **NEVER commit changes without explicit permission from the product owner.** Always stage changes and ask for approval before running `git commit`.

Following these rules keeps TinyBrain aligned with the PRD vision while enabling future agents to contribute confidently.
