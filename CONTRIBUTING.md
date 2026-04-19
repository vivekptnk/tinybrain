# Contributing to TinyBrain

Thank you for your interest in contributing to TinyBrain! This guide covers everything you need to get started.

## Development Setup

### Prerequisites

- **macOS 14** (Sonoma) or later
- **Xcode 16+** (free from the App Store)
- **Swift 5.10+** (bundled with Xcode)
- **Python 3.11+** (for model conversion only)
- **Apple Silicon** (M1 or newer) recommended; Intel Macs work but Metal performance will differ

### Clone and Build

```bash
git clone https://github.com/vivekptnk/tinybrain.git
cd tinybrain
swift build
```

### Run Tests

```bash
swift test --skip TinyBrainDemoTests
```

> **Note:** `--skip TinyBrainDemoTests` is needed due to a known Xcode beta linker issue with SwiftUI test targets. All other tests pass cleanly.

### Run the Demo

```bash
swift run tinybrain-chat
```

### Open in Xcode

```bash
open Package.swift
```

Select the **ChatDemo** scheme and press `Cmd+R`. If the text field doesn't accept input on macOS 15 (Tahoe), go to Edit Scheme > Run > Options and uncheck "Use the sandbox" — this is a known SPM executable limitation.

### Python Environment (Model Conversion)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
Sources/
  TinyBrainRuntime/    — Core engine: tensors, model runner, KV cache, quantization, sampler
  TinyBrainMetal/      — Metal GPU backend, kernels, buffer pool
  TinyBrainTokenizer/  — BPE tokenizer, HuggingFace adapter, TokenizerLoader
  TinyBrainDemo/       — SwiftUI chat app, X-Ray visualizations
  TinyBrainBench/      — CLI benchmarking tool
Examples/ChatDemo/     — App entry point
Tests/                 — Test suites for all modules
Scripts/               — Python model converter
docs/                  — Architecture docs, ADRs, task reports
```

Keep module boundaries clean. `TinyBrainMetal` internals (MTLDevice, MTLCommandQueue) must not leak beyond backend interfaces.

## Test-Driven Development

TinyBrain follows strict **TDD methodology**. All core functionality must be developed using the Red-Green-Refactor cycle:

### 1. RED — Write a Failing Test

Define the desired behavior before writing any implementation:

```swift
func testMatMulBasic() throws {
    let a = Tensor<Float>([[1, 2], [3, 4]])
    let b = Tensor<Float>([[5, 6], [7, 8]])
    let result = a.matmul(b)
    XCTAssertEqual(result[0, 0], 19, accuracy: 1e-5)
    XCTAssertEqual(result[0, 1], 22, accuracy: 1e-5)
}
```

### 2. GREEN — Write Minimal Code to Pass

Implement just enough to make the test pass. No extra features, no premature optimization.

### 3. REFACTOR — Improve While Green

Clean up the implementation. Add educational comments. Ensure tests still pass.

### Numerical Accuracy Requirements

- **Float32 operations:** relative error < 1e-5
- **Float16 operations:** relative error < 1e-3
- **Quantized operations:** perplexity delta <= 1% vs FP16 reference

### Test Naming Convention

Use `test<Operation><Scenario>`:

- `testMatMulBasic`
- `testMatMulBroadcast`
- `testSoftmaxNumericalStability`
- `testQuantizePerChannel`

### Edge Cases to Cover

- Empty tensors
- Mismatched shapes (should fail gracefully)
- Numerical stability (NaN, Inf inputs)
- Thread safety for shared resources

## Metal / GPU Development

### CPU Fallbacks Are Required

Every Metal operation must have a CPU fallback. Tests must run headless on CI without GPU acceleration:

```swift
if MetalBackend.isAvailable {
    // GPU path
} else {
    // CPU fallback via Accelerate
}
```

### Buffer Management

- Use `MetalBufferPool` for buffer allocation — never allocate `MTLBuffer` directly in hot paths.
- Release buffers back to the pool when done.
- Avoid per-token allocations in the inference loop.

### Debugging Metal Kernels

If Metal output doesn't match CPU:

1. Enable debug logging: `TinyBrainBackend.debugLogging = true`
2. Test with small, known matrices first
3. Compare GPU vs CPU results element-by-element
4. Check threadgroup sizes and tile dimensions
5. Use Xcode's Metal debugger (Product > Profile > Metal System Trace)

See `docs/Metal-Debugging-Guide.md` for detailed troubleshooting.

### Performance Reality: AMX vs GPU

M4 Max uses AMX (Apple Matrix Extension) via Accelerate, which competes with or beats GPU for single matmul operations. GPU wins for **batched workflows** (chained attention operations). See `docs/adr/002-gpu-resident-tensors-amx-reality.md` for the full analysis.

## Code Conventions

### Swift Style

- Target Swift 5.10+, iOS 17, macOS 14
- Prefer Swift Package Manager
- Use Swift Concurrency (`async/await`, `AsyncSequence`) for streaming
- Favor protocol-oriented design for pluggable components
- Document public APIs with DocC-compatible comments
- Value semantics for data types (e.g., `Tensor` is a struct with CoW)
- Reference types only for resource managers (buffer pools, caches)

### Commit Messages

Use semantic prefixes matching the module:

```
feat/runtime: Add top-p sampling strategy
core/metal: Optimize tiled matmul threadgroup size
fix/tokenizer: Handle empty string edge case in BPE
ui/demo: Add temperature slider to settings panel
docs: Update architecture overview with KV cache diagrams
chore: Update SwiftLint configuration
```

Reference task IDs when applicable: `feat/runtime: Implement KV cache (TB-004)`

### Module Boundaries

| Module | Responsibility | Dependencies |
|--------|---------------|--------------|
| `TinyBrainRuntime` | Core engine, public API | Accelerate |
| `TinyBrainMetal` | GPU backend | Metal, TinyBrainRuntime |
| `TinyBrainTokenizer` | Text ↔ tokens | Foundation |
| `TinyBrainDemo` | SwiftUI app | All modules |
| `TinyBrainBench` | Benchmarks | TinyBrainRuntime, TinyBrainMetal |

## Pull Request Requirements

1. **Tests first** — new functionality must include tests written before the implementation.
2. **All tests pass** — run `swift test --skip TinyBrainDemoTests` before submitting.
3. **No warnings** — `swift build` must complete without warnings.
4. **Focused scope** — one logical change per PR. Don't mix unrelated changes.
5. **Reference task IDs** — link to the relevant TB-XXX task in the PR description.
6. **Update docs** — if your change affects architecture, public API, or setup instructions, update the relevant docs.
7. **Benchmark critical changes** — performance-sensitive changes should include before/after numbers.

### PR Description Template

```markdown
## What

Brief description of the change.

## Why

Context and motivation. Link to task (e.g., TB-008).

## How

Technical approach. Key design decisions.

## Testing

- [ ] New tests added
- [ ] All existing tests pass
- [ ] Benchmarked (if performance-sensitive)
```

## Architecture Decisions

Significant technical decisions are recorded in `docs/adr/`. Before proposing a major API or architecture change, check existing ADRs and consider writing a new one. For design discussions, create a document under `docs/rfcs/` before coding.

## Getting Help

- **Architecture:** `docs/overview.md`
- **Project rules:** `AGENTS.md`
- **ADRs:** `docs/adr/`
- **Issues:** [GitHub Issues](https://github.com/vivekptnk/tinybrain/issues)
- **Discussions:** [GitHub Discussions](https://github.com/vivekptnk/tinybrain/discussions)
