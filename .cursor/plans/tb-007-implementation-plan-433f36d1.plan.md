<!-- 433f36d1-3a96-43e1-a0ab-0667ad1cda65 fc3053d8-d7d1-419b-8a3c-621a2efc4a7b -->
# TB-007: Benchmark Harness, Documentation, and Release Prep (TDD Methodology)

**Core Principle:** Follow Red-Green-Refactor cycle for all implementation work, as proven successful in TB-001 through TB-006 (163 tests).

## Phase 1: Fix macOS TextField Issue (Convert to Xcode Project)

**Problem:** SwiftUI TextField doesn't accept input in SPM executables on macOS Tahoe (identified in TB-006)

**Solution:** Convert from pure SPM to Xcode project with proper app bundle

### TDD Approach:

**RED (Write failing test):**

- Create manual test plan: "Type in TextField and verify input appears"
- Document expected behavior vs current behavior

**GREEN (Implement fix):**

1. Generate Xcode project: `swift package generate-xcodeproj`
2. Add `Info.plist` with `CFBundleIdentifier` for ChatDemo app target
3. Configure proper macOS app bundle structure in Xcode
4. Verify TextField accepts keyboard input

**REFACTOR (Document):**

- Update `README.md` with new build instructions (Xcode + SPM hybrid)
- Document both workflows: SPM for CLI/tests, Xcode for demo app
- Update `AGENTS.md` with Xcode workflow

**Files to modify:**

- `Package.swift`, `Examples/ChatDemo/Info.plist`, `README.md`, `AGENTS.md`

## Phase 2: PyTorch → TBF Model Converter (TDD)

**Purpose:** Enable benchmarking with real TinyLlama weights (not toy models)

### TDD Approach:

**RED (Write tests first):**

- `Tests/test_convert_model.py`:
                - `test_load_pytorch_checkpoint()` - Load .pt/.safetensors
                - `test_extract_weights_shapes()` - Verify tensor dimensions
                - `test_quantize_int8_accuracy()` - Check quantization error < 1%
                - `test_tbf_format_compliance()` - Validate TBF spec
                - `test_roundtrip_swift()` - Convert → Load in Swift → verify shapes

**GREEN (Implement converter):**

1. Create `Scripts/convert_model.py` Python script
2. Load PyTorch checkpoint (`.pt` or `.safetensors`)
3. Extract weights: embeddings, Q/K/V/O projections, MLP, layer norms
4. Quantize to INT8 per-channel (symmetric quantization)
5. Write TBF format matching `docs/tbf-format-spec.md`
6. Add CLI: `python convert_model.py --input model.pt --output model.tbf --quantize int8`

**REFACTOR (Optimize):**

- Add progress bars, error handling
- Validate all tests pass
- Document converter usage

**Dependencies:** `torch`, `safetensors`, `numpy` (add to `requirements.txt`)

**Validation:** Convert TinyLlama-1.1B from HuggingFace, load in `ModelRunner`

## Phase 3: Benchmark Harness CLI (TDD)

**Goal:** Reproducible performance measurements across devices

### TDD Approach:

**RED (Write tests first):**

- `Tests/TinyBrainBenchTests/`:
                - `testYAMLScenarioLoading()` - Parse YAML scenarios
                - `testJSONOutputFormat()` - Validate JSON schema
                - `testMemoryTracking()` - Verify memory metrics accuracy
                - `testDeviceInfoReporting()` - Check device metadata
                - `testWarmupIterations()` - Ensure warmup runs before timing

**GREEN (Implement harness):**

1. Enhance `Sources/TinyBrainBench/TinyBrainBench.swift` with YAML scenario support
2. Create YAML schema for benchmark scenarios (see example below)
3. Add JSON output format for CI/automation
4. Implement energy estimation (via `ProcessInfo.processInfo.thermalState`)
5. Add memory tracking using `task_info` APIs

**REFACTOR (Polish):**

- Clean up CLI interface
- Add `--help` documentation
- Validate tests pass

**YAML Example:**

```yaml
scenarios:
 - name: "TinyLlama INT8 - Short Prompt"
    model: "Models/tinyllama-1.1b-int8.tbf"
    prompts:
   - "Explain quantum physics"
    max_tokens: 50
    backend: auto
```

**CLI Flags:**

- `--scenario <path>`, `--output json|markdown`, `--device-info`, `--warmup <n>`

**Files to create:**

- `benchmarks/scenarios.yml`, `Scripts/run_benchmarks.sh`

## Phase 4: Baseline Benchmarks (Execution Phase)

**Devices:** M4 Max + iPhone 16 Pro

**Note:** No TDD here - this is measurement/documentation phase

### Tasks:

1. Run benchmarks on M4 Max (macOS):

                        - TinyLlama INT8: tokens/sec, ms/token, memory usage
                        - Varying prompt lengths: 10, 50, 100 tokens
                        - CPU vs Metal backend comparison

2. Run benchmarks on iPhone 16 Pro (iOS):

                        - Same scenarios as macOS
                        - Note thermal throttling if any
                        - Battery/energy impact (qualitative)

3. Store results: `benchmarks/baseline-m4-max.md`, `benchmarks/baseline-iphone16pro.md`
4. Generate comparison tables
5. Create plots (optional): tokens/sec vs prompt length

**Success criteria:**

- Document whether project meets PRD target: <150 ms/token on iPhone 15 Pro class
- Honest assessment if targets missed (like TB-003/TB-004 AMX findings)

## Phase 5: Documentation Expansion

**Note:** Documentation doesn't require TDD, but should be validated via peer review

### DocC Tutorials (create in `Sources/TinyBrain/TinyBrain.docc/`):

1. **RuntimeInternals.tutorial**

                        - How transformer layers are executed
                        - KV-cache lifecycle
                        - Quantization pipeline

2. **MetalDeepDive.tutorial**

                        - Tiled MatMul kernel walkthrough
                        - Buffer pool architecture
                        - GPU-resident tensors
                        - When Metal wins vs CPU (AMX findings)

3. **BuildingChatApp.tutorial**

                        - Step-by-step ChatDemo walkthrough
                        - Integrating tokenizer + sampler
                        - Streaming tokens to SwiftUI

### Additional Docs:

1. `docs/benchmarking.md` - Methodology, how to run, interpreting results
2. `docs/faq.md` - Troubleshooting, performance tuning, model compatibility
3. Update `docs/overview.md` - TB-006 completion, current state

## Phase 6: Release Notes

**Create:** `docs/releases/v0.1.0.md`

### Contents:

1. Features (Swift-native runtime, INT8 quant, Metal, KV-cache, tokenizer, sampler, SwiftUI)
2. Supported Models (TinyLlama 1.1B via converter)
3. Installation (Xcode project setup, CLI)
4. Performance Metrics (M4 Max + iPhone 16 Pro results)
5. Known Issues (honest assessment, AMX findings)
6. What's Next (INT4, FlashAttention roadmap)

## Phase 7: CI Setup (GitHub Actions)

**Create:** `.github/workflows/ci.yml`

### Pipeline:

1. Lint (SwiftLint)
2. Format check (SwiftFormat)
3. Build (debug + release)
4. Test (all 163+ tests) ← **Critical: All TDD tests must pass**
5. Benchmark smoke tests (reduced workloads)
6. Build documentation
7. Upload DocC archive as artifact

**Platform matrix:** macOS 14 (Xcode 16), iOS Simulator (optional)

**Triggers:** Push to main, pull requests, manual dispatch

## Phase 8: Final Quality Sweep

### Checklist (`docs/release-checklist.md`):

- [ ] All tests passing (run `make test`) ← **TDD validation**
- [ ] Linting clean (run `make lint`)
- [ ] Format check clean (run `make format-check`)
- [ ] Documentation builds without errors
- [ ] Demo app runs on macOS (TextField working)
- [ ] Demo app runs on iOS (iPhone 16 Pro tested)
- [ ] Benchmarks executed on both devices
- [ ] README updated with latest info
- [ ] License and attribution verified
- [ ] AGENTS.md reflects current workflow
- [ ] No sensitive data in repo
- [ ] Git tags prepared: `v0.1.0`

### Final actions:

1. Run full test suite one more time
2. Generate final documentation
3. Commit all changes
4. Create release tag
5. Push to GitHub

## TDD Summary

**Phases with TDD:**

- Phase 2 (Model Converter): 5+ tests written first
- Phase 3 (Benchmark Harness): 5+ tests written first

**Phases without TDD:**

- Phase 1 (Xcode fix): Manual testing (UI-based)
- Phase 4 (Benchmarks): Measurement/data collection
- Phase 5 (Documentation): Peer review
- Phase 6 (Release notes): Documentation
- Phase 7 (CI): Infrastructure
- Phase 8 (Quality): Validation

**Total Expected Test Count:** 163 (existing) + 10+ (TB-007) = **173+ tests**

## Key Files to Create/Modify

**New files:**

- `.github/workflows/ci.yml`
- `benchmarks/scenarios.yml`
- `benchmarks/baseline-m4-max.md`
- `benchmarks/baseline-iphone16pro.md`
- `Scripts/convert_model.py`
- `Scripts/run_benchmarks.sh`
- `requirements.txt`
- `Examples/ChatDemo/Info.plist`
- `docs/benchmarking.md`
- `docs/faq.md`
- `docs/releases/v0.1.0.md`
- `docs/release-checklist.md`
- `Sources/TinyBrain/TinyBrain.docc/RuntimeInternals.tutorial`
- `Sources/TinyBrain/TinyBrain.docc/MetalDeepDive.tutorial`
- `Sources/TinyBrain/TinyBrain.docc/BuildingChatApp.tutorial`
- `Tests/test_convert_model.py`
- `Tests/TinyBrainBenchTests/` (new test target)

**Modified files:**

- `Sources/TinyBrainBench/TinyBrainBench.swift`
- `README.md`
- `AGENTS.md`
- `docs/overview.md`
- `Package.swift`

## Success Criteria

1. TextField input works on macOS ✅
2. Benchmark CLI produces structured JSON/Markdown outputs ✅
3. Real TinyLlama model converted and benchmarked ✅
4. Documentation includes 3+ tutorials (Runtime, Metal, Chat) ✅
5. Release notes drafted with honest metrics ✅
6. CI pipeline runs on GitHub Actions ✅
7. Release checklist completed with evidence ✅
8. **All TDD tests pass (173+ total)** ✅

## Estimated Effort

- Phase 1 (Xcode fix): 2-3 hours
- Phase 2 (Model converter + tests): 6-8 hours (2 hours testing)
- Phase 3 (Benchmark harness + tests): 8-10 hours (2 hours testing)
- Phase 4 (Baseline benchmarks): 3-4 hours
- Phase 5 (Documentation): 8-10 hours
- Phase 6 (Release notes): 2-3 hours
- Phase 7 (CI setup): 3-4 hours
- Phase 8 (Quality sweep): 2-3 hours

**Total:** ~35-45 hours for complete TB-007 (includes TDD time)

### To-dos

- [ ] Convert to Xcode project with proper app bundle (fixes TextField on macOS)
- [ ] Build PyTorch → TBF model converter script
- [ ] Enhance tinybrain-bench CLI with YAML scenarios and JSON output
- [ ] Run benchmarks on M4 Max and iPhone 16 Pro with real TinyLlama model
- [ ] Create 3 DocC tutorials (Runtime, Metal, Chat App)
- [ ] Write benchmarking.md, faq.md, update overview.md
- [ ] Draft v0.1.0 release notes with metrics and known issues
- [ ] Create GitHub Actions CI pipeline
- [ ] Complete release checklist and final verification