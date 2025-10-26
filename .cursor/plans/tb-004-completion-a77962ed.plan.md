<!-- a77962ed-1a32-4b00-8e02-ee50d70683d6 899c4533-202c-493a-90b2-783a5d71d798 -->
# Complete TB-004 Using TDD (Red-Green-Refactor)

## Overview

Address the 3 incomplete TB-004 work items using strict TDD methodology: write failing tests first (RED), implement minimal code to pass (GREEN), then refactor (REFACTOR).

## Work Item #3: .tbf Weight Format with mmap Support

### Phase 1: RED - Write Specification & Failing Tests

**1. Create TBF Format Specification**

- Document `docs/tbf-format-spec.md` with exact byte layout
- Magic bytes "TBFM" (4 bytes), version UInt32, config JSON, quantization metadata
- Weight sections with 4KB page alignment for mmap efficiency

**2. Write Failing TBF Format Tests**

Create `Tests/TinyBrainRuntimeTests/TBFFormatTests.swift`:

```swift
func testSaveToyModelToTBF() 
    // RED: ModelWeights.save(to:) doesn't exist yet
    
func testLoadToyModelFromTBF()
    // RED: ModelWeights.load(from:) doesn't exist yet
    
func testRoundTripPreservesWeights()
    // RED: Save then load, verify weights identical
    
func testMmapDoesNotLoadEntireFile()
    // RED: Verify file not fully loaded into RAM
    
func testInvalidMagicBytesThrowsError()
    // RED: Error handling for corrupted files
    
func testVersionMismatchThrowsError()
    // RED: Version validation
```

Run tests - all should fail with "method not implemented"

### Phase 2: GREEN - Implement Just Enough to Pass

**3. Implement ModelWeights.save(to:)**

Add to `Sources/TinyBrainRuntime/ModelWeights.swift`:

- Write TBFM magic bytes and version
- Serialize ModelConfig to JSON
- Write quantization metadata arrays
- Write weight blobs with 4KB alignment
- Make `testSaveToyModelToTBF` pass

**4. Implement ModelWeights.load(from:)**

Add to `Sources/TinyBrainRuntime/ModelWeights.swift`:

- Use `mmap()` for zero-copy loading
- Parse/validate header and magic bytes
- Deserialize config JSON
- Create QuantizedTensor instances from mmap'd regions
- Make all TBFFormatTests pass (green)

### Phase 3: REFACTOR - Improve Implementation

**5. Refactor TBF Code**

- Extract header parsing into helper methods
- Add comprehensive error types
- Optimize alignment calculations
- Keep all tests passing

## Work Item #8: BLEU & Perplexity Regression Tests

### Phase 1: RED - Write Failing Quality Tests

**1. Create Test Fixtures**

- Create `Tests/TinyBrainRuntimeTests/Fixtures/prompts/`
- Add 5 sample prompts with reference outputs
- Store as JSON: `{prompt: [tokens], reference: [tokens], expected_text: "..."}`

**2. Write Failing Metrics Tests**

Create `Tests/TinyBrainRuntimeTests/QualityRegressionTests.swift`:

```swift
func testPerplexityCalculation()
    // RED: perplexity() function doesn't exist
    
func testBLEUScoreCalculation()
    // RED: bleuScore() function doesn't exist
    
func testINT8PerplexityVsFP32()
    // RED: Compare quantized vs float, assert ≤1% delta
    
func testINT8BLEUScoreVsFP32()
    // RED: Compare quantized vs float BLEU
    
func testMultiplePromptsRegression()
    // RED: Test on all fixture prompts
```

Run tests - all should fail

### Phase 2: GREEN - Implement Metrics

**3. Implement Perplexity**

Create `Sources/TinyBrainRuntime/Metrics.swift`:

```swift
public func perplexity(logits: [Tensor<Float>], targetTokens: [Int]) -> Float {
    // Formula: exp(-mean(log(P(target_token))))
}
```

- Make `testPerplexityCalculation` pass

**4. Implement BLEU Score**

Add to `Metrics.swift`:

```swift
public func bleuScore(candidate: [Int], reference: [Int], maxN: Int = 4) -> Float {
    // Compute n-gram precision (n=1,2,3,4) + brevity penalty
}
```

- Make `testBLEUScoreCalculation` pass

**5. Run Regression Tests**

- Generate outputs from INT8 and FP32 models
- Compute perplexity and BLEU for both
- Make all QualityRegressionTests pass

### Phase 3: REFACTOR - Improve Metrics Code

**6. Refactor Metrics**

- Extract n-gram counting into helper
- Optimize log probability calculations
- Add edge case handling
- Keep tests passing

**7. Document Metrics**

- Update `docs/overview.md` with metrics section
- Add interpretation guide: "Perplexity of X means Y"
- Include baseline benchmark results

## Work Item #11: Fresh Metal Benchmarks & Documentation

### Phase 1: RED - Define Benchmark Expectations

**1. Write Benchmark Validation Tests**

Add to `Tests/TinyBrainMetalTests/PerformanceBenchmarks.swift`:

```swift
func testBenchmarkResultsAreDocumented()
    // RED: Verify benchmark results match docs
    // Parse metal-vs-cpu.md, check numbers are fresh
```

### Phase 2: GREEN - Run Benchmarks & Update Docs

**2. Run Comprehensive Benchmarks**

- Execute: `swift test --filter MetalPerformanceBenchmarks`
- Test sizes: 512×512, 1024×1024, 1536×1536, 2048×2048
- Record 10 iterations per size, compute mean/stddev
- Clean system (no background load)

**3. Update Documentation**

Update `benchmarks/metal-vs-cpu.md`:

- Replace tables at lines 36-42 with fresh results
- Add "TB-004 Results with Persistent Buffers" section
- Document variance (0.7-1.3× range)
- Explain AMX reality vs original 3-8× goal
- Add comparison: without vs with persistent buffers

**4. Create Final Benchmark Report**

Create `docs/TB-004-BENCHMARK-FINAL.md`:

- Hardware: M4 Max specs
- Methodology: iterations, warmup, measurement
- Results table for all sizes
- Analysis: why AMX beats GPU for single ops
- Future: batched attention, fused kernels
- Conclusion: ready for TB-005

**5. Update Test Comments**

Update `Tests/TinyBrainMetalTests/PerformanceBenchmarks.swift`:

- Lines 243-256: reflect final validated results
- Document actual speedup ranges
- Note when GPU wins (batched ops)

**6. Finalize TB-004 Status**

Update `docs/tasks/TB-004.md`:

- Lines 23-26: final benchmark results
- Line 44: mark work item #11 complete
- Add performance summary section

### Phase 3: REFACTOR - Polish Documentation

**7. Polish All Docs**

- Ensure consistency across all benchmark docs
- Add cross-references between docs
- Verify all acceptance criteria met

## TDD Workflow Summary

For each work item:

1. **RED**: Write failing tests that define expected behavior
2. **GREEN**: Implement minimal code to make tests pass
3. **REFACTOR**: Improve code quality while keeping tests green

## Files to Create/Modify

### New Files

- `docs/tbf-format-spec.md` - TBF format specification
- `Sources/TinyBrainRuntime/Metrics.swift` - Perplexity & BLEU
- `Tests/TinyBrainRuntimeTests/TBFFormatTests.swift` - TBF tests (RED phase)
- `Tests/TinyBrainRuntimeTests/QualityRegressionTests.swift` - Quality tests (RED phase)
- `Tests/TinyBrainRuntimeTests/Fixtures/prompts/sample_prompts.json` - Test data
- `docs/TB-004-BENCHMARK-FINAL.md` - Final benchmark report

### Modified Files

- `Sources/TinyBrainRuntime/ModelWeights.swift` - Add save/load (GREEN phase)
- `benchmarks/metal-vs-cpu.md` - Fresh benchmark data
- `Tests/TinyBrainMetalTests/PerformanceBenchmarks.swift` - Update comments
- `docs/tasks/TB-004.md` - Mark work items complete
- `docs/overview.md` - Add metrics documentation

## Acceptance Criteria

- ✅ All tests written BEFORE implementation (TDD)
- ✅ `.tbf` format loads via mmap without full RAM load
- ✅ Round-trip save/load passes with zero corruption
- ✅ Perplexity shows INT8 within 1% of FP32
- ✅ BLEU demonstrates acceptable quality
- ✅ Fresh benchmarks documented
- ✅ Work items #3, #8, #11 complete in TB-004.md
- ✅ Ready for TB-005

### To-dos

- [ ] Create comprehensive TBF format specification document with exact byte layout
- [ ] Implement ModelWeights.save() with mmap-friendly alignment
- [ ] Implement ModelWeights.load() with mmap for zero-copy weight loading
- [ ] Write TBFFormatTests for round-trip save/load validation
- [ ] Implement perplexity and BLEU score calculations in Metrics.swift
- [ ] Create sample prompts and reference outputs for regression tests
- [ ] Write QualityRegressionTests comparing INT8 vs FP32 using BLEU/perplexity
- [ ] Run fresh Metal benchmarks across all matrix sizes with 10 iterations each
- [ ] Update metal-vs-cpu.md with fresh results and create TB-004-BENCHMARK-FINAL.md
- [ ] Update PerformanceBenchmarks.swift comments to reflect final validated results
- [ ] Mark work items #3, #8, #11 complete in TB-004.md and update overview.md