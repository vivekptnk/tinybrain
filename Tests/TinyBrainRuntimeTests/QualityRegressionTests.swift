import XCTest
@testable import TinyBrainRuntime
import TinyBrainMetal
import Foundation

/// **TB-004 Work Item #8:** Quality Regression Tests (TDD RED Phase)
///
/// WHAT: Test BLEU and perplexity metrics for INT8 vs FP32 models
/// WHY: Validates quantization doesn't degrade model quality beyond acceptable threshold
/// HOW: Compare metrics on sample prompts, assert ≤1% perplexity delta
///
/// **TDD Phase:** RED - These tests should FAIL until Metrics.swift is implemented
final class QualityRegressionTests: XCTestCase {
    
    // MARK: - Test Fixtures
    
    struct PromptFixture: Codable {
        let id: String
        let prompt: [Int]
        let reference: [Int]
        let description: String
    }
    
    var fixtures: [PromptFixture] = []
    
    override func setUp() {
        super.setUp()
        
        // Load test fixtures
        let fixturesURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("sample_prompts.json")
        
        if let fixtureData = try? Data(contentsOf: fixturesURL) {
            fixtures = (try? JSONDecoder().decode([PromptFixture].self, from: fixtureData)) ?? []
        }
        
        XCTAssertFalse(fixtures.isEmpty, "Should load test fixtures")
    }
    
    // MARK: - Perplexity Tests
    
    /// **RED:** Test perplexity calculation
    ///
    /// WHAT: Calculate perplexity from logits and target tokens
    /// WHY: Validates perplexity metric implementation
    /// HOW: Compute perplexity = exp(-mean(log(P(target))))
    /// EXPECTED: Should FAIL - perplexity() doesn't exist yet
    func testPerplexityCalculation() throws {
        // Simple test case: uniform distribution
        let vocabSize = 10
        let logits = [
            Tensor<Float>.filled(shape: TensorShape(vocabSize), value: 0.1)
        ]
        let targets = [5]  // Target token ID
        
        // RED: This should fail - perplexity() function doesn't exist
        let ppl = try perplexity(logits: logits, targetTokens: targets)
        
        // Perplexity for uniform distribution over 10 tokens should be ~10
        XCTAssertGreaterThan(ppl, 8.0, "Perplexity should be close to vocab size for uniform dist")
        XCTAssertLessThan(ppl, 12.0, "Perplexity should be close to vocab size for uniform dist")
    }
    
    /// **RED:** Test perplexity with known probabilities
    ///
    /// WHAT: Calculate perplexity with controlled probability distribution
    /// WHY: Validates numerical correctness
    /// HOW: Create logits with known softmax outputs
    /// ACCURACY: < 0.1% error
    func testPerplexityWithKnownProbabilities() throws {
        // Create logits that result in known probabilities
        let vocabSize = 4
        
        // Logits: [10, 0, 0, 0] -> softmax ≈ [0.9999, 0.0001, 0.0001, 0.0001]
        let logits = [
            Tensor<Float>(shape: TensorShape(vocabSize), data: [10.0, 0.0, 0.0, 0.0])
        ]
        let targets = [0]  // Targeting the high-probability token
        
        let ppl = try perplexity(logits: logits, targetTokens: targets)
        
        // Perplexity should be very low (~1.0) since we're predicting the likely token
        XCTAssertLessThan(ppl, 1.1, "Perplexity should be ~1 for high-confidence correct prediction")
        XCTAssertGreaterThan(ppl, 0.99, "Perplexity should be ~1 for high-confidence correct prediction")
    }
    
    // MARK: - BLEU Score Tests
    
    /// **RED:** Test BLEU score calculation
    ///
    /// WHAT: Calculate BLEU score between candidate and reference
    /// WHY: Validates BLEU metric implementation
    /// HOW: Compute n-gram precision with brevity penalty
    /// EXPECTED: Should FAIL - bleuScore() doesn't exist yet
    func testBLEUScoreCalculation() throws {
        // Perfect match
        let candidate = [1, 2, 3, 4, 5]
        let reference = [1, 2, 3, 4, 5]
        
        // RED: This should fail - bleuScore() function doesn't exist
        let score = bleuScore(candidate: candidate, reference: reference)
        
        // Perfect match should have BLEU = 1.0
        XCTAssertEqual(score, 1.0, accuracy: 0.001, "Perfect match should have BLEU = 1.0")
    }
    
    /// **RED:** Test BLEU score with partial match
    ///
    /// WHAT: Calculate BLEU for partially matching sequences
    /// WHY: Validates BLEU handles imperfect matches
    /// HOW: Test with sequences that share some n-grams
    /// EXPECTED: 0 < BLEU < 1
    func testBLEUScorePartialMatch() throws {
        let candidate = [1, 2, 3, 4, 5]
        let reference = [1, 2, 3, 6, 7]  // First 3 tokens match
        
        let score = bleuScore(candidate: candidate, reference: reference)
        
        // Partial match should have 0 < BLEU < 1
        XCTAssertGreaterThan(score, 0.0, "Partial match should have positive BLEU")
        XCTAssertLessThan(score, 1.0, "Partial match should have BLEU < 1.0")
    }
    
    /// **RED:** Test BLEU score with no match
    ///
    /// WHAT: Calculate BLEU for completely different sequences
    /// WHY: Validates BLEU handles mismatches
    /// HOW: Test with non-overlapping sequences
    /// EXPECTED: BLEU ≈ 0
    func testBLEUScoreNoMatch() throws {
        let candidate = [1, 2, 3, 4, 5]
        let reference = [10, 20, 30, 40, 50]  // No overlap
        
        let score = bleuScore(candidate: candidate, reference: reference)
        
        // No match should have BLEU ≈ 0
        XCTAssertLessThan(score, 0.1, "No match should have BLEU ≈ 0")
    }
    
    // MARK: - INT8 vs FP32 Regression Tests
    
    /// **RED:** Test INT8 perplexity vs FP32
    ///
    /// WHAT: Compare perplexity between quantized and float models
    /// WHY: Validates INT8 quantization doesn't degrade quality > 1%
    /// HOW: Run same prompts through both models, measure perplexity delta
    /// ACCEPTANCE: Perplexity delta ≤ 1% (per TB-004 spec)
    func testINT8PerplexityVsFP32() throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 128
        )
        
        // Create FP32 and INT8 models from same seed
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 2025)
        let fp32Weights = FloatModelWeights.random(config: config, rng: &rng)
        let int8Weights = fp32Weights.quantized()
        
        let fp32Runner = FloatReferenceRunner(weights: fp32Weights)
        let int8Runner = ModelRunner(weights: int8Weights)
        
        // Test on first fixture
        guard let fixture = fixtures.first else {
            XCTFail("No fixtures loaded")
            return
        }
        
        // Generate logits from both models
        var fp32Logits: [Tensor<Float>] = []
        var int8Logits: [Tensor<Float>] = []
        
        for token in fixture.prompt {
            fp32Logits.append(fp32Runner.step(tokenId: token))
            int8Logits.append(int8Runner.step(tokenId: token))
        }
        
        // Calculate perplexity for both
        let fp32PPL = try perplexity(logits: fp32Logits, targetTokens: fixture.reference)
        let int8PPL = try perplexity(logits: int8Logits, targetTokens: fixture.reference)
        
        // Calculate delta
        let delta = abs(fp32PPL - int8PPL) / fp32PPL
        
        print("FP32 Perplexity: \(fp32PPL)")
        print("INT8 Perplexity: \(int8PPL)")
        print("Delta: \(delta * 100)%")
        
        // TB-004 acceptance: ≤1% perplexity delta
        XCTAssertLessThan(delta, 0.01, "INT8 perplexity should be within 1% of FP32 (got \(delta * 100)%)")
    }
    
    /// **RED:** Test INT8 BLEU score vs FP32
    ///
    /// WHAT: Compare BLEU scores between quantized and float models
    /// WHY: Validates INT8 produces similar outputs to FP32
    /// HOW: Generate sequences from both, compute BLEU
    /// EXPECTED: High BLEU (>0.8) indicating similar outputs
    func testINT8BLEUScoreVsFP32() throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 128
        )
        
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 2025)
        let fp32Weights = FloatModelWeights.random(config: config, rng: &rng)
        let int8Weights = fp32Weights.quantized()
        
        let fp32Runner = FloatReferenceRunner(weights: fp32Weights)
        let int8Runner = ModelRunner(weights: int8Weights)
        
        guard let fixture = fixtures.first else {
            XCTFail("No fixtures loaded")
            return
        }
        
        // Generate sequences
        var fp32Generated: [Int] = []
        var int8Generated: [Int] = []
        
        for token in fixture.prompt {
            let fp32Logits = fp32Runner.step(tokenId: token)
            let int8Logits = int8Runner.step(tokenId: token)
            
            // Argmax sampling
            fp32Generated.append(fp32Logits.data.enumerated().max(by: { $0.1 < $1.1 })!.0)
            int8Generated.append(int8Logits.data.enumerated().max(by: { $0.1 < $1.1 })!.0)
        }
        
        let bleu = bleuScore(candidate: int8Generated, reference: fp32Generated)
        
        print("FP32 output: \(fp32Generated)")
        print("INT8 output: \(int8Generated)")
        print("BLEU: \(bleu)")
        
        // INT8 should produce similar outputs to FP32
        XCTAssertGreaterThan(bleu, 0.7, "INT8 BLEU vs FP32 should be high (got \(bleu))")
    }
    
    /// **RED:** Test multiple prompts regression
    ///
    /// WHAT: Run quality metrics on all test fixtures
    /// WHY: Validates consistency across different input patterns
    /// HOW: Iterate fixtures, collect perplexity deltas
    /// EXPECTED: All deltas ≤ 1%
    func testMultiplePromptsRegression() throws {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 64,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 128
        )
        
        var rng: any RandomNumberGenerator = SeededGenerator(seed: 2025)
        let fp32Weights = FloatModelWeights.random(config: config, rng: &rng)
        let int8Weights = fp32Weights.quantized()
        
        var maxDelta: Float = 0.0
        
        for fixture in fixtures {
            let fp32Runner = FloatReferenceRunner(weights: fp32Weights)
            let int8Runner = ModelRunner(weights: int8Weights)
            
            var fp32Logits: [Tensor<Float>] = []
            var int8Logits: [Tensor<Float>] = []
            
            for token in fixture.prompt {
                fp32Logits.append(fp32Runner.step(tokenId: token))
                int8Logits.append(int8Runner.step(tokenId: token))
            }
            
            let fp32PPL = try perplexity(logits: fp32Logits, targetTokens: fixture.reference)
            let int8PPL = try perplexity(logits: int8Logits, targetTokens: fixture.reference)
            
            let delta = abs(fp32PPL - int8PPL) / fp32PPL
            maxDelta = max(maxDelta, delta)
            
            print("[\(fixture.id)] FP32: \(fp32PPL), INT8: \(int8PPL), Delta: \(delta * 100)%")
        }
        
        XCTAssertLessThan(maxDelta, 0.01, "Max perplexity delta across all fixtures should be ≤1%")
    }

    // MARK: - CHA-108: TinyLlama INT4 vs INT8 Real-Model Regression

    // MARK: - CHA-109: Gemma 2B INT4 vs INT8 Real-Model Regression

    /// Asserts CHA-109's v0.2.0 DoD on a real Gemma 2B model: RTN INT4
    /// quantization (group=32) keeps perplexity within **6%** of the INT8
    /// baseline on the pinned `CHA-109-v1` WikiText-2 slice tokenized with
    /// the Gemma tokenizer.
    ///
    /// Skipped in CI — `Models/gemma-2b-int8.tbf` is gitignored. To run
    /// locally:
    ///   1. `python3 Scripts/convert_model.py --input <hf_dir> --output Models/gemma-2b-int8.tbf --quantize int8 --auto-config`
    ///   2. `python3 Scripts/pretokenize_wikitext.py --model gemma`
    ///   3. `swift test --filter testGemmaINT4VsINT8Perplexity`
    func testGemmaINT4VsINT8Perplexity() throws {
        let modelPath = "Models/gemma-2b-int8.tbf"
        let weightsINT8: ModelWeights
        do {
            weightsINT8 = try ModelLoader.load(from: modelPath)
        } catch {
            throw XCTSkip("Gemma 2B .tbf not available at \(modelPath) — convert first with Scripts/convert_model.py")
        }

        let sliceURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("wikitext2_gemma_slice.json")
        guard FileManager.default.fileExists(atPath: sliceURL.path) else {
            throw XCTSkip("Gemma wikitext2 slice missing at \(sliceURL.path) — regenerate with: python3 Scripts/pretokenize_wikitext.py --model gemma")
        }
        let slice = try PerplexitySlice.load(from: sliceURL)
        XCTAssertEqual(slice.seed, "CHA-109-v1",
                       "Slice seed drifted — regenerate with: python3 Scripts/pretokenize_wikitext.py --model gemma")
        XCTAssertGreaterThanOrEqual(slice.tokens.count, 32,
                                    "Need ≥32 tokens for a meaningful perplexity estimate")

        if TinyBrainBackend.metalBackend == nil, MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let resultINT8 = try PerplexityHarness.computePerplexity(weights: weightsINT8, slice: slice)
        let weightsINT4 = PerplexityHarness.convertToINT4(weightsINT8, groupSize: 32)
        let resultINT4 = try PerplexityHarness.computePerplexity(weights: weightsINT4, slice: slice)

        let pplINT8 = resultINT8.perplexity
        let pplINT4 = resultINT4.perplexity
        let delta = abs(pplINT4 - pplINT8) / pplINT8

        print("""
        🧪 CHA-109 Gemma 2B INT4 vs INT8 perplexity
           slice: \(slice.source) (\(slice.tokens.count) tokens, seed=\(slice.seed))
           INT8: ppl=\(pplINT8) over \(resultINT8.numPredictions) preds in \(String(format: "%.2fs", resultINT8.elapsedSeconds))
           INT4: ppl=\(pplINT4) over \(resultINT4.numPredictions) preds in \(String(format: "%.2fs", resultINT4.elapsedSeconds))
           Δ: \(String(format: "%+.3f%%", delta * 100))
        """)

        XCTAssertGreaterThan(pplINT8, 0, "INT8 perplexity must be positive")
        XCTAssertGreaterThan(pplINT4, 0, "INT4 perplexity must be positive")
        XCTAssertLessThanOrEqual(delta, 0.06,
            "INT4 perplexity must stay within 6% of INT8 baseline per CHA-104 v0.2.0 DoD (got \(String(format: "%.3f%%", delta * 100)))")
    }



    /// Asserts CHA-104's v0.2.0 DoD on a real model: RTN INT4 quantization
    /// (group=32) keeps perplexity within **6 %** of the INT8 baseline on
    /// the pinned `CHA-108-v1` WikiText-2 slice.
    ///
    /// v0.2.1 restores the 1 % bound via GPTQ/AWQ calibration in
    /// [CHA-156](/CHA/issues/CHA-156); this guard ratchets us forward as
    /// that work lands.
    ///
    /// Currently skipped in CI because the 1.2 GB TinyLlama `.tbf` is
    /// gitignored. When the model is available, the test runs the harness
    /// end-to-end and asserts the 6 % bound; drift surfaces as a regression.
    ///
    /// The pinned slice is 65 tokens / 64 predictions. The scalar per-head
    /// attention loop in `ModelRunner.attention` drops throughput below
    /// 0.1 tok/s on M-series once the KV cache grows past ~100 positions,
    /// so the slice length is gated on that path moving to Metal/Accelerate.
    func testTinyLlamaINT4VsINT8Perplexity() throws {
        let modelPath = "Models/tinyllama-1.1b-int8.tbf"
        let weightsINT8: ModelWeights
        do {
            weightsINT8 = try ModelLoader.load(from: modelPath)
        } catch {
            throw XCTSkip("TinyLlama .tbf not available at \(modelPath)")
        }

        let sliceURL = URL(fileURLWithPath: #file)
            .deletingLastPathComponent()
            .appendingPathComponent("Fixtures")
            .appendingPathComponent("wikitext2_slice.json")
        guard FileManager.default.fileExists(atPath: sliceURL.path) else {
            throw XCTSkip("Pinned WikiText-2 slice missing at \(sliceURL.path)")
        }
        let slice = try PerplexitySlice.load(from: sliceURL)
        XCTAssertEqual(slice.seed, "CHA-108-v1", "Slice seed drifted — regenerate fixture with Scripts/pretokenize_wikitext.py")
        XCTAssertGreaterThanOrEqual(slice.tokens.count, 32, "Need ≥32 tokens for a meaningful perplexity estimate")

        if TinyBrainBackend.metalBackend == nil, MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let resultINT8 = try PerplexityHarness.computePerplexity(weights: weightsINT8, slice: slice)
        let weightsINT4 = PerplexityHarness.convertToINT4(weightsINT8, groupSize: 32)
        let resultINT4 = try PerplexityHarness.computePerplexity(weights: weightsINT4, slice: slice)

        let pplINT8 = resultINT8.perplexity
        let pplINT4 = resultINT4.perplexity
        let delta = abs(pplINT4 - pplINT8) / pplINT8

        print("""
        🧪 CHA-108 TinyLlama INT4 vs INT8 perplexity
           slice: \(slice.source) (\(slice.tokens.count) tokens, seed=\(slice.seed))
           INT8: ppl=\(pplINT8) over \(resultINT8.numPredictions) preds in \(String(format: "%.2fs", resultINT8.elapsedSeconds))
           INT4: ppl=\(pplINT4) over \(resultINT4.numPredictions) preds in \(String(format: "%.2fs", resultINT4.elapsedSeconds))
           Δ: \(String(format: "%+.3f%%", delta * 100))
        """)

        XCTAssertGreaterThan(pplINT8, 0, "INT8 perplexity must be positive")
        XCTAssertGreaterThan(pplINT4, 0, "INT4 perplexity must be positive")
        XCTAssertLessThanOrEqual(delta, 0.06,
            "INT4 perplexity must stay within 6% of INT8 baseline per CHA-104 v0.2.0 DoD (got \(String(format: "%.3f%%", delta * 100)))")
    }
}

// MARK: - Helper: FloatReferenceRunner (from ModelRunnerQuantizationTests)

private struct FloatLinearLayer {
    let weights: Tensor<Float>
    let bias: Tensor<Float>?
    
    func apply(toRow input: Tensor<Float>) -> Tensor<Float> {
        var output = input.matmul(weights)
        if let bias = bias {
            var data = output.data
            let cols = output.shape.dimensions[1]
            let rows = output.shape.dimensions[0]
            for row in 0..<rows {
                for col in 0..<cols {
                    data[row * cols + col] += bias[col]
                }
            }
            output = Tensor(shape: output.shape, data: data)
        }
        return output
    }
}

private struct FloatAttentionWeights {
    let query: FloatLinearLayer
    let key: FloatLinearLayer
    let value: FloatLinearLayer
    let output: FloatLinearLayer
}

private struct FloatFeedForwardWeights {
    let up: FloatLinearLayer
    let down: FloatLinearLayer
}

private struct FloatTransformerLayer {
    let attention: FloatAttentionWeights
    let feedForward: FloatFeedForwardWeights
}

private struct FloatModelWeights {
    let config: ModelConfig
    let tokenEmbeddings: Tensor<Float>
    let layers: [FloatTransformerLayer]
    let output: FloatLinearLayer
    
    static func random(config: ModelConfig, rng: inout any RandomNumberGenerator) -> FloatModelWeights {
        func makeWeights(rows: Int, cols: Int) -> Tensor<Float> {
            Tensor<Float>.random(shape: TensorShape(rows, cols), mean: 0, std: 0.02, using: &rng)
        }
        
        func makeBias(_ count: Int) -> Tensor<Float> {
            Tensor<Float>.random(shape: TensorShape(count), mean: 0, std: 0.02, using: &rng)
        }
        
        func linear(outputDim: Int) -> FloatLinearLayer {
            FloatLinearLayer(weights: makeWeights(rows: config.hiddenDim, cols: outputDim), bias: makeBias(outputDim))
        }
        
        var layers: [FloatTransformerLayer] = []
        for _ in 0..<config.numLayers {
            let attention = FloatAttentionWeights(
                query: linear(outputDim: config.hiddenDim),
                key: linear(outputDim: config.hiddenDim),
                value: linear(outputDim: config.hiddenDim),
                output: linear(outputDim: config.hiddenDim)
            )
            
            let ffnHidden = config.hiddenDim * 4
            let feedForward = FloatFeedForwardWeights(
                up: linear(outputDim: ffnHidden),
                down: FloatLinearLayer(weights: makeWeights(rows: ffnHidden, cols: config.hiddenDim), bias: makeBias(config.hiddenDim))
            )
            layers.append(FloatTransformerLayer(attention: attention, feedForward: feedForward))
        }
        
        let embeddings = Tensor<Float>.random(shape: TensorShape(config.vocabSize, config.hiddenDim), mean: 0, std: 0.02, using: &rng)
        let output = FloatLinearLayer(weights: makeWeights(rows: config.hiddenDim, cols: config.vocabSize), bias: Tensor<Float>.zeros(shape: TensorShape(config.vocabSize)))
        
        return FloatModelWeights(config: config, tokenEmbeddings: embeddings, layers: layers, output: output)
    }
    
    func quantized() -> ModelWeights {
        func quantize(_ layer: FloatLinearLayer) -> LinearLayerWeights {
            LinearLayerWeights(floatWeights: layer.weights, bias: layer.bias)
        }
        
        let attentionLayers = layers.map { layer in
            TransformerLayerWeights(
                attention: AttentionProjectionWeights(
                    query: quantize(layer.attention.query),
                    key: quantize(layer.attention.key),
                    value: quantize(layer.attention.value),
                    output: quantize(layer.attention.output)
                ),
                feedForward: FeedForwardWeights(
                    up: quantize(layer.feedForward.up),
                    down: quantize(layer.feedForward.down)
                )
            )
        }
        
        return ModelWeights(config: config, tokenEmbeddings: tokenEmbeddings, layers: attentionLayers, output: LinearLayerWeights(floatWeights: output.weights, bias: output.bias))
    }
}

private final class FloatReferenceRunner {
    private let weights: FloatModelWeights
    private let kvCache: KVCache
    private var position: Int = 0
    
    init(weights: FloatModelWeights) {
        self.weights = weights
        self.kvCache = KVCache(numLayers: weights.config.numLayers, hiddenDim: weights.config.hiddenDim, maxTokens: weights.config.maxSeqLen, pageSize: 16)
    }
    
    func step(tokenId: Int) -> Tensor<Float> {
        var hiddenRow = weights.tokenEmbeddings.row(tokenId).asRowMatrix()
        
        for (layerIndex, layer) in weights.layers.enumerated() {
            hiddenRow = applyLayer(hiddenRow, layer: layer, layerIndex: layerIndex)
        }
        
        let logits = weights.output.apply(toRow: hiddenRow).squeezedRowVector()
        position += 1
        return logits
    }
    
    private func applyLayer(_ hidden: Tensor<Float>, layer: FloatTransformerLayer, layerIndex: Int) -> Tensor<Float> {
        let attnOut = attention(hidden, layer: layer.attention, layerIndex: layerIndex)
        let residual = hidden + attnOut
        let ffnUp = layer.feedForward.up.apply(toRow: residual).gelu()
        let ffnDown = layer.feedForward.down.apply(toRow: ffnUp)
        return residual + ffnDown
    }
    
    private func attention(_ hidden: Tensor<Float>, layer: FloatAttentionWeights, layerIndex: Int) -> Tensor<Float> {
        let query = layer.query.apply(toRow: hidden)
        let keyVec = layer.key.apply(toRow: hidden).squeezedRowVector()
        let valueVec = layer.value.apply(toRow: hidden).squeezedRowVector()
        
        kvCache.append(layer: layerIndex, key: keyVec, value: valueVec, position: position)
        
        let seqLen = position + 1
        let keys = kvCache.getKeys(layer: layerIndex, range: 0..<seqLen)
        let values = kvCache.getValues(layer: layerIndex, range: 0..<seqLen)
        let scale = 1.0 / sqrt(max(1.0, Float(weights.config.hiddenDim) / Float(weights.config.numHeads)))
        let scores = (query.matmul(keys.transpose())) * scale
        let weightsTensor = scores.softmax()
        let context = weightsTensor.matmul(values)
        return layer.output.apply(toRow: context)
    }
}

