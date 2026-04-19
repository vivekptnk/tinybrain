import XCTest
@testable import TinyBrainRuntime
import Foundation

/// Fast, model-free coverage for the CHA-108 perplexity harness.
///
/// These tests exercise `PerplexityHarness.convertToINT4` and
/// `PerplexityHarness.computePerplexity` on a toy `ModelWeights` so the
/// shared infrastructure has CI coverage even when the 1.2 GB TinyLlama
/// `.tbf` is absent (which is the common case).
final class PerplexityHarnessTests: XCTestCase {

    // MARK: - PerplexitySlice

    func testPerplexitySliceRoundTripsJSON() throws {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ppl_slice_\(UUID().uuidString).json")
        let payload = """
        {
          "source": "test",
          "tokenizer": "toy",
          "bos_token_id": 1,
          "seed": "unit-test-v1",
          "num_tokens": 4,
          "tokens": [1, 2, 3, 4],
          "notes": "roundtrip"
        }
        """
        try payload.write(to: tmp, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: tmp) }

        let slice = try PerplexitySlice.load(from: tmp)
        XCTAssertEqual(slice.source, "test")
        XCTAssertEqual(slice.tokenizer, "toy")
        XCTAssertEqual(slice.bosTokenId, 1)
        XCTAssertEqual(slice.seed, "unit-test-v1")
        XCTAssertEqual(slice.numTokens, 4)
        XCTAssertEqual(slice.tokens, [1, 2, 3, 4])
        XCTAssertEqual(slice.notes, "roundtrip")
    }

    // MARK: - convertToINT4

    func testConvertToINT4PreservesStructureOnToyModel() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 100, maxSeqLen: 64)
        let int8Weights = ModelWeights.makeToyModel(config: config, seed: 1)

        let int4Weights = PerplexityHarness.convertToINT4(int8Weights, groupSize: 32)

        // Config and layer count untouched.
        XCTAssertEqual(int4Weights.config.numLayers, int8Weights.config.numLayers)
        XCTAssertEqual(int4Weights.config.hiddenDim, int8Weights.config.hiddenDim)
        XCTAssertEqual(int4Weights.layers.count, int8Weights.layers.count)
        XCTAssertEqual(int4Weights.tokenEmbeddings.shape, int8Weights.tokenEmbeddings.shape,
                       "Embeddings stay in their original FP32 form")
        XCTAssertEqual(int4Weights.output.weights.shape, int8Weights.output.weights.shape)

        // Every linear weight should now be INT4.
        XCTAssertEqual(int4Weights.output.weights.precision, .int4,
                       "Output projection must be INT4 after conversion")
        for (i, layer) in int4Weights.layers.enumerated() {
            XCTAssertEqual(layer.attention.query.weights.precision, .int4, "layer \(i) Q")
            XCTAssertEqual(layer.attention.key.weights.precision, .int4, "layer \(i) K")
            XCTAssertEqual(layer.attention.value.weights.precision, .int4, "layer \(i) V")
            XCTAssertEqual(layer.attention.output.weights.precision, .int4, "layer \(i) O")
            XCTAssertEqual(layer.feedForward.up.weights.precision, .int4, "layer \(i) FFN up")
            XCTAssertEqual(layer.feedForward.down.weights.precision, .int4, "layer \(i) FFN down")
        }
    }

    func testConvertToINT4IsDeterministic() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 100, maxSeqLen: 64)
        let int8Weights = ModelWeights.makeToyModel(config: config, seed: 7)

        let a = PerplexityHarness.convertToINT4(int8Weights, groupSize: 64)
        let b = PerplexityHarness.convertToINT4(int8Weights, groupSize: 64)

        XCTAssertEqual(a.output.weights.data, b.output.weights.data,
                       "Same input weights + group size must produce identical INT4 payloads")
        XCTAssertEqual(a.output.weights.scales, b.output.weights.scales,
                       "Per-group scales are deterministic too")
    }

    // MARK: - computePerplexity

    private func makeSlice(tokens: [Int], seed: String = "harness-unit-v1") -> PerplexitySlice {
        // PerplexitySlice's auto-synthesised init is internal; we build one
        // via JSON round-trip so the helper stays on the public surface.
        let json = """
        {
          "source": "unit test",
          "tokenizer": "toy",
          "bos_token_id": \(tokens.first ?? 1),
          "seed": "\(seed)",
          "num_tokens": \(tokens.count),
          "tokens": \(tokens),
          "notes": "synthetic"
        }
        """
        return try! JSONDecoder().decode(PerplexitySlice.self, from: Data(json.utf8))
    }

    func testComputePerplexityIsPositiveAndDeterministic() throws {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 100, maxSeqLen: 64)
        let weights = ModelWeights.makeToyModel(config: config, seed: 2025)
        let slice = makeSlice(tokens: [1, 5, 12, 34, 7, 18])

        let first = try PerplexityHarness.computePerplexity(weights: weights, slice: slice)
        let second = try PerplexityHarness.computePerplexity(weights: weights, slice: slice)

        XCTAssertEqual(first.numPredictions, slice.tokens.count - 1)
        XCTAssertGreaterThan(first.perplexity, 0)
        XCTAssertTrue(first.perplexity.isFinite, "Perplexity must be finite, got \(first.perplexity)")
        XCTAssertEqual(first.perplexity, second.perplexity, accuracy: 1e-4,
                       "Same weights + same slice = same perplexity")
    }

    func testComputePerplexityApproachesVocabSizeOnRandomToyModel() throws {
        // A freshly-initialised toy model assigns roughly uniform probability
        // over the vocabulary, so next-token perplexity should sit near the
        // vocab size (≈10× above, ≈2× below). This catches gross breakage in
        // the log-softmax path (e.g. wrong axis, missing log, double `exp`).
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 50, maxSeqLen: 64)
        let weights = ModelWeights.makeToyModel(config: config, seed: 11)
        let slice = makeSlice(tokens: Array(1...20))

        let result = try PerplexityHarness.computePerplexity(weights: weights, slice: slice)

        XCTAssertGreaterThan(result.perplexity, Float(config.vocabSize) / 10,
                             "ppl=\(result.perplexity) far below vocab=\(config.vocabSize) suggests a broken log-softmax")
        XCTAssertLessThan(result.perplexity, Float(config.vocabSize) * 10,
                          "ppl=\(result.perplexity) far above vocab=\(config.vocabSize) suggests a broken log-softmax")
    }
}
