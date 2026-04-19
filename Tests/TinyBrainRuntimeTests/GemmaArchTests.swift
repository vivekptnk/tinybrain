import XCTest
@testable import TinyBrainRuntime
import Foundation

/// CHA-109 — runtime-visible Gemma divergences from the default LLaMA path.
///
/// Covers:
///   1. `ModelConfig` decodes `architecture` from the TBF header and
///      defaults to `"llama"` when the field is absent (backward compat).
///   2. `Tensor.rmsNormWithOffset` applies `(1 + w)` scale, not `w`.
///   3. `ModelRunner.step()` takes different code paths for `"llama"` vs
///      `"gemma"` on a toy model with known weights — same inputs produce
///      measurably different logits when arch is flipped, confirming the
///      embed-scale + offset-RMSNorm changes are actually being applied.
final class GemmaArchTests: XCTestCase {

    // MARK: - ModelConfig decoding

    func testModelConfigDecodesArchitectureWhenPresent() throws {
        let json = """
        {
          "numLayers": 2, "hiddenDim": 8, "numHeads": 2,
          "vocabSize": 16, "maxSeqLen": 32, "intermediateDim": 16,
          "architecture": "gemma"
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ModelConfig.self, from: json)
        XCTAssertEqual(config.architecture, "gemma")
        XCTAssertTrue(config.isGemmaStyle)
    }

    func testModelConfigDefaultsArchitectureToLlamaForLegacyTBF() throws {
        // Older .tbf files written before CHA-109 omit the `architecture`
        // key. Decoding must still succeed with `"llama"` semantics.
        let json = """
        {
          "numLayers": 2, "hiddenDim": 8, "numHeads": 2,
          "vocabSize": 16, "maxSeqLen": 32, "intermediateDim": 16
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ModelConfig.self, from: json)
        XCTAssertEqual(config.architecture, "llama")
        XCTAssertFalse(config.isGemmaStyle)
    }

    func testModelConfigExplicitInitDefaultsToLlama() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 4, numHeads: 1, vocabSize: 8)
        XCTAssertEqual(config.architecture, "llama")
        XCTAssertFalse(config.isGemmaStyle)
    }

    // MARK: - rmsNormWithOffset

    func testRmsNormWithOffsetAppliesOnePlusWeightScale() {
        // For input x = [1, 1, 1, 1]: rms = 1, so (x / rms) = x.
        // With weight = [0, 0, 0, 0], standard rmsNorm → all zeros.
        // With rmsNormWithOffset → (1 + 0) * 1 = 1 everywhere.
        let x = Tensor<Float>(shape: TensorShape(1, 4), data: [1, 1, 1, 1])
        let zeroWeight = Tensor<Float>(shape: TensorShape(4), data: [0, 0, 0, 0])

        let standard = x.rmsNorm(weight: zeroWeight)
        let gemmaStyle = x.rmsNormWithOffset(weight: zeroWeight)

        for i in 0..<4 {
            XCTAssertEqual(standard.data[i], 0.0, accuracy: 1e-4,
                           "Standard RMSNorm with zero weight must zero out output")
            XCTAssertEqual(gemmaStyle.data[i], 1.0, accuracy: 1e-4,
                           "Gemma-style RMSNorm with zero weight must preserve input (1+0=1)")
        }
    }

    func testRmsNormWithOffsetMatchesManualFormula() {
        // Arbitrary input + non-trivial weights: compare against the
        // reference `(x / rms(x)) * (1 + w)` formula directly.
        let x = Tensor<Float>(shape: TensorShape(1, 4), data: [2.0, -1.0, 3.0, 0.5])
        let w = Tensor<Float>(shape: TensorShape(4), data: [0.1, -0.2, 0.05, 0.3])
        let eps: Float = 1e-5

        let meanSquare: Float = (4.0 + 1.0 + 9.0 + 0.25) / 4.0  // = 3.5625
        let invRMS: Float = 1.0 / sqrt(meanSquare + eps)

        let result = x.rmsNormWithOffset(weight: w, epsilon: eps)

        let expected: [Float] = [
             2.0  * invRMS * (1.0 + 0.1),
            -1.0  * invRMS * (1.0 - 0.2),
             3.0  * invRMS * (1.0 + 0.05),
             0.5  * invRMS * (1.0 + 0.3),
        ]

        for i in 0..<4 {
            XCTAssertEqual(result.data[i], expected[i], accuracy: 1e-4,
                           "rmsNormWithOffset mismatch at index \(i)")
        }
    }

    // MARK: - ModelRunner arch branching

    func testModelRunnerProducesDifferentLogitsForGemmaVsLlama() {
        // Same weights + same input token → different logits when arch
        // flag flips, because the Gemma path applies `sqrt(hiddenDim)`
        // embed scale AND `(1 + w)` RMSNorm scaling.
        let llamaConfig = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4,
                                      vocabSize: 32, maxSeqLen: 32)
        let llamaWeights = ModelWeights.makeToyModel(config: llamaConfig, seed: 7)

        // Reuse the same tensors but rewrap with a gemma config so both
        // runners see identical weight values.
        let gemmaConfig = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4,
                                      vocabSize: 32, maxSeqLen: 32,
                                      architecture: "gemma")
        let gemmaWeights = ModelWeights(
            config: gemmaConfig,
            tokenEmbeddings: llamaWeights.tokenEmbeddings,
            layers: llamaWeights.layers,
            output: llamaWeights.output,
            finalNormWeights: llamaWeights.finalNormWeights
        )

        let llamaRunner = ModelRunner(weights: llamaWeights)
        let gemmaRunner = ModelRunner(weights: gemmaWeights)

        let llamaLogits = llamaRunner.step(tokenId: 3).data
        let gemmaLogits = gemmaRunner.step(tokenId: 3).data

        XCTAssertEqual(llamaLogits.count, gemmaLogits.count)

        var maxAbsDelta: Float = 0
        for i in 0..<llamaLogits.count {
            maxAbsDelta = max(maxAbsDelta, abs(llamaLogits[i] - gemmaLogits[i]))
        }
        XCTAssertGreaterThan(maxAbsDelta, 1e-3,
                             "Gemma arch path must measurably change logits vs LLaMA path on identical weights")
    }

    func testGemmaRunnerIsDeterministic() {
        // Same weights + same input on two fresh runners → identical
        // logits, so the Gemma code path is pure and safe to regression-test.
        let config = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4,
                                 vocabSize: 32, maxSeqLen: 32,
                                 architecture: "gemma")
        let weights = ModelWeights.makeToyModel(config: config, seed: 11)

        let r1 = ModelRunner(weights: weights)
        let r2 = ModelRunner(weights: weights)

        let l1 = r1.step(tokenId: 5).data
        let l2 = r2.step(tokenId: 5).data

        XCTAssertEqual(l1.count, l2.count)
        for i in 0..<l1.count {
            XCTAssertEqual(l1[i], l2[i], accuracy: 1e-6,
                           "Gemma step must be deterministic at index \(i)")
        }
    }
}
