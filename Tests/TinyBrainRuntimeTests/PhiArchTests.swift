import XCTest
@testable import TinyBrainRuntime
import Foundation

/// CHA-111 — runtime-visible Phi-2 divergences from the default LLaMA path.
///
/// Covers:
///   1. `ModelConfig` decodes `architecture: "phi"`, sets `isPhiStyle`, and
///      round-trips `partialRotaryFactor`.
///   2. `Tensor.layerNorm(weight:bias:)` applies correct affine transform.
///   3. `ModelRunner.step()` takes the phi code path — same weights produce
///      measurably different logits when the arch flag is flipped.
///   4. Partial RoPE: rotating fewer dims than headDim produces different
///      attention patterns than full RoPE.
///   5. TBF round-trip preserves `inputNormBias`, `finalNormBias`, and
///      attention/FFN biases.
final class PhiArchTests: XCTestCase {

    // MARK: - ModelConfig decoding

    func testModelConfigDecodesPhiArchitecture() throws {
        let json = """
        {
          "numLayers": 2, "hiddenDim": 8, "numHeads": 2,
          "vocabSize": 16, "maxSeqLen": 32, "intermediateDim": 16,
          "architecture": "phi", "partialRotaryFactor": 0.4
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ModelConfig.self, from: json)
        XCTAssertEqual(config.architecture, "phi")
        XCTAssertTrue(config.isPhiStyle)
        XCTAssertFalse(config.isGemmaStyle)
        XCTAssertEqual(config.partialRotaryFactor, 0.4, accuracy: 1e-6)
    }

    func testModelConfigPartialRotaryFactorDefaultsToOne() throws {
        let json = """
        {
          "numLayers": 2, "hiddenDim": 8, "numHeads": 2,
          "vocabSize": 16, "maxSeqLen": 32, "intermediateDim": 16
        }
        """.data(using: .utf8)!
        let config = try JSONDecoder().decode(ModelConfig.self, from: json)
        XCTAssertEqual(config.partialRotaryFactor, 1.0, accuracy: 1e-6)
        XCTAssertFalse(config.isPhiStyle)
    }

    func testModelConfigRotaryDimsComputedCorrectly() {
        // headDim = 80 (2560/32), partialRotaryFactor = 0.4 → rotaryDims = 32
        let config = ModelConfig(numLayers: 32, hiddenDim: 2560, numHeads: 32, vocabSize: 51200,
                                  partialRotaryFactor: 0.4)
        XCTAssertEqual(config.headDim, 80)
        XCTAssertEqual(config.rotaryDims, 32)
    }

    func testModelConfigFullRotaryDims() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4, vocabSize: 32)
        XCTAssertEqual(config.headDim, 4)
        XCTAssertEqual(config.rotaryDims, 4)  // 1.0 * 4 = 4
    }

    func testModelConfigCodablePreservesPartialRotaryFactor() throws {
        let config = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4, vocabSize: 32,
                                  architecture: "phi", partialRotaryFactor: 0.4)
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(ModelConfig.self, from: encoded)
        XCTAssertEqual(decoded.architecture, "phi")
        XCTAssertTrue(decoded.isPhiStyle)
        XCTAssertEqual(decoded.partialRotaryFactor, 0.4, accuracy: 1e-6)
    }

    // MARK: - LayerNorm with weight and bias

    func testLayerNormWithWeightAndBiasMatchesFormula() {
        // x = [2, -1, 3, 0.5]
        // mean = (2 - 1 + 3 + 0.5) / 4 = 4.5 / 4 = 1.125
        // var  = ((2-1.125)^2 + (-1-1.125)^2 + (3-1.125)^2 + (0.5-1.125)^2) / 4
        let x = Tensor<Float>(shape: TensorShape(1, 4), data: [2.0, -1.0, 3.0, 0.5])
        let w = Tensor<Float>(shape: TensorShape(4), data: [0.5, 1.0, 2.0, 0.25])
        let b = Tensor<Float>(shape: TensorShape(4), data: [0.1, -0.1, 0.2, 0.0])
        let eps: Float = 1e-5

        let result = x.layerNorm(weight: w, bias: b, epsilon: eps)

        // Reference implementation
        let vals: [Float] = [2.0, -1.0, 3.0, 0.5]
        let mean = vals.reduce(0, +) / 4.0
        let variance = vals.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 4.0
        let invStd = 1.0 / sqrt(variance + eps)
        let wArr: [Float] = [0.5, 1.0, 2.0, 0.25]
        let bArr: [Float] = [0.1, -0.1, 0.2, 0.0]
        let expected = vals.enumerated().map { i, v in
            (v - mean) * invStd * wArr[i] + bArr[i]
        }

        XCTAssertEqual(result.data.count, 4)
        for i in 0..<4 {
            XCTAssertEqual(result.data[i], expected[i], accuracy: 1e-4,
                           "layerNorm(weight:bias:) mismatch at index \(i)")
        }
    }

    func testLayerNormWithWeightNoBias() {
        let x = Tensor<Float>(shape: TensorShape(4), data: [1.0, 1.0, 1.0, 1.0])
        let w = Tensor<Float>(shape: TensorShape(4), data: [2.0, 2.0, 2.0, 2.0])
        // Constant input: mean=1, variance=0, normalized=0, scaled by weight=0 everywhere
        let result = x.layerNorm(weight: w)
        for i in 0..<4 {
            XCTAssertEqual(result.data[i], 0.0, accuracy: 1e-5)
        }
    }

    func testLayerNormBiasShiftApplied() {
        // Zero input after normalization → output should equal bias
        let x = Tensor<Float>(shape: TensorShape(4), data: [5.0, 5.0, 5.0, 5.0])
        let w = Tensor<Float>(shape: TensorShape(4), data: [1.0, 1.0, 1.0, 1.0])
        let b = Tensor<Float>(shape: TensorShape(4), data: [0.3, 0.6, -0.2, 1.0])
        let result = x.layerNorm(weight: w, bias: b)
        for i in 0..<4 {
            XCTAssertEqual(result.data[i], b.data[i], accuracy: 1e-5,
                           "Bias not applied correctly at index \(i)")
        }
    }

    // MARK: - Phi runner produces different logits than LLaMA

    func testPhiRunnerDiffersFromLlamaOnIdenticalWeights() {
        let llamaConfig = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4,
                                      vocabSize: 32, maxSeqLen: 32)
        let llamaWeights = ModelWeights.makeToyModel(config: llamaConfig, seed: 42)

        let phiConfig = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4,
                                    vocabSize: 32, maxSeqLen: 32,
                                    architecture: "phi", partialRotaryFactor: 0.5)
        let phiWeights = ModelWeights(
            config: phiConfig,
            tokenEmbeddings: llamaWeights.tokenEmbeddings,
            layers: llamaWeights.layers,
            output: llamaWeights.output,
            finalNormWeights: llamaWeights.finalNormWeights
        )

        let llamaRunner = ModelRunner(weights: llamaWeights)
        let phiRunner   = ModelRunner(weights: phiWeights)

        let llamaLogits = llamaRunner.step(tokenId: 5).data
        let phiLogits   = phiRunner.step(tokenId: 5).data

        XCTAssertEqual(llamaLogits.count, phiLogits.count)

        var maxDelta: Float = 0
        for i in 0..<llamaLogits.count {
            maxDelta = max(maxDelta, abs(llamaLogits[i] - phiLogits[i]))
        }
        XCTAssertGreaterThan(maxDelta, 1e-3,
            "Phi arch path must produce measurably different logits than LLaMA on identical weights")
    }

    func testPhiRunnerIsDeterministic() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 16, numHeads: 4,
                                  vocabSize: 32, maxSeqLen: 32,
                                  architecture: "phi", partialRotaryFactor: 0.5)
        let weights = ModelWeights.makeToyModel(config: config, seed: 17)

        let r1 = ModelRunner(weights: weights)
        let r2 = ModelRunner(weights: weights)

        let l1 = r1.step(tokenId: 3).data
        let l2 = r2.step(tokenId: 3).data

        XCTAssertEqual(l1.count, l2.count)
        for i in 0..<l1.count {
            XCTAssertEqual(l1[i], l2[i], accuracy: 1e-6,
                           "Phi step must be deterministic at index \(i)")
        }
    }

    // MARK: - Partial RoPE

    func testPartialRoPEProducesDifferentLogitsThanFullRoPE() {
        let fullConfig = ModelConfig(numLayers: 1, hiddenDim: 16, numHeads: 4,
                                      vocabSize: 32, architecture: "phi",
                                      partialRotaryFactor: 1.0)
        let partialConfig = ModelConfig(numLayers: 1, hiddenDim: 16, numHeads: 4,
                                         vocabSize: 32, architecture: "phi",
                                         partialRotaryFactor: 0.5)

        let seed: UInt64 = 99
        let baseWeights = ModelWeights.makeToyModel(config: fullConfig, seed: seed)

        let partialWeights = ModelWeights(
            config: partialConfig,
            tokenEmbeddings: baseWeights.tokenEmbeddings,
            layers: baseWeights.layers,
            output: baseWeights.output
        )

        let fullRunner    = ModelRunner(weights: baseWeights)
        let partialRunner = ModelRunner(weights: partialWeights)

        // Step twice to expose positional difference
        _ = fullRunner.step(tokenId: 1)
        _ = partialRunner.step(tokenId: 1)
        let fullLogits    = fullRunner.step(tokenId: 2).data
        let partialLogits = partialRunner.step(tokenId: 2).data

        var maxDelta: Float = 0
        for i in 0..<fullLogits.count {
            maxDelta = max(maxDelta, abs(fullLogits[i] - partialLogits[i]))
        }
        XCTAssertGreaterThan(maxDelta, 1e-4,
            "Partial RoPE (factor=0.5) must produce different logits than full RoPE (factor=1.0)")
    }

    // MARK: - TBF round-trip with phi biases

    func testTBFRoundTripPreservesPhiBiases() throws {
        let config = ModelConfig(numLayers: 2, hiddenDim: 8, numHeads: 2, vocabSize: 16,
                                  maxSeqLen: 32, intermediateDim: 16,
                                  architecture: "phi", partialRotaryFactor: 0.4)

        let normW = Tensor<Float>(shape: TensorShape(8), data: (0..<8).map { Float($0) * 0.1 })
        let normB = Tensor<Float>(shape: TensorShape(8), data: (0..<8).map { Float($0) * 0.05 })

        func makeLinear(inDim: Int, outDim: Int) -> LinearLayerWeights {
            LinearLayerWeights(
                floatWeights: Tensor<Float>.random(shape: TensorShape(inDim, outDim), mean: 0, std: 0.02),
                bias: Tensor<Float>.random(shape: TensorShape(outDim), mean: 0, std: 0.01)
            )
        }

        var layers: [TransformerLayerWeights] = []
        for _ in 0..<config.numLayers {
            let attn = AttentionProjectionWeights(
                query:  makeLinear(inDim: 8, outDim: 8),
                key:    makeLinear(inDim: 8, outDim: 8),
                value:  makeLinear(inDim: 8, outDim: 8),
                output: makeLinear(inDim: 8, outDim: 8)
            )
            let ff = FeedForwardWeights(
                up:   makeLinear(inDim: 8, outDim: 16),
                down: makeLinear(inDim: 16, outDim: 8)
            )
            layers.append(TransformerLayerWeights(
                attention: attn, feedForward: ff,
                inputNormWeights: normW, inputNormBias: normB
            ))
        }

        let finalNormBias = Tensor<Float>(shape: TensorShape(8), data: (0..<8).map { Float($0) * 0.02 })
        let original = ModelWeights(
            config: config,
            tokenEmbeddings: Tensor<Float>.random(shape: TensorShape(16, 8), mean: 0, std: 0.02),
            layers: layers,
            output: makeLinear(inDim: 8, outDim: 16),
            finalNormWeights: normW,
            finalNormBias: finalNormBias
        )

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let path = tempDir.appendingPathComponent("phi_test.tbf").path
        try original.save(to: path)
        let loaded = try ModelWeights.load(from: path)

        XCTAssertEqual(loaded.config.architecture, "phi")
        XCTAssertEqual(loaded.config.partialRotaryFactor, 0.4, accuracy: 1e-6)

        // Final norm bias round-trips
        XCTAssertNotNil(loaded.finalNormBias)
        if let lb = loaded.finalNormBias {
            for i in 0..<8 {
                XCTAssertEqual(lb.data[i], finalNormBias.data[i], accuracy: 1e-6,
                               "finalNormBias mismatch at index \(i)")
            }
        }

        // Layer inputNormBias round-trips
        XCTAssertNotNil(loaded.layers[0].inputNormBias)
        if let lb = loaded.layers[0].inputNormBias {
            for i in 0..<8 {
                XCTAssertEqual(lb.data[i], normB.data[i], accuracy: 1e-6,
                               "inputNormBias mismatch at index \(i)")
            }
        }

        // Attention query bias round-trips
        XCTAssertNotNil(loaded.layers[0].attention.query.bias)

        // Forward pass produces finite logits
        let runner = ModelRunner(weights: loaded)
        let logits = runner.step(tokenId: 0)
        XCTAssertEqual(logits.shape, TensorShape(config.vocabSize))
        XCTAssertTrue(logits.data.allSatisfy { $0.isFinite },
                      "Phi-style forward pass must produce finite logits")
    }
}
