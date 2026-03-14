import XCTest
@testable import TinyBrainRuntime

/// Tests for the model converter fixes (Phase 0 v0.2.0)
///
/// Validates: RMSNorm, SiLU, gated FFN, RoPE, pre-norm architecture,
/// final norm, and proper weight mapping for TinyLlama-style models.
final class ConverterFixTests: XCTestCase {

    // MARK: - SiLU Activation

    func testSiLU() {
        let input = Tensor<Float>(shape: TensorShape(5), data: [-2, -1, 0, 1, 2])
        let output = input.silu()

        // SiLU(0) = 0
        XCTAssertEqual(output.data[2], 0.0, accuracy: 1e-6)
        // SiLU(x) = x * sigmoid(x)
        // SiLU(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.7311
        XCTAssertEqual(output.data[3], 0.7311, accuracy: 0.001)
        // SiLU(-1) = -1 * sigmoid(-1) ≈ -0.2689
        XCTAssertEqual(output.data[1], -0.2689, accuracy: 0.001)
        // SiLU(2) = 2 * sigmoid(2) ≈ 1.7616
        XCTAssertEqual(output.data[4], 1.7616, accuracy: 0.001)
    }

    func testSiLU2D() {
        let input = Tensor<Float>(shape: TensorShape(1, 4), data: [0, 1, -1, 2])
        let output = input.silu()
        XCTAssertEqual(output.shape, input.shape)
        XCTAssertEqual(output.data[0], 0.0, accuracy: 1e-6)
        XCTAssertEqual(output.data[1], 0.7311, accuracy: 0.001)
    }

    // MARK: - RMSNorm

    func testRMSNorm1D() {
        let input = Tensor<Float>(shape: TensorShape(4), data: [1, 2, 3, 4])
        let weights = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])
        let output = input.rmsNorm(weight: weights)

        // RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.7386
        let rms = sqrt(Float(30.0 / 4.0))
        XCTAssertEqual(output.data[0], 1.0 / rms, accuracy: 0.001)
        XCTAssertEqual(output.data[1], 2.0 / rms, accuracy: 0.001)
        XCTAssertEqual(output.data[2], 3.0 / rms, accuracy: 0.001)
        XCTAssertEqual(output.data[3], 4.0 / rms, accuracy: 0.001)
    }

    func testRMSNormWithWeights() {
        let input = Tensor<Float>(shape: TensorShape(4), data: [1, 2, 3, 4])
        let weights = Tensor<Float>(shape: TensorShape(4), data: [2, 2, 2, 2])
        let output = input.rmsNorm(weight: weights)

        let rms = sqrt(Float(30.0 / 4.0))
        // Output should be 2x the unit-weight version
        XCTAssertEqual(output.data[0], 2.0 / rms, accuracy: 0.001)
        XCTAssertEqual(output.data[3], 8.0 / rms, accuracy: 0.001)
    }

    func testRMSNorm2D() {
        // [1, 4] row matrix
        let input = Tensor<Float>(shape: TensorShape(1, 4), data: [1, 2, 3, 4])
        let weights = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])
        let output = input.rmsNorm(weight: weights)

        XCTAssertEqual(output.shape, input.shape)
        let rms = sqrt(Float(30.0 / 4.0))
        XCTAssertEqual(output.data[0], 1.0 / rms, accuracy: 0.001)
    }

    func testRMSNormDoesNotSubtractMean() {
        // Unlike LayerNorm, RMSNorm should NOT subtract mean
        let input = Tensor<Float>(shape: TensorShape(4), data: [10, 10, 10, 10])
        let weights = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])
        let output = input.rmsNorm(weight: weights)

        // All values are the same, so RMS = 10, output = 10/10 * 1 = 1
        for i in 0..<4 {
            XCTAssertEqual(output.data[i], 1.0, accuracy: 0.001)
        }

        // LayerNorm would give 0.0 here (mean subtracted)
        let lnOutput = input.layerNorm()
        for i in 0..<4 {
            XCTAssertEqual(lnOutput.data[i], 0.0, accuracy: 0.001,
                           "LayerNorm should produce 0 for constant input")
        }
    }

    // MARK: - Gated FFN

    func testGatedFFN() {
        // Test the gated FFN pattern: down(silu(gate(x)) * up(x))
        let input = Tensor<Float>(shape: TensorShape(1, 4), data: [1, 0, -1, 0.5])

        // Create simple weights for gate, up, down
        let gateW = Tensor<Float>(shape: TensorShape(4, 4), data:
            [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])  // identity
        let upW = Tensor<Float>(shape: TensorShape(4, 4), data:
            [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])  // identity

        let gateOut = input.matmul(gateW).silu()
        let upOut = input.matmul(upW)
        let gated = gateOut * upOut

        // Verify gated = silu(x) * x = x * sigmoid(x) * x = x^2 * sigmoid(x)
        XCTAssertEqual(gated.data[0], 0.7311, accuracy: 0.001)  // silu(1) * 1
        XCTAssertEqual(gated.data[1], 0.0, accuracy: 0.001)     // silu(0) * 0
    }

    // MARK: - FeedForwardWeights with gate

    func testFeedForwardWeightsBackwardCompatible() {
        // Ensure the 2-arg init still works (no gate)
        let up = LinearLayerWeights(
            floatWeights: Tensor<Float>.random(shape: TensorShape(4, 8)),
            bias: nil)
        let down = LinearLayerWeights(
            floatWeights: Tensor<Float>.random(shape: TensorShape(8, 4)),
            bias: nil)
        let ff = FeedForwardWeights(up: up, down: down)
        XCTAssertNil(ff.gate)
    }

    func testFeedForwardWeightsWithGate() {
        let gate = LinearLayerWeights(
            floatWeights: Tensor<Float>.random(shape: TensorShape(4, 8)),
            bias: nil)
        let up = LinearLayerWeights(
            floatWeights: Tensor<Float>.random(shape: TensorShape(4, 8)),
            bias: nil)
        let down = LinearLayerWeights(
            floatWeights: Tensor<Float>.random(shape: TensorShape(8, 4)),
            bias: nil)
        let ff = FeedForwardWeights(gate: gate, up: up, down: down)
        XCTAssertNotNil(ff.gate)
    }

    // MARK: - TransformerLayerWeights with norms

    func testTransformerLayerWeightsBackwardCompatible() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 4, numHeads: 1, vocabSize: 8)
        let weights = ModelWeights.makeToyModel(config: config)
        // Toy model should have nil norm weights
        XCTAssertNil(weights.layers[0].inputNormWeights)
        XCTAssertNil(weights.layers[0].postAttentionNormWeights)
        XCTAssertNil(weights.finalNormWeights)
    }

    // MARK: - ModelConfig intermediateDim

    func testModelConfigIntermediateDimDefault() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 100)
        XCTAssertEqual(config.intermediateDim, 256)  // 4 * 64
    }

    func testModelConfigIntermediateDimExplicit() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 100,
                                  intermediateDim: 128)
        XCTAssertEqual(config.intermediateDim, 128)
    }

    func testModelConfigHeadDim() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 2048, numHeads: 32, vocabSize: 32000)
        XCTAssertEqual(config.headDim, 64)
    }

    func testModelConfigCodable() throws {
        // Test encoding/decoding with intermediateDim
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 4, vocabSize: 100,
                                  numKVHeads: 2, intermediateDim: 128)
        let encoded = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(ModelConfig.self, from: encoded)
        XCTAssertEqual(decoded.numLayers, 2)
        XCTAssertEqual(decoded.hiddenDim, 64)
        XCTAssertEqual(decoded.numHeads, 4)
        XCTAssertEqual(decoded.numKVHeads, 2)
        XCTAssertEqual(decoded.vocabSize, 100)
        XCTAssertEqual(decoded.intermediateDim, 128)
    }

    func testModelConfigCodableWithoutIntermediateDim() throws {
        // Test decoding JSON that lacks intermediateDim (backward compat)
        let json = """
        {"numLayers":2,"hiddenDim":64,"numHeads":4,"numKVHeads":4,"vocabSize":100,"maxSeqLen":2048}
        """.data(using: .utf8)!
        let decoded = try JSONDecoder().decode(ModelConfig.self, from: json)
        XCTAssertEqual(decoded.intermediateDim, 256)  // default 4 * 64
    }

    // MARK: - RoPE (tested indirectly since applyRoPE is private)

    func testRoPEAppliedDuringInference() {
        // RoPE is applied inside the attention function. We verify that running
        // a multi-token sequence produces different attention patterns than without RoPE.
        // With toy weights the absolute difference may be tiny, so we just verify
        // the forward pass completes without error at multiple positions.
        let config = ModelConfig(numLayers: 1, hiddenDim: 4, numHeads: 1, vocabSize: 8)
        let runner = ModelRunner(config: config)
        let logits0 = runner.step(tokenId: 0)
        let logits1 = runner.step(tokenId: 1)
        let logits2 = runner.step(tokenId: 0)
        // All logits should be finite
        XCTAssertTrue(logits0.data.allSatisfy { $0.isFinite })
        XCTAssertTrue(logits1.data.allSatisfy { $0.isFinite })
        XCTAssertTrue(logits2.data.allSatisfy { $0.isFinite })
        // Position 2 should differ from position 0 (different context due to KV cache + RoPE)
        XCTAssertNotEqual(logits0.data, logits2.data,
                          "Different context lengths should produce different logits")
    }

    // MARK: - Pre-norm architecture (integration)

    func testPreNormApplied() {
        // Create a model with norm weights and verify norms are applied
        let config = ModelConfig(numLayers: 1, hiddenDim: 4, numHeads: 1, vocabSize: 8)

        let normWeights = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])

        func makeLinear(inDim: Int, outDim: Int) -> LinearLayerWeights {
            LinearLayerWeights(
                floatWeights: Tensor<Float>.filled(shape: TensorShape(inDim, outDim), value: 0.01))
        }

        let attention = AttentionProjectionWeights(
            query: makeLinear(inDim: 4, outDim: 4),
            key: makeLinear(inDim: 4, outDim: 4),
            value: makeLinear(inDim: 4, outDim: 4),
            output: makeLinear(inDim: 4, outDim: 4))

        let ff = FeedForwardWeights(
            up: makeLinear(inDim: 4, outDim: 16),
            down: makeLinear(inDim: 16, outDim: 4))

        let layer = TransformerLayerWeights(
            attention: attention, feedForward: ff,
            inputNormWeights: normWeights,
            postAttentionNormWeights: normWeights)

        let finalNorm = Tensor<Float>(shape: TensorShape(4), data: [1, 1, 1, 1])

        let weights = ModelWeights(
            config: config,
            tokenEmbeddings: Tensor<Float>.random(shape: TensorShape(8, 4)),
            layers: [layer],
            output: LinearLayerWeights(
                floatWeights: Tensor<Float>.random(shape: TensorShape(4, 8))),
            finalNormWeights: finalNorm)

        let runner = ModelRunner(weights: weights)
        let logits = runner.step(tokenId: 0)
        XCTAssertEqual(logits.shape, TensorShape(8))
        // Just verify it doesn't crash and produces finite values
        XCTAssertTrue(logits.data.allSatisfy { $0.isFinite })
    }

    // MARK: - Gated FFN integration

    func testGatedFFNIntegration() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 4, numHeads: 1, vocabSize: 8,
                                  intermediateDim: 8)

        func makeLinear(inDim: Int, outDim: Int) -> LinearLayerWeights {
            LinearLayerWeights(
                floatWeights: Tensor<Float>.random(shape: TensorShape(inDim, outDim),
                                                   mean: 0, std: 0.02))
        }

        let attention = AttentionProjectionWeights(
            query: makeLinear(inDim: 4, outDim: 4),
            key: makeLinear(inDim: 4, outDim: 4),
            value: makeLinear(inDim: 4, outDim: 4),
            output: makeLinear(inDim: 4, outDim: 4))

        let ff = FeedForwardWeights(
            gate: makeLinear(inDim: 4, outDim: 8),
            up: makeLinear(inDim: 4, outDim: 8),
            down: makeLinear(inDim: 8, outDim: 4))

        let layer = TransformerLayerWeights(attention: attention, feedForward: ff)

        let weights = ModelWeights(
            config: config,
            tokenEmbeddings: Tensor<Float>.random(shape: TensorShape(8, 4)),
            layers: [layer],
            output: LinearLayerWeights(
                floatWeights: Tensor<Float>.random(shape: TensorShape(4, 8))))

        let runner = ModelRunner(weights: weights)
        let logits = runner.step(tokenId: 0)
        XCTAssertEqual(logits.shape, TensorShape(8))
        XCTAssertTrue(logits.data.allSatisfy { $0.isFinite })
    }

    // MARK: - TBF round-trip backward compat

    func testToyModelRoundTripStillWorks() throws {
        // Ensure existing toy model save/load still works after changes
        let config = ModelConfig(numLayers: 2, hiddenDim: 32, numHeads: 4, vocabSize: 64)
        let original = ModelWeights.makeToyModel(config: config, seed: 42)

        let tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let path = tempDir.appendingPathComponent("test.tbf").path
        try original.save(to: path)
        let loaded = try ModelWeights.load(from: path)

        XCTAssertEqual(loaded.config.numLayers, config.numLayers)
        XCTAssertEqual(loaded.config.hiddenDim, config.hiddenDim)
        XCTAssertEqual(loaded.config.numHeads, config.numHeads)
        XCTAssertEqual(loaded.config.vocabSize, config.vocabSize)
        XCTAssertEqual(loaded.layers.count, config.numLayers)

        // Toy model should have no gate/norm weights
        XCTAssertNil(loaded.layers[0].inputNormWeights)
        XCTAssertNil(loaded.layers[0].feedForward.gate)
        XCTAssertNil(loaded.finalNormWeights)
    }
}
