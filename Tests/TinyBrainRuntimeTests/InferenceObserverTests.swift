/// Tests for InferenceObserver hooks (TB-010: X-Ray Mode)
///
/// Verifies that observer callbacks fire with correct data shapes
/// during inference, and that no callbacks fire when observer is nil.

import XCTest
@testable import TinyBrainRuntime

/// Mock observer that records all callbacks
final class MockInferenceObserver: InferenceObserver {
    var attentionCalls: [(layerIndex: Int, weights: [Float], position: Int)] = []
    var layerEntryCalls: [(layerIndex: Int, norm: Float, position: Int)] = []
    var logitsCalls: [(logits: [Float], position: Int)] = []
    var finalHiddenStateCalls: [(hiddenState: [Float], position: Int)] = []

    func didComputeAttention(layerIndex: Int, weights: [Float], position: Int) {
        attentionCalls.append((layerIndex, weights, position))
    }

    func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int) {
        layerEntryCalls.append((layerIndex, hiddenStateNorm, position))
    }

    func didComputeLogits(logits: [Float], position: Int) {
        logitsCalls.append((logits, position))
    }

    func didComputeFinalHiddenState(_ hiddenState: [Float], position: Int) {
        finalHiddenStateCalls.append((hiddenState, position))
    }
}

final class InferenceObserverTests: XCTestCase {

    func testObserverReceivesCallbacksDuringStep() {
        // Given: a toy model with 2 layers and an observer attached
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 50)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        let observer = MockInferenceObserver()
        runner.observer = observer

        // When: we run one inference step
        _ = runner.step(tokenId: 1)

        // Then: observer should have received callbacks for each layer
        XCTAssertEqual(observer.layerEntryCalls.count, 2, "Should fire didEnterLayer for each of 2 layers")
        XCTAssertEqual(observer.attentionCalls.count, 2, "Should fire didComputeAttention for each of 2 layers")
        XCTAssertEqual(observer.logitsCalls.count, 1, "Should fire didComputeLogits once per step")

        // Verify layer indices
        XCTAssertEqual(observer.layerEntryCalls[0].layerIndex, 0)
        XCTAssertEqual(observer.layerEntryCalls[1].layerIndex, 1)
        XCTAssertEqual(observer.attentionCalls[0].layerIndex, 0)
        XCTAssertEqual(observer.attentionCalls[1].layerIndex, 1)

        // Verify attention weights shape: numHeads * sequenceLength = 2 * 1 = 2 for first token
        XCTAssertEqual(observer.attentionCalls[0].weights.count, 2,
                       "First token: numHeads(2) * seqLen(1) attention weights")

        // Verify attention weights sum to ~numHeads (each head's softmax sums to 1.0)
        let weightSum = observer.attentionCalls[0].weights.reduce(0, +)
        XCTAssertEqual(weightSum, 2.0, accuracy: 1e-5, "Attention weights should sum to numHeads")

        // Verify logits shape matches vocab size
        XCTAssertEqual(observer.logitsCalls[0].logits.count, 50,
                       "Logits should have vocabSize elements")

        // Verify hidden state norms are positive
        for call in observer.layerEntryCalls {
            XCTAssertGreaterThan(call.norm, 0, "Hidden state norm should be positive")
        }
    }

    func testObserverPositionIncrementsAcrossSteps() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 1, vocabSize: 20)
        let weights = ModelWeights.makeToyModel(config: config, seed: 99)
        let runner = ModelRunner(weights: weights)
        let observer = MockInferenceObserver()
        runner.observer = observer

        // Run 3 steps
        _ = runner.step(tokenId: 0)
        _ = runner.step(tokenId: 1)
        _ = runner.step(tokenId: 2)

        // Position should increment: 0, 1, 2
        XCTAssertEqual(observer.logitsCalls.count, 3)
        XCTAssertEqual(observer.logitsCalls[0].position, 0)
        XCTAssertEqual(observer.logitsCalls[1].position, 1)
        XCTAssertEqual(observer.logitsCalls[2].position, 2)

        // Attention weights should grow: [1], [2], [3]
        XCTAssertEqual(observer.attentionCalls[0].weights.count, 1)
        XCTAssertEqual(observer.attentionCalls[1].weights.count, 2)
        XCTAssertEqual(observer.attentionCalls[2].weights.count, 3)
    }

    func testNoCallbacksWhenObserverIsNil() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 50)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        // observer is nil by default

        // This should not crash or produce side effects
        let logits = runner.step(tokenId: 1)
        XCTAssertEqual(logits.shape.dimensions, [50], "Inference should work normally without observer")
    }

    func testObserverCanBeDetachedMidInference() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 1, vocabSize: 20)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        let observer = MockInferenceObserver()

        // Attach, run one step
        runner.observer = observer
        _ = runner.step(tokenId: 0)
        XCTAssertEqual(observer.logitsCalls.count, 1)

        // Detach, run another step
        runner.observer = nil
        _ = runner.step(tokenId: 1)
        XCTAssertEqual(observer.logitsCalls.count, 1, "Should not receive callbacks after detach")
    }

    // MARK: - Final Hidden State Hook Tests

    func testFinalHiddenStateHookFiresDuringStep() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 50)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        let observer = MockInferenceObserver()
        runner.observer = observer

        _ = runner.step(tokenId: 1)

        // Hook should fire once per step
        XCTAssertEqual(observer.finalHiddenStateCalls.count, 1,
                       "didComputeFinalHiddenState should fire once per step")

        // Shape should be [hiddenDim]
        XCTAssertEqual(observer.finalHiddenStateCalls[0].hiddenState.count, 64,
                       "Hidden state should have hiddenDim elements")

        // Position should match
        XCTAssertEqual(observer.finalHiddenStateCalls[0].position, 0)
    }

    func testFinalHiddenStatePositionIncrementsAcrossSteps() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 1, vocabSize: 20)
        let weights = ModelWeights.makeToyModel(config: config, seed: 99)
        let runner = ModelRunner(weights: weights)
        let observer = MockInferenceObserver()
        runner.observer = observer

        _ = runner.step(tokenId: 0)
        _ = runner.step(tokenId: 1)
        _ = runner.step(tokenId: 2)

        XCTAssertEqual(observer.finalHiddenStateCalls.count, 3)
        XCTAssertEqual(observer.finalHiddenStateCalls[0].position, 0)
        XCTAssertEqual(observer.finalHiddenStateCalls[1].position, 1)
        XCTAssertEqual(observer.finalHiddenStateCalls[2].position, 2)
    }

    // MARK: - extractEmbedding Tests

    func testExtractEmbeddingShape() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 50)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)

        let embedding = runner.extractEmbedding(for: [1, 2, 3])

        // Shape should be [1, hiddenDim]
        XCTAssertEqual(embedding.shape.dimensions, [1, 64],
                       "Embedding should have shape [1, hiddenDim]")
    }

    func testExtractEmbeddingResetsState() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 1, vocabSize: 20)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)

        // Run some steps first
        _ = runner.step(tokenId: 0)
        _ = runner.step(tokenId: 1)
        XCTAssertEqual(runner.currentPosition, 2)

        // extractEmbedding should reset and produce a clean result
        let embedding = runner.extractEmbedding(for: [5])
        XCTAssertEqual(embedding.shape.dimensions, [1, 32])
        XCTAssertEqual(runner.currentPosition, 1)
    }

    func testExtractEmbeddingDeterministic() {
        let config = ModelConfig(numLayers: 1, hiddenDim: 32, numHeads: 1, vocabSize: 20)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)

        let runner1 = ModelRunner(weights: weights)
        let runner2 = ModelRunner(weights: weights)

        let emb1 = runner1.extractEmbedding(for: [1, 2, 3])
        let emb2 = runner2.extractEmbedding(for: [1, 2, 3])

        XCTAssertEqual(emb1.data, emb2.data, "Embeddings should be deterministic")
    }

    func testExtractEmbeddingDiffersForDifferentInputs() {
        let config = ModelConfig(numLayers: 2, hiddenDim: 64, numHeads: 2, vocabSize: 50)
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)

        let emb1 = runner.extractEmbedding(for: [1, 2, 3])
        let emb2 = runner.extractEmbedding(for: [4, 5, 6])

        XCTAssertNotEqual(emb1.data, emb2.data,
                          "Different inputs should produce different embeddings")
    }
}
