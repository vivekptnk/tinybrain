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

    func didComputeAttention(layerIndex: Int, weights: [Float], position: Int) {
        attentionCalls.append((layerIndex, weights, position))
    }

    func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int) {
        layerEntryCalls.append((layerIndex, hiddenStateNorm, position))
    }

    func didComputeLogits(logits: [Float], position: Int) {
        logitsCalls.append((logits, position))
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

        // Verify attention weights shape: should be [sequenceLength] = [1] for first token
        XCTAssertEqual(observer.attentionCalls[0].weights.count, 1,
                       "First token should have attention over 1 position")

        // Verify attention weights sum to ~1.0 (softmax output)
        let weightSum = observer.attentionCalls[0].weights.reduce(0, +)
        XCTAssertEqual(weightSum, 1.0, accuracy: 1e-5, "Attention weights should sum to 1.0")

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
}
