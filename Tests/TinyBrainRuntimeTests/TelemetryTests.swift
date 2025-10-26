import XCTest
@testable import TinyBrainRuntime

final class TelemetryTests: XCTestCase {
    func testSampleDetailedReturnsCorrectProbabilityAndEntropy_TemperatureOnly() {
        // Logits over 4 tokens
        let logits = Tensor<Float>(shape: TensorShape(4), data: [0.0, 1.0, 0.0, 2.0])
        var config = SamplerConfig(temperature: 0.5, seed: 123)
        let history: [Int] = []

        let result = Sampler.sampleDetailed(logits: logits, config: &config, history: history)

        // Recompute final probabilities according to the same path:
        // 1) no repetition penalty, 2) no top-k/p, 3) temperature scaling, 4) softmax
        let scaled = logits.data.map { $0 / 0.5 }
        let scaledTensor = Tensor<Float>(shape: logits.shape, data: scaled)
        let finalProbs = scaledTensor.softmax().data

        // Probability should match probability of selected token in final distribution
        XCTAssertEqual(result.probability, finalProbs[result.tokenId], accuracy: 1e-6)

        // Entropy should match -Σ p log p of the final distribution
        var expectedEntropy: Float = 0
        for p in finalProbs where p > 0 {
            expectedEntropy -= p * log(p)
        }
        XCTAssertEqual(result.entropy, expectedEntropy, accuracy: 1e-6)
    }

    func testSampleDetailedRespectsTopKFiltering() {
        // Construct logits where the top-2 are indices 3 and 1
        let logits = Tensor<Float>(shape: TensorShape(5), data: [0.1, 0.8, 0.2, 1.0, 0.3])
        var config = SamplerConfig(temperature: 1.0, topK: 2, seed: 42)
        let history: [Int] = []

        let result = Sampler.sampleDetailed(logits: logits, config: &config, history: history)

        // Manually apply top-k filter, then temperature (1.0), then softmax
        let sorted = logits.data.enumerated().sorted { $0.element > $1.element }
        let keep = Set(sorted.prefix(2).map { $0.offset })
        var filtered = logits.data
        for i in 0..<filtered.count { if !keep.contains(i) { filtered[i] = -Float.infinity } }
        let filteredTensor = Tensor<Float>(shape: logits.shape, data: filtered)
        let finalProbs = filteredTensor.softmax().data

        // Token must be one of the top-2 indices
        XCTAssertTrue(keep.contains(result.tokenId))

        // Probability should match the final distribution at that index
        XCTAssertEqual(result.probability, finalProbs[result.tokenId], accuracy: 1e-6)

        // Entropy sanity: should be finite and >= 0
        XCTAssertTrue(result.entropy.isFinite)
        XCTAssertGreaterThanOrEqual(result.entropy, 0)
    }
}


