import TinyBrainRuntime
import XCTest
@testable import TinyBrainRAG

final class ModelRunnerGeneratorTests: XCTestCase {
    func testGenerateStreamDoesNotYieldStopTokenWhenSampledImmediately() async throws {
        let stopToken = 3
        let generator = makeGenerator(argmaxToken: stopToken)
        let config = greedyConfig(maxTokens: 4, stopTokens: [stopToken])

        let tokens = try await collectTokenIDs(from: generator, config: config)

        XCTAssertEqual(tokens, [], "Sampled EOS/stop token should terminate generation without leaking into answer text")
    }

    func testGenerateStreamYieldsNonStopToken() async throws {
        let sampledToken = 2
        let generator = makeGenerator(argmaxToken: sampledToken)
        let config = greedyConfig(maxTokens: 1, stopTokens: [5])

        let tokens = try await collectTokenIDs(from: generator, config: config)

        XCTAssertEqual(tokens, [sampledToken], "Non-stop sampled tokens should still stream normally")
    }

    private func makeGenerator(argmaxToken: Int) -> ModelRunnerGenerator {
        ModelRunnerGenerator(runner: ModelRunner(weights: makeArgmaxModel(argmaxToken: argmaxToken)))
    }

    private func makeArgmaxModel(argmaxToken: Int) -> ModelWeights {
        let vocabSize = 6
        let hiddenDim = 2
        precondition(argmaxToken >= 0 && argmaxToken < vocabSize)

        let config = ModelConfig(
            numLayers: 0,
            hiddenDim: hiddenDim,
            numHeads: 1,
            vocabSize: vocabSize,
            maxSeqLen: 8
        )
        let tokenEmbeddings = Tensor<Float>(
            shape: TensorShape(vocabSize, hiddenDim),
            data: [Float](repeating: 0, count: vocabSize * hiddenDim)
        )
        let outputWeights = Tensor<Float>(
            shape: TensorShape(hiddenDim, vocabSize),
            data: [Float](repeating: 0, count: hiddenDim * vocabSize)
        )
        let outputBias = Tensor<Float>(
            shape: TensorShape(vocabSize),
            data: (0..<vocabSize).map { $0 == argmaxToken ? 10 : -10 }
        )

        return ModelWeights(
            config: config,
            tokenEmbeddings: tokenEmbeddings,
            layers: [],
            output: LinearLayerWeights(floatWeights: outputWeights, bias: outputBias)
        )
    }

    private func greedyConfig(maxTokens: Int, stopTokens: [Int]) -> GenerationConfig {
        GenerationConfig(
            maxTokens: maxTokens,
            sampler: SamplerConfig(temperature: 0, topK: 1),
            stopTokens: stopTokens
        )
    }

    private func collectTokenIDs(
        from generator: ModelRunnerGenerator,
        config: GenerationConfig
    ) async throws -> [Int] {
        var tokens: [Int] = []
        for try await output in generator.generateStream(prompt: [0], config: config) {
            tokens.append(output.tokenId)
        }
        return tokens
    }
}
