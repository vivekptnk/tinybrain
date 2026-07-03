import Foundation
import TinyBrainRuntime

/// Streams answer tokens for a retrieval-augmented prompt.
///
/// The protocol keeps ``RAGEngine`` independent from the concrete model runner
/// so tests can inject scripted generators while production uses TinyBrain's
/// ``ModelRunner``.
public protocol AnswerGenerator: Sendable {
    /// Produces generated token outputs for an already-tokenized prompt.
    func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error>
}

/// Actor-isolated TinyBrain model adapter for ``AnswerGenerator``.
///
/// ``ModelRunner`` owns mutable KV-cache state and is not thread-safe. This
/// actor keeps every `step(tokenId:)` call on one serialized executor.
public actor ModelRunnerGenerator: AnswerGenerator {
    private let runner: ModelRunner

    /// Creates a generator around a loaded TinyBrain model runner.
    public init(runner: ModelRunner) {
        self.runner = runner
    }

    /// Streams sampled tokens from TinyBrain while preserving actor isolation.
    public nonisolated func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.generate(
                        prompt: prompt,
                        config: config,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func generate(
        prompt: [Int],
        config: GenerationConfig,
        continuation: AsyncThrowingStream<TokenOutput, Error>.Continuation
    ) throws {
        try Task.checkCancellation()

        var mutableConfig = config
        runner.reset()

        let sanitizedPrompt = prompt.map { max(0, min($0, runner.config.vocabSize - 1)) }
        var currentToken = sanitizedPrompt.last ?? 0
        if !sanitizedPrompt.isEmpty {
            for token in sanitizedPrompt.dropLast() {
                try Task.checkCancellation()
                _ = runner.step(tokenId: token)
            }
        }

        var history = Array(sanitizedPrompt)
        var generated = 0
        while generated < mutableConfig.maxTokens {
            try Task.checkCancellation()

            let logits = runner.step(tokenId: currentToken)
            let sampled = Sampler.sampleDetailed(
                logits: logits,
                config: &mutableConfig.sampler,
                history: history
            )

            if mutableConfig.stopTokens.contains(sampled.tokenId) {
                break
            }

            let output = TokenOutput(
                tokenId: sampled.tokenId,
                probability: sampled.probability,
                entropy: sampled.entropy,
                timestamp: Date(),
                strategy: strategySummary(for: mutableConfig.sampler),
                energyJoules: nil
            )

            continuation.yield(output)

            currentToken = sampled.tokenId
            history.append(sampled.tokenId)
            generated += 1
        }
    }

    private nonisolated func strategySummary(for sampler: SamplerConfig) -> String? {
        var parts: [String] = []
        parts.append(String(format: "temp=%.2f", sampler.temperature))
        if let topK = sampler.topK {
            parts.append("topK=\(topK)")
        }
        if let topP = sampler.topP {
            parts.append(String(format: "topP=%.2f", topP))
        }
        if sampler.repetitionPenalty != 1.0 {
            parts.append(String(format: "penalty=%.2f", sampler.repetitionPenalty))
        }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }
}
