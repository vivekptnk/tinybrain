import Foundation
import TinyBrainRuntime
import TinyBrainTokenizer

/// ``AgentGenerating`` adapter for TinyBrain's ``ModelRunner``.
///
/// `ModelRunner` owns mutable KV-cache state and is not thread-safe. Use this
/// adapter through ``AgentLoop``, whose actor isolation serializes generation
/// turns for a run.
public final class ModelRunnerAgentGenerator: AgentGenerating {
    private let runner: ModelRunner
    private let tokenizer: any Tokenizer
    private let tokenLookup: TokenizerTokenLookup

    /// Creates a model-backed agent generator.
    public init(runner: ModelRunner, tokenizer: any Tokenizer) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.tokenLookup = TokenizerTokenLookup(tokenizer: tokenizer)
    }

    /// Generates one agent turn using `ModelRunner.generateStream`.
    public func generate(_ request: AgentGenerationRequest) async throws -> AgentGenerationResult {
        let promptTokens = encodePrompt(request.prompt)
        let toolConfig: ToolCallingConfig?
        let constraintMode: ConstraintMode
        if request.mode == .finalAnswer || request.toolDefinitions.isEmpty {
            toolConfig = nil
            constraintMode = .none
        } else {
            toolConfig = ToolCallingConfig(
                tools: request.toolDefinitions,
                toolChoice: request.toolChoice,
                maxIterations: 1
            )
            constraintMode = request.constraintMode
        }

        let generationConfig = GenerationConfig(
            maxTokens: request.maxTokens,
            sampler: request.sampler,
            stopTokens: effectiveStopTokens(request.stopTokens),
            constraintMode: constraintMode,
            toolCallingConfig: toolConfig
        )

        var generatedTokenIDs: [Int] = []
        for try await output in runner.generateStream(
            prompt: promptTokens,
            config: generationConfig,
            tokenLookup: tokenLookup
        ) {
            try Task.checkCancellation()
            generatedTokenIDs.append(output.tokenId)
        }

        return AgentGenerationResult(
            text: tokenizer.decode(generatedTokenIDs),
            tokenCount: generatedTokenIDs.count
        )
    }

    private func encodePrompt(_ prompt: String) -> [Int] {
        var tokens = tokenizer.encode(prompt)
        if let bpeTokenizer = tokenizer as? BPETokenizer, bpeTokenizer.addsBosToken {
            tokens.insert(bpeTokenizer.bosToken, at: 0)
        }
        return tokens
    }

    private func effectiveStopTokens(_ configured: [Int]) -> [Int] {
        var tokens = configured
        if let bpeTokenizer = tokenizer as? BPETokenizer,
           !tokens.contains(bpeTokenizer.eosToken) {
            tokens.append(bpeTokenizer.eosToken)
        }
        return tokens
    }
}

private struct TokenizerTokenLookup: TokenLookup {
    let tokenizer: any Tokenizer

    var vocabularySize: Int {
        tokenizer.vocabularySize
    }

    func decode(tokenId: Int) -> String {
        tokenizer.decode([tokenId])
    }
}
