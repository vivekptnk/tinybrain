/// Runtime construction for the bundled Agent demo.
///
/// P1 registers only the `retrieve` tool over an in-memory on-device corpus.

import Foundation
import NaturalLanguage
import ProximaEmbeddings
import TinyBrainAgent
import TinyBrainRAG
import TinyBrainRuntime
import TinyBrainTokenizer

/// Summary of the indexed demo corpus backing an agent runtime.
public struct AgentCorpusSummary: Equatable, Sendable {
    /// Number of source notes.
    public let noteCount: Int

    /// Number of chunks inserted into the index.
    public let chunkCount: Int

    /// Embedder description.
    public let embedder: String

    /// Whether the deterministic stub embedder was used.
    public let isStubEmbedder: Bool
}

/// Constructed agent runtime and corpus metadata.
public struct AgentRuntimeContext {
    /// Agent loop with a weak observer already attached.
    public let loop: AgentLoop

    /// Indexed corpus metadata.
    public let corpus: AgentCorpusSummary
}

/// Builds the P1 agent runtime from the active chat model.
public enum AgentRuntimeFactory {
    /// Number of plan-act-observe steps in the P1 demo.
    public static let maxSteps = 3

    /// Builds an agent loop, RAG index, RAG engine, and retrieve tool.
    public static func makeRuntime(
        weights: ModelWeights,
        tokenizer: any Tokenizer,
        promptStyle: ModelPromptStyle,
        sampler: SamplerConfig,
        observer: AgentTraceObserver
    ) async throws -> AgentRuntimeContext {
        let indexBundle = makeIndex()
        let engine = RAGEngine(
            index: indexBundle.index,
            generator: EmptyRAGAnswerGenerator(),
            tokenizer: tokenizer,
            retrievalK: 3
        )
        let chunks = try await engine.index(
            documents: AgentDemoCorpus.documents,
            chunkingConfig: ChunkingConfig(targetTokens: 96, overlapTokens: 8)
        )

        let registry = ToolRegistry()
        await registry.register(
            BuiltInAgentTools.retrieve(engine.retrieveTool(defaultK: 3, maxK: 5))
        )

        let runner = ModelRunner(weights: weights)
        let config = AgentConfig(
            maxSteps: maxSteps,
            toolChoice: .required,
            constraintMode: .strict,
            perStepTokenBudget: 160,
            contextBudget: 2_048,
            sampler: sampler,
            stopTokens: TinyBrainChatStops.stopTokenIDs(for: tokenizer, promptStyle: promptStyle),
            promptStyle: promptStyle.agentPromptStyle
        )

        let loop = AgentLoop(
            runner: runner,
            tokenizer: tokenizer,
            registry: registry,
            config: config,
            observer: observer
        )

        return AgentRuntimeContext(
            loop: loop,
            corpus: AgentCorpusSummary(
                noteCount: AgentDemoCorpus.notes.count,
                chunkCount: chunks.count,
                embedder: indexBundle.summary,
                isStubEmbedder: indexBundle.isStub
            )
        )
    }

    private static func makeIndex() -> (index: RAGIndex, summary: String, isStub: Bool) {
        if let provider = try? NLEmbeddingProvider(language: .english) {
            return (
                RAGIndex(embedder: provider),
                "NLEmbeddingProvider (.english, \(provider.dimension)d)",
                false
            )
        }

        let provider = DeterministicStubEmbedder(dimension: 64, seed: 42)
        return (
            RAGIndex(embedder: provider),
            "DeterministicStubEmbedder (64d, seed 42)",
            true
        )
    }
}

private struct EmptyRAGAnswerGenerator: AnswerGenerator {
    func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}

private extension ModelPromptStyle {
    var agentPromptStyle: AgentPromptStyle {
        switch self {
        case .qwenChatML:
            return .qwenChatML
        case .zephyrChat:
            return .zephyrChat
        case .rawCompletion:
            return .rawCompletion
        }
    }
}
