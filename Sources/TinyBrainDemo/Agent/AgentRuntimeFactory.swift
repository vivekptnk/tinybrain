/// Runtime construction for the bundled Agent demo.
///
/// P1 registers only the `retrieve` tool over a persisted on-device corpus.

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

    /// How the index became ready for this runtime.
    public let indexPreparation: AgentIndexPreparationSummary
}

/// Where the demo RAG index came from for the current runtime.
public enum AgentIndexPreparationSource: Equatable, Sendable {
    /// The corpus was chunked, embedded, indexed, and saved to disk.
    case indexed

    /// The corpus was chunked, embedded, and indexed, but persistence failed.
    case indexedNotPersisted

    /// A matching persisted ProximaKit index was loaded from disk.
    case loaded
}

/// User-visible index preparation timing and cache identity.
public struct AgentIndexPreparationSummary: Equatable, Sendable {
    /// Whether the index was built or loaded.
    public let source: AgentIndexPreparationSource

    /// Wall-clock time spent preparing the index.
    public let elapsedSeconds: TimeInterval

    /// Stable corpus/embedder fingerprint used in the `.pxkt` filename.
    public let fingerprint: String

    /// Persisted ProximaKit index file.
    public let storageURL: URL
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

    /// Greedy factual decoding verified by the Qwen agent smoke test.
    static let factualSampler = SamplerConfig(temperature: 0.0, topK: 1)

    /// Fingerprint namespace for the persisted demo index.
    ///
    /// Fingerprint assumptions:
    /// - Tokenizer identity is deliberately not hashed. This is safe only while
    ///   every demo note is a single chunk; all 15 notes are under 96 tokens, so
    ///   chunk boundaries are tokenizer-independent.
    /// - Bump this value on any `DocumentChunker` algorithm change or
    ///   `DocumentChunk` `Codable` schema change.
    /// - The fingerprint is filename-only for source identity, not a
    ///   per-content integrity checksum. The trust boundary is the app's own
    ///   Application Support directory.
    static let indexFormatVersion = "TinyBrain.AgentDemoIndex.v1"

    /// ProximaKit HNSW binary index extension.
    static let indexFileExtension = "pxkt"

    /// Stable chunking used for the bundled corpus index.
    static let demoChunkingConfig = ChunkingConfig(targetTokens: 96, overlapTokens: 8)

    /// Default on-disk index location.
    static var defaultIndexDirectory: URL {
        let applicationSupport = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library", isDirectory: true)
            .appendingPathComponent("Application Support", isDirectory: true)

        return applicationSupport
            .appendingPathComponent("TinyBrain", isDirectory: true)
            .appendingPathComponent("AgentIndex", isDirectory: true)
    }

    /// Builds the Demo agent configuration used for tool calls and final answers.
    static func makeAgentConfig(
        tokenizer: any Tokenizer,
        promptStyle: ModelPromptStyle
    ) -> AgentConfig {
        AgentConfig(
            maxSteps: maxSteps,
            toolChoice: .required,
            constraintMode: .strict,
            perStepTokenBudget: 160,
            contextBudget: 2_048,
            sampler: factualSampler,
            stopTokens: TinyBrainChatStops.stopTokenIDs(for: tokenizer, promptStyle: promptStyle),
            promptStyle: promptStyle.agentPromptStyle
        )
    }

    /// Builds an agent loop, RAG index, RAG engine, and retrieve tool.
    public static func makeRuntime(
        weights: ModelWeights,
        tokenizer: any Tokenizer,
        promptStyle: ModelPromptStyle,
        sampler _: SamplerConfig,
        observer: AgentTraceObserver,
        indexDirectory: URL? = nil
    ) async throws -> AgentRuntimeContext {
        let indexBundle = try await makeIndex(
            tokenizer: tokenizer,
            indexDirectory: indexDirectory ?? defaultIndexDirectory
        )
        let engine = RAGEngine(
            index: indexBundle.index,
            generator: EmptyRAGAnswerGenerator(),
            tokenizer: tokenizer,
            retrievalK: 3
        )

        let registry = ToolRegistry()
        await registry.register(
            BuiltInAgentTools.retrieve(engine.retrieveTool(defaultK: 3, maxK: 5))
        )

        let runner = ModelRunner(weights: weights)
        let config = makeAgentConfig(tokenizer: tokenizer, promptStyle: promptStyle)

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
                chunkCount: indexBundle.chunkCount,
                embedder: indexBundle.summary,
                isStubEmbedder: indexBundle.isStub,
                indexPreparation: indexBundle.preparation
            )
        )
    }

    static func makeIndex(
        tokenizer: any Tokenizer,
        indexDirectory: URL = defaultIndexDirectory,
        provider: AgentIndexProvider = makeIndexProvider(),
        documents: [RAGDocument] = AgentDemoCorpus.documents
    ) async throws -> AgentIndexBundle {
        let chunks = chunks(
            for: documents,
            tokenizer: tokenizer,
            chunkingConfig: demoChunkingConfig
        )
        let fingerprint = indexFingerprint(
            documents: documents,
            embedderIdentity: provider.identity,
            dimension: provider.dimension,
            chunkingConfig: demoChunkingConfig
        )
        let indexURL = indexDirectory
            .appendingPathComponent(fingerprint)
            .appendingPathExtension(indexFileExtension)

        if FileManager.default.fileExists(atPath: indexURL.path) {
            let started = Date()
            do {
                let loaded = try provider.loadIndex(indexURL)
                try await loaded.validateStoredChunks(expectedCount: chunks.count)
                let elapsed = Date().timeIntervalSince(started)
                logIndexPersistence(
                    "loaded \(chunks.count) chunks from \(indexURL.path) in \(formatSeconds(elapsed))s"
                )
                return AgentIndexBundle(
                    index: loaded,
                    summary: provider.summary,
                    isStub: provider.isStub,
                    chunkCount: chunks.count,
                    preparation: AgentIndexPreparationSummary(
                        source: .loaded,
                        elapsedSeconds: elapsed,
                        fingerprint: fingerprint,
                        storageURL: indexURL
                    )
                )
            } catch {
                logIndexPersistence(
                    "cache load failed at \(indexURL.path): \(error.localizedDescription). Deleting and rebuilding."
                )
                do {
                    try FileManager.default.removeItem(at: indexURL)
                } catch {
                    logIndexPersistence(
                        "cache delete failed at \(indexURL.path): \(error.localizedDescription). Continuing rebuild."
                    )
                }
            }
        }

        let started = Date()
        let index = provider.makeEmptyIndex()
        try await index.add(chunks)
        try await index.validateStoredChunks(expectedCount: chunks.count)
        let source: AgentIndexPreparationSource
        do {
            try FileManager.default.createDirectory(
                at: indexDirectory,
                withIntermediateDirectories: true
            )
            try await index.save(to: indexURL)
            let elapsed = Date().timeIntervalSince(started)
            logIndexPersistence(
                "indexed \(chunks.count) chunks and saved \(indexURL.path) in \(formatSeconds(elapsed))s"
            )
            source = .indexed
        } catch {
            let elapsed = Date().timeIntervalSince(started)
            logIndexPersistence(
                "indexed \(chunks.count) chunks in \(formatSeconds(elapsed))s but persistence failed at \(indexURL.path): \(error.localizedDescription). Serving in-memory index."
            )
            source = .indexedNotPersisted
        }
        let elapsed = Date().timeIntervalSince(started)
        return AgentIndexBundle(
            index: index,
            summary: provider.summary,
            isStub: provider.isStub,
            chunkCount: chunks.count,
            preparation: AgentIndexPreparationSummary(
                source: source,
                elapsedSeconds: elapsed,
                fingerprint: fingerprint,
                storageURL: indexURL
            )
        )
    }

    static func indexFingerprint(
        documents: [RAGDocument],
        embedderIdentity: String,
        dimension: Int,
        chunkingConfig: ChunkingConfig
    ) -> String {
        var hasher = StableIndexHasher()
        hasher.append(indexFormatVersion)
        hasher.append(embedderIdentity)
        hasher.append(dimension)
        hasher.append(chunkingConfig.targetTokens)
        hasher.append(chunkingConfig.overlapTokens)
        hasher.append(chunkingConfig.respectParagraphs ? 1 : 0)
        hasher.append(documents.count)
        for document in documents {
            hasher.append(document.sourcePath)
            hasher.append(document.text)
        }
        return hasher.hexDigest
    }

    static func stubIndexProvider(dimension: Int = 64, seed: UInt64 = 42) -> AgentIndexProvider {
        let provider = DeterministicStubEmbedder(dimension: dimension, seed: seed)
        return AgentIndexProvider(
            summary: "DeterministicStubEmbedder (\(dimension)d, seed \(seed))",
            identity: "TinyBrainRAG.DeterministicStubEmbedder.seed=\(seed)",
            dimension: provider.dimension,
            isStub: true,
            makeEmptyIndex: { RAGIndex(embedder: provider) },
            loadIndex: { try RAGIndex.load(from: $0, embedder: provider) }
        )
    }

    private static func makeIndexProvider() -> AgentIndexProvider {
        if let provider = try? NLEmbeddingProvider(language: .english) {
            return AgentIndexProvider(
                summary: "NLEmbeddingProvider (.english, \(provider.dimension)d)",
                identity: "ProximaEmbeddings.NLEmbeddingProvider.language=english",
                dimension: provider.dimension,
                isStub: false,
                makeEmptyIndex: { RAGIndex(embedder: provider) },
                loadIndex: { try RAGIndex.load(from: $0, embedder: provider) }
            )
        }

        return stubIndexProvider(dimension: 64, seed: 42)
    }

    private static func chunks(
        for documents: [RAGDocument],
        tokenizer: any Tokenizer,
        chunkingConfig: ChunkingConfig
    ) -> [DocumentChunk] {
        let chunker = DocumentChunker(tokenizer: tokenizer, config: chunkingConfig)
        return documents.flatMap { document in
            chunker.chunk(document.text, sourcePath: document.sourcePath)
        }
    }

    private static func logIndexPersistence(_ message: String) {
        print("[TinyBrain AgentIndex] \(message)")
    }

    private static func formatSeconds(_ elapsed: TimeInterval) -> String {
        String(format: "%.3f", elapsed)
    }
}

struct AgentIndexProvider {
    let summary: String
    let identity: String
    let dimension: Int
    let isStub: Bool
    let makeEmptyIndex: () -> RAGIndex
    let loadIndex: (URL) throws -> RAGIndex
}

struct AgentIndexBundle {
    let index: RAGIndex
    let summary: String
    let isStub: Bool
    let chunkCount: Int
    let preparation: AgentIndexPreparationSummary
}

private struct StableIndexHasher {
    private var hash: UInt64 = 0xcbf2_9ce4_8422_2325

    var hexDigest: String {
        String(format: "%016llx", hash)
    }

    mutating func append(_ value: Int) {
        append(String(value))
    }

    mutating func append(_ value: String) {
        appendLength(value.utf8.count)
        for byte in value.utf8 {
            combine(byte)
        }
        combine(0x1f)
    }

    private mutating func appendLength(_ length: Int) {
        var value = UInt64(length)
        for _ in 0..<8 {
            combine(UInt8(truncatingIfNeeded: value))
            value >>= 8
        }
    }

    private mutating func combine(_ byte: UInt8) {
        hash ^= UInt64(byte)
        hash &*= 0x0000_0100_0000_01b3
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
