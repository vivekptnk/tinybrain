import Foundation
import TinyBrainRuntime
import TinyBrainTokenizer

/// Text document supplied to ``RAGEngine`` for chunking and indexing.
public struct RAGDocument: Equatable, Sendable {
    /// Full document text.
    public let text: String

    /// Source identifier shown in citations and retrieval output.
    public let sourcePath: String

    /// Creates an indexable RAG document.
    public init(text: String, sourcePath: String) {
        self.text = text
        self.sourcePath = sourcePath
    }
}

/// Final non-streaming answer returned by ``RAGEngine``.
public struct RAGResponse: Equatable, Sendable {
    /// Complete generated answer text.
    public let answer: String

    /// Citations parsed from `answer`.
    public let citations: [Citation]

    /// Retrieved passages, including scores, before prompt-budget filtering.
    public let passages: [RetrievedPassage]

    /// Creates a response value.
    public init(answer: String, citations: [Citation], passages: [RetrievedPassage]) {
        self.answer = answer
        self.citations = citations
        self.passages = passages
    }
}

/// Events emitted by ``RAGEngine.answerStream(_:)``.
public enum RAGEvent: Equatable, Sendable {
    /// Retrieval has completed and the passages are available for display.
    case passages([RetrievedPassage])

    /// One decoded answer text delta.
    case token(String)

    /// Generation finished and citations have been resolved.
    case done([Citation])
}

private struct IncrementalDetokenizer {
    private let tokenizer: any Tokenizer
    private var tokenIDs: [Int] = []
    private var emittedText = ""
    private(set) var decodedText = ""

    init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    mutating func append(_ tokenID: Int) -> String? {
        tokenIDs.append(tokenID)
        decodedText = tokenizer.decode(tokenIDs)

        guard decodedText.hasPrefix(emittedText) else {
            return nil
        }

        let delta = String(decodedText.dropFirst(emittedText.count))
        emittedText = decodedText
        return delta.isEmpty ? nil : delta
    }
}

/// Retrieval-augmented generation orchestrator.
///
/// The engine owns the shared pipeline used by both answers and the
/// `retrieve` tool: chunk/index, retrieve, prompt, stream, parse citations.
public actor RAGEngine {
    private let ragIndex: RAGIndex
    private let generator: any AnswerGenerator
    private let tokenizer: any Tokenizer
    private let promptBuilder: RAGPromptBuilder
    private let retrievalK: Int
    private let generationConfig: GenerationConfig

    /// Creates a RAG engine over an existing index.
    public init(
        index: RAGIndex,
        generator: any AnswerGenerator,
        tokenizer: any Tokenizer,
        promptBuilder: RAGPromptBuilder? = nil,
        promptTemplate: PromptTemplate = .none,
        retrievalK: Int = 4,
        generationConfig: GenerationConfig = GenerationConfig(maxTokens: 256)
    ) {
        precondition(retrievalK > 0, "retrievalK must be positive")
        self.ragIndex = index
        self.generator = generator
        self.tokenizer = tokenizer
        self.promptBuilder = promptBuilder ?? RAGPromptBuilder(
            tokenizer: tokenizer,
            template: promptTemplate
        )
        self.retrievalK = retrievalK
        self.generationConfig = generationConfig
    }

    /// Chunks and indexes in-memory documents.
    ///
    /// - Returns: The chunks inserted into the underlying index.
    public func index(
        documents: [RAGDocument],
        chunkingConfig: ChunkingConfig = .init()
    ) async throws -> [DocumentChunk] {
        guard !documents.isEmpty else { return [] }

        let chunker = DocumentChunker(tokenizer: tokenizer, config: chunkingConfig)
        let chunks = documents.flatMap { document in
            chunker.chunk(document.text, sourcePath: document.sourcePath)
        }
        try Task.checkCancellation()
        try await ragIndex.add(chunks)
        return chunks
    }

    /// Recursively indexes UTF-8 `.txt` and `.md` files from a folder.
    ///
    /// - Returns: The chunks inserted into the underlying index.
    public func index(
        folderAt url: URL,
        chunkingConfig: ChunkingConfig = .init(),
        fileExtensions: Set<String> = ["txt", "md"]
    ) async throws -> [DocumentChunk] {
        let chunker = DocumentChunker(tokenizer: tokenizer, config: chunkingConfig)
        let urls = try textFileURLs(in: url, fileExtensions: fileExtensions)

        var chunks: [DocumentChunk] = []
        for fileURL in urls {
            try Task.checkCancellation()
            chunks.append(contentsOf: try chunker.chunk(fileAt: fileURL))
        }

        try await ragIndex.add(chunks)
        return chunks
    }

    /// Retrieval only: the shared code path used by answers and the tool seam.
    public func retrieve(_ query: String, k: Int) async throws -> [RetrievedPassage] {
        try Task.checkCancellation()
        return try await ragIndex.search(query, k: k)
    }

    /// Runs the full RAG pipeline and returns the completed response.
    public func answer(_ question: String) async throws -> RAGResponse {
        var answer = ""
        var passages: [RetrievedPassage] = []
        var citations: [Citation] = []

        for try await event in answerStream(question) {
            switch event {
            case .passages(let retrieved):
                passages = retrieved
            case .token(let token):
                answer += token
            case .done(let parsed):
                citations = parsed
            }
        }

        return RAGResponse(answer: answer, citations: citations, passages: passages)
    }

    /// Streams retrieved passages, decoded answer tokens, and final citations.
    public nonisolated func answerStream(_ question: String) -> AsyncThrowingStream<RAGEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await self.runAnswerStream(question, continuation: continuation)
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

    private func runAnswerStream(
        _ question: String,
        continuation: AsyncThrowingStream<RAGEvent, Error>.Continuation
    ) async throws {
        let retrieved = try await retrieve(question, k: retrievalK)
        continuation.yield(.passages(retrieved))

        guard !retrieved.isEmpty else {
            let answer = "I could not find any relevant passages in the indexed documents."
            continuation.yield(.token(answer))
            continuation.yield(.done([]))
            return
        }

        let built = promptBuilder.build(question: question, passages: retrieved)
        guard !built.included.isEmpty else {
            let answer = "I found passages, but none fit within the prompt budget."
            continuation.yield(.token(answer))
            continuation.yield(.done([]))
            return
        }

        var config = generationConfig
        config.maxTokens = min(config.maxTokens, promptBuilder.generationHeadroom)
        if let bpeTokenizer = tokenizer as? BPETokenizer,
           !config.stopTokens.contains(bpeTokenizer.eosToken) {
            config.stopTokens.append(bpeTokenizer.eosToken)
        }

        let promptTokens = promptBuilder.tokenIDs(for: built.prompt)
        let stream = generator.generateStream(prompt: promptTokens, config: config)

        var detokenizer = IncrementalDetokenizer(tokenizer: tokenizer)
        for try await output in stream {
            try Task.checkCancellation()
            if let delta = detokenizer.append(output.tokenId) {
                continuation.yield(.token(delta))
            }
        }

        let citations = CitationParser(passages: built.included).parse(detokenizer.decodedText)
        continuation.yield(.done(citations))
    }

    private func textFileURLs(
        in folderURL: URL,
        fileExtensions: Set<String>
    ) throws -> [URL] {
        let resourceKeys: [URLResourceKey] = [.isRegularFileKey]
        guard let enumerator = FileManager.default.enumerator(
            at: folderURL,
            includingPropertiesForKeys: resourceKeys,
            options: [.skipsHiddenFiles]
        ) else {
            return []
        }

        var urls: [URL] = []
        for case let fileURL as URL in enumerator {
            let values = try fileURL.resourceValues(forKeys: Set(resourceKeys))
            guard values.isRegularFile == true else { continue }
            let ext = fileURL.pathExtension.lowercased()
            guard fileExtensions.contains(ext) else { continue }
            urls.append(fileURL)
        }
        return urls.sorted { $0.path < $1.path }
    }
}
