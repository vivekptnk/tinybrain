import Foundation
import TinyBrainRuntime

/// Errors thrown by the `retrieve` tool adapter.
public enum RetrievalToolError: Error, Equatable, LocalizedError, Sendable {
    /// The required `query` argument was absent or empty.
    case missingQuery

    public var errorDescription: String? {
        switch self {
        case .missingQuery:
            return "The retrieve tool requires a non-empty query string."
        }
    }
}

/// Programmatic passage metadata returned by ``RetrievalTool`` alongside prose.
public struct RetrievedPassageRecord: Codable, Equatable, Sendable {
    /// Zero-based rank in the rendered retrieve result.
    public let rank: Int

    /// Original source path for the retrieved chunk.
    public let sourcePath: String

    /// Lower-is-better vector distance.
    public let distance: Float

    /// Retrieved chunk text, preserving its original line breaks.
    public let excerpt: String

    /// Stable chunk identifier when the backing corpus provides one.
    public let chunkId: String?

    /// Creates a structured retrieval record.
    public init(
        rank: Int,
        sourcePath: String,
        distance: Float,
        excerpt: String,
        chunkId: String? = nil
    ) {
        self.rank = rank
        self.sourcePath = sourcePath
        self.distance = distance
        self.excerpt = excerpt
        self.chunkId = chunkId
    }
}

/// Combined retrieve-tool output for programmatic consumers.
public struct RetrievalToolOutput: Equatable, Sendable {
    /// Prose result returned to the model.
    public let content: String

    /// Structured records used to render ``content``.
    public let records: [RetrievedPassageRecord]

    /// Creates a retrieval output.
    public init(content: String, records: [RetrievedPassageRecord]) {
        self.content = content
        self.records = records
    }
}

/// Tool-calling adapter that exposes RAG retrieval as `retrieve`.
public struct RetrievalTool {
    /// Tool name used by the model and dispatcher.
    public static let name = "retrieve"

    /// JSON-schema tool definition registered with TinyBrain tool calling.
    public let definition: ToolDefinition

    /// Optional maximum vector distance a retrieved passage may have to be returned.
    ///
    /// Lower cosine distance means a stronger semantic match. `nil` disables
    /// filtering for callers that need legacy retrieval behavior.
    public let maxDistance: Float?

    private let defaultK: Int
    private let maxK: Int
    private let retrieve: @Sendable (String, Int) async throws -> [RetrievedPassage]

    /// Creates a retrieval tool from any shared retrieval implementation.
    public init(
        defaultK: Int = 4,
        maxK: Int = 8,
        maxDistance: Float? = 0.85,
        retrieve: @escaping @Sendable (String, Int) async throws -> [RetrievedPassage]
    ) {
        precondition(defaultK > 0, "defaultK must be positive")
        precondition(maxK >= defaultK, "maxK must be at least defaultK")
        if let maxDistance {
            precondition(maxDistance >= 0, "maxDistance must be non-negative")
        }
        self.defaultK = defaultK
        self.maxK = maxK
        self.maxDistance = maxDistance
        self.retrieve = retrieve
        self.definition = ToolDefinition(
            name: Self.name,
            description: "Search the user's indexed documents for passages relevant to a query.",
            parameters: .object(
                properties: [
                    JSONSchemaProperty(
                        name: "query",
                        schema: .string,
                        description: "What to search for",
                        required: true
                    ),
                    JSONSchemaProperty(
                        name: "k",
                        schema: .integer,
                        description: "How many passages to return",
                        required: false
                    )
                ],
                required: ["query"]
            )
        )
    }

    /// Handles a TinyBrain tool call and returns numbered passages.
    public func handle(_ call: ToolCall) async throws -> String {
        try await handleWithRecords(call).content
    }

    /// Handles a TinyBrain tool call and returns prose plus structured records.
    public func handleWithRecords(_ call: ToolCall) async throws -> RetrievalToolOutput {
        guard let rawQuery = call.arguments["query"] as? String else {
            throw RetrievalToolError.missingQuery
        }

        let query = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else {
            throw RetrievalToolError.missingQuery
        }

        let requestedK = intArgument(call.arguments["k"]) ?? defaultK
        let boundedK = min(max(requestedK, 1), maxK)
        let passages = try await retrieve(query, boundedK)
        let filteredPassages = filtered(passages)
        guard !filteredPassages.isEmpty else {
            if let maxDistance, let bestDistance = passages.map(\.distance).min() {
                return RetrievalToolOutput(
                    content: noRelevantPassagesMessage(
                        query: query,
                        bestDistance: bestDistance,
                        maxDistance: maxDistance
                    ),
                    records: []
                )
            }
            return RetrievalToolOutput(content: "No relevant passages found.", records: [])
        }

        let records = Self.records(from: filteredPassages)
        return RetrievalToolOutput(content: Self.render(records: records), records: records)
    }

    /// Renders structured records into the legacy prose format read by models.
    public static func render(records: [RetrievedPassageRecord]) -> String {
        records
            .map { record in
                let text = record.excerpt.components(separatedBy: .newlines).joined(separator: " ")
                let distance = String(format: "%.3f", record.distance)
                return "[\(record.rank + 1)] \(text) (source: \(record.sourcePath), distance: \(distance))"
            }
            .joined(separator: "\n")
    }

    /// Registers this tool with a closure dispatcher.
    public func register(on dispatcher: ClosureToolDispatcher) {
        dispatcher.register(definition.name) { call in
            try await handle(call)
        }
    }

    private static func records(from passages: [RetrievedPassage]) -> [RetrievedPassageRecord] {
        passages.enumerated().map { index, passage in
            RetrievedPassageRecord(
                rank: index,
                sourcePath: passage.chunk.sourcePath,
                distance: passage.distance,
                excerpt: passage.chunk.text
            )
        }
    }

    private func intArgument(_ value: Any?) -> Int? {
        switch value {
        case let int as Int:
            return int
        case let number as NSNumber:
            return number.intValue
        case let string as String:
            return Int(string)
        default:
            return nil
        }
    }

    private func filtered(_ passages: [RetrievedPassage]) -> [RetrievedPassage] {
        guard let maxDistance else {
            return passages
        }

        return passages.filter { $0.distance <= maxDistance }
    }

    private func noRelevantPassagesMessage(
        query: String,
        bestDistance: Float,
        maxDistance: Float
    ) -> String {
        let best = String(format: "%.3f", bestDistance)
        let threshold = String(format: "%.2f", maxDistance)
        return "No sufficiently relevant passages found for '\(query)' (best distance \(best), threshold \(threshold)). The knowledge base may not cover this topic."
    }
}

public extension RAGEngine {
    /// Creates the `retrieve` tool backed by this engine's retrieval method.
    func retrieveTool(defaultK: Int = 4, maxK: Int = 8, maxDistance: Float? = 0.85) -> RetrievalTool {
        RetrievalTool(defaultK: defaultK, maxK: maxK, maxDistance: maxDistance) { query, k in
            try await self.retrieve(query, k: k)
        }
    }
}
