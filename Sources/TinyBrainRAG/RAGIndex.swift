import Foundation
import ProximaKit

/// Errors surfaced by the RAG index wrapper.
public enum RAGIndexError: Error, Equatable, LocalizedError, Sendable {
    /// The persisted index dimension does not match the supplied embedder.
    case dimensionMismatch(expected: Int, got: Int)

    /// The persisted index has a different number of live entries than expected.
    case metadataCountMismatch(expected: Int, got: Int)

    /// A persisted live entry is missing or does not decode as a `DocumentChunk`.
    case invalidStoredMetadata(index: Int)

    public var errorDescription: String? {
        switch self {
        case .dimensionMismatch(let expected, let got):
            return "Embedder dimension \(got) does not match index dimension \(expected)."
        case .metadataCountMismatch(let expected, let got):
            return "Index stores \(got) live chunks, expected \(expected)."
        case .invalidStoredMetadata(let index):
            return "Index metadata at live entry \(index) is missing or is not a DocumentChunk."
        }
    }
}

/// A retrieved chunk with its vector distance and retrieval rank.
public struct RetrievedPassage: Equatable, Sendable {
    /// The decoded chunk metadata stored with the index entry.
    public let chunk: DocumentChunk

    /// ProximaKit search distance. Lower values are more similar.
    public let distance: Float

    /// Zero-based retrieval order.
    public let rank: Int

    /// Creates a retrieved passage.
    public init(chunk: DocumentChunk, distance: Float, rank: Int) {
        self.chunk = chunk
        self.distance = distance
        self.rank = rank
    }
}

/// Metadata-carrying retrieval index backed by ProximaKit `HNSWIndex`.
public actor RAGIndex {
    private let embedder: any TextEmbedder
    private let index: HNSWIndex
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    /// Creates an empty HNSW index whose dimension comes from `embedder`.
    public init(
        embedder: any TextEmbedder,
        metric: any DistanceMetric = CosineDistance()
    ) {
        self.embedder = embedder
        self.index = HNSWIndex(dimension: embedder.dimension, metric: metric)
        precondition(
            embedder.dimension == index.dimension,
            "Embedder dimension must match HNSW index dimension"
        )
    }

    private init(embedder: any TextEmbedder, loadedIndex: HNSWIndex) throws {
        guard embedder.dimension == loadedIndex.dimension else {
            throw RAGIndexError.dimensionMismatch(expected: loadedIndex.dimension, got: embedder.dimension)
        }
        self.embedder = embedder
        self.index = loadedIndex
    }

    /// Number of vectors stored in the underlying index.
    public var count: Int {
        get async {
            await index.count
        }
    }

    /// Embeds and inserts chunks, storing each `DocumentChunk` as JSON metadata.
    public func add(_ chunks: [DocumentChunk]) async throws {
        guard !chunks.isEmpty else { return }

        let vectors = try await embedder.embedBatch(chunks.map(\.text))
        for (chunk, vector) in zip(chunks, vectors) {
            let metadata = try encoder.encode(chunk)
            try await index.add(vector, id: UUID(), metadata: metadata)
        }
    }

    /// Searches the index and resolves JSON metadata back into document chunks.
    public func search(_ query: String, k: Int) async throws -> [RetrievedPassage] {
        guard k > 0 else { return [] }

        let queryVector = try await embedder.embed(query)
        guard queryVector.dimension == index.dimension else {
            return []
        }

        let results = await index.search(query: queryVector, k: k)
        var passages: [RetrievedPassage] = []
        passages.reserveCapacity(results.count)
        for result in results {
            guard let chunk = result.decodeMetadata(as: DocumentChunk.self) else {
                continue
            }
            passages.append(RetrievedPassage(
                chunk: chunk,
                distance: result.distance,
                rank: passages.count
            ))
        }
        return passages
    }

    /// Persists the underlying ProximaKit binary index, including chunk metadata.
    public func save(to url: URL) async throws {
        try await index.save(to: url)
    }

    /// Validates that every live persisted entry stores decodable chunk metadata.
    ///
    /// Use this after loading a cached index before serving retrieval from it.
    public func validateStoredChunks(expectedCount: Int) async throws {
        let entries = await index.liveEntries()
        guard entries.count == expectedCount else {
            throw RAGIndexError.metadataCountMismatch(expected: expectedCount, got: entries.count)
        }

        for (entryIndex, entry) in entries.enumerated() {
            guard let metadata = entry.metadata,
                  (try? decoder.decode(DocumentChunk.self, from: metadata)) != nil else {
                throw RAGIndexError.invalidStoredMetadata(index: entryIndex)
            }
        }
    }

    /// Loads a persisted ProximaKit index and validates it against `embedder`.
    public static func load(from url: URL, embedder: any TextEmbedder) throws -> RAGIndex {
        let loadedIndex = try HNSWIndex.load(from: url)
        return try RAGIndex(embedder: embedder, loadedIndex: loadedIndex)
    }
}
