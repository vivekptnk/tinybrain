import Foundation
import ProximaKit
import XCTest
@testable import TinyBrainRAG

private struct QueryDimensionMismatchEmbedder: TextEmbedder {
    let dimension: Int
    let base: DeterministicStubEmbedder

    init(dimension: Int) {
        self.dimension = dimension
        self.base = DeterministicStubEmbedder(dimension: dimension, seed: 42)
    }

    func embed(_ text: String) async throws -> Vector {
        if text == "wrong dimension" {
            return Vector([Float](repeating: 1, count: dimension + 1))
        }
        return try await base.embed(text)
    }
}

final class RAGIndexTests: XCTestCase {
    func testAddSearchReturnsNearestChunk() async throws {
        let embedder = DeterministicStubEmbedder(dimension: 64, seed: 7)
        let index = RAGIndex(embedder: embedder)
        let chunks = [
            RAGTestSupport.chunk("apple battery screen", sourcePath: "device.md", ordinal: 0),
            RAGTestSupport.chunk("apple pie recipe", sourcePath: "kitchen.md", ordinal: 1),
            RAGTestSupport.chunk("quantum field theory", sourcePath: "physics.md", ordinal: 2)
        ]

        try await index.add(chunks)
        let results = try await index.search("apple battery", k: 2)

        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0].chunk, chunks[0])
        XCTAssertEqual(results[0].rank, 0)
    }

    func testSearchReturnsRankOrderedDistances() async throws {
        let embedder = DeterministicStubEmbedder(dimension: 64, seed: 11)
        let index = RAGIndex(embedder: embedder)
        try await index.add([
            RAGTestSupport.chunk("apple battery screen", ordinal: 0),
            RAGTestSupport.chunk("apple pie recipe", ordinal: 1),
            RAGTestSupport.chunk("quantum field theory", ordinal: 2)
        ])

        let results = try await index.search("apple battery", k: 3)

        XCTAssertEqual(results.map(\.rank), Array(results.indices))
        XCTAssertEqual(results.map(\.distance), results.map(\.distance).sorted())
    }

    func testCountAfterAddingChunks() async throws {
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 32, seed: 1))

        try await index.add([
            RAGTestSupport.chunk("one", ordinal: 0),
            RAGTestSupport.chunk("two", ordinal: 1)
        ])

        let count = await index.count
        XCTAssertEqual(count, 2)
    }

    func testEmptyIndexSearchReturnsNoResults() async throws {
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 32, seed: 1))

        let results = try await index.search("anything", k: 5)

        XCTAssertEqual(results, [])
    }

    func testSaveLoadRoundTripPreservesSearchResultsAndMetadata() async throws {
        let embedder = DeterministicStubEmbedder(dimension: 64, seed: 5)
        let index = RAGIndex(embedder: embedder)
        let chunks = [
            RAGTestSupport.chunk("private local documents", sourcePath: "rag.md", ordinal: 0),
            RAGTestSupport.chunk("banana bread recipe", sourcePath: "food.md", ordinal: 1)
        ]
        try await index.add(chunks)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainRAGIndexTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("rag-index.pxkt")
        try await index.save(to: url)

        let loaded = try RAGIndex.load(from: url, embedder: embedder)
        let results = try await loaded.search("private local", k: 1)

        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].chunk, chunks[0])
    }

    func testSearchWithMismatchedEmbeddingVectorReturnsEmpty() async throws {
        let embedder = QueryDimensionMismatchEmbedder(dimension: 32)
        let index = RAGIndex(embedder: embedder)
        try await index.add([RAGTestSupport.chunk("stored with correct dimension", ordinal: 0)])

        let results = try await index.search("wrong dimension", k: 1)

        XCTAssertEqual(results, [])
    }

    func testLoadRejectsEmbedderDimensionMismatch() async throws {
        let embedder = DeterministicStubEmbedder(dimension: 32, seed: 9)
        let index = RAGIndex(embedder: embedder)
        try await index.add([RAGTestSupport.chunk("dimension check", ordinal: 0)])

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainRAGDimensionTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let url = directory.appendingPathComponent("rag-index.pxkt")
        try await index.save(to: url)

        XCTAssertThrowsError(
            try RAGIndex.load(
                from: url,
                embedder: DeterministicStubEmbedder(dimension: 16, seed: 9)
            )
        ) { error in
            XCTAssertEqual(error as? RAGIndexError, .dimensionMismatch(expected: 32, got: 16))
        }
    }
}
