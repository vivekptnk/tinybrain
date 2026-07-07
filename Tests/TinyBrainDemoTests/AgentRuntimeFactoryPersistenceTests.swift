import Foundation
import XCTest
@testable import TinyBrainDemo
import TinyBrainRAG
import TinyBrainTokenizer

private struct AgentPersistenceTestTokenizer: Tokenizer {
    let vocabularySize = 512

    func encode(_ text: String) -> [Int] {
        let words = text.split { $0.isWhitespace || $0.isNewline }
        return Array(0..<words.count)
    }

    func decode(_ tokens: [Int]) -> String {
        tokens.map(String.init).joined(separator: " ")
    }
}

final class AgentRuntimeFactoryPersistenceTests: XCTestCase {
    private let tokenizer = AgentPersistenceTestTokenizer()

    func testMakeIndexBuildsSavesThenLoadsFromDiskWithSameSearchResults() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = AgentRuntimeFactory.stubIndexProvider(dimension: 64, seed: 101)
        let cold = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: Self.documents
        )
        let freshResults = try await cold.index.search("atlas owner mira", k: 1)

        let warm = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: Self.documents
        )
        let loadedResults = try await warm.index.search("atlas owner mira", k: 1)

        XCTAssertEqual(cold.preparation.source, .indexed)
        XCTAssertEqual(warm.preparation.source, .loaded)
        XCTAssertEqual(cold.preparation.fingerprint, warm.preparation.fingerprint)
        XCTAssertEqual(cold.preparation.storageURL.pathExtension, AgentRuntimeFactory.indexFileExtension)
        XCTAssertTrue(FileManager.default.fileExists(atPath: warm.preparation.storageURL.path))
        XCTAssertEqual(freshResults.first?.chunk.sourcePath, loadedResults.first?.chunk.sourcePath)
        XCTAssertEqual(freshResults.first?.distance, loadedResults.first?.distance)
        print(
            "Agent index persistence timing: cold=\(format(cold.preparation.elapsedSeconds))s warm=\(format(warm.preparation.elapsedSeconds))s"
        )
    }

    func testDemoCorpusColdWarmTimingIsMeasured() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = AgentRuntimeFactory.stubIndexProvider(dimension: 64, seed: 42)
        let cold = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider
        )
        let warm = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider
        )

        XCTAssertEqual(cold.preparation.source, .indexed)
        XCTAssertEqual(warm.preparation.source, .loaded)
        XCTAssertEqual(cold.chunkCount, AgentDemoCorpus.notes.count)
        XCTAssertEqual(warm.chunkCount, AgentDemoCorpus.notes.count)
        print(
            "Demo corpus index persistence timing: cold=\(format(cold.preparation.elapsedSeconds))s warm=\(format(warm.preparation.elapsedSeconds))s chunks=\(cold.chunkCount)"
        )
    }

    func testDefaultDemoCorpusColdWarmTimingIsMeasured() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let cold = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory
        )
        let warm = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory
        )

        XCTAssertEqual(cold.preparation.source, .indexed)
        XCTAssertEqual(warm.preparation.source, .loaded)
        XCTAssertEqual(cold.chunkCount, AgentDemoCorpus.notes.count)
        XCTAssertEqual(warm.chunkCount, AgentDemoCorpus.notes.count)
        print(
            "Default demo corpus index persistence timing: embedder=\(cold.summary) cold=\(format(cold.preparation.elapsedSeconds))s warm=\(format(warm.preparation.elapsedSeconds))s chunks=\(cold.chunkCount)"
        )
    }

    func testFingerprintMismatchBuildsNewIndexFile() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = AgentRuntimeFactory.stubIndexProvider(dimension: 64, seed: 102)
        let first = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: Self.documents
        )
        var changedDocuments = Self.documents
        changedDocuments[0] = RAGDocument(
            text: "Project Atlas review lock moved to September and has a different owner.",
            sourcePath: "atlas.md"
        )

        let changed = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: changedDocuments
        )

        XCTAssertEqual(first.preparation.source, .indexed)
        XCTAssertEqual(changed.preparation.source, .indexed)
        XCTAssertNotEqual(first.preparation.fingerprint, changed.preparation.fingerprint)
        XCTAssertNotEqual(first.preparation.storageURL, changed.preparation.storageURL)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.preparation.storageURL.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: changed.preparation.storageURL.path))
    }

    func testCorruptedIndexFileRebuildsInsteadOfCrashing() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = AgentRuntimeFactory.stubIndexProvider(dimension: 64, seed: 103)
        let cold = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: Self.documents
        )
        try Data([0, 1, 2, 3]).write(to: cold.preparation.storageURL, options: .atomic)

        let rebuilt = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: Self.documents
        )
        let results = try await rebuilt.index.search("coffee recipe", k: 1)

        XCTAssertEqual(rebuilt.preparation.source, .indexed)
        XCTAssertEqual(rebuilt.preparation.fingerprint, cold.preparation.fingerprint)
        XCTAssertEqual(results.first?.chunk.sourcePath, "coffee.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: rebuilt.preparation.storageURL.path))
    }

    func testWrongDimensionLoadRebuildsInsteadOfCrashing() async throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = AgentRuntimeFactory.stubIndexProvider(dimension: 64, seed: 104)
        let cold = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: provider,
            documents: Self.documents
        )
        let wrongDimensionOnLoad = AgentIndexProvider(
            summary: provider.summary,
            identity: provider.identity,
            dimension: provider.dimension,
            isStub: provider.isStub,
            makeEmptyIndex: provider.makeEmptyIndex,
            loadIndex: { url in
                try RAGIndex.load(
                    from: url,
                    embedder: DeterministicStubEmbedder(dimension: 16, seed: 104)
                )
            }
        )

        let rebuilt = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: directory,
            provider: wrongDimensionOnLoad,
            documents: Self.documents
        )
        let results = try await rebuilt.index.search("atlas owner mira", k: 1)

        XCTAssertEqual(rebuilt.preparation.source, .indexed)
        XCTAssertEqual(rebuilt.preparation.fingerprint, cold.preparation.fingerprint)
        XCTAssertEqual(results.first?.chunk.sourcePath, "atlas.md")
    }

    func testUnwritableIndexDirectoryFallsBackToInMemoryIndex() async throws {
        let unwritableParent = try makeTemporaryDirectory()
        let indexDirectory = unwritableParent.appendingPathComponent("AgentIndex", isDirectory: true)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o500],
            ofItemAtPath: unwritableParent.path
        )
        defer {
            try? FileManager.default.setAttributes(
                [.posixPermissions: 0o700],
                ofItemAtPath: unwritableParent.path
            )
            try? FileManager.default.removeItem(at: unwritableParent)
        }

        let provider = AgentRuntimeFactory.stubIndexProvider(dimension: 64, seed: 105)
        let built = try await AgentRuntimeFactory.makeIndex(
            tokenizer: tokenizer,
            indexDirectory: indexDirectory,
            provider: provider,
            documents: Self.documents
        )
        let results = try await built.index.search("atlas owner mira", k: 1)

        XCTAssertEqual(built.preparation.source, .indexedNotPersisted)
        XCTAssertEqual(results.first?.chunk.sourcePath, "atlas.md")
        XCTAssertFalse(FileManager.default.fileExists(atPath: built.preparation.storageURL.path))
    }

    private static let documents = [
        RAGDocument(
            text: "Project Atlas review lock is August 14 2026 and the owner is Mira Chen.",
            sourcePath: "atlas.md"
        ),
        RAGDocument(
            text: "The courier coffee recipe uses 18 grams coffee and 288 grams water.",
            sourcePath: "coffee.md"
        ),
        RAGDocument(
            text: "TinyBrain RAG retrieval returns ranked passages with lower distances for stronger matches.",
            sourcePath: "rag.md"
        )
    ]

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainAgentIndexTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private func format(_ elapsed: TimeInterval) -> String {
        String(format: "%.3f", elapsed)
    }
}
