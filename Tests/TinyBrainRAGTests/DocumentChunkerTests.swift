import Foundation
import XCTest
@testable import TinyBrainRAG

final class DocumentChunkerTests: XCTestCase {
    func testChunkEmptyDocumentReturnsNoChunks() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(tokenizer: tokenizer)

        XCTAssertEqual(chunker.chunk("", sourcePath: "empty.md"), [])
        XCTAssertEqual(chunker.chunk("  \n\t  ", sourcePath: "empty.md"), [])
    }

    func testChunkSmallDocumentProducesSingleChunk() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(
            tokenizer: tokenizer,
            config: ChunkingConfig(targetTokens: 32, overlapTokens: 4)
        )
        let text = "Hello world!"

        let chunks = chunker.chunk(text, sourcePath: "hello.md")

        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].text, text)
        XCTAssertEqual(chunks[0].sourcePath, "hello.md")
        XCTAssertEqual(chunks[0].ordinal, 0)
        XCTAssertEqual(chunks[0].tokenRange.count, tokenizer.encode(text).count)
    }

    func testChunkPrefersParagraphBoundary() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(
            tokenizer: tokenizer,
            config: ChunkingConfig(targetTokens: 12, overlapTokens: 0)
        )
        let text = "Hello world!\n\nTinyBrain TinyBrain TinyBrain!"

        let chunks = chunker.chunk(text, sourcePath: "paragraphs.md")

        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].text, "Hello world!")
        XCTAssertTrue(chunks[1].text.hasPrefix("TinyBrain"))
    }

    func testChunkPrefersSentenceBoundary() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(
            tokenizer: tokenizer,
            config: ChunkingConfig(targetTokens: 10, overlapTokens: 0)
        )
        let text = "Hello world! TinyBrain TinyBrain TinyBrain! café café café."

        let chunks = chunker.chunk(text, sourcePath: "sentences.md")

        XCTAssertGreaterThanOrEqual(chunks.count, 2)
        XCTAssertEqual(chunks[0].text, "Hello world!")
        XCTAssertTrue(chunks[1].text.hasPrefix("TinyBrain"))
    }

    func testChunkHardSplitsTextWithoutBoundaries() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let config = ChunkingConfig(targetTokens: 9, overlapTokens: 2)
        let chunker = DocumentChunker(tokenizer: tokenizer, config: config)
        let text = String(repeating: "abcdefghij", count: 4)

        let chunks = chunker.chunk(text, sourcePath: "runon.md")

        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { tokenizer.encode($0.text).count <= config.targetTokens })
    }

    func testChunkOverlapsAdjacentTokenRanges() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(
            tokenizer: tokenizer,
            config: ChunkingConfig(targetTokens: 10, overlapTokens: 3)
        )
        let text = String(repeating: "abcdefghij", count: 5)

        let chunks = chunker.chunk(text, sourcePath: "overlap.md")

        XCTAssertGreaterThan(chunks.count, 2)
        for index in chunks.indices.dropFirst() {
            XCTAssertLessThan(chunks[index].tokenRange.lowerBound, chunks[index - 1].tokenRange.upperBound)
            XCTAssertGreaterThan(chunks[index].tokenRange.upperBound, chunks[index - 1].tokenRange.upperBound)
        }
    }

    func testChunkTokenBudgetRespectedForEveryChunk() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let config = ChunkingConfig(targetTokens: 14, overlapTokens: 4)
        let chunker = DocumentChunker(tokenizer: tokenizer, config: config)
        let text = """
        Hello world! TinyBrain TinyBrain TinyBrain.
        café café café café café. Hello world!
        """

        let chunks = chunker.chunk(text, sourcePath: "budget.md")

        XCTAssertFalse(chunks.isEmpty)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(tokenizer.encode(chunk.text).count, config.targetTokens)
        }
    }

    func testChunkHandlesUnicodeText() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(
            tokenizer: tokenizer,
            config: ChunkingConfig(targetTokens: 8, overlapTokens: 2)
        )

        let chunks = chunker.chunk("café café café. TinyBrain café!", sourcePath: "unicode.md")

        XCTAssertFalse(chunks.isEmpty)
        XCTAssertTrue(chunks.map(\.text).joined(separator: " ").contains("café"))
    }

    func testChunkFileAtURLUsesFilePath() throws {
        let tokenizer = try RAGTestSupport.tokenizer()
        let chunker = DocumentChunker(
            tokenizer: tokenizer,
            config: ChunkingConfig(targetTokens: 32, overlapTokens: 4)
        )
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainRAGTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let fileURL = directory.appendingPathComponent("note.md")
        try "Hello world!".write(to: fileURL, atomically: true, encoding: .utf8)

        let chunks = try chunker.chunk(fileAt: fileURL)

        XCTAssertEqual(chunks.count, 1)
        XCTAssertEqual(chunks[0].sourcePath, fileURL.path)
    }
}
