import Foundation
import XCTest
import TinyBrainTokenizer
@testable import TinyBrainRAG

enum RAGTestSupport {
    static func tokenizer(file: StaticString = #filePath, line: UInt = #line) throws -> any Tokenizer {
        let url = try XCTUnwrap(
            Bundle.module.url(forResource: "test_vocab", withExtension: "json"),
            file: file,
            line: line
        )
        return try TokenizerLoader.load(from: url.path)
    }

    static func chunk(
        _ text: String,
        sourcePath: String = "notes.md",
        ordinal: Int = 0
    ) -> DocumentChunk {
        let length = max(1, text.count)
        return DocumentChunk(
            text: text,
            sourcePath: sourcePath,
            tokenRange: (ordinal * 100)..<(ordinal * 100 + length),
            ordinal: ordinal
        )
    }

    static func passage(
        _ text: String,
        rank: Int,
        distance: Float? = nil,
        sourcePath: String = "notes.md"
    ) -> RetrievedPassage {
        RetrievedPassage(
            chunk: chunk(text, sourcePath: sourcePath, ordinal: rank),
            distance: distance ?? Float(rank) / 10,
            rank: rank
        )
    }
}
