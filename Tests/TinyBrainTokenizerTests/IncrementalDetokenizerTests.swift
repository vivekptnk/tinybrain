import XCTest
@testable import TinyBrainTokenizer

private struct BoundaryTokenizer: Tokenizer {
    let vocabularySize = 32

    private let pieces: [Int: String] = [
        10: "▁Hello",
        11: "▁world",
        12: "."
    ]

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(_ tokens: [Int]) -> String {
        var decoded = ""
        for token in tokens {
            decoded += pieces[token]?.replacingOccurrences(of: "▁", with: " ") ?? ""
        }
        if decoded.hasPrefix(" ") {
            decoded.removeFirst()
        }
        return decoded
    }
}

final class IncrementalDetokenizerTests: XCTestCase {
    func testAppendEmitsStableSuffixDeltasAcrossSentencePieceBoundaries() {
        var detokenizer = IncrementalDetokenizer(tokenizer: BoundaryTokenizer())

        let deltas = [10, 11, 12].compactMap { detokenizer.append($0) }

        XCTAssertEqual(deltas.joined(), "Hello world.")
        XCTAssertEqual(detokenizer.decodedText, "Hello world.")
        XCTAssertFalse(deltas.joined().contains("Helloworld"))
    }

    func testAppendReturnsNilWhenFullDecodeNoLongerHasStablePrefix() {
        struct RewritingTokenizer: Tokenizer {
            let vocabularySize = 4

            func encode(_ text: String) -> [Int] { [] }

            func decode(_ tokens: [Int]) -> String {
                tokens.count == 1 ? "draft" : "final"
            }
        }

        var detokenizer = IncrementalDetokenizer(tokenizer: RewritingTokenizer())

        XCTAssertEqual(detokenizer.append(0), "draft")
        XCTAssertNil(detokenizer.append(1))
        XCTAssertEqual(detokenizer.decodedText, "final")
    }
}
