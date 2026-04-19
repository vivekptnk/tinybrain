// Tests for the lenient search-output parser.

import Cartographer
import XCTest
@testable import TinyBrainCartographerBridge

final class ResponseParserTests: XCTestCase {

    private let projectID = UUID()

    private func makeCorpus(_ n: Int) -> [Annotation] {
        (0..<n).map { i in
            Annotation(
                type: .pin,
                coordinate: GeoCoordinate(latitude: 0, longitude: 0),
                title: "a\(i)",
                projectID: projectID
            )
        }
    }

    func testParsesPlainArray() {
        let corpus = makeCorpus(3)
        let ids = ResponseParser.parseSearchIndices("[2, 0, 1]", corpus: corpus)
        XCTAssertEqual(ids, [corpus[2].id, corpus[0].id, corpus[1].id])
    }

    func testStripsSurroundingChatter() {
        let corpus = makeCorpus(3)
        let ids = ResponseParser.parseSearchIndices(
            "Answer: [1]. That was easy.",
            corpus: corpus
        )
        XCTAssertEqual(ids, [corpus[1].id])
    }

    func testDropsOutOfRangeIndices() {
        let corpus = makeCorpus(2)
        let ids = ResponseParser.parseSearchIndices("[5, 1, -1, 0]", corpus: corpus)
        XCTAssertEqual(ids, [corpus[1].id, corpus[0].id])
    }

    func testDropsDuplicates() {
        let corpus = makeCorpus(2)
        let ids = ResponseParser.parseSearchIndices("[0, 0, 1, 1, 0]", corpus: corpus)
        XCTAssertEqual(ids, [corpus[0].id, corpus[1].id])
    }

    func testBareNumbersWithoutBrackets() {
        let corpus = makeCorpus(3)
        let ids = ResponseParser.parseSearchIndices("2 then 0 then 1", corpus: corpus)
        XCTAssertEqual(ids, [corpus[2].id, corpus[0].id, corpus[1].id])
    }

    func testEmptyInputReturnsEmpty() {
        let ids = ResponseParser.parseSearchIndices("", corpus: makeCorpus(3))
        XCTAssertTrue(ids.isEmpty)
    }

    func testEmptyCorpusReturnsEmpty() {
        let ids = ResponseParser.parseSearchIndices("[1, 2, 3]", corpus: [])
        XCTAssertTrue(ids.isEmpty)
    }
}
