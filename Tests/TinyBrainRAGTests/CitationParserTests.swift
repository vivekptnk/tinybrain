import XCTest
@testable import TinyBrainRAG

final class CitationParserTests: XCTestCase {
    func testParseSingleCitationMapsToPassage() {
        let passages = [RAGTestSupport.passage("Local inference.", rank: 0)]
        let parser = CitationParser(passages: passages)

        let citations = parser.parse("TinyBrain runs locally [1].")

        XCTAssertEqual(citations.count, 1)
        XCTAssertEqual(citations[0].marker, 1)
        XCTAssertEqual(citations[0].passage, passages[0])
    }

    func testParseMultipleCitations() {
        let passages = (0..<3).map { RAGTestSupport.passage("passage \($0)", rank: $0) }
        let parser = CitationParser(passages: passages)

        let citations = parser.parse("First [1], then [2] and [3].")

        XCTAssertEqual(citations.map(\.marker), [1, 2, 3])
        XCTAssertEqual(citations.compactMap(\.passage), passages)
    }

    func testParseDuplicateMarkersKeepsEachOccurrence() {
        let passages = [RAGTestSupport.passage("Repeated.", rank: 0)]
        let parser = CitationParser(passages: passages)

        let citations = parser.parse("Repeated [1]. Still repeated [1].")

        XCTAssertEqual(citations.map(\.marker), [1, 1])
        XCTAssertEqual(citations.map(\.passage), [passages[0], passages[0]])
    }

    func testParseOutOfRangeCitationHasNilPassage() {
        let parser = CitationParser(passages: [RAGTestSupport.passage("Only one.", rank: 0)])

        let citations = parser.parse("Unsupported claim [99].")

        XCTAssertEqual(citations.count, 1)
        XCTAssertEqual(citations[0].marker, 99)
        XCTAssertNil(citations[0].passage)
    }

    func testParseZeroCitationHasNilPassage() {
        let parser = CitationParser(passages: [RAGTestSupport.passage("Only one.", rank: 0)])

        let citations = parser.parse("Unsupported claim [0].")

        XCTAssertEqual(citations.count, 1)
        XCTAssertEqual(citations[0].marker, 0)
        XCTAssertNil(citations[0].passage)
    }

    func testParseMalformedMarkersIgnored() {
        let parser = CitationParser(passages: [RAGTestSupport.passage("Only one.", rank: 0)])

        let citations = parser.parse("Ignore [a], [1,2], [ ], and unclosed [1.")

        XCTAssertEqual(citations, [])
    }

    func testParseNoCitationsReturnsEmpty() {
        let parser = CitationParser(passages: [RAGTestSupport.passage("Only one.", rank: 0)])

        XCTAssertEqual(parser.parse("No citations here."), [])
    }

    func testParseTwoDigitMarker() {
        let passages = (0..<12).map { RAGTestSupport.passage("passage \($0)", rank: $0) }
        let parser = CitationParser(passages: passages)

        let citations = parser.parse("The twelfth passage applies [12].")

        XCTAssertEqual(citations.count, 1)
        XCTAssertEqual(citations[0].marker, 12)
        XCTAssertEqual(citations[0].passage, passages[11])
    }

    func testParseCitationRangeMatchesAnswerSubstring() {
        let parser = CitationParser(passages: [RAGTestSupport.passage("Only one.", rank: 0)])
        let answer = "TinyBrain cites [1] precisely."

        let citation = parser.parse(answer)[0]

        XCTAssertEqual(String(answer[citation.range]), "[1]")
    }
}
