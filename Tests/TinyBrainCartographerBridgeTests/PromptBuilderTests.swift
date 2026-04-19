// Tests for the search / summarize prompt shaping.

import Cartographer
import XCTest
@testable import TinyBrainCartographerBridge

final class PromptBuilderTests: XCTestCase {

    private let projectID = UUID()

    private func annotation(
        title: String = "",
        body: String = "",
        metadata: [String: String] = [:]
    ) -> Annotation {
        Annotation(
            type: .pin,
            coordinate: GeoCoordinate(latitude: 0, longitude: 0),
            title: title,
            body: body,
            metadata: metadata,
            projectID: projectID
        )
    }

    // MARK: - Search

    func testSearchPromptIncludesQueryAndNumberedCorpus() {
        let a = annotation(title: "Alpha", body: "first")
        let b = annotation(title: "Bravo", body: "second")
        let prompt = PromptBuilder.search(query: "needle", corpus: [a, b])

        XCTAssertTrue(prompt.contains("Query: needle"))
        XCTAssertTrue(prompt.contains("0: Alpha — first"))
        XCTAssertTrue(prompt.contains("1: Bravo — second"))
        XCTAssertTrue(prompt.contains("Answer:"))
    }

    func testSearchPromptHandlesUntitledAnnotations() {
        let a = annotation()
        let prompt = PromptBuilder.search(query: "x", corpus: [a])
        XCTAssertTrue(prompt.contains("0: [untitled pin]"))
    }

    func testSearchPromptIncludesMetadataValuesSortedByKey() {
        let a = annotation(title: "T", metadata: ["zzz": "last", "aaa": "first"])
        let prompt = PromptBuilder.search(query: "x", corpus: [a])
        XCTAssertTrue(prompt.contains("aaa=first, zzz=last"))
    }

    // MARK: - Summarize

    func testSummarizePromptAsksForShortProse() {
        let a = annotation(title: "A", body: "body a")
        let prompt = PromptBuilder.summarize(annotations: [a])
        XCTAssertTrue(prompt.contains("three short"))
        XCTAssertTrue(prompt.contains("Do not use lists"))
        XCTAssertTrue(prompt.contains("- A — body a"))
        XCTAssertTrue(prompt.contains("Summary:"))
    }
}
