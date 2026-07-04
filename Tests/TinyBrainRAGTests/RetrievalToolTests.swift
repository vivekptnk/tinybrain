import TinyBrainRuntime
import TinyBrainTokenizer
import XCTest
@testable import TinyBrainRAG

private struct ToolTestTokenizer: Tokenizer {
    let vocabularySize = 1_114_112

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(_ tokens: [Int]) -> String {
        String(String.UnicodeScalarView(tokens.compactMap { UnicodeScalar($0) }))
    }
}

private struct EmptyAnswerGenerator: AnswerGenerator {
    func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}

private actor CapturedK {
    private(set) var values: [Int] = []

    func append(_ value: Int) {
        values.append(value)
    }
}

final class RetrievalToolTests: XCTestCase {
    func testRetrieveToolSchemaRequiresQueryAndAllowsOptionalK() {
        let tool = RetrievalTool { _, _ in [] }

        XCTAssertEqual(tool.definition.name, "retrieve")
        guard case .object(let properties, let required) = tool.definition.parameters else {
            return XCTFail("Retrieve tool parameters should be an object schema")
        }

        XCTAssertEqual(required, ["query"])
        XCTAssertEqual(properties.first(where: { $0.name == "query" })?.schema, .string)
        XCTAssertEqual(properties.first(where: { $0.name == "query" })?.required, true)
        XCTAssertEqual(properties.first(where: { $0.name == "k" })?.schema, .integer)
        XCTAssertEqual(properties.first(where: { $0.name == "k" })?.required, false)
    }

    func testRetrieveToolDispatchMapsResultsToNumberedPassages() async throws {
        let passage = RAGTestSupport.passage("TinyBrain retrieval stays local.", rank: 0, sourcePath: "rag.md")
        let tool = RetrievalTool(defaultK: 2, maxK: 4) { query, k in
            XCTAssertEqual(query, "local retrieval")
            XCTAssertEqual(k, 2)
            return [passage]
        }
        let dispatcher = ClosureToolDispatcher()
        tool.register(on: dispatcher)

        let result = try await dispatcher.dispatch(ToolCall(
            id: "call-1",
            name: "retrieve",
            arguments: ["query": "local retrieval", "k": 2]
        ))

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("[1] TinyBrain retrieval stays local."))
        XCTAssertTrue(result.content.contains("source: rag.md"))
    }

    func testRetrieveToolDropsPassagesBeyondDistanceFloor() async throws {
        let tool = RetrievalTool(defaultK: 3, maxK: 3, maxDistance: 0.85) { _, _ in
            [
                RAGTestSupport.passage("Relevant Atlas timing.", rank: 0, distance: 0.581, sourcePath: "atlas.md"),
                RAGTestSupport.passage("Near-boundary Atlas owner.", rank: 1, distance: 0.850, sourcePath: "atlas.md"),
                RAGTestSupport.passage("Noisy kitchen note.", rank: 2, distance: 0.965, sourcePath: "kitchen.md")
            ]
        }

        let output = try await tool.handle(ToolCall(
            id: "filter",
            name: "retrieve",
            arguments: ["query": "Project Atlas timing", "k": 3]
        ))

        XCTAssertTrue(output.contains("Relevant Atlas timing."), output)
        XCTAssertTrue(output.contains("Near-boundary Atlas owner."), output)
        XCTAssertFalse(output.contains("Noisy kitchen note."), output)
        XCTAssertFalse(output.contains("0.965"), output)
    }

    func testRetrieveToolAllPassagesBeyondDistanceFloorReturnsNonErrorNoRelevantMessage() async throws {
        let tool = RetrievalTool(defaultK: 1, maxK: 1, maxDistance: 0.85) { _, _ in
            [
                RAGTestSupport.passage(
                    "Atlas review lock is August 14, 2026.",
                    rank: 0,
                    distance: 0.965,
                    sourcePath: "ops/project-atlas.md"
                )
            ]
        }
        let dispatcher = ClosureToolDispatcher()
        tool.register(on: dispatcher)

        let result = try await dispatcher.dispatch(ToolCall(
            id: "vague",
            name: "retrieve",
            arguments: ["query": "What does it do", "k": 1]
        ))

        XCTAssertFalse(result.isError)
        XCTAssertEqual(
            result.content,
            "No sufficiently relevant passages found for 'What does it do' (best distance 0.965, threshold 0.85). The knowledge base may not cover this topic."
        )
    }

    func testRetrieveToolNilDistanceFloorPreservesHighDistanceResults() async throws {
        let tool = RetrievalTool(defaultK: 1, maxK: 1, maxDistance: nil) { _, _ in
            [
                RAGTestSupport.passage(
                    "High-distance results are still returned when the floor is disabled.",
                    rank: 0,
                    distance: 0.965,
                    sourcePath: "legacy.md"
                )
            ]
        }

        let output = try await tool.handle(ToolCall(
            id: "legacy",
            name: "retrieve",
            arguments: ["query": "legacy behavior", "k": 1]
        ))

        XCTAssertTrue(output.contains("High-distance results are still returned"), output)
        XCTAssertTrue(output.contains("distance: 0.965"), output)
    }

    func testRetrieveToolKeepsPassageExactlyAtDistanceFloor() async throws {
        let tool = RetrievalTool(defaultK: 1, maxK: 1, maxDistance: 0.85) { _, _ in
            [
                RAGTestSupport.passage(
                    "The boundary passage should survive the floor.",
                    rank: 0,
                    distance: 0.85,
                    sourcePath: "boundary.md"
                )
            ]
        }

        let output = try await tool.handle(ToolCall(
            id: "boundary",
            name: "retrieve",
            arguments: ["query": "boundary", "k": 1]
        ))

        XCTAssertTrue(output.contains("The boundary passage should survive the floor."), output)
        XCTAssertTrue(output.contains("distance: 0.850"), output)
    }

    func testRetrieveToolBoundsKToConfiguredRange() async throws {
        let captured = CapturedK()
        let tool = RetrievalTool(defaultK: 3, maxK: 5) { _, k in
            await captured.append(k)
            return []
        }

        _ = try await tool.handle(ToolCall(id: "high", name: "retrieve", arguments: ["query": "alpha", "k": 99]))
        _ = try await tool.handle(ToolCall(id: "low", name: "retrieve", arguments: ["query": "alpha", "k": -2]))
        _ = try await tool.handle(ToolCall(id: "default", name: "retrieve", arguments: ["query": "alpha"]))

        let capturedValues = await captured.values
        XCTAssertEqual(capturedValues, [5, 1, 3])
    }

    func testRetrieveToolMissingQueryReturnsDispatcherErrorResult() async throws {
        let tool = RetrievalTool { _, _ in [] }
        let dispatcher = ClosureToolDispatcher()
        tool.register(on: dispatcher)

        let result = try await dispatcher.dispatch(ToolCall(
            id: "bad-call",
            name: "retrieve",
            arguments: [:]
        ))

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("retrieve tool requires"))
    }

    func testRetrieveToolUsesSameResultsAsEngineRetrieve() async throws {
        let tokenizer = ToolTestTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 31))
        let engine = RAGEngine(index: index, generator: EmptyAnswerGenerator(), tokenizer: tokenizer, retrievalK: 2)
        _ = try await engine.index(
            documents: [
                RAGDocument(
                    text: "Device battery diagnostics and screen calibration stay on device.",
                    sourcePath: "device.md"
                ),
                RAGDocument(
                    text: "Pasta water and olive oil belong in the kitchen note.",
                    sourcePath: "kitchen.md"
                )
            ],
            chunkingConfig: ChunkingConfig(targetTokens: 120, overlapTokens: 0)
        )

        let direct = try await engine.retrieve("battery diagnostics", k: 1)
        let tool = await engine.retrieveTool(defaultK: 1, maxK: 4)
        let toolOutput = try await tool.handle(ToolCall(
            id: "call-2",
            name: "retrieve",
            arguments: ["query": "battery diagnostics", "k": 1]
        ))

        XCTAssertEqual(direct.count, 1)
        XCTAssertTrue(toolOutput.contains(direct[0].chunk.text))
    }
}
