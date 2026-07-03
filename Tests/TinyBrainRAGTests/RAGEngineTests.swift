import Foundation
import TinyBrainRuntime
import TinyBrainTokenizer
import XCTest
@testable import TinyBrainRAG

private struct CharacterTokenizer: Tokenizer {
    let vocabularySize = 1_114_112

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(_ tokens: [Int]) -> String {
        String(String.UnicodeScalarView(tokens.compactMap { UnicodeScalar($0) }))
    }
}

private struct SentencePieceBoundaryTokenizer: Tokenizer {
    let vocabularySize = 1_114_112
    private let pieces: [Int: String] = [
        10: "▁Steep",
        11: "▁cold",
        12: "▁brew",
        13: "▁for",
        14: "▁16",
        15: "▁hours",
        16: "▁[1]",
        17: "."
    ]

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(_ tokens: [Int]) -> String {
        var decoded = ""
        for token in tokens {
            if let piece = pieces[token] {
                decoded += piece.replacingOccurrences(of: "▁", with: " ")
            } else if let scalar = UnicodeScalar(token) {
                decoded += String(scalar)
            }
        }

        if decoded.hasPrefix(" ") {
            decoded.removeFirst()
        }
        return decoded
    }
}

private actor ScriptedGeneratorState {
    private(set) var prompts: [[Int]] = []
    private(set) var maxTokens: [Int] = []
    private(set) var stopTokens: [[Int]] = []
    private(set) var yieldedTokens = 0
    private(set) var terminations = 0

    func record(prompt: [Int], config: GenerationConfig) {
        prompts.append(prompt)
        maxTokens.append(config.maxTokens)
        stopTokens.append(config.stopTokens)
    }

    func tokenYielded() {
        yieldedTokens += 1
    }

    func terminated() {
        terminations += 1
    }
}

private struct ScriptedGenerator: AnswerGenerator {
    let tokenIDs: [Int]
    let delayNanoseconds: UInt64
    let state: ScriptedGeneratorState

    init(
        answer: String,
        tokenizer: any Tokenizer,
        delayNanoseconds: UInt64 = 0,
        state: ScriptedGeneratorState = ScriptedGeneratorState()
    ) {
        self.tokenIDs = tokenizer.encode(answer)
        self.delayNanoseconds = delayNanoseconds
        self.state = state
    }

    init(
        tokenIDs: [Int],
        delayNanoseconds: UInt64 = 0,
        state: ScriptedGeneratorState = ScriptedGeneratorState()
    ) {
        self.tokenIDs = tokenIDs
        self.delayNanoseconds = delayNanoseconds
        self.state = state
    }

    func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    await state.record(prompt: prompt, config: config)
                    for tokenID in tokenIDs.prefix(config.maxTokens) {
                        try Task.checkCancellation()
                        if delayNanoseconds > 0 {
                            try await Task.sleep(nanoseconds: delayNanoseconds)
                        }
                        await state.tokenYielded()
                        continuation.yield(TokenOutput(tokenId: tokenID, probability: 1))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
                Task {
                    await state.terminated()
                }
            }
        }
    }
}

final class RAGEngineTests: XCTestCase {
    func testAnswerEndToEndResolvesCitationsToRetrievedPassages() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 21))
        let generator = ScriptedGenerator(
            answer: "Battery diagnostics stay local [1].",
            tokenizer: tokenizer
        )
        let engine = RAGEngine(index: index, generator: generator, tokenizer: tokenizer, retrievalK: 2)
        _ = try await engine.index(documents: sampleDocuments(), chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0))

        let response = try await engine.answer("How do battery diagnostics work?")

        XCTAssertEqual(response.passages.first?.chunk.sourcePath, "device.md")
        XCTAssertEqual(response.answer, "Battery diagnostics stay local [1].")
        XCTAssertEqual(response.citations.count, 1)
        XCTAssertEqual(response.citations[0].marker, 1)
        XCTAssertEqual(response.citations[0].passage?.chunk.sourcePath, response.passages[0].chunk.sourcePath)
    }

    func testAnswerFeedsNumberedBudgetedPromptToGenerator() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 22))
        let state = ScriptedGeneratorState()
        let generator = ScriptedGenerator(answer: "I know from the first passage [1].", tokenizer: tokenizer, state: state)
        let builder = RAGPromptBuilder(
            tokenizer: tokenizer,
            budget: PromptBudget(contextWindow: 470, generationHeadroom: 80)
        )
        let engine = RAGEngine(
            index: index,
            generator: generator,
            tokenizer: tokenizer,
            promptBuilder: builder,
            retrievalK: 1,
            generationConfig: GenerationConfig(maxTokens: 120)
        )
        _ = try await engine.index(
            documents: [RAGDocument(text: "Battery diagnostics stay local.", sourcePath: "device.md")],
            chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0)
        )

        _ = try await engine.answer("Battery diagnostics?")

        let prompts = await state.prompts
        let prompt = try XCTUnwrap(prompts.first)
        let decoded = tokenizer.decode(prompt)
        XCTAssertTrue(decoded.contains("[1]"))
        XCTAssertTrue(decoded.contains("Question:"))
        XCTAssertTrue(decoded.contains("Answer:"))
        XCTAssertLessThanOrEqual(prompt.count, 390)
        let maxTokens = await state.maxTokens
        XCTAssertEqual(maxTokens.first, 80)
    }

    func testAnswerWithZephyrTemplatePassesWrappedPromptAndSingleBOSToGenerator() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 29))
        let state = ScriptedGeneratorState()
        let generator = ScriptedGenerator(
            answer: "Cold brew should steep for 16 hours [1].",
            tokenizer: tokenizer,
            state: state
        )
        let engine = RAGEngine(
            index: index,
            generator: generator,
            tokenizer: tokenizer,
            promptTemplate: .zephyr,
            retrievalK: 1
        )
        _ = try await engine.index(
            documents: [
                RAGDocument(
                    text: "Cold brew coffee should steep for 16 hours.",
                    sourcePath: "coffee.md"
                )
            ],
            chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0)
        )

        _ = try await engine.answer("How long should I steep cold brew coffee?")

        let prompts = await state.prompts
        let prompt = try XCTUnwrap(prompts.first)
        XCTAssertEqual(prompt.first, 1)
        XCTAssertNotEqual(prompt.dropFirst().first, 1)

        let decoded = tokenizer.decode(Array(prompt.dropFirst()))
        XCTAssertTrue(decoded.hasPrefix("<|system|>\n"))
        XCTAssertTrue(decoded.contains("[1] Cold brew coffee should steep for 16 hours."))
        XCTAssertTrue(decoded.contains("For example, cite like: The steep time is 16 hours [1].</s>"))
        XCTAssertTrue(decoded.contains("</s>\n<|user|>\nHow long should I steep cold brew coffee?</s>\n<|assistant|>\n"))
        XCTAssertFalse(decoded.contains("Question:"))
        XCTAssertFalse(decoded.contains("Answer:"))
    }

    func testAnswerWithNoCitationsReturnsEmptyCitationList() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 23))
        let generator = ScriptedGenerator(answer: "The answer has no marker.", tokenizer: tokenizer)
        let engine = RAGEngine(index: index, generator: generator, tokenizer: tokenizer)
        _ = try await engine.index(documents: sampleDocuments(), chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0))

        let response = try await engine.answer("What does TinyBrain do locally?")

        XCTAssertEqual(response.answer, "The answer has no marker.")
        XCTAssertEqual(response.citations, [])
        XCTAssertFalse(response.passages.isEmpty)
    }

    func testAnswerOnEmptyIndexDoesNotCallGenerator() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 24))
        let state = ScriptedGeneratorState()
        let generator = ScriptedGenerator(answer: "Should not run [1].", tokenizer: tokenizer, state: state)
        let engine = RAGEngine(index: index, generator: generator, tokenizer: tokenizer)

        let response = try await engine.answer("Anything indexed?")

        XCTAssertEqual(response.passages, [])
        XCTAssertEqual(response.citations, [])
        XCTAssertTrue(response.answer.contains("could not find any relevant passages"))
        let promptCount = await state.prompts.count
        XCTAssertEqual(promptCount, 0)
    }

    func testAnswerStreamEmitsPassagesTokensThenDone() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 25))
        let generator = ScriptedGenerator(answer: "Local answer [1].", tokenizer: tokenizer)
        let engine = RAGEngine(index: index, generator: generator, tokenizer: tokenizer, retrievalK: 1)
        _ = try await engine.index(documents: sampleDocuments(), chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0))

        var events: [RAGEvent] = []
        for try await event in engine.answerStream("How is TinyBrain local?") {
            events.append(event)
        }

        guard case .passages(let passages) = events.first else {
            return XCTFail("First event should contain retrieved passages")
        }
        XCTAssertFalse(passages.isEmpty)
        XCTAssertTrue(events.dropFirst().contains { event in
            if case .token = event { return true }
            return false
        })
        guard case .done(let citations) = events.last else {
            return XCTFail("Last event should contain citations")
        }
        XCTAssertEqual(citations.count, 1)
    }

    func testAnswerStreamDetokenizesAccumulatedSentencePieceBoundaries() async throws {
        let tokenizer = SentencePieceBoundaryTokenizer()
        let generatedIDs = [10, 11, 12, 13, 14, 15, 16, 17]
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 28))
        let generator = ScriptedGenerator(tokenIDs: generatedIDs)
        let engine = RAGEngine(index: index, generator: generator, tokenizer: tokenizer, retrievalK: 1)
        _ = try await engine.index(
            documents: [RAGDocument(text: "Cold brew coffee should steep for 16 hours.", sourcePath: "coffee.md")],
            chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0)
        )

        var streamed = ""
        var citations: [Citation] = []
        for try await event in engine.answerStream("How long should I steep cold brew coffee?") {
            switch event {
            case .token(let token):
                streamed += token
            case .done(let parsed):
                citations = parsed
            case .passages:
                break
            }
        }

        let fullDecode = tokenizer.decode(generatedIDs)
        XCTAssertEqual(streamed, fullDecode)
        XCTAssertTrue(streamed.contains("Steep cold brew for 16 hours"))
        XCTAssertFalse(streamed.contains("Steepcoldbrew"))
        XCTAssertEqual(citations.map(\.marker), [1])
    }

    func testAnswerStreamCancellationStopsGenerator() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 26))
        let state = ScriptedGeneratorState()
        let generator = ScriptedGenerator(
            answer: String(repeating: "x", count: 200),
            tokenizer: tokenizer,
            delayNanoseconds: 5_000_000,
            state: state
        )
        let engine = RAGEngine(index: index, generator: generator, tokenizer: tokenizer, retrievalK: 1)
        _ = try await engine.index(documents: sampleDocuments(), chunkingConfig: ChunkingConfig(targetTokens: 180, overlapTokens: 0))

        let consumer = Task {
            var tokenCount = 0
            for try await event in engine.answerStream("How is TinyBrain local?") {
                if case .token = event {
                    tokenCount += 1
                    if tokenCount == 1 {
                        break
                    }
                }
            }
            return tokenCount
        }

        let consumed = try await consumer.value
        try await Task.sleep(nanoseconds: 30_000_000)

        XCTAssertEqual(consumed, 1)
        let yieldedTokens = await state.yieldedTokens
        let terminations = await state.terminations
        XCTAssertLessThan(yieldedTokens, 200)
        XCTAssertGreaterThan(terminations, 0)
    }

    func testIndexFolderRecursivelyIndexesTextAndMarkdownFiles() async throws {
        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 27))
        let engine = RAGEngine(index: index, generator: ScriptedGenerator(answer: "", tokenizer: tokenizer), tokenizer: tokenizer)
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainRAGEngineFolder-\(UUID().uuidString)")
        let nested = directory.appendingPathComponent("nested")
        try FileManager.default.createDirectory(at: nested, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let rootNote = directory.appendingPathComponent("root.md")
        let nestedNote = nested.appendingPathComponent("nested.txt")
        let ignored = nested.appendingPathComponent("ignored.json")
        try "TinyBrain folder indexing note.".write(to: rootNote, atomically: true, encoding: .utf8)
        try "Nested retrieval note for local search.".write(to: nestedNote, atomically: true, encoding: .utf8)
        try "{\"ignored\":true}".write(to: ignored, atomically: true, encoding: .utf8)

        let chunks = try await engine.index(
            folderAt: directory,
            chunkingConfig: ChunkingConfig(targetTokens: 120, overlapTokens: 0)
        )

        XCTAssertEqual(chunks.count, 2)
        let indexedPaths = Set(chunks.map { URL(fileURLWithPath: $0.sourcePath).resolvingSymlinksInPath().path })
        let expectedPaths = Set([rootNote, nestedNote].map { $0.resolvingSymlinksInPath().path })
        XCTAssertEqual(indexedPaths, expectedPaths)
        let indexCount = await index.count
        XCTAssertEqual(indexCount, 2)
    }

    private func sampleDocuments() -> [RAGDocument] {
        [
            RAGDocument(
                text: "Apple battery repair notes mention screen calibration and private device diagnostics.",
                sourcePath: "device.md"
            ),
            RAGDocument(
                text: "Sourdough starter care uses flour, water, and a warm kitchen shelf.",
                sourcePath: "kitchen.md"
            ),
            RAGDocument(
                text: "TinyBrain keeps local inference transparent with Swift actors and on-device retrieval.",
                sourcePath: "runtime.md"
            )
        ]
    }
}
