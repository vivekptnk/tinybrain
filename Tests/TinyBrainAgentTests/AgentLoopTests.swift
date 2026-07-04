import Foundation
import TinyBrainMetal
import TinyBrainRAG
import TinyBrainRuntime
import TinyBrainTokenizer
import XCTest
@testable import TinyBrainAgent

private struct CharacterTokenizer: Tokenizer {
    let vocabularySize = 1_114_112

    func encode(_ text: String) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(_ tokens: [Int]) -> String {
        String(String.UnicodeScalarView(tokens.compactMap { UnicodeScalar($0) }))
    }
}

private actor ScriptedAgentGenerator: AgentGenerating {
    private var outputs: [String]
    private let delayNanoseconds: UInt64
    private(set) var requests: [AgentGenerationRequest] = []

    init(outputs: [String], delayNanoseconds: UInt64 = 0) {
        self.outputs = outputs
        self.delayNanoseconds = delayNanoseconds
    }

    func generate(_ request: AgentGenerationRequest) async throws -> AgentGenerationResult {
        requests.append(request)
        if delayNanoseconds > 0 {
            try await Task.sleep(nanoseconds: delayNanoseconds)
        }
        try Task.checkCancellation()
        let output = outputs.isEmpty ? "" : outputs.removeFirst()
        return AgentGenerationResult(text: output, tokenCount: output.count)
    }
}

private final class TraceRecorder: AgentTraceObserver {
    private let lock = NSLock()
    private var storage: [String] = []

    var lines: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func stepStarted(_ event: AgentStepStarted) {
        append("stepStarted index=\(event.index) promptTokens=\(event.promptTokens)")
    }

    func toolCallProposed(_ event: AgentToolCallProposed) {
        append("toolCallProposed index=\(event.index) name=\(event.call.name) args=\(event.call.arguments)")
    }

    func toolExecuted(_ event: AgentToolExecuted) {
        append("toolExecuted index=\(event.index) isError=\(event.result.isError) content=\(event.result.content)")
    }

    func finalAnswer(_ event: AgentFinalAnswer) {
        append("finalAnswer answer=\(event.answer)")
    }

    func budgetExhausted(_ event: AgentBudgetExhausted) {
        append("budgetExhausted maxSteps=\(event.maxSteps)")
    }

    func cancelled(_ event: AgentCancelled) {
        append("cancelled reason=\(event.reason)")
    }

    private func append(_ line: String) {
        lock.lock()
        storage.append(line)
        lock.unlock()
    }
}

private struct NoopAnswerGenerator: AnswerGenerator {
    func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}

final class AgentLoopTests: XCTestCase {
    func testLoopTerminatesOnFinalAnswer() async throws {
        let registry = ToolRegistry()
        let generator = ScriptedAgentGenerator(outputs: ["The answer is local."])
        let loop = AgentLoop(
            generator: generator,
            tokenizer: CharacterTokenizer(),
            registry: registry,
            config: AgentConfig(maxSteps: 3)
        )

        let events = try await collectEvents(from: loop.run("Answer directly."))

        guard case .finalAnswer(let final) = events.last else {
            return XCTFail("Expected final answer event, got \(events)")
        }
        XCTAssertEqual(final.answer, "The answer is local.")
        XCTAssertEqual(final.transcript.terminationReason, .finalAnswer)
        XCTAssertEqual(final.transcript.steps, [])
    }

    func testLoopRespectsMaxStepsAndForcesFinalAnswer() async throws {
        // Prove-can-fail: this was verified RED during development by allowing
        // `stepIndex <= maxSteps`; the scripted second tool call executed, and
        // this test saw two `toolExecuted` events instead of one.
        let registry = ToolRegistry()
        await registry.register(RegisteredTool(definition: echoDefinition) { call in
            call.arguments["text"] as? String ?? ""
        })
        let generator = ScriptedAgentGenerator(outputs: [
            #"{"name":"echo","arguments":{"text":"first"}}"#,
            "Forced answer from observed tool result."
        ])
        let loop = AgentLoop(
            generator: generator,
            tokenizer: CharacterTokenizer(),
            registry: registry,
            config: AgentConfig(maxSteps: 1)
        )

        let events = try await collectEvents(from: loop.run("Use the tool once."))

        XCTAssertEqual(events.toolExecutions.count, 1)
        XCTAssertEqual(events.budgetExhaustions.count, 1)
        XCTAssertEqual(events.finalAnswers.last?.answer, "Forced answer from observed tool result.")
        XCTAssertEqual(events.finalAnswers.last?.transcript.steps.map(\.toolCall?.name), ["echo"])
    }

    func testToolErrorsLoopBackAsToolResults() async throws {
        enum FailingTool: Error { case failed }

        let registry = ToolRegistry()
        await registry.register(RegisteredTool(definition: echoDefinition) { _ in
            throw FailingTool.failed
        })
        let generator = ScriptedAgentGenerator(outputs: [
            #"{"name":"echo","arguments":{"text":"fail"}}"#,
            "I saw the tool error."
        ])
        let loop = AgentLoop(
            generator: generator,
            tokenizer: CharacterTokenizer(),
            registry: registry,
            config: AgentConfig(maxSteps: 3)
        )

        let events = try await collectEvents(from: loop.run("Call failing tool."))

        let execution = try XCTUnwrap(events.toolExecutions.first)
        XCTAssertTrue(execution.result.isError)
        XCTAssertTrue(execution.result.content.contains("Error:"))
        XCTAssertEqual(events.finalAnswers.last?.answer, "I saw the tool error.")
    }

    func testSandboxDenialIsObservedAsErrorResultNotCrash() async throws {
        let fixture = try AgentLoopFixture()
        try "secret".write(to: fixture.outsideFile, atomically: true, encoding: .utf8)
        let policy = SandboxPolicy(readableRoots: [fixture.root])
        let registry = ToolRegistry()
        await registry.register(BuiltInAgentTools.readFile(policy: policy))
        let generator = ScriptedAgentGenerator(outputs: [
            #"{"name":"read_file","arguments":{"path":"\#(fixture.outsideFile.path)"}}"#,
            "I cannot read that file."
        ])
        let loop = AgentLoop(
            generator: generator,
            tokenizer: CharacterTokenizer(),
            registry: registry,
            config: AgentConfig(maxSteps: 2)
        )

        let events = try await collectEvents(from: loop.run("Read a denied file."))

        let execution = try XCTUnwrap(events.toolExecutions.first)
        XCTAssertTrue(execution.result.isError)
        XCTAssertTrue(execution.result.content.contains("outside the readable sandbox"))
        XCTAssertEqual(events.finalAnswers.last?.answer, "I cannot read that file.")
    }

    func testCancellationStopsWithinOneStep() async throws {
        let generator = ScriptedAgentGenerator(
            outputs: [#"{"name":"echo","arguments":{"text":"slow"}}"#],
            delayNanoseconds: 1_000_000_000
        )
        let registry = ToolRegistry()
        await registry.register(RegisteredTool(definition: echoDefinition) { call in
            call.arguments["text"] as? String ?? ""
        })
        let loop = AgentLoop(
            generator: generator,
            tokenizer: CharacterTokenizer(),
            registry: registry,
            config: AgentConfig(maxSteps: 4)
        )

        let task = Task {
            for try await _ in loop.run("Start, then cancel.") {}
        }

        try await Task.sleep(nanoseconds: 50_000_000)
        task.cancel()
        _ = await task.result

        let requests = await generator.requests
        XCTAssertLessThanOrEqual(requests.count, 1)
    }

    func testRetrieveThenReadFileProducesTranscriptAndTrace() async throws {
        let fixture = try AgentLoopFixture()
        let factFile = fixture.root.appendingPathComponent("facts.md")
        try "The offline launch phrase is Aurora Quartz.".write(
            to: factFile,
            atomically: true,
            encoding: .utf8
        )
        try "Unrelated notes.".write(
            to: fixture.root.appendingPathComponent("other.md"),
            atomically: true,
            encoding: .utf8
        )

        let tokenizer = CharacterTokenizer()
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 64, seed: 42))
        let engine = RAGEngine(
            index: index,
            generator: NoopAnswerGenerator(),
            tokenizer: tokenizer
        )
        _ = try await engine.index(
            folderAt: fixture.root,
            chunkingConfig: ChunkingConfig(targetTokens: 120, overlapTokens: 0)
        )

        let registry = ToolRegistry()
        await registry.register(BuiltInAgentTools.retrieve(engine.retrieveTool(defaultK: 1, maxK: 2)))
        await registry.register(BuiltInAgentTools.readFile(policy: SandboxPolicy(readableRoots: [fixture.root])))

        let generator = ScriptedAgentGenerator(outputs: [
            #"{"name":"retrieve","arguments":{"query":"offline launch phrase","k":1}}"#,
            #"{"name":"read_file","arguments":{"path":"\#(factFile.path)"}}"#,
            "The offline launch phrase is Aurora Quartz."
        ])
        let observer = TraceRecorder()
        let loop = AgentLoop(
            generator: generator,
            tokenizer: tokenizer,
            registry: registry,
            config: AgentConfig(maxSteps: 3),
            observer: observer
        )

        let events = try await collectEvents(from: loop.run("What is the offline launch phrase?"))

        XCTAssertEqual(events.toolExecutions.map(\.result.isError), [false, false])
        XCTAssertEqual(events.finalAnswers.last?.answer, "The offline launch phrase is Aurora Quartz.")
        XCTAssertEqual(events.finalAnswers.last?.transcript.steps.map(\.toolCall?.name), ["retrieve", "read_file"])
        XCTAssertEqual(
            observer.lines.map { $0.components(separatedBy: " ").first ?? "" },
            ["stepStarted", "toolCallProposed", "toolExecuted", "stepStarted", "toolCallProposed", "toolExecuted", "stepStarted", "finalAnswer"]
        )
    }

    func testQwenRetrieveSmokeUsesRealModelWhenEnabled() async throws {
        guard ProcessInfo.processInfo.environment["TINYBRAIN_RUN_QWEN_SMOKE"] == "1" else {
            throw XCTSkip("Set TINYBRAIN_RUN_QWEN_SMOKE=1 to run the real Qwen agent smoke")
        }

        let modelPath = "Models/qwen2.5-1.5b-int8.tbf"
        let tokenizerPath = "Models/qwen2.5-1.5b-raw/tokenizer.json"
        guard FileManager.default.fileExists(atPath: resolveProjectPath(modelPath)) else {
            throw XCTSkip("\(modelPath) not available")
        }
        guard FileManager.default.fileExists(atPath: resolveProjectPath(tokenizerPath)) else {
            throw XCTSkip("\(tokenizerPath) not available")
        }

        if MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let fixture = try AgentLoopFixture()
        let factFile = fixture.root.appendingPathComponent("aster.md")
        try """
        Project Aster offline launch code phrase is Aurora Quartz.
        This fact is only available in the local document.
        """.write(to: factFile, atomically: true, encoding: .utf8)
        try "Project Borealis uses a different placeholder phrase.".write(
            to: fixture.root.appendingPathComponent("borealis.md"),
            atomically: true,
            encoding: .utf8
        )
        try "TinyBrain agents execute tools on-device without network calls.".write(
            to: fixture.root.appendingPathComponent("agent.md"),
            atomically: true,
            encoding: .utf8
        )

        let weights = try ModelLoader.load(from: resolveProjectPath(modelPath))
        let tokenizer = try TokenizerLoader.loadHuggingFace(from: resolveProjectPath(tokenizerPath))
        let index = RAGIndex(embedder: DeterministicStubEmbedder(dimension: 96, seed: 73))
        let engine = RAGEngine(index: index, generator: NoopAnswerGenerator(), tokenizer: tokenizer)
        _ = try await engine.index(
            folderAt: fixture.root,
            chunkingConfig: ChunkingConfig(targetTokens: 160, overlapTokens: 0)
        )

        let registry = ToolRegistry()
        await registry.register(BuiltInAgentTools.retrieve(engine.retrieveTool(defaultK: 2, maxK: 3)))
        let observer = TraceRecorder()
        let loop = AgentLoop(
            generator: ModelRunnerAgentGenerator(
                runner: ModelRunner(weights: weights),
                tokenizer: tokenizer
            ),
            tokenizer: tokenizer,
            registry: registry,
            config: AgentConfig(
                maxSteps: 1,
                toolChoice: .required,
                constraintMode: .strict,
                perStepTokenBudget: 160,
                contextBudget: 2_048,
                sampler: SamplerConfig(temperature: 0.0, topK: 1)
            ),
            observer: observer
        )

        let events = try await collectEvents(
            from: loop.run("Use retrieval to answer: what is Project Aster's offline launch code phrase?")
        )

        let retrieveCalls = events.toolCalls.filter { $0.call.name == "retrieve" }
        XCTAssertGreaterThanOrEqual(retrieveCalls.count, 1)
        let final = try XCTUnwrap(events.finalAnswers.last)
        XCTAssertTrue(final.answer.lowercased().contains("aurora quartz"), final.answer)

        print("TINYBRAIN_AGENT_TRACE_BEGIN")
        for line in observer.lines {
            print(line)
        }
        print("TINYBRAIN_AGENT_TRACE_END")
    }

    private var echoDefinition: ToolDefinition {
        ToolDefinition(
            name: "echo",
            description: "Echoes input text.",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "text", schema: .string, required: true)
            ], required: ["text"])
        )
    }

    private func collectEvents(
        from stream: AsyncThrowingStream<AgentEvent, Error>
    ) async throws -> [AgentEvent] {
        var events: [AgentEvent] = []
        for try await event in stream {
            events.append(event)
        }
        return events
    }

    private func resolveProjectPath(_ relativePath: String) -> String {
        let cwd = FileManager.default.currentDirectoryPath
        let direct = URL(fileURLWithPath: relativePath)
        if FileManager.default.fileExists(atPath: direct.path) {
            return direct.path
        }
        return URL(fileURLWithPath: cwd).appendingPathComponent(relativePath).path
    }
}

private extension Array where Element == AgentEvent {
    var toolCalls: [AgentToolCallProposed] {
        compactMap {
            if case .toolCallProposed(let event) = $0 { return event }
            return nil
        }
    }

    var toolExecutions: [AgentToolExecuted] {
        compactMap {
            if case .toolExecuted(let event) = $0 { return event }
            return nil
        }
    }

    var finalAnswers: [AgentFinalAnswer] {
        compactMap {
            if case .finalAnswer(let event) = $0 { return event }
            return nil
        }
    }

    var budgetExhaustions: [AgentBudgetExhausted] {
        compactMap {
            if case .budgetExhausted(let event) = $0 { return event }
            return nil
        }
    }
}

private struct AgentLoopFixture {
    let root: URL
    let outsideFile: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainAgentLoopTests-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("root", isDirectory: true)
        let outside = base.appendingPathComponent("outside", isDirectory: true)
        outsideFile = outside.appendingPathComponent("secret.txt")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    }
}
