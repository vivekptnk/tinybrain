import Foundation
import TinyBrainRuntime
import XCTest
@testable import TinyBrainAgent

private final class AuditRecorder {
    private let lock = NSLock()
    private var storage: [AuditEvent] = []

    var events: [AuditEvent] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func record(_ event: AuditEvent) {
        lock.lock()
        storage.append(event)
        lock.unlock()
    }
}

final class ToolRegistryTests: XCTestCase {
    func testRegisterDefinitionsAndDispatcherStayInSync() async throws {
        let registry = ToolRegistry()
        let definition = ToolDefinition(
            name: "echo",
            description: "Echoes input text.",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "text", schema: .string, required: true)
            ], required: ["text"])
        )

        await registry.register(RegisteredTool(definition: definition) { call in
            call.arguments["text"] as? String ?? ""
        })

        let definitions = await registry.definitions
        XCTAssertEqual(definitions, [definition])

        let dispatcher = await registry.dispatcher()
        let result = try await dispatcher.dispatch(
            ToolCall(id: "call_1", name: "echo", arguments: ["text": "hello"])
        )

        XCTAssertEqual(result, ToolResult(callId: "call_1", content: "hello"))
    }

    func testUnknownToolReturnsErrorToolResult() async throws {
        let registry = ToolRegistry()
        let dispatcher = await registry.dispatcher()

        let result = try await dispatcher.dispatch(
            ToolCall(id: "call_unknown", name: "missing", arguments: [:])
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("missing"))
    }

    func testSystemPromptIncludesEachToolOnce() async {
        let registry = ToolRegistry()
        await registry.register(RegisteredTool(definition: currentTimeDefinition) { _ in "now" })
        await registry.register(RegisteredTool(definition: echoDefinition) { _ in "echo" })

        let prompt = await registry.systemPrompt(choice: .auto)

        XCTAssertEqual(prompt.components(separatedBy: "### current_time").count - 1, 1)
        XCTAssertEqual(prompt.components(separatedBy: "### echo").count - 1, 1)
        XCTAssertTrue(prompt.contains("You may call a tool"))
    }

    private var currentTimeDefinition: ToolDefinition {
        ToolDefinition(
            name: "current_time",
            description: "Returns the current time.",
            parameters: .object(properties: [], required: [])
        )
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
}

final class SandboxPolicyTests: XCTestCase {
    func testReadFileAdmitsGrantedRootAndAuditsEffect() async throws {
        let fixture = try SandboxFixture()
        try fixture.write("notes/local.md", contents: "TinyBrain keeps agent traces local.")
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(
            readableRoots: [fixture.root],
            audit: recorder.record
        )

        let result = try await dispatch(
            BuiltInAgentTools.readFile(policy: policy),
            arguments: ["path": fixture.url("notes/local.md").path]
        )

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("agent traces local"))
        XCTAssertEqual(recorder.events.map(\.decision), [.allowed])
        XCTAssertEqual(recorder.events.first?.operation, .readFile)
    }

    func testTraversalEscapeIsRejectedAndAudited() async throws {
        // Prove-can-fail: this was verified RED during development by changing
        // SandboxPolicy to prefix-check the raw path before standardization; the
        // traversal path was then admitted and this assertion failed.
        let fixture = try SandboxFixture()
        try fixture.writeOutside("secret.txt", contents: "outside")
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(readableRoots: [fixture.root], audit: recorder.record)
        let escapedPath = fixture.root
            .appendingPathComponent("..")
            .appendingPathComponent(fixture.outside.lastPathComponent)
            .appendingPathComponent("secret.txt")
            .path

        let result = try await dispatch(
            BuiltInAgentTools.readFile(policy: policy),
            arguments: ["path": escapedPath]
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("outside the readable sandbox"))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testSymlinkEscapeIsRejectedAndAudited() async throws {
        let fixture = try SandboxFixture()
        try fixture.writeOutside("secret.txt", contents: "outside")
        let link = fixture.url("link.txt")
        try FileManager.default.createSymbolicLink(
            at: link,
            withDestinationURL: fixture.outside.appendingPathComponent("secret.txt")
        )
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(readableRoots: [fixture.root], audit: recorder.record)

        let result = try await dispatch(
            BuiltInAgentTools.readFile(policy: policy),
            arguments: ["path": link.path]
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("outside the readable sandbox"))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testWriteFileSymlinkedDirectoryNewFileIsRejectedAndAudited() async throws {
        let fixture = try SandboxFixture()
        let symlinkedDirectory = fixture.url("sub")
        try FileManager.default.createSymbolicLink(at: symlinkedDirectory, withDestinationURL: fixture.outside)
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(
            readableRoots: [fixture.root],
            writableRoots: [fixture.root],
            audit: recorder.record
        )

        let result = try await dispatch(
            BuiltInAgentTools.writeFile(policy: policy),
            arguments: [
                "path": fixture.url("sub/evil.txt").path,
                "content": "outside write"
            ]
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("outside the writable sandbox"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.outside.appendingPathComponent("evil.txt").path))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testWriteFileSymlinkedDirectoryIntermediateCreationIsRejectedAndAudited() async throws {
        let fixture = try SandboxFixture()
        let symlinkedDirectory = fixture.url("sub")
        try FileManager.default.createSymbolicLink(at: symlinkedDirectory, withDestinationURL: fixture.outside)
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(
            readableRoots: [fixture.root],
            writableRoots: [fixture.root],
            audit: recorder.record
        )

        let result = try await dispatch(
            BuiltInAgentTools.writeFile(policy: policy),
            arguments: [
                "path": fixture.url("sub/nested/evil.txt").path,
                "content": "outside write"
            ]
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("outside the writable sandbox"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.outside.appendingPathComponent("nested").path))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testReadFileSymlinkedDirectoryMissingPathIsRejectedAndAudited() async throws {
        let fixture = try SandboxFixture()
        let symlinkedDirectory = fixture.url("sub")
        try FileManager.default.createSymbolicLink(at: symlinkedDirectory, withDestinationURL: fixture.outside)
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(readableRoots: [fixture.root], audit: recorder.record)

        let result = try await dispatch(
            BuiltInAgentTools.readFile(policy: policy),
            arguments: ["path": fixture.url("sub/missing.txt").path]
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("outside the readable sandbox"))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testListDirSymlinkedDirectoryMissingPathIsRejectedAndAudited() async throws {
        let fixture = try SandboxFixture()
        let symlinkedDirectory = fixture.url("sub")
        try FileManager.default.createSymbolicLink(at: symlinkedDirectory, withDestinationURL: fixture.outside)
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(readableRoots: [fixture.root], audit: recorder.record)

        let result = try await dispatch(
            BuiltInAgentTools.listDirectory(policy: policy),
            arguments: ["path": fixture.url("sub/missing").path]
        )

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.content.contains("outside the readable sandbox"))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testWriteWithoutGrantReturnsErrorToolResult() async throws {
        let fixture = try SandboxFixture()
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(readableRoots: [fixture.root], audit: recorder.record)

        let result = try await dispatch(
            BuiltInAgentTools.writeFile(policy: policy),
            arguments: [
                "path": fixture.url("created.txt").path,
                "content": "should not write"
            ]
        )

        XCTAssertTrue(result.isError)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.url("created.txt").path))
        XCTAssertEqual(recorder.events.map(\.decision), [.denied])
    }

    func testDryRunWritePerformsNoWriteAndAuditsDryRun() async throws {
        let fixture = try SandboxFixture()
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(
            readableRoots: [fixture.root],
            writableRoots: [fixture.root],
            dryRun: true,
            audit: recorder.record
        )

        let result = try await dispatch(
            BuiltInAgentTools.writeFile(policy: policy),
            arguments: [
                "path": fixture.url("created.txt").path,
                "content": "dry run"
            ]
        )

        XCTAssertTrue(result.isError)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fixture.url("created.txt").path))
        XCTAssertEqual(recorder.events.map(\.decision), [.dryRun])
    }

    func testReadFileHonorsSizeCap() async throws {
        let fixture = try SandboxFixture()
        try fixture.write("large.txt", contents: "abcdef")
        let policy = SandboxPolicy(readableRoots: [fixture.root], maxReadBytes: 3)

        let result = try await dispatch(
            BuiltInAgentTools.readFile(policy: policy),
            arguments: ["path": fixture.url("large.txt").path]
        )

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.hasPrefix("abc"))
        XCTAssertTrue(result.content.contains("[truncated"))
    }

    func testListDirReturnsEntriesAndAuditsEffect() async throws {
        let fixture = try SandboxFixture()
        try fixture.write("a.txt", contents: "a")
        try fixture.write("nested/b.txt", contents: "b")
        let recorder = AuditRecorder()
        let policy = SandboxPolicy(readableRoots: [fixture.root], audit: recorder.record)

        let result = try await dispatch(
            BuiltInAgentTools.listDirectory(policy: policy),
            arguments: ["path": fixture.root.path]
        )

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("a.txt"))
        XCTAssertTrue(result.content.contains("nested/"))
        XCTAssertEqual(recorder.events.map(\.decision), [.allowed])
    }

    func testCurrentTimeReturnsISO8601Payload() async throws {
        let fixed = Date(timeIntervalSince1970: 1_704_067_200)

        let result = try await dispatch(
            BuiltInAgentTools.currentTime(clock: { fixed }),
            arguments: [:]
        )

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.content.contains("\"utc\""))
        XCTAssertTrue(result.content.contains("2024-01-01T00:00:00Z"))
    }

    private func dispatch(
        _ tool: RegisteredTool,
        arguments: [String: Any],
        file: StaticString = #filePath,
        line: UInt = #line
    ) async throws -> ToolResult {
        let registry = ToolRegistry()
        await registry.register(tool)
        let dispatcher = await registry.dispatcher()
        return try await dispatcher.dispatch(
            ToolCall(id: "call_1", name: tool.definition.name, arguments: arguments)
        )
    }
}

private struct SandboxFixture {
    let root: URL
    let outside: URL

    init() throws {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainAgentTests-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("root", isDirectory: true)
        outside = base.appendingPathComponent("outside", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
    }

    func url(_ relativePath: String) -> URL {
        root.appendingPathComponent(relativePath)
    }

    func write(_ relativePath: String, contents: String) throws {
        let destination = url(relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try contents.write(to: destination, atomically: true, encoding: .utf8)
    }

    func writeOutside(_ relativePath: String, contents: String) throws {
        let destination = outside.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try contents.write(to: destination, atomically: true, encoding: .utf8)
    }
}
