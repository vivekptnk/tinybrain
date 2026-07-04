import Foundation
import TinyBrainRAG
import TinyBrainRuntime

/// Factory for the agent runtime's built-in tools.
public enum BuiltInAgentTools {
    /// Creates the `retrieve` tool backed by TinyBrainRAG's retrieval adapter.
    public static func retrieve(_ retrievalTool: RetrievalTool) -> RegisteredTool {
        RegisteredTool(definition: retrievalTool.definition) { call in
            try await retrievalTool.handle(call)
        }
    }

    /// Creates the `read_file` tool gated by a sandbox policy.
    public static func readFile(policy: SandboxPolicy) -> RegisteredTool {
        RegisteredTool(definition: readFileDefinition) { call in
            let path = try stringArgument("path", in: call)
            return try policy.readFile(atPath: path)
        }
    }

    /// Creates the `list_dir` tool gated by a sandbox policy.
    public static func listDirectory(policy: SandboxPolicy) -> RegisteredTool {
        RegisteredTool(definition: listDirectoryDefinition) { call in
            let path = try stringArgument("path", in: call)
            return try policy.listDirectory(atPath: path)
        }
    }

    /// Creates the `write_file` tool gated by a sandbox policy.
    public static func writeFile(policy: SandboxPolicy) -> RegisteredTool {
        RegisteredTool(definition: writeFileDefinition) { call in
            let path = try stringArgument("path", in: call)
            let content = try stringArgument("content", in: call)
            return try policy.writeFile(atPath: path, content: content)
        }
    }

    /// Creates the pure `current_time` tool.
    public static func currentTime(clock: @escaping () -> Date = Date.init) -> RegisteredTool {
        RegisteredTool(definition: currentTimeDefinition) { _ in
            let date = clock()
            let utcFormatter = ISO8601DateFormatter()
            utcFormatter.timeZone = TimeZone(secondsFromGMT: 0)

            let localFormatter = ISO8601DateFormatter()
            localFormatter.timeZone = .current

            return AgentJSON.objectString([
                "utc": utcFormatter.string(from: date),
                "local": localFormatter.string(from: date),
                "timeZone": TimeZone.current.identifier
            ])
        }
    }

    /// Schema for `read_file`.
    public static let readFileDefinition = ToolDefinition(
        name: "read_file",
        description: "Read a UTF-8 text file from an explicitly granted sandbox root.",
        parameters: .object(properties: [
            JSONSchemaProperty(
                name: "path",
                schema: .string,
                description: "Absolute path or sandbox-relative path to read",
                required: true
            )
        ], required: ["path"])
    )

    /// Schema for `list_dir`.
    public static let listDirectoryDefinition = ToolDefinition(
        name: "list_dir",
        description: "List visible entries in a directory under an explicitly granted sandbox root.",
        parameters: .object(properties: [
            JSONSchemaProperty(
                name: "path",
                schema: .string,
                description: "Absolute path or sandbox-relative directory path to list",
                required: true
            )
        ], required: ["path"])
    )

    /// Schema for `write_file`.
    public static let writeFileDefinition = ToolDefinition(
        name: "write_file",
        description: "Write UTF-8 text to a file under an explicitly granted writable sandbox root.",
        parameters: .object(properties: [
            JSONSchemaProperty(
                name: "path",
                schema: .string,
                description: "Absolute path or sandbox-relative file path to write",
                required: true
            ),
            JSONSchemaProperty(
                name: "content",
                schema: .string,
                description: "UTF-8 text to write",
                required: true
            )
        ], required: ["path", "content"])
    )

    /// Schema for `current_time`.
    public static let currentTimeDefinition = ToolDefinition(
        name: "current_time",
        description: "Return the current local and UTC time in ISO-8601 format.",
        parameters: .object(properties: [], required: [])
    )

    private static func stringArgument(_ name: String, in call: ToolCall) throws -> String {
        guard let value = call.arguments[name] as? String, !value.isEmpty else {
            throw SandboxPolicyError.missingArgument(name)
        }
        return value
    }
}
