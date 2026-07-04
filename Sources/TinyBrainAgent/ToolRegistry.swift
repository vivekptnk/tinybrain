import Foundation
import TinyBrainRuntime

/// A tool definition paired with its async implementation.
public struct RegisteredTool {
    /// Schema and prompt metadata consumed by the model.
    public let definition: ToolDefinition

    /// Async handler that returns tool output text or throws on failure.
    public let handler: (ToolCall) async throws -> String

    /// Creates a registered tool.
    public init(
        definition: ToolDefinition,
        handler: @escaping (ToolCall) async throws -> String
    ) {
        self.definition = definition
        self.handler = handler
    }
}

/// Single source of truth for agent tool schemas, parsing, and dispatch.
public actor ToolRegistry {
    private var toolsByName: [String: RegisteredTool] = [:]
    private var insertionOrder: [String] = []

    /// Creates an empty registry.
    public init() {}

    /// Registers or replaces a tool by name.
    public func register(_ tool: RegisteredTool) {
        if toolsByName[tool.definition.name] == nil {
            insertionOrder.append(tool.definition.name)
        }
        toolsByName[tool.definition.name] = tool
    }

    /// Tool definitions in registration order.
    public var definitions: [ToolDefinition] {
        insertionOrder.compactMap { toolsByName[$0]?.definition }
    }

    /// Builds a parser over the registered tool definitions.
    public func parser() -> ToolCallParser {
        ToolCallParser(tools: definitions)
    }

    /// Builds a dispatcher over the currently registered handlers.
    public func dispatcher() -> ToolDispatcher {
        RegistryToolDispatcher(
            handlers: toolsByName.mapValues(\.handler)
        )
    }

    /// Builds the tool system-prompt section from the registered definitions.
    public func systemPrompt(choice: ToolChoice) -> String {
        ToolCallingConfig(tools: definitions, toolChoice: choice).buildSystemPrompt()
    }
}

private final class RegistryToolDispatcher: ToolDispatcher {
    private let handlers: [String: (ToolCall) async throws -> String]

    init(handlers: [String: (ToolCall) async throws -> String]) {
        self.handlers = handlers
    }

    func dispatch(_ call: ToolCall) async throws -> ToolResult {
        guard let handler = handlers[call.name] else {
            return ToolResult(
                callId: call.id,
                content: "Error: No handler registered for tool '\(call.name)'",
                isError: true
            )
        }

        do {
            return ToolResult(
                callId: call.id,
                content: try await handler(call),
                isError: false
            )
        } catch {
            return ToolResult(
                callId: call.id,
                content: "Error: \(error.localizedDescription)",
                isError: true
            )
        }
    }
}
