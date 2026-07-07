import Foundation
import TinyBrainRAG
import TinyBrainRuntime

/// Structured output a tool may expose to TinyBrain Agent in addition to prose.
public protocol AgentToolStructuredOutput: Sendable {
    /// Result content returned to the model.
    var content: String { get }

    /// Structured retrieved passages, when this tool produced retrieval output.
    var passages: [RetrievedPassageRecord]? { get }
}

/// Default structured tool output container for agent tool handlers.
public struct AgentToolOutput: AgentToolStructuredOutput, Equatable, Sendable {
    /// Result content returned to the model.
    public let content: String

    /// Structured retrieved passages, when available.
    public let passages: [RetrievedPassageRecord]?

    /// Creates a structured tool output.
    public init(content: String, passages: [RetrievedPassageRecord]? = nil) {
        self.content = content
        self.passages = passages
    }
}

/// Result returned by the agent-aware dispatcher.
public struct AgentToolDispatchResult: Equatable {
    /// Runtime-compatible tool result fed back into the transcript.
    public let result: ToolResult

    /// Optional structured retrieved passages for programmatic consumers.
    public let passages: [RetrievedPassageRecord]?

    /// Creates an agent dispatch result.
    public init(result: ToolResult, passages: [RetrievedPassageRecord]? = nil) {
        self.result = result
        self.passages = passages
    }
}

/// Dispatcher that preserves runtime ``ToolResult`` text and optional agent metadata.
public protocol AgentToolDispatching: AnyObject {
    /// Dispatches a tool call and returns text plus optional structured payloads.
    func dispatch(_ call: ToolCall) async -> AgentToolDispatchResult
}

/// A tool definition paired with its async implementation.
public struct RegisteredTool {
    /// Schema and prompt metadata consumed by the model.
    public let definition: ToolDefinition

    /// Async handler that returns tool output text or throws on failure.
    public let handler: (ToolCall) async throws -> String

    let structuredHandler: ((ToolCall) async throws -> AgentToolOutput)?

    /// Creates a registered tool.
    public init(
        definition: ToolDefinition,
        handler: @escaping (ToolCall) async throws -> String
    ) {
        self.definition = definition
        self.handler = handler
        self.structuredHandler = nil
    }

    /// Creates a registered tool that can return prose plus structured metadata.
    public init<Output: AgentToolStructuredOutput>(
        definition: ToolDefinition,
        structuredHandler: @escaping (ToolCall) async throws -> Output
    ) {
        self.definition = definition
        self.handler = { call in
            try await structuredHandler(call).content
        }
        self.structuredHandler = { call in
            let output = try await structuredHandler(call)
            return AgentToolOutput(content: output.content, passages: output.passages)
        }
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

    /// Builds an agent-aware dispatcher over the currently registered handlers.
    public func agentDispatcher() -> AgentToolDispatching {
        RegistryAgentToolDispatcher(tools: toolsByName)
    }

    /// Builds the tool system-prompt section from the registered definitions.
    public func systemPrompt(choice: ToolChoice) -> String {
        ToolCallingConfig(tools: definitions, toolChoice: choice).buildSystemPrompt()
    }
}

private final class RegistryAgentToolDispatcher: AgentToolDispatching {
    private let tools: [String: RegisteredTool]

    init(tools: [String: RegisteredTool]) {
        self.tools = tools
    }

    func dispatch(_ call: ToolCall) async -> AgentToolDispatchResult {
        guard let tool = tools[call.name] else {
            return AgentToolDispatchResult(
                result: ToolResult(
                    callId: call.id,
                    content: "Error: No handler registered for tool '\(call.name)'",
                    isError: true
                )
            )
        }

        do {
            if let structuredHandler = tool.structuredHandler {
                let output = try await structuredHandler(call)
                return AgentToolDispatchResult(
                    result: ToolResult(callId: call.id, content: output.content, isError: false),
                    passages: output.passages
                )
            }

            return AgentToolDispatchResult(
                result: ToolResult(callId: call.id, content: try await tool.handler(call), isError: false)
            )
        } catch {
            return AgentToolDispatchResult(
                result: ToolResult(
                    callId: call.id,
                    content: "Error: \(error.localizedDescription)",
                    isError: true
                )
            )
        }
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
