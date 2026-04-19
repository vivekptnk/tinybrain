/// Tool dispatch protocol and closure-based implementation
///
/// **TB-Tool-007:** Dispatches parsed tool calls to registered handlers
///
/// ## Educational Overview
///
/// After the model generates a tool call and it's parsed + validated,
/// we need to actually *run* the tool. The dispatcher routes calls
/// to registered handlers and returns results.
///
/// **Design:** Protocol-oriented so apps can implement custom dispatchers
/// (e.g., sandboxed execution, rate limiting, logging).
///
/// **Example:**
/// ```swift
/// let dispatcher = ClosureToolDispatcher()
/// dispatcher.register("get_weather") { call in
///     let city = call.arguments["city"] as? String ?? "unknown"
///     return "72°F in \(city)"
/// }
/// let result = try await dispatcher.dispatch(toolCall)
/// ```

import Foundation

// MARK: - Dispatch Error

/// Errors produced during tool dispatch
public enum ToolDispatchError: Error, Equatable, CustomStringConvertible {
    /// No handler registered for the given tool name
    case unknownTool(String)
    /// The handler threw an error
    case handlerFailed(String)

    public var description: String {
        switch self {
        case .unknownTool(let name):
            return "No handler registered for tool '\(name)'"
        case .handlerFailed(let message):
            return "Tool handler failed: \(message)"
        }
    }
}

// MARK: - Tool Dispatcher Protocol

/// Protocol for dispatching tool calls to handlers
///
/// Implement this protocol for custom dispatch strategies:
/// sandboxed execution, retry logic, rate limiting, etc.
public protocol ToolDispatcher {
    /// Dispatch a tool call and return the result
    ///
    /// - Parameter call: The parsed tool call to dispatch
    /// - Returns: Result from the tool handler
    /// - Throws: `ToolDispatchError` if the tool is unknown or handler fails
    func dispatch(_ call: ToolCall) async throws -> ToolResult
}

// MARK: - Closure Tool Dispatcher

/// Simple dispatcher that routes tool calls to registered closures
///
/// Ideal for straightforward tool implementations where each tool
/// is a single async function.
public final class ClosureToolDispatcher: ToolDispatcher {
    /// Type alias for tool handler closures
    public typealias Handler = (ToolCall) async throws -> String

    /// Registered handlers keyed by tool name
    private var handlers: [String: Handler] = [:]

    public init() {}

    /// Register a handler for a named tool
    ///
    /// - Parameters:
    ///   - name: Tool name (must match `ToolDefinition.name`)
    ///   - handler: Async closure that processes the call and returns a string result
    public func register(_ name: String, handler: @escaping Handler) {
        handlers[name] = handler
    }

    /// Dispatch a tool call to its registered handler
    ///
    /// - Parameter call: The tool call to dispatch
    /// - Returns: `ToolResult` with the handler's output
    /// - Throws: `ToolDispatchError.unknownTool` if no handler is registered
    public func dispatch(_ call: ToolCall) async throws -> ToolResult {
        guard let handler = handlers[call.name] else {
            throw ToolDispatchError.unknownTool(call.name)
        }

        do {
            let content = try await handler(call)
            return ToolResult(callId: call.id, content: content, isError: false)
        } catch {
            return ToolResult(
                callId: call.id,
                content: "Error: \(error.localizedDescription)",
                isError: true
            )
        }
    }
}

// MARK: - Tool Orchestration Loop

/// Standalone orchestration function for tool-augmented generation
///
/// This is the core loop that coordinates generation, parsing, dispatch,
/// and re-prompting. It's a free function — not baked into ModelRunner.
///
/// **Flow:**
/// 1. Generate text with constrained sampler → detect tool call
/// 2. Parse tool call → validate against schema
/// 3. Dispatch to handler → get result
/// 4. Format result as context for continued generation
/// 5. Repeat until model produces final text or max iterations
///
/// - Parameters:
///   - generatedText: The model's output text to check for tool calls
///   - parser: Parser to extract tool calls from text
///   - dispatcher: Dispatcher to route calls to handlers
/// - Returns: `ToolResult` if a tool call was found and dispatched, nil otherwise
public func processToolCall(
    generatedText: String,
    parser: inout ToolCallParser,
    dispatcher: ToolDispatcher
) async throws -> ToolResult? {
    parser.reset()
    parser.feed(generatedText)

    guard parser.hasCompleteJSON else { return nil }

    let parseResult = parser.extractToolCall()
    switch parseResult {
    case .success(let call):
        return try await dispatcher.dispatch(call)
    case .failure:
        return nil
    }
}

/// Format a tool result as context text for continued generation
///
/// - Parameter result: The tool result to format
/// - Returns: Text that can be appended to the conversation for re-prompting
public func formatToolResult(_ result: ToolResult) -> String {
    if result.isError {
        return "[Tool Error] \(result.content)"
    }
    return "[Tool Result] \(result.content)"
}
