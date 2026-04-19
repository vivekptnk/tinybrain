/// Tool/function calling data types
///
/// **TB-Tool-004:** Foundation types for tool registration, invocation, and results
///
/// ## Educational Overview
///
/// Tool calling lets LLMs invoke external functions. The model generates
/// a structured JSON call (`ToolCall`), the app dispatches it to the
/// matching handler, and the result (`ToolResult`) feeds back into
/// continued generation.
///
/// **Flow:**
/// ```
/// Model output → ToolCall (parsed JSON) → Dispatch → ToolResult → Resume
/// ```
///
/// **Example:**
/// ```swift
/// let tool = ToolDefinition(
///     name: "get_weather",
///     description: "Get weather for a city",
///     parameters: .object(properties: [
///         JSONSchemaProperty(name: "location", schema: .string, description: "City name", required: true)
///     ], required: ["location"])
/// )
/// ```

import Foundation

// MARK: - Tool Definition

/// Describes a tool the model can call
///
/// Each tool has a name, description, and a JSON Schema describing
/// the expected arguments. The schema is used both for prompt
/// generation (so the model knows the API) and for validation
/// (so malformed calls are caught before dispatch).
public struct ToolDefinition: Equatable, Codable {
    /// Unique tool name (used by the model to invoke it)
    public let name: String

    /// Human-readable description of what the tool does
    public let description: String

    /// JSON Schema describing the function parameters (must be `.object`)
    public let parameters: JSONSchema

    public init(
        name: String,
        description: String,
        parameters: JSONSchema
    ) {
        self.name = name
        self.description = description
        self.parameters = parameters
    }
}

// MARK: - Tool Call

/// A parsed tool invocation extracted from model output
///
/// The model generates JSON matching `{"name": "...", "arguments": {...}}`.
/// `ToolCallParser` converts that text into a `ToolCall` value.
public struct ToolCall: Equatable {
    /// Unique ID for correlating call → result
    public let id: String

    /// Name of the tool to invoke (must match a `ToolDefinition.name`)
    public let name: String

    /// Parsed argument values keyed by parameter name
    ///
    /// Values are JSON-compatible types: `String`, `NSNumber`, `[Any]`,
    /// `[String: Any]`, `NSNull`. Use `SchemaValidator` to type-check.
    public let arguments: [String: Any]

    public init(id: String, name: String, arguments: [String: Any]) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    // Manual Equatable — [String: Any] isn't automatically Equatable
    public static func == (lhs: ToolCall, rhs: ToolCall) -> Bool {
        guard lhs.id == rhs.id, lhs.name == rhs.name else { return false }
        guard lhs.arguments.count == rhs.arguments.count else { return false }
        // Compare JSON-serialized forms for deep equality
        guard let lData = try? JSONSerialization.data(withJSONObject: lhs.arguments),
              let rData = try? JSONSerialization.data(withJSONObject: rhs.arguments) else {
            return false
        }
        return lData == rData
    }
}

// MARK: - Tool Result

/// The outcome of dispatching a `ToolCall` to its handler
///
/// Carries the result content (or error message) back to the
/// generation loop so the model can incorporate it.
public struct ToolResult: Equatable {
    /// Correlates back to `ToolCall.id`
    public let callId: String

    /// Result content (success message or error description)
    public let content: String

    /// Whether this result represents an error
    public let isError: Bool

    public init(callId: String, content: String, isError: Bool = false) {
        self.callId = callId
        self.content = content
        self.isError = isError
    }
}
