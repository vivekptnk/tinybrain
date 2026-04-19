/// Tool call parser for extracting function calls from model output
///
/// **TB-Tool-005:** Parses model-generated text to extract structured tool calls
///
/// ## Educational Overview
///
/// When a model generates a tool call, it produces JSON text like:
/// ```json
/// {"name": "get_weather", "arguments": {"city": "Paris"}}
/// ```
///
/// The parser handles:
/// - Complete JSON extraction from surrounding text
/// - Streaming accumulation (tokens arrive one at a time)
/// - Schema validation of extracted arguments
/// - Error reporting with context
///
/// **Usage:**
/// ```swift
/// var parser = ToolCallParser()
/// parser.feed("Here's the tool call: ")
/// parser.feed("{\"name\": \"get_weather\"")
/// parser.feed(", \"arguments\": {\"city\": \"Paris\"}}")
/// if let call = parser.extractToolCall() { ... }
/// ```

import Foundation

// MARK: - Parse Error

/// Errors produced during tool call parsing
public enum ToolCallParseError: Error, Equatable, CustomStringConvertible {
    /// No valid JSON object found in the text
    case noJSONFound
    /// JSON is valid but missing required fields
    case missingField(String)
    /// JSON parsing failed
    case invalidJSON(String)
    /// Arguments don't match the tool's parameter schema
    case schemaViolation(SchemaError)

    public var description: String {
        switch self {
        case .noJSONFound:
            return "No valid JSON tool call found in output"
        case .missingField(let field):
            return "Tool call JSON missing required field: '\(field)'"
        case .invalidJSON(let detail):
            return "Invalid JSON in tool call: \(detail)"
        case .schemaViolation(let error):
            return "Tool call arguments violate schema: \(error)"
        }
    }
}

// MARK: - Tool Call Parser

/// Stateful parser that accumulates tokens and extracts tool calls
///
/// Designed for streaming: feed tokens one at a time via `feed(_:)`,
/// then check `extractToolCall()` to see if a complete call has been parsed.
public struct ToolCallParser {
    /// Accumulated text buffer
    private var buffer: String = ""

    /// Counter for generating unique call IDs
    private var callCounter: Int = 0

    /// Available tool definitions for schema validation
    private let tools: [ToolDefinition]

    /// Initialize parser with optional tool definitions for validation
    ///
    /// - Parameter tools: Tool definitions to validate against (empty = skip validation)
    public init(tools: [ToolDefinition] = []) {
        self.tools = tools
    }

    /// Feed a token string into the parser buffer
    public mutating func feed(_ token: String) {
        buffer += token
    }

    /// Reset the parser buffer
    public mutating func reset() {
        buffer = ""
    }

    /// The current accumulated buffer contents
    public var currentBuffer: String { buffer }

    /// Attempt to extract a tool call from the accumulated buffer
    ///
    /// Scans for the outermost `{...}` in the buffer. If found and valid,
    /// returns a `ToolCall`. Otherwise returns a parse error.
    ///
    /// - Returns: Parsed `ToolCall` or error explaining why parsing failed
    public mutating func extractToolCall() -> Result<ToolCall, ToolCallParseError> {
        guard let jsonString = extractJSONObject(from: buffer) else {
            return .failure(.noJSONFound)
        }

        guard let data = jsonString.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .failure(.invalidJSON("Could not parse JSON object"))
        }

        guard let name = json["name"] as? String else {
            return .failure(.missingField("name"))
        }

        let arguments: [String: Any]
        if let args = json["arguments"] as? [String: Any] {
            arguments = args
        } else if let argsString = json["arguments"] as? String,
                  let argsData = argsString.data(using: .utf8),
                  let argsDict = try? JSONSerialization.jsonObject(with: argsData) as? [String: Any] {
            // Handle case where arguments is a JSON string
            arguments = argsDict
        } else {
            arguments = [:]
        }

        // Validate against schema if tool definition is available
        if let tool = tools.first(where: { $0.name == name }) {
            let validationResult = SchemaValidator.validate(arguments, against: tool.parameters)
            if case .failure(let schemaError) = validationResult {
                return .failure(.schemaViolation(schemaError))
            }
        }

        callCounter += 1
        let callId = "call_\(callCounter)"
        let call = ToolCall(id: callId, name: name, arguments: arguments)
        return .success(call)
    }

    /// Check if the buffer contains a potentially complete JSON object
    public var hasCompleteJSON: Bool {
        extractJSONObject(from: buffer) != nil
    }

    // MARK: - JSON Extraction

    /// Extract the first complete JSON object from text
    ///
    /// Finds matching `{` ... `}` pairs, handling nested objects and strings.
    private func extractJSONObject(from text: String) -> String? {
        guard let startIdx = text.firstIndex(of: "{") else { return nil }

        var depth = 0
        var inString = false
        var escaped = false
        var endIdx: String.Index?

        for idx in text[startIdx...].indices {
            let char = text[idx]

            if escaped {
                escaped = false
                continue
            }

            if char == "\\" && inString {
                escaped = true
                continue
            }

            if char == "\"" {
                inString.toggle()
                continue
            }

            if inString { continue }

            if char == "{" {
                depth += 1
            } else if char == "}" {
                depth -= 1
                if depth == 0 {
                    endIdx = idx
                    break
                }
            }
        }

        guard let end = endIdx else { return nil }
        return String(text[startIdx...end])
    }
}
