/// Tool calling configuration and system prompt generation
///
/// **TB-Tool-006:** Configuration for tool-augmented generation
///
/// ## Educational Overview
///
/// When using tool calling, the model needs to know:
/// 1. What tools are available (names, descriptions, parameter schemas)
/// 2. Whether it *must* call a tool, *may* call a tool, or *should not*
///
/// This configuration gets injected into the system prompt so the model
/// understands its available capabilities.
///
/// **Example:**
/// ```swift
/// let config = ToolCallingConfig(
///     tools: [weatherTool, calculatorTool],
///     toolChoice: .auto
/// )
/// let systemPrompt = config.buildSystemPrompt()
/// ```

import Foundation

// MARK: - Tool Choice

/// Controls whether and how the model should use tools
public enum ToolChoice: Equatable {
    /// Model decides whether to call a tool (default behavior)
    case auto
    /// Model must NOT call any tool (text-only response)
    case none
    /// Model MUST call at least one tool
    case required
    /// Model MUST call this specific tool
    case specific(name: String)
}

// MARK: - Tool Calling Config

/// Configuration for tool-augmented generation
public struct ToolCallingConfig: Equatable {
    /// Available tool definitions
    public let tools: [ToolDefinition]

    /// How the model should choose tools
    public let toolChoice: ToolChoice

    /// Maximum number of tool call rounds before forcing a text response
    public let maxIterations: Int

    public init(
        tools: [ToolDefinition],
        toolChoice: ToolChoice = .auto,
        maxIterations: Int = 5
    ) {
        self.tools = tools
        self.toolChoice = toolChoice
        self.maxIterations = maxIterations
    }

    /// Build a system prompt section describing available tools
    ///
    /// Generates a structured prompt that tells the model what tools
    /// are available and how to invoke them.
    ///
    /// - Returns: System prompt text for tool descriptions
    public func buildSystemPrompt() -> String {
        guard !tools.isEmpty else { return "" }

        var prompt = "You have access to the following tools:\n\n"

        for tool in tools {
            prompt += "### \(tool.name)\n"
            prompt += "\(tool.description)\n"
            prompt += "Parameters: \(schemaDescription(tool.parameters))\n\n"
        }

        prompt += "To call a tool, respond with a JSON object:\n"
        prompt += "{\"name\": \"tool_name\", \"arguments\": {...}}\n\n"

        switch toolChoice {
        case .auto:
            prompt += "You may call a tool if it would help answer the user's request."
        case .none:
            prompt += "Do NOT call any tools. Respond with text only."
        case .required:
            prompt += "You MUST call at least one tool to respond."
        case .specific(let name):
            prompt += "You MUST call the '\(name)' tool."
        }

        return prompt
    }

    // MARK: - Schema Description

    /// Generate a human-readable description of a JSON schema for prompting
    private func schemaDescription(_ schema: JSONSchema) -> String {
        switch schema {
        case .string: return "string"
        case .number: return "number"
        case .integer: return "integer"
        case .boolean: return "boolean"
        case .null: return "null"
        case .enum(let values): return "enum(\(values.joined(separator: "|")))"
        case .array(let items): return "array<\(schemaDescription(items))>"
        case .object(let properties, let required):
            let props = properties.map { prop in
                let req = required.contains(prop.name) ? " (required)" : ""
                let desc = prop.description.map { " — \($0)" } ?? ""
                return "  \(prop.name): \(schemaDescription(prop.schema))\(req)\(desc)"
            }
            return "{\n\(props.joined(separator: "\n"))\n}"
        }
    }
}
