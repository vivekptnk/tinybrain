/// Public API surface for TinyBrain tool calling
///
/// **TB-Tool-008:** Clean re-exports and ergonomic builder API
///
/// ## Educational Overview
///
/// This file provides the public API that app developers interact with.
/// It re-exports all tool calling types and provides a result-builder DSL
/// for ergonomic tool registration.
///
/// **Example:**
/// ```swift
/// let tools = ToolKit {
///     Tool("get_weather") {
///         Parameter("location", type: .string, description: "City name")
///         Parameter("unit", type: .enum(values: ["celsius", "fahrenheit"]))
///     } description: "Get current weather for a location"
/// }
/// ```

import Foundation

// MARK: - Parameter Builder

/// A parameter descriptor for the builder DSL
public struct Parameter {
    /// Parameter name
    public let name: String
    /// Parameter type schema
    public let type: JSONSchema
    /// Human-readable description
    public let description: String?
    /// Whether this parameter is required
    public let isRequired: Bool

    public init(
        _ name: String,
        type: JSONSchema,
        description: String? = nil,
        isRequired: Bool = true
    ) {
        self.name = name
        self.type = type
        self.description = description
        self.isRequired = isRequired
    }
}

// MARK: - Parameter Result Builder

/// Result builder for collecting parameters in a declarative block
@resultBuilder
public struct ParameterBuilder {
    public static func buildBlock(_ components: Parameter...) -> [Parameter] {
        components
    }

    public static func buildOptional(_ component: [Parameter]?) -> [Parameter] {
        component ?? []
    }

    public static func buildEither(first component: [Parameter]) -> [Parameter] {
        component
    }

    public static func buildEither(second component: [Parameter]) -> [Parameter] {
        component
    }
}

// MARK: - Tool Builder Entry

/// A tool descriptor for the builder DSL
public struct Tool {
    /// The built tool definition
    public let definition: ToolDefinition

    /// Create a tool with a parameter builder block
    ///
    /// - Parameters:
    ///   - name: Tool name
    ///   - parameters: Parameter builder block
    ///   - description: Tool description
    public init(
        _ name: String,
        @ParameterBuilder parameters: () -> [Parameter],
        description: String
    ) {
        let params = parameters()
        let properties = params.map { param in
            JSONSchemaProperty(
                name: param.name,
                schema: param.type,
                description: param.description,
                required: param.isRequired
            )
        }
        let required = params.filter(\.isRequired).map(\.name)
        self.definition = ToolDefinition(
            name: name,
            description: description,
            parameters: .object(properties: properties, required: required)
        )
    }

    /// Create a tool with no parameters
    public init(_ name: String, description: String) {
        self.definition = ToolDefinition(
            name: name,
            description: description,
            parameters: .object(properties: [], required: [])
        )
    }
}

// MARK: - ToolKit Result Builder

/// Result builder for collecting tools in a declarative block
@resultBuilder
public struct ToolKitBuilder {
    public static func buildBlock(_ components: Tool...) -> [Tool] {
        components
    }

    public static func buildOptional(_ component: [Tool]?) -> [Tool] {
        component ?? []
    }

    public static func buildEither(first component: [Tool]) -> [Tool] {
        component
    }

    public static func buildEither(second component: [Tool]) -> [Tool] {
        component
    }
}

// MARK: - ToolKit

/// Container for tool definitions built with the DSL
///
/// **Usage:**
/// ```swift
/// let kit = ToolKit {
///     Tool("greet") {
///         Parameter("name", type: .string, description: "Person to greet")
///     } description: "Greet someone by name"
///
///     Tool("add") {
///         Parameter("a", type: .number, description: "First number")
///         Parameter("b", type: .number, description: "Second number")
///     } description: "Add two numbers"
/// }
///
/// let config = ToolCallingConfig(tools: kit.definitions)
/// ```
public struct ToolKit {
    /// The built tool definitions
    public let definitions: [ToolDefinition]

    /// Create a tool kit with a builder block
    public init(@ToolKitBuilder _ builder: () -> [Tool]) {
        self.definitions = builder().map(\.definition)
    }
}
