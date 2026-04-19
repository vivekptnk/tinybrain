/// JSON Schema types for structured output generation
///
/// **TB-Tool-001:** Foundation types for constraining LLM output to valid JSON
///
/// ## Educational Overview
///
/// JSON Schema defines the *shape* of valid JSON. By expressing a schema
/// in Swift types, the constrained sampler can mask invalid tokens at
/// each generation step, guaranteeing well-formed output.
///
/// **Example:**
/// ```swift
/// let schema: JSONSchema = .object(properties: [
///     JSONSchemaProperty(name: "name", schema: .string, description: "User name", required: true),
///     JSONSchemaProperty(name: "age", schema: .integer, description: "User age", required: true)
/// ], required: ["name", "age"])
/// ```

import Foundation

// MARK: - JSON Schema

/// A subset of JSON Schema sufficient for tool-calling structured output
///
/// Each case maps to a JSON Schema `type` keyword. Compound types
/// (`.object`, `.array`) carry their sub-schemas recursively.
public indirect enum JSONSchema: Equatable, Codable {
    /// JSON string value
    case string
    /// JSON number (floating-point)
    case number
    /// JSON integer
    case integer
    /// JSON boolean (`true` / `false`)
    case boolean
    /// JSON array with uniform item schema
    case array(items: JSONSchema)
    /// JSON object with named properties
    case object(properties: [JSONSchemaProperty], required: [String])
    /// Enumeration of allowed string values
    case `enum`(values: [String])
    /// JSON null
    case null

    // MARK: - Codable

    private enum CodingKeys: String, CodingKey {
        case type, items, properties, required
        case enumValues = "enum"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)

        switch type {
        case "string":
            // Check if it's an enum
            if let values = try container.decodeIfPresent([String].self, forKey: .enumValues) {
                self = .enum(values: values)
            } else {
                self = .string
            }
        case "number":
            self = .number
        case "integer":
            self = .integer
        case "boolean":
            self = .boolean
        case "null":
            self = .null
        case "array":
            let items = try container.decode(JSONSchema.self, forKey: .items)
            self = .array(items: items)
        case "object":
            let props = try container.decodeIfPresent([JSONSchemaProperty].self, forKey: .properties) ?? []
            let req = try container.decodeIfPresent([String].self, forKey: .required) ?? []
            self = .object(properties: props, required: req)
        default:
            throw DecodingError.dataCorruptedError(
                forKey: .type,
                in: container,
                debugDescription: "Unknown JSON Schema type: \(type)"
            )
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)

        switch self {
        case .string:
            try container.encode("string", forKey: .type)
        case .number:
            try container.encode("number", forKey: .type)
        case .integer:
            try container.encode("integer", forKey: .type)
        case .boolean:
            try container.encode("boolean", forKey: .type)
        case .null:
            try container.encode("null", forKey: .type)
        case .array(let items):
            try container.encode("array", forKey: .type)
            try container.encode(items, forKey: .items)
        case .object(let properties, let required):
            try container.encode("object", forKey: .type)
            try container.encode(properties, forKey: .properties)
            try container.encode(required, forKey: .required)
        case .enum(let values):
            try container.encode("string", forKey: .type)
            try container.encode(values, forKey: .enumValues)
        }
    }
}

// MARK: - JSON Schema Property

/// A named property within a JSON Schema object
///
/// Carries the property name, its sub-schema, an optional description,
/// and whether the property is required.
public struct JSONSchemaProperty: Equatable, Codable {
    /// Property name (JSON key)
    public let name: String
    /// Schema for the property value
    public let schema: JSONSchema
    /// Human-readable description of the property
    public let description: String?
    /// Whether this property must be present
    public let required: Bool

    public init(
        name: String,
        schema: JSONSchema,
        description: String? = nil,
        required: Bool = false
    ) {
        self.name = name
        self.schema = schema
        self.description = description
        self.required = required
    }
}
