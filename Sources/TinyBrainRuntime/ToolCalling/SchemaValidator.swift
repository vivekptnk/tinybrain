/// Schema validation for structured JSON output
///
/// **TB-Tool-002:** Validates parsed JSON values against a `JSONSchema`
///
/// ## Educational Overview
///
/// After the constrained sampler produces JSON text, we parse it and
/// validate the result against the original schema. This is a safety
/// net: even if the sampler's state machine has a bug, the validator
/// catches malformed output before it reaches user code.
///
/// **Usage:**
/// ```swift
/// let schema: JSONSchema = .object(properties: [
///     JSONSchemaProperty(name: "name", schema: .string, required: true)
/// ], required: ["name"])
///
/// let value: Any = ["name": "Alice"]
/// switch SchemaValidator.validate(value, against: schema) {
/// case .success: print("Valid!")
/// case .failure(let error): print("Invalid: \(error)")
/// }
/// ```

import Foundation

// MARK: - Schema Error

/// Errors produced during schema validation
///
/// Each error carries a `path` showing where in the JSON tree the
/// violation occurred, making debugging straightforward.
public enum SchemaError: Error, Equatable, CustomStringConvertible {
    /// Value type does not match expected schema type
    case typeMismatch(path: String, expected: String, actual: String)
    /// Required property is missing from object
    case missingRequired(path: String, property: String)
    /// String value is not in the allowed enum set
    case invalidEnum(path: String, value: String, allowed: [String])
    /// Catch-all for unexpected validation failures
    case validationFailed(path: String, message: String)

    public var description: String {
        switch self {
        case .typeMismatch(let path, let expected, let actual):
            return "\(path): expected \(expected), got \(actual)"
        case .missingRequired(let path, let property):
            return "\(path): missing required property '\(property)'"
        case .invalidEnum(let path, let value, let allowed):
            return "\(path): '\(value)' not in \(allowed)"
        case .validationFailed(let path, let message):
            return "\(path): \(message)"
        }
    }
}

// MARK: - Schema Validator

/// Validates parsed JSON values against a JSONSchema
///
/// All methods are static — no instance state needed.
public struct SchemaValidator {

    /// Validate a parsed JSON value against a schema
    ///
    /// - Parameters:
    ///   - value: A parsed JSON value (`String`, `NSNumber`, `[Any]`, `[String: Any]`, `NSNull`)
    ///   - schema: The schema to validate against
    ///   - path: JSON path prefix for error messages (default: "$")
    /// - Returns: `.success` or `.failure(SchemaError)`
    public static func validate(
        _ value: Any,
        against schema: JSONSchema,
        path: String = "$"
    ) -> Result<Void, SchemaError> {
        switch schema {
        case .string:
            guard value is String else {
                return .failure(.typeMismatch(path: path, expected: "string", actual: typeName(value)))
            }
            return .success(())

        case .number:
            guard isNumber(value) else {
                return .failure(.typeMismatch(path: path, expected: "number", actual: typeName(value)))
            }
            return .success(())

        case .integer:
            guard isInteger(value) else {
                return .failure(.typeMismatch(path: path, expected: "integer", actual: typeName(value)))
            }
            return .success(())

        case .boolean:
            guard isBool(value) else {
                return .failure(.typeMismatch(path: path, expected: "boolean", actual: typeName(value)))
            }
            return .success(())

        case .null:
            guard value is NSNull else {
                return .failure(.typeMismatch(path: path, expected: "null", actual: typeName(value)))
            }
            return .success(())

        case .enum(let allowed):
            guard let str = value as? String else {
                return .failure(.typeMismatch(path: path, expected: "string", actual: typeName(value)))
            }
            guard allowed.contains(str) else {
                return .failure(.invalidEnum(path: path, value: str, allowed: allowed))
            }
            return .success(())

        case .array(let itemSchema):
            guard let arr = value as? [Any] else {
                return .failure(.typeMismatch(path: path, expected: "array", actual: typeName(value)))
            }
            for (index, element) in arr.enumerated() {
                let result = validate(element, against: itemSchema, path: "\(path)[\(index)]")
                if case .failure = result { return result }
            }
            return .success(())

        case .object(let properties, let required):
            guard let dict = value as? [String: Any] else {
                return .failure(.typeMismatch(path: path, expected: "object", actual: typeName(value)))
            }
            // Check required properties
            for req in required {
                guard dict[req] != nil else {
                    return .failure(.missingRequired(path: path, property: req))
                }
            }
            // Validate each declared property that is present
            for prop in properties {
                if let propValue = dict[prop.name] {
                    let result = validate(propValue, against: prop.schema, path: "\(path).\(prop.name)")
                    if case .failure = result { return result }
                }
            }
            return .success(())
        }
    }

    // MARK: - Type Helpers

    private static func typeName(_ value: Any) -> String {
        if value is String { return "string" }
        if value is NSNull { return "null" }
        if isBool(value) { return "boolean" }
        if isNumber(value) { return "number" }
        if value is [Any] { return "array" }
        if value is [String: Any] { return "object" }
        return String(describing: type(of: value))
    }

    private static func isBool(_ value: Any) -> Bool {
        // NSNumber wraps both Bool and numbers; CFBooleanGetTypeID distinguishes
        guard let number = value as? NSNumber else { return false }
        return CFGetTypeID(number) == CFBooleanGetTypeID()
    }

    private static func isNumber(_ value: Any) -> Bool {
        guard let number = value as? NSNumber else { return false }
        // Exclude booleans
        return CFGetTypeID(number) != CFBooleanGetTypeID()
    }

    private static func isInteger(_ value: Any) -> Bool {
        guard isNumber(value), let number = value as? NSNumber else { return false }
        return number.doubleValue == number.doubleValue.rounded()
    }
}
