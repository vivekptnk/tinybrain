/// Tests for the tool-calling / structured output subsystem
///
/// **TB-Tool-Tests:** Covers JSONSchema Codable, SchemaValidator, and ConstrainedSampler
///
/// ## Educational Overview
///
/// Structured output generation has three layers:
/// 1. **Types** (JSONSchema) — describe the shape of valid output
/// 2. **Validation** (SchemaValidator) — verify parsed JSON after generation
/// 3. **Constrained Sampling** — enforce structure *during* generation
///
/// Each layer has its own test group below.

import XCTest
@testable import TinyBrainRuntime

// MARK: - JSONSchema Codable Tests

final class JSONSchemaTests: XCTestCase {

    /// **Test:** String schema roundtrips through JSON encoding/decoding
    func testStringCodableRoundtrip() throws {
        let schema: JSONSchema = .string
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, .string, "String schema should survive Codable roundtrip")
    }

    /// **Test:** Number schema roundtrips
    func testNumberCodableRoundtrip() throws {
        let schema: JSONSchema = .number
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, .number)
    }

    /// **Test:** Integer schema roundtrips
    func testIntegerCodableRoundtrip() throws {
        let schema: JSONSchema = .integer
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, .integer)
    }

    /// **Test:** Boolean schema roundtrips
    func testBooleanCodableRoundtrip() throws {
        let schema: JSONSchema = .boolean
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, .boolean)
    }

    /// **Test:** Null schema roundtrips
    func testNullCodableRoundtrip() throws {
        let schema: JSONSchema = .null
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, .null)
    }

    /// **Test:** Enum schema roundtrips with values preserved
    func testEnumCodableRoundtrip() throws {
        let schema: JSONSchema = .enum(values: ["red", "green", "blue"])
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, schema, "Enum values should be preserved through roundtrip")
    }

    /// **Test:** Array schema roundtrips with item type preserved
    func testArrayCodableRoundtrip() throws {
        let schema: JSONSchema = .array(items: .integer)
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, schema, "Array item schema should survive roundtrip")
    }

    /// **Test:** Object schema roundtrips with properties and required list
    ///
    /// **Educational:**
    /// Object schemas are the most complex — they carry nested properties,
    /// each with their own sub-schemas. This test verifies the full tree survives.
    func testObjectCodableRoundtrip() throws {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "name", schema: .string, description: "User name", required: true),
                JSONSchemaProperty(name: "age", schema: .integer, description: "User age", required: true),
                JSONSchemaProperty(name: "tags", schema: .array(items: .string), description: "Tags", required: false)
            ],
            required: ["name", "age"]
        )
        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, schema, "Complex object schema should survive Codable roundtrip")
    }

    /// **Test:** Nested object within object roundtrips
    func testNestedObjectCodableRoundtrip() throws {
        let inner: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "street", schema: .string, required: true)
            ],
            required: ["street"]
        )
        let outer: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "address", schema: inner, required: true)
            ],
            required: ["address"]
        )
        let data = try JSONEncoder().encode(outer)
        let decoded = try JSONDecoder().decode(JSONSchema.self, from: data)
        XCTAssertEqual(decoded, outer, "Nested object schemas should roundtrip correctly")
    }

    /// **Test:** JSONSchemaProperty preserves all fields
    func testPropertyFieldsPreserved() throws {
        let prop = JSONSchemaProperty(
            name: "email",
            schema: .string,
            description: "User email address",
            required: true
        )
        XCTAssertEqual(prop.name, "email")
        XCTAssertEqual(prop.schema, .string)
        XCTAssertEqual(prop.description, "User email address")
        XCTAssertTrue(prop.required)
    }
}

// MARK: - SchemaValidator Tests

final class SchemaValidatorTests: XCTestCase {

    // MARK: String Validation

    /// **Test:** Valid string passes validation
    func testValidString() {
        let result = SchemaValidator.validate("hello", against: .string)
        XCTAssertNoThrow(try result.get(), "A String value should validate against .string schema")
    }

    /// **Test:** Non-string fails against string schema
    func testInvalidString() {
        let result = SchemaValidator.validate(42 as NSNumber, against: .string)
        if case .failure(let error) = result {
            if case .typeMismatch(_, let expected, _) = error {
                XCTAssertEqual(expected, "string")
            } else {
                XCTFail("Expected typeMismatch error")
            }
        } else {
            XCTFail("Expected validation failure for number against string schema")
        }
    }

    // MARK: Number Validation

    /// **Test:** Valid number passes
    func testValidNumber() {
        let result = SchemaValidator.validate(3.14 as NSNumber, against: .number)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Integer also passes number validation
    func testIntegerAsNumber() {
        let result = SchemaValidator.validate(42 as NSNumber, against: .number)
        XCTAssertNoThrow(try result.get(), "Integers are valid numbers")
    }

    /// **Test:** String fails against number schema
    func testInvalidNumber() {
        let result = SchemaValidator.validate("not a number", against: .number)
        if case .failure = result {
            // Expected
        } else {
            XCTFail("String should not validate as number")
        }
    }

    // MARK: Integer Validation

    /// **Test:** Whole number passes integer validation
    func testValidInteger() {
        let result = SchemaValidator.validate(42 as NSNumber, against: .integer)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Floating point fails integer validation
    func testFloatFailsInteger() {
        let result = SchemaValidator.validate(3.14 as NSNumber, against: .integer)
        if case .failure = result {
            // Expected: 3.14 is not an integer
        } else {
            XCTFail("3.14 should not validate as integer")
        }
    }

    // MARK: Boolean Validation

    /// **Test:** true/false pass boolean validation
    func testValidBoolean() {
        let trueResult = SchemaValidator.validate(true as NSNumber, against: .boolean)
        let falseResult = SchemaValidator.validate(false as NSNumber, against: .boolean)
        XCTAssertNoThrow(try trueResult.get())
        XCTAssertNoThrow(try falseResult.get())
    }

    /// **Test:** Number fails boolean validation
    ///
    /// **Educational:**
    /// NSNumber wraps both Bool and Int. The validator uses CFBooleanGetTypeID
    /// to distinguish `true`/`false` from `1`/`0`.
    func testNumberFailsBoolean() {
        let result = SchemaValidator.validate(1 as NSNumber, against: .boolean)
        if case .failure = result {
            // Expected: 1 is a number, not a boolean
        } else {
            XCTFail("NSNumber(1) should not validate as boolean")
        }
    }

    // MARK: Null Validation

    /// **Test:** NSNull passes null validation
    func testValidNull() {
        let result = SchemaValidator.validate(NSNull(), against: .null)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Non-null fails null validation
    func testNonNullFailsNull() {
        let result = SchemaValidator.validate("something", against: .null)
        if case .failure = result {
            // Expected
        } else {
            XCTFail("String should not validate as null")
        }
    }

    // MARK: Enum Validation

    /// **Test:** Valid enum value passes
    func testValidEnum() {
        let schema: JSONSchema = .enum(values: ["red", "green", "blue"])
        let result = SchemaValidator.validate("red", against: schema)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Invalid enum value fails with descriptive error
    func testInvalidEnum() {
        let schema: JSONSchema = .enum(values: ["red", "green", "blue"])
        let result = SchemaValidator.validate("yellow", against: schema)
        if case .failure(let error) = result {
            if case .invalidEnum(_, let value, let allowed) = error {
                XCTAssertEqual(value, "yellow")
                XCTAssertEqual(allowed, ["red", "green", "blue"])
            } else {
                XCTFail("Expected invalidEnum error")
            }
        } else {
            XCTFail("'yellow' should not validate against [red, green, blue]")
        }
    }

    // MARK: Array Validation

    /// **Test:** Valid array of integers passes
    func testValidArray() {
        let schema: JSONSchema = .array(items: .integer)
        let value: [Any] = [1 as NSNumber, 2 as NSNumber, 3 as NSNumber]
        let result = SchemaValidator.validate(value, against: schema)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Empty array passes (no items to validate)
    func testEmptyArray() {
        let schema: JSONSchema = .array(items: .string)
        let result = SchemaValidator.validate([Any](), against: schema)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Array with wrong item type fails with correct path
    func testArrayWithWrongItemType() {
        let schema: JSONSchema = .array(items: .integer)
        let value: [Any] = [1 as NSNumber, "oops", 3 as NSNumber]
        let result = SchemaValidator.validate(value, against: schema)
        if case .failure(let error) = result {
            if case .typeMismatch(let path, _, _) = error {
                XCTAssertEqual(path, "$[1]", "Error path should point to index 1")
            } else {
                XCTFail("Expected typeMismatch at array index")
            }
        } else {
            XCTFail("Mixed array should fail integer item validation")
        }
    }

    // MARK: Object Validation

    /// **Test:** Valid object with all required properties passes
    func testValidObject() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "name", schema: .string, required: true),
                JSONSchemaProperty(name: "age", schema: .integer, required: true)
            ],
            required: ["name", "age"]
        )
        let value: [String: Any] = ["name": "Alice", "age": 30 as NSNumber]
        let result = SchemaValidator.validate(value, against: schema)
        XCTAssertNoThrow(try result.get())
    }

    /// **Test:** Missing required property fails
    func testMissingRequired() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "name", schema: .string, required: true)
            ],
            required: ["name"]
        )
        let value: [String: Any] = ["age": 30 as NSNumber]
        let result = SchemaValidator.validate(value, against: schema)
        if case .failure(let error) = result {
            if case .missingRequired(_, let property) = error {
                XCTAssertEqual(property, "name")
            } else {
                XCTFail("Expected missingRequired error")
            }
        } else {
            XCTFail("Object missing 'name' should fail validation")
        }
    }

    /// **Test:** Wrong property type fails with nested path
    func testWrongPropertyType() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "age", schema: .integer, required: true)
            ],
            required: ["age"]
        )
        let value: [String: Any] = ["age": "not a number"]
        let result = SchemaValidator.validate(value, against: schema)
        if case .failure(let error) = result {
            if case .typeMismatch(let path, _, _) = error {
                XCTAssertEqual(path, "$.age", "Error path should include property name")
            }
        } else {
            XCTFail("String 'age' should fail integer validation")
        }
    }

    /// **Test:** Nested object validation with correct path
    func testNestedObjectValidation() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(
                    name: "address",
                    schema: .object(
                        properties: [
                            JSONSchemaProperty(name: "city", schema: .string, required: true)
                        ],
                        required: ["city"]
                    ),
                    required: true
                )
            ],
            required: ["address"]
        )
        let value: [String: Any] = ["address": ["city": 123 as NSNumber] as [String: Any]]
        let result = SchemaValidator.validate(value, against: schema)
        if case .failure(let error) = result {
            if case .typeMismatch(let path, _, _) = error {
                XCTAssertEqual(path, "$.address.city", "Should show full nested path")
            }
        } else {
            XCTFail("Numeric city should fail string validation")
        }
    }

    /// **Test:** Optional property can be absent
    func testOptionalPropertyAbsent() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "name", schema: .string, required: true),
                JSONSchemaProperty(name: "nickname", schema: .string, required: false)
            ],
            required: ["name"]
        )
        let value: [String: Any] = ["name": "Alice"]
        let result = SchemaValidator.validate(value, against: schema)
        XCTAssertNoThrow(try result.get(), "Optional property 'nickname' can be absent")
    }
}

// MARK: - ConstrainedSampler Tests

/// Mock tokenizer for constrained sampler tests
struct MockTokenLookup: TokenLookup {
    let tokens: [String]

    var vocabularySize: Int { tokens.count }

    func decode(tokenId: Int) -> String {
        guard tokenId >= 0, tokenId < tokens.count else { return "" }
        return tokens[tokenId]
    }
}

final class ConstrainedSamplerTests: XCTestCase {

    // MARK: State Machine Initialization

    /// **Test:** Object schema starts in expectingOpenBrace state
    ///
    /// **Educational:**
    /// The state machine must begin by requiring `{` for object schemas.
    /// This ensures the first generated character is always valid JSON.
    func testObjectSchemaStartsExpectingBrace() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "key", schema: .string, required: true)
            ],
            required: ["key"]
        )
        let sampler = ConstrainedSampler(schema: schema)
        XCTAssertFalse(sampler.isComplete)
        XCTAssertEqual(sampler.stateDescription, "expecting '{'")
    }

    /// **Test:** Array schema starts in expectingOpenBracket state
    func testArraySchemaStartsExpectingBracket() {
        let schema: JSONSchema = .array(items: .string)
        let sampler = ConstrainedSampler(schema: schema)
        XCTAssertEqual(sampler.stateDescription, "expecting '['")
    }

    /// **Test:** Primitive schema starts in expectingValue state
    func testPrimitiveSchemaStartsExpectingValue() {
        let sampler = ConstrainedSampler(schema: .string)
        XCTAssertEqual(sampler.stateDescription, "expecting value(string)")
    }

    // MARK: State Transitions

    /// **Test:** Advancing through `{` transitions to expecting key
    func testOpenBraceTransitionsToKey() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "x", schema: .string, required: true)
            ],
            required: ["x"]
        )
        var sampler = ConstrainedSampler(schema: schema)
        sampler.advance(token: "{")
        XCTAssertEqual(sampler.stateDescription, "expecting key")
    }

    /// **Test:** After key, transitions through colon to value
    func testKeyTransitionsThroughColonToValue() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "x", schema: .integer, required: true)
            ],
            required: ["x"]
        )
        var sampler = ConstrainedSampler(schema: schema)
        sampler.advance(token: "{")
        sampler.advance(token: "\"x\"")
        XCTAssertEqual(sampler.stateDescription, "expecting ':'")
        sampler.advance(token: ":")
        XCTAssertEqual(sampler.stateDescription, "expecting value(integer)")
    }

    /// **Test:** Complete object generation reaches complete or comma/close state
    func testCompleteObjectFlow() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "n", schema: .integer, required: true)
            ],
            required: ["n"]
        )
        var sampler = ConstrainedSampler(schema: schema)
        sampler.advance(token: "{")
        sampler.advance(token: "\"n\"")
        sampler.advance(token: ":")
        sampler.advance(token: "42")
        // After value, should be expecting comma or close
        XCTAssertEqual(sampler.stateDescription, "expecting ',' or '}'")
        sampler.advance(token: "}")
        XCTAssertTrue(sampler.isComplete, "Object should be complete after closing brace")
    }

    // MARK: Logit Masking

    /// **Test:** Strict mode masks non-brace tokens when expecting open brace
    ///
    /// **Educational:**
    /// In strict mode, any token that doesn't start with `{` gets
    /// `-infinity` logits, making it impossible to sample.
    func testStrictMaskingAtOpenBrace() {
        let schema: JSONSchema = .object(properties: [], required: [])
        var sampler = ConstrainedSampler(schema: schema, mode: .strict)

        let tokenizer = MockTokenLookup(tokens: ["{", "hello", "}", "42", "["])
        var logits = Tensor<Float>(shape: TensorShape(5), data: [1.0, 1.0, 1.0, 1.0, 1.0])

        sampler.maskLogits(&logits, tokenizer: tokenizer)

        XCTAssertEqual(logits.data[0], 1.0, "'{' should keep its logit")
        XCTAssertEqual(logits.data[1], -Float.infinity, "'hello' should be masked")
        XCTAssertEqual(logits.data[2], -Float.infinity, "'}' should be masked at open position")
        XCTAssertEqual(logits.data[3], -Float.infinity, "'42' should be masked")
        XCTAssertEqual(logits.data[4], -Float.infinity, "'[' should be masked for object schema")
    }

    /// **Test:** None mode does not modify logits
    func testNoneModeSkipsMasking() {
        let schema: JSONSchema = .object(properties: [], required: [])
        var sampler = ConstrainedSampler(schema: schema, mode: .none)

        let tokenizer = MockTokenLookup(tokens: ["{", "hello"])
        var logits = Tensor<Float>(shape: TensorShape(2), data: [1.0, 1.0])

        sampler.maskLogits(&logits, tokenizer: tokenizer)

        XCTAssertEqual(logits.data[0], 1.0)
        XCTAssertEqual(logits.data[1], 1.0, "None mode should leave all logits untouched")
    }

    /// **Test:** Guided mode applies bias instead of -infinity
    func testGuidedModeBiasesInvalid() {
        let schema: JSONSchema = .object(properties: [], required: [])
        var sampler = ConstrainedSampler(schema: schema, mode: .guided)

        let tokenizer = MockTokenLookup(tokens: ["{", "hello"])
        var logits = Tensor<Float>(shape: TensorShape(2), data: [1.0, 1.0])

        sampler.maskLogits(&logits, tokenizer: tokenizer)

        XCTAssertEqual(logits.data[0], 1.0, "'{' should keep its logit in guided mode")
        XCTAssertTrue(logits.data[1] < 0, "'hello' should be negatively biased, not -infinity")
        XCTAssertGreaterThan(logits.data[1], -Float.infinity, "Guided mode should bias, not hard-mask")
    }

    // MARK: Constraint Mode

    /// **Test:** ConstraintMode equality
    func testConstraintModeEquality() {
        XCTAssertEqual(ConstraintMode.strict, ConstraintMode.strict)
        XCTAssertEqual(ConstraintMode.guided, ConstraintMode.guided)
        XCTAssertEqual(ConstraintMode.none, ConstraintMode.none)
        XCTAssertNotEqual(ConstraintMode.strict, ConstraintMode.guided)
    }
}

// MARK: - GenerationConfig Extension Tests

final class GenerationConfigToolCallingTests: XCTestCase {

    /// **Test:** Default GenerationConfig has no output schema (backward compatible)
    func testDefaultConfigHasNoSchema() {
        let config = GenerationConfig()
        XCTAssertNil(config.outputSchema, "Default config should have nil outputSchema")
        XCTAssertEqual(config.constraintMode, .none, "Default constraint mode should be .none")
    }

    /// **Test:** Config with output schema preserves it
    func testConfigWithSchema() {
        let schema: JSONSchema = .object(
            properties: [
                JSONSchemaProperty(name: "answer", schema: .string, required: true)
            ],
            required: ["answer"]
        )
        let config = GenerationConfig(
            maxTokens: 200,
            outputSchema: schema,
            constraintMode: .strict
        )
        XCTAssertEqual(config.outputSchema, schema)
        XCTAssertEqual(config.constraintMode, .strict)
        XCTAssertEqual(config.maxTokens, 200)
    }

    /// **Test:** Default GenerationConfig has no toolCallingConfig (backward compatible)
    func testDefaultConfigHasNoToolCallingConfig() {
        let config = GenerationConfig()
        XCTAssertNil(config.toolCallingConfig)
    }

    /// **Test:** Config with toolCallingConfig preserves it
    func testConfigWithToolCallingConfig() {
        let tool = ToolDefinition(
            name: "get_weather",
            description: "Get weather",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "city", schema: .string, required: true)
            ], required: ["city"])
        )
        let toolConfig = ToolCallingConfig(tools: [tool], toolChoice: .required)
        let config = GenerationConfig(
            maxTokens: 300,
            toolCallingConfig: toolConfig
        )
        XCTAssertNotNil(config.toolCallingConfig)
        XCTAssertEqual(config.toolCallingConfig?.tools.count, 1)
        XCTAssertEqual(config.toolCallingConfig?.tools.first?.name, "get_weather")
        XCTAssertEqual(config.toolCallingConfig?.toolChoice, .required)
        XCTAssertEqual(config.maxTokens, 300)
    }
}

// MARK: - TokenOutput Extension Tests

final class TokenOutputToolCallingTests: XCTestCase {

    /// **Test:** Default TokenOutput has nil constraintState
    func testDefaultTokenOutputHasNoConstraintState() {
        let output = TokenOutput(tokenId: 1, probability: 0.9)
        XCTAssertNil(output.constraintState)
    }

    /// **Test:** TokenOutput with constraintState preserves it
    func testTokenOutputWithConstraintState() {
        let output = TokenOutput(
            tokenId: 42,
            probability: 0.85,
            constraintState: "expecting key"
        )
        XCTAssertEqual(output.constraintState, "expecting key")
        XCTAssertEqual(output.tokenId, 42)
    }
}

// MARK: - ToolDefinition Tests

final class ToolDefinitionTests: XCTestCase {

    /// **Test:** ToolDefinition stores all fields correctly
    func testToolDefinitionCreation() {
        let tool = ToolDefinition(
            name: "get_weather",
            description: "Get weather for a city",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "city", schema: .string, description: "City name", required: true)
            ], required: ["city"])
        )
        XCTAssertEqual(tool.name, "get_weather")
        XCTAssertEqual(tool.description, "Get weather for a city")
        if case .object(let props, let req) = tool.parameters {
            XCTAssertEqual(props.count, 1)
            XCTAssertEqual(req, ["city"])
        } else {
            XCTFail("Expected object schema")
        }
    }

    /// **Test:** ToolDefinition Codable roundtrip
    func testToolDefinitionCodableRoundtrip() throws {
        let tool = ToolDefinition(
            name: "search",
            description: "Search the web",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "query", schema: .string, required: true),
                JSONSchemaProperty(name: "limit", schema: .integer, required: false)
            ], required: ["query"])
        )
        let data = try JSONEncoder().encode(tool)
        let decoded = try JSONDecoder().decode(ToolDefinition.self, from: data)
        XCTAssertEqual(decoded, tool)
    }

    /// **Test:** ToolDefinition equality
    func testToolDefinitionEquality() {
        let a = ToolDefinition(name: "foo", description: "A", parameters: .object(properties: [], required: []))
        let b = ToolDefinition(name: "foo", description: "A", parameters: .object(properties: [], required: []))
        let c = ToolDefinition(name: "bar", description: "B", parameters: .object(properties: [], required: []))
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}

// MARK: - ToolCall Tests

final class ToolCallTests: XCTestCase {

    /// **Test:** ToolCall creation and field access
    func testToolCallCreation() {
        let call = ToolCall(id: "call_1", name: "get_weather", arguments: ["city": "Paris"])
        XCTAssertEqual(call.id, "call_1")
        XCTAssertEqual(call.name, "get_weather")
        XCTAssertEqual(call.arguments["city"] as? String, "Paris")
    }

    /// **Test:** ToolCall equality with matching arguments
    func testToolCallEquality() {
        let a = ToolCall(id: "1", name: "foo", arguments: ["x": "y"])
        let b = ToolCall(id: "1", name: "foo", arguments: ["x": "y"])
        XCTAssertEqual(a, b)
    }

    /// **Test:** ToolCall inequality with different arguments
    func testToolCallInequality() {
        let a = ToolCall(id: "1", name: "foo", arguments: ["x": "y"])
        let b = ToolCall(id: "1", name: "foo", arguments: ["x": "z"])
        XCTAssertNotEqual(a, b)
    }
}

// MARK: - ToolResult Tests

final class ToolResultTests: XCTestCase {

    /// **Test:** Successful tool result
    func testSuccessResult() {
        let result = ToolResult(callId: "call_1", content: "72°F in Paris")
        XCTAssertEqual(result.callId, "call_1")
        XCTAssertEqual(result.content, "72°F in Paris")
        XCTAssertFalse(result.isError)
    }

    /// **Test:** Error tool result
    func testErrorResult() {
        let result = ToolResult(callId: "call_2", content: "City not found", isError: true)
        XCTAssertTrue(result.isError)
        XCTAssertEqual(result.content, "City not found")
    }

    /// **Test:** ToolResult equality
    func testResultEquality() {
        let a = ToolResult(callId: "1", content: "ok")
        let b = ToolResult(callId: "1", content: "ok")
        XCTAssertEqual(a, b)
    }
}

// MARK: - ToolCallParser Tests

final class ToolCallParserTests: XCTestCase {

    let weatherTool = ToolDefinition(
        name: "get_weather",
        description: "Get weather",
        parameters: .object(properties: [
            JSONSchemaProperty(name: "city", schema: .string, description: "City", required: true)
        ], required: ["city"])
    )

    /// **Test:** Parse complete valid tool call JSON
    func testParseCompleteJSON() {
        var parser = ToolCallParser(tools: [weatherTool])
        parser.feed("{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}")
        XCTAssertTrue(parser.hasCompleteJSON)
        let result = parser.extractToolCall()
        if case .success(let call) = result {
            XCTAssertEqual(call.name, "get_weather")
            XCTAssertEqual(call.arguments["city"] as? String, "Paris")
        } else {
            XCTFail("Expected successful parse, got \(result)")
        }
    }

    /// **Test:** Streaming tokens accumulate to a valid call
    func testStreamingAccumulation() {
        var parser = ToolCallParser(tools: [weatherTool])
        parser.feed("{\"name\":")
        XCTAssertFalse(parser.hasCompleteJSON, "Incomplete JSON should not be ready")
        parser.feed(" \"get_weather\", ")
        parser.feed("\"arguments\": ")
        parser.feed("{\"city\": \"NYC\"}}")
        XCTAssertTrue(parser.hasCompleteJSON)
        let result = parser.extractToolCall()
        if case .success(let call) = result {
            XCTAssertEqual(call.name, "get_weather")
            XCTAssertEqual(call.arguments["city"] as? String, "NYC")
        } else {
            XCTFail("Expected successful streaming parse")
        }
    }

    /// **Test:** Text before JSON is ignored
    func testTextBeforeJSON() {
        var parser = ToolCallParser(tools: [weatherTool])
        parser.feed("I'll call a tool now: {\"name\": \"get_weather\", \"arguments\": {\"city\": \"London\"}}")
        let result = parser.extractToolCall()
        if case .success(let call) = result {
            XCTAssertEqual(call.arguments["city"] as? String, "London")
        } else {
            XCTFail("Should parse JSON even with preceding text")
        }
    }

    /// **Test:** Missing name field returns error
    func testMissingNameField() {
        var parser = ToolCallParser()
        parser.feed("{\"arguments\": {\"city\": \"Paris\"}}")
        let result = parser.extractToolCall()
        if case .failure(let error) = result {
            if case .missingField(let field) = error {
                XCTAssertEqual(field, "name")
            } else {
                XCTFail("Expected missingField error, got \(error)")
            }
        } else {
            XCTFail("Expected failure for missing name")
        }
    }

    /// **Test:** No JSON found returns error
    func testNoJSONFound() {
        var parser = ToolCallParser()
        parser.feed("Just some plain text with no JSON")
        let result = parser.extractToolCall()
        if case .failure(.noJSONFound) = result {
            // Expected
        } else {
            XCTFail("Expected noJSONFound error")
        }
    }

    /// **Test:** Malformed JSON returns error
    func testMalformedJSON() {
        var parser = ToolCallParser()
        parser.feed("{not valid json}")
        let result = parser.extractToolCall()
        if case .failure(.invalidJSON) = result {
            // Expected
        } else {
            XCTFail("Expected invalidJSON error")
        }
    }

    /// **Test:** Schema violation is caught
    func testSchemaViolation() {
        let strictTool = ToolDefinition(
            name: "add",
            description: "Add numbers",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "a", schema: .integer, required: true),
                JSONSchemaProperty(name: "b", schema: .integer, required: true)
            ], required: ["a", "b"])
        )
        var parser = ToolCallParser(tools: [strictTool])
        // Pass string "one" instead of integer
        parser.feed("{\"name\": \"add\", \"arguments\": {\"a\": \"one\", \"b\": 2}}")
        let result = parser.extractToolCall()
        if case .failure(.schemaViolation) = result {
            // Expected — "one" is not an integer
        } else {
            XCTFail("Expected schemaViolation for non-integer argument")
        }
    }

    /// **Test:** Parser reset clears state
    func testReset() {
        var parser = ToolCallParser()
        parser.feed("{\"partial\":")
        XCTAssertFalse(parser.currentBuffer.isEmpty)
        parser.reset()
        XCTAssertTrue(parser.currentBuffer.isEmpty)
    }

    /// **Test:** Empty arguments default to empty dict
    func testEmptyArguments() {
        var parser = ToolCallParser()
        parser.feed("{\"name\": \"no_args\"}")
        let result = parser.extractToolCall()
        if case .success(let call) = result {
            XCTAssertEqual(call.name, "no_args")
            XCTAssertTrue(call.arguments.isEmpty)
        } else {
            XCTFail("Should succeed with empty arguments")
        }
    }

    /// **Test:** Nested JSON in arguments is handled correctly
    func testNestedArguments() {
        var parser = ToolCallParser()
        parser.feed("{\"name\": \"complex\", \"arguments\": {\"data\": {\"nested\": true}}}")
        let result = parser.extractToolCall()
        if case .success(let call) = result {
            let data = call.arguments["data"] as? [String: Any]
            XCTAssertNotNil(data)
            XCTAssertEqual(data?["nested"] as? Bool, true)
        } else {
            XCTFail("Should parse nested arguments")
        }
    }

    /// **Test:** Call counter increments across extractions
    func testCallIdIncrement() {
        var parser = ToolCallParser()
        parser.feed("{\"name\": \"a\", \"arguments\": {}}")
        let r1 = parser.extractToolCall()
        parser.reset()
        parser.feed("{\"name\": \"b\", \"arguments\": {}}")
        let r2 = parser.extractToolCall()
        if case .success(let c1) = r1, case .success(let c2) = r2 {
            XCTAssertEqual(c1.id, "call_1")
            XCTAssertEqual(c2.id, "call_2")
        } else {
            XCTFail("Both parses should succeed")
        }
    }
}

// MARK: - ToolCallingConfig Tests

final class ToolCallingConfigTests: XCTestCase {

    let sampleTool = ToolDefinition(
        name: "search",
        description: "Search the web",
        parameters: .object(properties: [
            JSONSchemaProperty(name: "query", schema: .string, description: "Search query", required: true)
        ], required: ["query"])
    )

    /// **Test:** Config stores tools and choice
    func testConfigCreation() {
        let config = ToolCallingConfig(tools: [sampleTool], toolChoice: .required)
        XCTAssertEqual(config.tools.count, 1)
        XCTAssertEqual(config.toolChoice, .required)
        XCTAssertEqual(config.maxIterations, 5)
    }

    /// **Test:** Default tool choice is .auto
    func testDefaultToolChoice() {
        let config = ToolCallingConfig(tools: [])
        XCTAssertEqual(config.toolChoice, .auto)
    }

    /// **Test:** System prompt contains tool name and description
    func testSystemPromptContainsToolInfo() {
        let config = ToolCallingConfig(tools: [sampleTool])
        let prompt = config.buildSystemPrompt()
        XCTAssertTrue(prompt.contains("search"), "Prompt should contain tool name")
        XCTAssertTrue(prompt.contains("Search the web"), "Prompt should contain tool description")
        XCTAssertTrue(prompt.contains("query"), "Prompt should contain parameter name")
    }

    /// **Test:** System prompt includes JSON format instruction
    func testSystemPromptFormat() {
        let config = ToolCallingConfig(tools: [sampleTool])
        let prompt = config.buildSystemPrompt()
        XCTAssertTrue(prompt.contains("{\"name\": \"tool_name\""), "Should include JSON format")
    }

    /// **Test:** Empty tools produce empty system prompt
    func testEmptyToolsEmptyPrompt() {
        let config = ToolCallingConfig(tools: [])
        XCTAssertEqual(config.buildSystemPrompt(), "")
    }

    /// **Test:** ToolChoice .none instruction in prompt
    func testToolChoiceNonePrompt() {
        let config = ToolCallingConfig(tools: [sampleTool], toolChoice: .none)
        let prompt = config.buildSystemPrompt()
        XCTAssertTrue(prompt.contains("Do NOT call any tools"))
    }

    /// **Test:** ToolChoice .specific instruction in prompt
    func testToolChoiceSpecificPrompt() {
        let config = ToolCallingConfig(tools: [sampleTool], toolChoice: .specific(name: "search"))
        let prompt = config.buildSystemPrompt()
        XCTAssertTrue(prompt.contains("MUST call the 'search' tool"))
    }

    /// **Test:** ToolChoice equality
    func testToolChoiceEquality() {
        XCTAssertEqual(ToolChoice.auto, ToolChoice.auto)
        XCTAssertEqual(ToolChoice.specific(name: "x"), ToolChoice.specific(name: "x"))
        XCTAssertNotEqual(ToolChoice.auto, ToolChoice.none)
        XCTAssertNotEqual(ToolChoice.specific(name: "a"), ToolChoice.specific(name: "b"))
    }
}

// MARK: - ToolDispatcher Tests

final class ToolDispatcherTests: XCTestCase {

    /// **Test:** Closure dispatcher routes to correct handler
    func testClosureDispatchSuccess() async throws {
        let dispatcher = ClosureToolDispatcher()
        dispatcher.register("greet") { call in
            let name = call.arguments["name"] as? String ?? "world"
            return "Hello, \(name)!"
        }

        let call = ToolCall(id: "call_1", name: "greet", arguments: ["name": "Alice"])
        let result = try await dispatcher.dispatch(call)
        XCTAssertEqual(result.callId, "call_1")
        XCTAssertEqual(result.content, "Hello, Alice!")
        XCTAssertFalse(result.isError)
    }

    /// **Test:** Unknown tool throws error
    func testUnknownToolThrows() async {
        let dispatcher = ClosureToolDispatcher()
        let call = ToolCall(id: "call_1", name: "nonexistent", arguments: [:])
        do {
            _ = try await dispatcher.dispatch(call)
            XCTFail("Should have thrown for unknown tool")
        } catch let error as ToolDispatchError {
            if case .unknownTool(let name) = error {
                XCTAssertEqual(name, "nonexistent")
            } else {
                XCTFail("Expected unknownTool error")
            }
        } catch {
            XCTFail("Unexpected error type: \(error)")
        }
    }

    /// **Test:** Handler error returns error result (not throw)
    func testHandlerErrorReturnsErrorResult() async throws {
        let dispatcher = ClosureToolDispatcher()
        dispatcher.register("fail") { _ in
            throw NSError(domain: "test", code: 42, userInfo: [NSLocalizedDescriptionKey: "test error"])
        }

        let call = ToolCall(id: "call_1", name: "fail", arguments: [:])
        let result = try await dispatcher.dispatch(call)
        XCTAssertTrue(result.isError, "Handler errors should produce error results")
        XCTAssertTrue(result.content.contains("error"), "Error content should describe the failure")
    }

    /// **Test:** Multiple handlers can be registered
    func testMultipleHandlers() async throws {
        let dispatcher = ClosureToolDispatcher()
        dispatcher.register("a") { _ in "result_a" }
        dispatcher.register("b") { _ in "result_b" }

        let ra = try await dispatcher.dispatch(ToolCall(id: "1", name: "a", arguments: [:]))
        let rb = try await dispatcher.dispatch(ToolCall(id: "2", name: "b", arguments: [:]))
        XCTAssertEqual(ra.content, "result_a")
        XCTAssertEqual(rb.content, "result_b")
    }
}

// MARK: - Tool Orchestration Tests

final class ToolOrchestrationTests: XCTestCase {

    /// **Test:** processToolCall extracts and dispatches a valid call
    func testProcessToolCallSuccess() async throws {
        let tool = ToolDefinition(
            name: "add",
            description: "Add two numbers",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "a", schema: .number, required: true),
                JSONSchemaProperty(name: "b", schema: .number, required: true)
            ], required: ["a", "b"])
        )

        let dispatcher = ClosureToolDispatcher()
        dispatcher.register("add") { call in
            let a = (call.arguments["a"] as? NSNumber)?.doubleValue ?? 0
            let b = (call.arguments["b"] as? NSNumber)?.doubleValue ?? 0
            return "\(a + b)"
        }

        var parser = ToolCallParser(tools: [tool])
        let result = try await processToolCall(
            generatedText: "{\"name\": \"add\", \"arguments\": {\"a\": 3, \"b\": 4}}",
            parser: &parser,
            dispatcher: dispatcher
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.content, "7.0")
        XCTAssertFalse(result?.isError ?? true)
    }

    /// **Test:** processToolCall returns nil for plain text
    func testProcessToolCallNoJSON() async throws {
        var parser = ToolCallParser()
        let dispatcher = ClosureToolDispatcher()
        let result = try await processToolCall(
            generatedText: "Just a normal text response",
            parser: &parser,
            dispatcher: dispatcher
        )
        XCTAssertNil(result, "No tool call in plain text")
    }

    /// **Test:** formatToolResult formats success and error correctly
    func testFormatToolResult() {
        let success = ToolResult(callId: "1", content: "42", isError: false)
        XCTAssertEqual(formatToolResult(success), "[Tool Result] 42")

        let error = ToolResult(callId: "2", content: "Not found", isError: true)
        XCTAssertEqual(formatToolResult(error), "[Tool Error] Not found")
    }
}

// MARK: - ToolKit Builder Tests

final class ToolKitBuilderTests: XCTestCase {

    /// **Test:** ToolKit DSL builds correct definitions
    func testToolKitBuilderBasic() {
        let kit = ToolKit {
            Tool("get_weather", parameters: {
                Parameter("location", type: .string, description: "City name")
                Parameter("unit", type: .enum(values: ["celsius", "fahrenheit"]), description: "Unit")
            }, description: "Get weather for a location")
        }

        XCTAssertEqual(kit.definitions.count, 1)
        let tool = kit.definitions[0]
        XCTAssertEqual(tool.name, "get_weather")
        XCTAssertEqual(tool.description, "Get weather for a location")

        if case .object(let props, let required) = tool.parameters {
            XCTAssertEqual(props.count, 2)
            XCTAssertEqual(props[0].name, "location")
            XCTAssertEqual(props[1].name, "unit")
            XCTAssertEqual(required, ["location", "unit"])
        } else {
            XCTFail("Expected object parameters")
        }
    }

    /// **Test:** ToolKit with multiple tools
    func testToolKitMultipleTools() {
        let kit = ToolKit {
            Tool("a", parameters: {
                Parameter("x", type: .string)
            }, description: "Tool A")

            Tool("b", parameters: {
                Parameter("y", type: .integer)
            }, description: "Tool B")
        }
        XCTAssertEqual(kit.definitions.count, 2)
        XCTAssertEqual(kit.definitions[0].name, "a")
        XCTAssertEqual(kit.definitions[1].name, "b")
    }

    /// **Test:** Tool with no parameters
    func testToolNoParameters() {
        let kit = ToolKit {
            Tool("ping", description: "Ping the server")
        }
        XCTAssertEqual(kit.definitions.count, 1)
        if case .object(let props, _) = kit.definitions[0].parameters {
            XCTAssertTrue(props.isEmpty)
        } else {
            XCTFail("Expected empty object parameters")
        }
    }

    /// **Test:** Optional parameter is not in required list
    func testOptionalParameter() {
        let kit = ToolKit {
            Tool("search", parameters: {
                Parameter("query", type: .string, description: "Search query")
                Parameter("limit", type: .integer, description: "Max results", isRequired: false)
            }, description: "Search")
        }
        if case .object(let props, let required) = kit.definitions[0].parameters {
            XCTAssertEqual(props.count, 2)
            XCTAssertEqual(required, ["query"], "Only required params should be in required list")
        } else {
            XCTFail("Expected object parameters")
        }
    }

    /// **Test:** ToolKit definitions can be used with ToolCallingConfig
    func testToolKitIntegrationWithConfig() {
        let kit = ToolKit {
            Tool("greet", parameters: {
                Parameter("name", type: .string)
            }, description: "Greet someone")
        }
        let config = ToolCallingConfig(tools: kit.definitions, toolChoice: .required)
        XCTAssertEqual(config.tools.count, 1)
        let prompt = config.buildSystemPrompt()
        XCTAssertTrue(prompt.contains("greet"))
        XCTAssertTrue(prompt.contains("MUST call"))
    }
}

// MARK: - Integration: Parse → Validate → Dispatch

final class ToolCallingIntegrationTests: XCTestCase {

    /// **Test:** End-to-end: parse tool call → validate → dispatch → format result
    func testEndToEndToolCalling() async throws {
        // 1. Define tools
        let kit = ToolKit {
            Tool("multiply", parameters: {
                Parameter("a", type: .number, description: "First number")
                Parameter("b", type: .number, description: "Second number")
            }, description: "Multiply two numbers")
        }

        // 2. Set up dispatcher
        let dispatcher = ClosureToolDispatcher()
        dispatcher.register("multiply") { call in
            let a = (call.arguments["a"] as? NSNumber)?.doubleValue ?? 0
            let b = (call.arguments["b"] as? NSNumber)?.doubleValue ?? 0
            return "\(a * b)"
        }

        // 3. Simulate model output
        let modelOutput = "Let me calculate that. {\"name\": \"multiply\", \"arguments\": {\"a\": 6, \"b\": 7}}"

        // 4. Parse
        var parser = ToolCallParser(tools: kit.definitions)
        parser.feed(modelOutput)
        XCTAssertTrue(parser.hasCompleteJSON)

        let parseResult = parser.extractToolCall()
        guard case .success(let call) = parseResult else {
            XCTFail("Parse should succeed: \(parseResult)")
            return
        }
        XCTAssertEqual(call.name, "multiply")

        // 5. Dispatch
        let result = try await dispatcher.dispatch(call)
        XCTAssertEqual(result.content, "42.0")
        XCTAssertFalse(result.isError)

        // 6. Format
        let formatted = formatToolResult(result)
        XCTAssertEqual(formatted, "[Tool Result] 42.0")
    }

    /// **Test:** Config system prompt → parse → dispatch round-trip
    func testConfigParseDispatchRoundtrip() async throws {
        let tool = ToolDefinition(
            name: "echo",
            description: "Echo input",
            parameters: .object(properties: [
                JSONSchemaProperty(name: "message", schema: .string, required: true)
            ], required: ["message"])
        )

        let config = ToolCallingConfig(tools: [tool])
        let prompt = config.buildSystemPrompt()
        XCTAssertTrue(prompt.contains("echo"), "System prompt should describe the tool")

        let dispatcher = ClosureToolDispatcher()
        dispatcher.register("echo") { call in
            call.arguments["message"] as? String ?? ""
        }

        var parser = ToolCallParser(tools: config.tools)
        let result = try await processToolCall(
            generatedText: "{\"name\": \"echo\", \"arguments\": {\"message\": \"hello\"}}",
            parser: &parser,
            dispatcher: dispatcher
        )

        XCTAssertNotNil(result)
        XCTAssertEqual(result?.content, "hello")
    }
}
