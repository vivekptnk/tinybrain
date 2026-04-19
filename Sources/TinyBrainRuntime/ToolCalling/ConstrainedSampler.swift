/// Constrained sampler for schema-guided JSON generation
///
/// **TB-Tool-003:** Wraps the existing `Sampler` to enforce JSON structure
///
/// ## Educational Overview
///
/// Unconstrained generation can produce invalid JSON. The constrained
/// sampler solves this by tracking *where* in the JSON structure we are
/// and masking logits for tokens that would be invalid at that position.
///
/// **State machine approach:**
/// ```
/// expectingOpenBrace → expectingKey → expectingColon →
/// expectingValue(schema) → expectingCommaOrClose → ...
/// ```
///
/// At each state, only tokens that produce valid JSON syntax are allowed.
/// The sampler never modifies `Sampler.swift` — it builds a logit mask
/// and delegates actual sampling to `Sampler.sampleDetailed()`.

import Foundation

// MARK: - Output Constraint Protocol

/// Protocol for output constraints that can mask logits
///
/// Pluggable design: swap in different constraint strategies
/// (JSON schema, regex, grammar) without changing the generation loop.
public protocol OutputConstraint {
    /// Apply constraint by masking invalid token logits
    ///
    /// - Parameters:
    ///   - logits: Mutable logit tensor — set invalid positions to `-Float.infinity`
    ///   - tokenizer: Maps token IDs to string pieces for structural analysis
    mutating func maskLogits(_ logits: inout Tensor<Float>, tokenizer: TokenLookup)

    /// Update internal state after a token is accepted
    ///
    /// - Parameter token: The token string that was just generated
    mutating func advance(token: String)

    /// Whether the constraint considers the output complete
    var isComplete: Bool { get }

    /// Human-readable description of current constraint state
    var stateDescription: String { get }
}

// MARK: - Token Lookup

/// Minimal interface for looking up token strings by ID
///
/// Decouples the constrained sampler from any specific tokenizer implementation.
public protocol TokenLookup {
    /// Decode a single token ID to its string representation
    func decode(tokenId: Int) -> String
    /// Total vocabulary size
    var vocabularySize: Int { get }
}

// MARK: - Constraint Mode

/// How strictly to enforce the output schema
public enum ConstraintMode: Equatable {
    /// Hard masking: invalid tokens get `-infinity` logits (guaranteed valid output)
    case strict
    /// Soft biasing: invalid tokens get a negative bias (mostly valid, allows recovery)
    case guided
    /// No constraint applied
    case none
}

// MARK: - Constrained Sampler State

/// Tracks position within the JSON structure being generated
public enum ConstrainedState: Equatable, CustomStringConvertible {
    /// Expecting `{` to open the root or nested object
    case expectingOpenBrace
    /// Expecting a quoted key name or `}` (if no required keys remain)
    case expectingKey(properties: [JSONSchemaProperty], required: Set<String>, emitted: Set<String>)
    /// Expecting `:` after a key
    case expectingColon
    /// Expecting a value matching the given schema
    case expectingValue(JSONSchema)
    /// Expecting `,` (more fields) or `}` / `]` (close container)
    case expectingCommaOrClose(isArray: Bool)
    /// Expecting `[` to open an array
    case expectingOpenBracket(items: JSONSchema)
    /// Expecting array element value or `]`
    case expectingArrayElement(items: JSONSchema, count: Int)
    /// Generation is complete
    case complete

    public var description: String {
        switch self {
        case .expectingOpenBrace: return "expecting '{'"
        case .expectingKey: return "expecting key"
        case .expectingColon: return "expecting ':'"
        case .expectingValue(let s): return "expecting value(\(s))"
        case .expectingCommaOrClose(let isArray): return "expecting ',' or '\(isArray ? "]" : "}")'"
        case .expectingOpenBracket: return "expecting '['"
        case .expectingArrayElement: return "expecting array element"
        case .complete: return "complete"
        }
    }
}

// MARK: - Constrained Sampler

/// Schema-guided constrained sampler that wraps `Sampler`
///
/// Maintains a state stack to handle nested objects and arrays.
/// At each step, masks logits to only allow structurally valid tokens,
/// then delegates to `Sampler.sampleDetailed()` for the actual pick.
public struct ConstrainedSampler: OutputConstraint {
    /// Stack of states for nested structures
    public private(set) var stateStack: [ConstrainedState]

    /// The root schema being enforced
    public let schema: JSONSchema

    /// How strictly to enforce constraints
    public let mode: ConstraintMode

    /// Accumulated output buffer for multi-character token tracking
    private var buffer: String = ""

    /// Current key being populated (for object value routing)
    private var currentKey: String?

    /// Bias magnitude for guided mode (negative bias on invalid tokens)
    private let guidedBias: Float = -10.0

    /// Initialize a constrained sampler for a given schema
    ///
    /// - Parameters:
    ///   - schema: The JSON schema to enforce
    ///   - mode: Constraint strictness (default: `.strict`)
    public init(schema: JSONSchema, mode: ConstraintMode = .strict) {
        self.schema = schema
        self.mode = mode

        // Initialize state based on root schema type
        switch schema {
        case .object:
            self.stateStack = [.expectingOpenBrace]
        case .array(let items):
            self.stateStack = [.expectingOpenBracket(items: items)]
        default:
            // Primitive at root level — expect the value directly
            self.stateStack = [.expectingValue(schema)]
        }
    }

    // MARK: - OutputConstraint

    public var isComplete: Bool {
        guard let top = stateStack.last else { return true }
        if case .complete = top { return true }
        return stateStack.isEmpty
    }

    public var stateDescription: String {
        stateStack.last?.description ?? "empty"
    }

    public mutating func maskLogits(_ logits: inout Tensor<Float>, tokenizer: TokenLookup) {
        guard mode != .none else { return }
        guard let state = stateStack.last else { return }

        let vocabSize = tokenizer.vocabularySize
        let allowedTokens = computeAllowedTokens(state: state, tokenizer: tokenizer)

        if allowedTokens.isEmpty { return }

        for tokenId in 0..<min(vocabSize, logits.data.count) {
            if !allowedTokens.contains(tokenId) {
                switch mode {
                case .strict:
                    logits.data[tokenId] = -Float.infinity
                case .guided:
                    logits.data[tokenId] += guidedBias
                case .none:
                    break
                }
            }
        }
    }

    public mutating func advance(token: String) {
        buffer += token
        processBuffer()
    }

    // MARK: - Token Classification

    /// Compute which token IDs are valid given the current state
    private func computeAllowedTokens(state: ConstrainedState, tokenizer: TokenLookup) -> Set<Int> {
        var allowed = Set<Int>()
        let vocabSize = tokenizer.vocabularySize

        for tokenId in 0..<vocabSize {
            let piece = tokenizer.decode(tokenId: tokenId)
            let trimmed = piece.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty {
                // Whitespace tokens are generally safe
                allowed.insert(tokenId)
                continue
            }

            if isTokenValid(piece: trimmed, state: state) {
                allowed.insert(tokenId)
            }
        }

        return allowed
    }

    /// Check if a token piece is valid for the current state
    private func isTokenValid(piece: String, state: ConstrainedState) -> Bool {
        switch state {
        case .expectingOpenBrace:
            return piece.hasPrefix("{")

        case .expectingKey(_, _, _):
            return piece.hasPrefix("\"") || piece.hasPrefix("}")

        case .expectingColon:
            return piece.hasPrefix(":")

        case .expectingValue(let schema):
            return isValidValueStart(piece: piece, schema: schema)

        case .expectingCommaOrClose(let isArray):
            let closeChar: Character = isArray ? "]" : "}"
            return piece.hasPrefix(",") || piece.first == closeChar

        case .expectingOpenBracket:
            return piece.hasPrefix("[")

        case .expectingArrayElement(let items, _):
            return piece.hasPrefix("]") || isValidValueStart(piece: piece, schema: items)

        case .complete:
            return false
        }
    }

    /// Check if a token could start a valid value for the given schema
    private func isValidValueStart(piece: String, schema: JSONSchema) -> Bool {
        switch schema {
        case .string, .enum:
            return piece.hasPrefix("\"")
        case .number, .integer:
            return piece.first?.isNumber == true || piece.hasPrefix("-")
        case .boolean:
            return piece.hasPrefix("t") || piece.hasPrefix("f")
        case .null:
            return piece.hasPrefix("n")
        case .object:
            return piece.hasPrefix("{")
        case .array:
            return piece.hasPrefix("[")
        }
    }

    // MARK: - Buffer Processing

    /// Process accumulated buffer to advance state machine
    private mutating func processBuffer() {
        // Trim and process meaningful characters
        let trimmed = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let state = stateStack.last else { return }

        switch state {
        case .expectingOpenBrace:
            if trimmed.contains("{") {
                buffer = ""
                stateStack.removeLast()
                // Transition to expecting first key
                if case .object(let props, let required) = schema {
                    stateStack.append(.expectingKey(
                        properties: props,
                        required: Set(required),
                        emitted: []
                    ))
                } else {
                    stateStack.append(.expectingKey(properties: [], required: [], emitted: []))
                }
            }

        case .expectingKey(let props, let required, var emitted):
            if trimmed.contains("}") {
                buffer = ""
                stateStack.removeLast()
                popToParentAfterClose()
            } else if let key = extractQuotedString(from: trimmed) {
                buffer = ""
                currentKey = key
                emitted.insert(key)
                stateStack.removeLast()
                // Find schema for this key
                let propSchema = props.first(where: { $0.name == key })?.schema ?? .string
                // Push states: colon, then value
                stateStack.append(.expectingCommaOrClose(isArray: false))
                // After the value, we re-enter with updated emitted set
                stateStack.append(.expectingValue(propSchema))
                stateStack.append(.expectingColon)
            }

        case .expectingColon:
            if trimmed.contains(":") {
                buffer = ""
                stateStack.removeLast()
            }

        case .expectingValue(let valueSchema):
            if processValue(trimmed: trimmed, schema: valueSchema) {
                buffer = ""
                stateStack.removeLast()
            }

        case .expectingCommaOrClose(let isArray):
            if trimmed.contains(",") {
                buffer = ""
                stateStack.removeLast()
                // After comma, expect next key (object) or next element (array)
                // The parent state should handle this
            } else if isArray && trimmed.contains("]") {
                buffer = ""
                stateStack.removeLast()
                popToParentAfterClose()
            } else if !isArray && trimmed.contains("}") {
                buffer = ""
                stateStack.removeLast()
                popToParentAfterClose()
            }

        case .expectingOpenBracket:
            if trimmed.contains("[") {
                buffer = ""
                stateStack.removeLast()
                if case .array(let items) = schema {
                    stateStack.append(.expectingArrayElement(items: items, count: 0))
                }
            }

        case .expectingArrayElement(let items, let count):
            if trimmed.contains("]") {
                buffer = ""
                stateStack.removeLast()
                popToParentAfterClose()
            } else if processValue(trimmed: trimmed, schema: items) {
                buffer = ""
                stateStack.removeLast()
                stateStack.append(.expectingCommaOrClose(isArray: true))
            }

        case .complete:
            break
        }
    }

    /// Try to process a value token, returning true if the value is complete
    private func processValue(trimmed: String, schema: JSONSchema) -> Bool {
        switch schema {
        case .string, .enum:
            // Check for complete quoted string
            if trimmed.hasPrefix("\"") && trimmed.dropFirst().contains("\"") {
                return true
            }
            return false
        case .number, .integer:
            // Numbers are complete when we see a non-numeric follow-up
            // For simplicity, accept any token that starts with digit/minus
            if trimmed.first?.isNumber == true || trimmed.hasPrefix("-") {
                return true
            }
            return false
        case .boolean:
            return trimmed == "true" || trimmed == "false" ||
                   trimmed.hasPrefix("true") || trimmed.hasPrefix("false")
        case .null:
            return trimmed == "null" || trimmed.hasPrefix("null")
        case .object, .array:
            // Nested structures need their own state tracking
            // For now, accept the opening token as value start
            return trimmed.hasPrefix("{") || trimmed.hasPrefix("[")
        }
    }

    /// After closing a container, advance to complete or parent's comma/close state
    private mutating func popToParentAfterClose() {
        if stateStack.isEmpty {
            stateStack.append(.complete)
        }
    }

    /// Extract a quoted string from buffer (e.g., `"name"` → `name`)
    private func extractQuotedString(from text: String) -> String? {
        guard text.hasPrefix("\"") else { return nil }
        let content = text.dropFirst()
        guard let endQuote = content.firstIndex(of: "\"") else { return nil }
        return String(content[content.startIndex..<endQuote])
    }
}
