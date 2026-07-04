/// Constrained sampler for schema-guided JSON generation
///
/// **TB-Tool-003:** Wraps the existing `Sampler` to enforce JSON structure.
///
/// The public seam is intentionally small: mask logits before sampling, then
/// advance the state machine with the accepted token text. Internally the
/// sampler treats each token as a whole string, so tokens such as `{"name"` or
/// split values such as `"hel` + `lo"` update the JSON parser correctly.

import Foundation

// MARK: - Output Constraint Protocol

/// Protocol for output constraints that can mask logits.
///
/// Pluggable design: swap in different constraint strategies such as JSON
/// schema, regex, or grammar constraints without changing the generation loop.
public protocol OutputConstraint {
    /// Applies the constraint by masking or biasing invalid token logits.
    ///
    /// - Parameters:
    ///   - logits: Mutable logit tensor. Strict constraints set invalid
    ///     positions to `-Float.infinity`.
    ///   - tokenizer: Maps token IDs to raw token text for structural analysis.
    mutating func maskLogits(_ logits: inout Tensor<Float>, tokenizer: TokenLookup)

    /// Updates internal state after a token is accepted.
    ///
    /// - Parameter token: The raw token text that was just generated.
    mutating func advance(token: String)

    /// Whether the constraint considers the output complete.
    var isComplete: Bool { get }

    /// Human-readable description of the current constraint state.
    var stateDescription: String { get }
}

// MARK: - Token Lookup

/// Minimal interface for looking up raw token text by ID.
///
/// The returned string must be the token piece used for constraint advancement,
/// not display text that has been post-processed in a sequence-dependent way.
/// This keeps generation-time parsing from corrupting tokenizer-owned spacing
/// or byte-level markers.
public protocol TokenLookup {
    /// Returns the raw token text for a single token ID.
    func decode(tokenId: Int) -> String

    /// Total vocabulary size.
    var vocabularySize: Int { get }
}

/// Cached token-string table for constrained decoding.
///
/// Build this once per generation from the caller's tokenizer bridge, then use
/// it anywhere a ``TokenLookup`` is required. This prevents full-vocabulary
/// token decoding from running once per generated token.
public struct TokenStringTable: TokenLookup {
    /// Raw token text by token ID.
    public let pieces: [String]

    /// Total vocabulary size.
    public var vocabularySize: Int { pieces.count }

    /// Builds a table by decoding every token ID exactly once.
    ///
    /// - Parameter tokenLookup: Token lookup for the active tokenizer/vocabulary.
    public init(tokenLookup: any TokenLookup) {
        self.pieces = (0..<tokenLookup.vocabularySize).map {
            tokenLookup.decode(tokenId: $0)
        }
    }

    /// Returns the cached raw token text for a token ID.
    public func decode(tokenId: Int) -> String {
        guard tokenId >= 0, tokenId < pieces.count else { return "" }
        return pieces[tokenId]
    }
}

// MARK: - Constraint Mode

/// How strictly to enforce the output schema.
public enum ConstraintMode: Equatable {
    /// Hard masking: invalid tokens get `-infinity` logits.
    case strict

    /// Soft biasing: invalid tokens get a negative bias.
    case guided

    /// No constraint applied.
    case none
}

// MARK: - Constrained Sampler

/// Schema-guided constrained sampler that wraps `Sampler`.
///
/// The sampler maintains a JSON-schema prefix parser and caches the allowed
/// token mask for each parser-state fingerprint. It never decodes the
/// vocabulary on repeated mask calls for the same generation session.
public struct ConstrainedSampler: OutputConstraint {
    /// The root schema being enforced.
    public let schema: JSONSchema

    /// How strictly to enforce constraints.
    public let mode: ConstraintMode

    /// Whether every accepted token so far is still a valid schema prefix.
    public private(set) var isValidPrefix: Bool = true

    /// Number of tokens allowed by the most recent mask operation.
    public private(set) var lastAllowedTokenCount: Int?

    /// Test-visible counter for allowed-mask cache misses.
    internal private(set) var allowedMaskCacheMisses = 0

    private var parser: JSONPrefixParser
    private var cachedTokenPieces: [String]?
    private var cachedTokenPieceCount: Int?
    private var allowedMaskCache: [String: [Bool]] = [:]
    private let guidedBias: Float = -10.0

    /// Initializes a constrained sampler for a given schema.
    ///
    /// - Parameters:
    ///   - schema: The JSON schema to enforce.
    ///   - mode: Constraint strictness. Defaults to `.strict`.
    public init(schema: JSONSchema, mode: ConstraintMode = .strict) {
        let parser = JSONPrefixParser(schema: schema)
        self.schema = schema
        self.mode = mode
        self.parser = parser
    }

    public var isComplete: Bool {
        isValidPrefix && parser.isComplete
    }

    public var stateDescription: String {
        isValidPrefix ? parser.stateDescription : "invalid"
    }

    public mutating func maskLogits(_ logits: inout Tensor<Float>, tokenizer: TokenLookup) {
        guard mode != .none else { return }
        guard isValidPrefix else {
            maskAll(&logits)
            lastAllowedTokenCount = 0
            return
        }

        let pieces = tokenPieces(from: tokenizer)
        let cacheKey = "\(pieces.count)|\(parser.fingerprint)"
        let allowedMask: [Bool]
        if let cached = allowedMaskCache[cacheKey] {
            allowedMask = cached
        } else {
            allowedMask = computeAllowedMask(pieces: pieces)
            allowedMaskCache[cacheKey] = allowedMask
            allowedMaskCacheMisses += 1
        }

        let allowedCount = allowedMask.reduce(0) { $0 + ($1 ? 1 : 0) }
        lastAllowedTokenCount = allowedCount

        guard allowedCount > 0 else {
            maskAll(&logits)
            return
        }

        for tokenId in 0..<logits.data.count {
            let allowed = tokenId < allowedMask.count && allowedMask[tokenId]
            if !allowed {
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
        guard isValidPrefix, !token.isEmpty else {
            isValidPrefix = false
            return
        }

        guard parser.consume(token) else {
            isValidPrefix = false
            return
        }

        parser.finishTokenBoundary()
    }

    private mutating func tokenPieces(from tokenizer: TokenLookup) -> [String] {
        if let table = tokenizer as? TokenStringTable {
            if cachedTokenPieceCount != table.vocabularySize {
                cachedTokenPieces = table.pieces
                cachedTokenPieceCount = table.vocabularySize
                allowedMaskCache.removeAll(keepingCapacity: true)
            }
            return table.pieces
        }

        if cachedTokenPieceCount == tokenizer.vocabularySize,
           let cachedTokenPieces {
            return cachedTokenPieces
        }

        let pieces = (0..<tokenizer.vocabularySize).map {
            tokenizer.decode(tokenId: $0)
        }
        cachedTokenPieces = pieces
        cachedTokenPieceCount = tokenizer.vocabularySize
        allowedMaskCache.removeAll(keepingCapacity: true)
        return pieces
    }

    private func computeAllowedMask(pieces: [String]) -> [Bool] {
        pieces.map { piece in
            guard !piece.isEmpty else { return false }
            var candidate = parser
            guard candidate.consume(piece) else { return false }
            return candidate.hasPathToCompletion
        }
    }

    private func maskAll(_ logits: inout Tensor<Float>) {
        switch mode {
        case .strict:
            for index in logits.data.indices {
                logits.data[index] = -Float.infinity
            }
        case .guided:
            for index in logits.data.indices {
                logits.data[index] += guidedBias
            }
        case .none:
            break
        }
    }

}

// MARK: - JSON Prefix Parser

private struct JSONPrefixParser: Equatable {
    private var state: State

    init(schema: JSONSchema) {
        self.state = .expectingValue(schema, after: .root)
    }

    var isComplete: Bool {
        if case .complete = state { return true }
        return false
    }

    var stateDescription: String {
        state.description
    }

    var fingerprint: String {
        state.fingerprint
    }

    var hasPathToCompletion: Bool {
        state.hasPathToCompletion
    }

    mutating func consume(_ text: String) -> Bool {
        for scalar in text.unicodeScalars {
            guard consume(scalar) else { return false }
        }
        return hasPathToCompletion
    }

    mutating func finishTokenBoundary() {
        if case .inNumber(let buffer, let integerOnly, let after) = state,
           Self.isCompleteNumber(buffer, integerOnly: integerOnly) {
            state = Self.stateAfterValue(after)
        }
    }

    private mutating func consume(_ scalar: UnicodeScalar) -> Bool {
        var reprocess = true
        var guardCount = 0

        while reprocess {
            guardCount += 1
            if guardCount > 8 { return false }
            reprocess = false

            switch state {
            case .expectingValue(let schema, let after):
                if scalar.isJSONWhitespace { return true }
                switch schema {
                case .object(let properties, let required):
                    guard scalar == JSONScalars.openBrace else { return false }
                    let context = ObjectContext(properties: properties, required: required)
                    state = .expectingObjectKey(context: context, mustEmitKey: false, after: after)
                    return state.hasPathToCompletion
                case .array(let items):
                    guard scalar == JSONScalars.openBracket else { return false }
                    state = .expectingArrayElement(items: items, count: 0, allowClose: true, after: after)
                    return true
                case .string:
                    guard scalar == JSONScalars.quote else { return false }
                    state = .inString(kind: .free(.string), prefix: "", escape: .none, after: after)
                    return true
                case .enum(let values):
                    guard scalar == JSONScalars.quote, !values.isEmpty else { return false }
                    state = .inString(kind: .enumeration(values), prefix: "", escape: .none, after: after)
                    return true
                case .boolean:
                    if scalar == JSONScalars.t {
                        state = .inLiteral(expected: "true", matched: 1, after: after)
                        return true
                    }
                    if scalar == JSONScalars.f {
                        state = .inLiteral(expected: "false", matched: 1, after: after)
                        return true
                    }
                    return false
                case .null:
                    guard scalar == JSONScalars.n else { return false }
                    state = .inLiteral(expected: "null", matched: 1, after: after)
                    return true
                case .integer, .number:
                    guard Self.canStartNumber(with: scalar) else { return false }
                    let text = String(scalar)
                    let integerOnly = schema == .integer
                    guard Self.isNumberPrefix(text, integerOnly: integerOnly) else { return false }
                    state = .inNumber(buffer: text, integerOnly: integerOnly, after: after)
                    return true
                }

            case .expectingObjectKey(let context, let mustEmitKey, let after):
                if scalar.isJSONWhitespace { return true }
                if scalar == JSONScalars.closeBrace {
                    guard !mustEmitKey && context.canClose else { return false }
                    state = Self.stateAfterValue(after)
                    return true
                }
                guard scalar == JSONScalars.quote,
                      !context.availableProperties.isEmpty else { return false }
                state = .inObjectKey(context: context, prefix: "", after: after)
                return true

            case .inObjectKey(let context, let prefix, let after):
                if scalar == JSONScalars.quote {
                    guard let property = context.availableProperties.first(where: { $0.name == prefix }) else {
                        return false
                    }
                    var updated = context
                    updated.emitted.insert(property.name)
                    state = .expectingColon(context: updated, key: property.name, after: after)
                    return updated.canEventuallyComplete
                }
                guard scalar != JSONScalars.backslash,
                      !scalar.isJSONControl else { return false }
                let next = prefix + String(scalar)
                guard context.availableProperties.contains(where: { $0.name.hasPrefix(next) }) else {
                    return false
                }
                state = .inObjectKey(context: context, prefix: next, after: after)
                return true

            case .expectingColon(let context, let key, let after):
                if scalar.isJSONWhitespace { return true }
                guard scalar == JSONScalars.colon,
                      let schema = context.property(named: key)?.schema else { return false }
                state = .expectingValue(schema, after: .objectValue(context: context, after: after))
                return true

            case .expectingObjectCommaOrClose(let context, let after):
                if scalar.isJSONWhitespace { return true }
                if scalar == JSONScalars.closeBrace {
                    guard context.canClose else { return false }
                    state = Self.stateAfterValue(after)
                    return true
                }
                guard scalar == JSONScalars.comma,
                      !context.availableProperties.isEmpty else { return false }
                state = .expectingObjectKey(context: context, mustEmitKey: true, after: after)
                return state.hasPathToCompletion

            case .expectingArrayElement(let items, let count, let allowClose, let after):
                if scalar.isJSONWhitespace { return true }
                if allowClose && scalar == JSONScalars.closeBracket {
                    state = Self.stateAfterValue(after)
                    return true
                }
                state = .expectingValue(items, after: .arrayElement(items: items, count: count, after: after))
                reprocess = true

            case .expectingArrayCommaOrClose(let items, let count, let after):
                if scalar.isJSONWhitespace { return true }
                if scalar == JSONScalars.closeBracket {
                    state = Self.stateAfterValue(after)
                    return true
                }
                guard scalar == JSONScalars.comma else { return false }
                state = .expectingArrayElement(items: items, count: count, allowClose: false, after: after)
                return true

            case .inString(let kind, let prefix, let escape, let after):
                guard consumeStringScalar(
                    scalar,
                    kind: kind,
                    prefix: prefix,
                    escape: escape,
                    after: after
                ) else {
                    return false
                }
                return true

            case .inLiteral(let expected, let matched, let after):
                let expectedScalars = Array(expected.unicodeScalars)
                guard matched < expectedScalars.count,
                      scalar == expectedScalars[matched] else { return false }
                let nextMatched = matched + 1
                if nextMatched == expectedScalars.count {
                    state = Self.stateAfterValue(after)
                } else {
                    state = .inLiteral(expected: expected, matched: nextMatched, after: after)
                }
                return true

            case .inNumber(let buffer, let integerOnly, let after):
                if Self.isNumberDelimiter(scalar) {
                    guard Self.isCompleteNumber(buffer, integerOnly: integerOnly) else { return false }
                    state = Self.stateAfterValue(after)
                    reprocess = true
                } else {
                    let next = buffer + String(scalar)
                    guard Self.isNumberPrefix(next, integerOnly: integerOnly) else { return false }
                    state = .inNumber(buffer: next, integerOnly: integerOnly, after: after)
                    return true
                }

            case .complete:
                return scalar.isJSONWhitespace
            }
        }

        return true
    }

    private mutating func consumeStringScalar(
        _ scalar: UnicodeScalar,
        kind: StringKind,
        prefix: String,
        escape: StringEscape,
        after: Continuation
    ) -> Bool {
        switch escape {
        case .none:
            if scalar == JSONScalars.quote {
                switch kind {
                case .free:
                    state = Self.stateAfterValue(after)
                    return true
                case .enumeration(let values):
                    guard values.contains(prefix) else { return false }
                    state = Self.stateAfterValue(after)
                    return true
                }
            }

            if scalar == JSONScalars.backslash {
                guard case .free = kind else { return false }
                state = .inString(kind: kind, prefix: prefix, escape: .escaped, after: after)
                return true
            }

            guard !scalar.isJSONControl else { return false }
            let next = prefix + String(scalar)
            switch kind {
            case .free:
                state = .inString(kind: kind, prefix: next, escape: .none, after: after)
                return true
            case .enumeration(let values):
                guard values.contains(where: { $0.hasPrefix(next) }) else { return false }
                state = .inString(kind: kind, prefix: next, escape: .none, after: after)
                return true
            }

        case .escaped:
            guard scalar.isJSONSimpleEscape || scalar == JSONScalars.u else { return false }
            let nextEscape: StringEscape = scalar == JSONScalars.u ? .unicodeDigits(0) : .none
            state = .inString(kind: kind, prefix: prefix, escape: nextEscape, after: after)
            return true

        case .unicodeDigits(let count):
            guard scalar.isJSONHexDigit else { return false }
            if count == 3 {
                state = .inString(kind: kind, prefix: prefix, escape: .none, after: after)
            } else {
                state = .inString(kind: kind, prefix: prefix, escape: .unicodeDigits(count + 1), after: after)
            }
            return true
        }
    }

    private static func stateAfterValue(_ continuation: Continuation) -> State {
        switch continuation {
        case .root:
            return .complete
        case .objectValue(let context, let after):
            return .expectingObjectCommaOrClose(context: context, after: after)
        case .arrayElement(let items, let count, let after):
            return .expectingArrayCommaOrClose(items: items, count: count + 1, after: after)
        }
    }

    private static func canStartNumber(with scalar: UnicodeScalar) -> Bool {
        scalar == JSONScalars.minus || scalar.isJSONDigit
    }

    private static func isNumberDelimiter(_ scalar: UnicodeScalar) -> Bool {
        scalar.isJSONWhitespace ||
            scalar == JSONScalars.comma ||
            scalar == JSONScalars.closeBrace ||
            scalar == JSONScalars.closeBracket
    }

    private static func isNumberPrefix(_ text: String, integerOnly: Bool) -> Bool {
        numberLexState(for: text, integerOnly: integerOnly)?.canReachCompleteNumber == true
    }

    private static func isCompleteNumber(_ text: String, integerOnly: Bool) -> Bool {
        numberLexState(for: text, integerOnly: integerOnly)?.isCompleteNumber == true
    }

    private static func numberLexState(for text: String, integerOnly: Bool) -> NumberLexState? {
        guard !text.isEmpty else { return nil }

        var state = NumberLexState.start
        for scalar in text.unicodeScalars {
            switch state {
            case .start:
                if scalar == JSONScalars.minus {
                    state = .afterMinus
                } else if scalar == "0" {
                    state = .zero
                } else if scalar.isJSONNonZeroDigit {
                    state = .intDigits
                } else {
                    return nil
                }
            case .afterMinus:
                if scalar == "0" {
                    state = .zero
                } else if scalar.isJSONNonZeroDigit {
                    state = .intDigits
                } else {
                    return nil
                }
            case .zero:
                if integerOnly {
                    return nil
                } else if scalar == "." {
                    state = .afterDot
                } else if scalar == "e" || scalar == "E" {
                    state = .afterExponent
                } else {
                    return nil
                }
            case .intDigits:
                if scalar.isJSONDigit {
                    state = .intDigits
                } else if !integerOnly && scalar == "." {
                    state = .afterDot
                } else if !integerOnly && (scalar == "e" || scalar == "E") {
                    state = .afterExponent
                } else {
                    return nil
                }
            case .afterDot:
                if scalar.isJSONDigit {
                    state = .fractionDigits
                } else {
                    return nil
                }
            case .fractionDigits:
                if scalar.isJSONDigit {
                    state = .fractionDigits
                } else if scalar == "e" || scalar == "E" {
                    state = .afterExponent
                } else {
                    return nil
                }
            case .afterExponent:
                if scalar == "+" || scalar == JSONScalars.minus {
                    state = .afterExponentSign
                } else if scalar.isJSONDigit {
                    state = .exponentDigits
                } else {
                    return nil
                }
            case .afterExponentSign:
                if scalar.isJSONDigit {
                    state = .exponentDigits
                } else {
                    return nil
                }
            case .exponentDigits:
                if scalar.isJSONDigit {
                    state = .exponentDigits
                } else {
                    return nil
                }
            }
        }

        return state
    }

    private enum NumberLexState {
        case start
        case afterMinus
        case zero
        case intDigits
        case afterDot
        case fractionDigits
        case afterExponent
        case afterExponentSign
        case exponentDigits

        var isCompleteNumber: Bool {
            switch self {
            case .zero, .intDigits, .fractionDigits, .exponentDigits:
                return true
            case .start, .afterMinus, .afterDot, .afterExponent, .afterExponentSign:
                return false
            }
        }

        var canReachCompleteNumber: Bool {
            switch self {
            case .start:
                return false
            case .afterMinus, .zero, .intDigits, .afterDot, .fractionDigits, .afterExponent, .afterExponentSign, .exponentDigits:
                return true
            }
        }
    }

    private indirect enum State: Equatable {
        case expectingValue(JSONSchema, after: Continuation)
        case expectingObjectKey(context: ObjectContext, mustEmitKey: Bool, after: Continuation)
        case inObjectKey(context: ObjectContext, prefix: String, after: Continuation)
        case expectingColon(context: ObjectContext, key: String, after: Continuation)
        case expectingObjectCommaOrClose(context: ObjectContext, after: Continuation)
        case expectingArrayElement(items: JSONSchema, count: Int, allowClose: Bool, after: Continuation)
        case expectingArrayCommaOrClose(items: JSONSchema, count: Int, after: Continuation)
        case inString(kind: StringKind, prefix: String, escape: StringEscape, after: Continuation)
        case inLiteral(expected: String, matched: Int, after: Continuation)
        case inNumber(buffer: String, integerOnly: Bool, after: Continuation)
        case complete

        var description: String {
            switch self {
            case .expectingValue(let schema, _):
                switch schema {
                case .object:
                    return "expecting '{'"
                case .array:
                    return "expecting '['"
                default:
                    return "expecting value(\(schema.constraintDescription))"
                }
            case .expectingObjectKey, .inObjectKey:
                return "expecting key"
            case .expectingColon:
                return "expecting ':'"
            case .expectingObjectCommaOrClose:
                return "expecting ',' or '}'"
            case .expectingArrayElement:
                return "expecting array element"
            case .expectingArrayCommaOrClose:
                return "expecting ',' or ']'"
            case .inString(let kind, _, _, _):
                return "expecting value(\(kind.schema.constraintDescription))"
            case .inLiteral(let expected, _, _):
                return "expecting value(\(expected == "null" ? "null" : "boolean"))"
            case .inNumber(_, let integerOnly, _):
                return "expecting value(\(integerOnly ? "integer" : "number"))"
            case .complete:
                return "complete"
            }
        }

        var fingerprint: String {
            switch self {
            case .expectingValue(let schema, let after):
                return "value(\(schema.fingerprint))|\(after.fingerprint)"
            case .expectingObjectKey(let context, let mustEmitKey, let after):
                return "objectKey(\(context.fingerprint),\(mustEmitKey))|\(after.fingerprint)"
            case .inObjectKey(let context, let prefix, let after):
                return "objectKeyPrefix(\(context.fingerprint),\(prefix))|\(after.fingerprint)"
            case .expectingColon(let context, let key, let after):
                return "colon(\(context.fingerprint),\(key))|\(after.fingerprint)"
            case .expectingObjectCommaOrClose(let context, let after):
                return "objectComma(\(context.fingerprint))|\(after.fingerprint)"
            case .expectingArrayElement(let items, let count, let allowClose, let after):
                return "arrayValue(\(items.fingerprint),\(count),\(allowClose))|\(after.fingerprint)"
            case .expectingArrayCommaOrClose(let items, let count, let after):
                return "arrayComma(\(items.fingerprint),\(count))|\(after.fingerprint)"
            case .inString(let kind, let prefix, let escape, let after):
                return "string(\(kind.fingerprint),\(kind.cachePrefix(prefix)),\(escape.fingerprint))|\(after.fingerprint)"
            case .inLiteral(let expected, let matched, let after):
                return "literal(\(expected),\(matched))|\(after.fingerprint)"
            case .inNumber(let buffer, let integerOnly, let after):
                return "number(\(integerOnly),\(buffer))|\(after.fingerprint)"
            case .complete:
                return "complete"
            }
        }

        var hasPathToCompletion: Bool {
            switch self {
            case .expectingValue(let schema, _):
                return schema.hasPossibleValue
            case .expectingObjectKey(let context, let mustEmitKey, _):
                let canClose = !mustEmitKey && context.canClose
                return canClose || (!context.availableProperties.isEmpty && context.canEventuallyComplete)
            case .inObjectKey(let context, let prefix, _):
                return context.availableProperties.contains { $0.name.hasPrefix(prefix) }
            case .expectingColon(let context, let key, _):
                return context.property(named: key) != nil
            case .expectingObjectCommaOrClose(let context, _):
                return context.canClose || (!context.availableProperties.isEmpty && context.canEventuallyComplete)
            case .expectingArrayElement(let items, _, let allowClose, _):
                return allowClose || items.hasPossibleValue
            case .expectingArrayCommaOrClose:
                return true
            case .inString(let kind, let prefix, let escape, _):
                switch escape {
                case .none:
                    return kind.canComplete(prefix: prefix)
                case .escaped, .unicodeDigits:
                    return true
                }
            case .inLiteral:
                return true
            case .inNumber(let buffer, let integerOnly, _):
                return JSONPrefixParser.isNumberPrefix(buffer, integerOnly: integerOnly)
            case .complete:
                return true
            }
        }
    }

    private indirect enum Continuation: Equatable {
        case root
        case objectValue(context: ObjectContext, after: Continuation)
        case arrayElement(items: JSONSchema, count: Int, after: Continuation)

        var fingerprint: String {
            switch self {
            case .root:
                return "root"
            case .objectValue(let context, let after):
                return "objectValue(\(context.fingerprint))|\(after.fingerprint)"
            case .arrayElement(let items, let count, let after):
                return "arrayValue(\(items.fingerprint),\(count))|\(after.fingerprint)"
            }
        }
    }

    private enum StringKind: Equatable {
        case free(JSONSchema)
        case enumeration([String])

        var schema: JSONSchema {
            switch self {
            case .free(let schema):
                return schema
            case .enumeration(let values):
                return .enum(values: values)
            }
        }

        var fingerprint: String {
            switch self {
            case .free:
                return "free"
            case .enumeration(let values):
                return "enum(\(values.joined(separator: "\u{1F}")))"
            }
        }

        func cachePrefix(_ prefix: String) -> String {
            switch self {
            case .free:
                return ""
            case .enumeration:
                return prefix
            }
        }

        func canComplete(prefix: String) -> Bool {
            switch self {
            case .free:
                return true
            case .enumeration(let values):
                return values.contains { $0.hasPrefix(prefix) }
            }
        }
    }

    private enum StringEscape: Equatable {
        case none
        case escaped
        case unicodeDigits(Int)

        var fingerprint: String {
            switch self {
            case .none:
                return "none"
            case .escaped:
                return "escaped"
            case .unicodeDigits(let count):
                return "u\(count)"
            }
        }
    }
}

private struct ObjectContext: Equatable {
    let properties: [JSONSchemaProperty]
    let required: Set<String>
    var emitted: Set<String>

    init(properties: [JSONSchemaProperty], required: [String], emitted: Set<String> = []) {
        self.properties = properties
        self.required = Set(required).union(properties.filter(\.required).map(\.name))
        self.emitted = emitted
    }

    var availableProperties: [JSONSchemaProperty] {
        properties.filter { !emitted.contains($0.name) }
    }

    var canClose: Bool {
        required.isSubset(of: emitted)
    }

    var canEventuallyComplete: Bool {
        let names = Set(properties.map(\.name))
        return required.subtracting(emitted).isSubset(of: names)
    }

    var fingerprint: String {
        let props = properties.map(\.name).joined(separator: ",")
        let requiredKey = required.sorted().joined(separator: ",")
        let emittedKey = emitted.sorted().joined(separator: ",")
        return "props=\(props);required=\(requiredKey);emitted=\(emittedKey)"
    }

    func property(named name: String) -> JSONSchemaProperty? {
        properties.first { $0.name == name }
    }
}

private enum JSONScalars {
    static let quote: UnicodeScalar = "\""
    static let backslash: UnicodeScalar = "\\"
    static let openBrace: UnicodeScalar = "{"
    static let closeBrace: UnicodeScalar = "}"
    static let openBracket: UnicodeScalar = "["
    static let closeBracket: UnicodeScalar = "]"
    static let colon: UnicodeScalar = ":"
    static let comma: UnicodeScalar = ","
    static let minus: UnicodeScalar = "-"
    static let t: UnicodeScalar = "t"
    static let f: UnicodeScalar = "f"
    static let n: UnicodeScalar = "n"
    static let u: UnicodeScalar = "u"
}

private extension UnicodeScalar {
    var isJSONWhitespace: Bool {
        self == " " || self == "\n" || self == "\r" || self == "\t"
    }

    var isJSONControl: Bool {
        value < 0x20
    }

    var isJSONDigit: Bool {
        value >= 48 && value <= 57
    }

    var isJSONNonZeroDigit: Bool {
        value >= 49 && value <= 57
    }

    var isJSONHexDigit: Bool {
        (value >= 48 && value <= 57) ||
            (value >= 65 && value <= 70) ||
            (value >= 97 && value <= 102)
    }

    var isJSONSimpleEscape: Bool {
        self == "\"" || self == "\\" || self == "/" ||
            self == "b" || self == "f" || self == "n" ||
            self == "r" || self == "t"
    }
}

private extension JSONSchema {
    var constraintDescription: String {
        switch self {
        case .string:
            return "string"
        case .number:
            return "number"
        case .integer:
            return "integer"
        case .boolean:
            return "boolean"
        case .array:
            return "array"
        case .object:
            return "object"
        case .enum(let values):
            return "enum(\(values.joined(separator: "|")))"
        case .null:
            return "null"
        }
    }

    var fingerprint: String {
        switch self {
        case .string:
            return "string"
        case .number:
            return "number"
        case .integer:
            return "integer"
        case .boolean:
            return "boolean"
        case .null:
            return "null"
        case .enum(let values):
            return "enum(\(values.joined(separator: "\u{1F}")))"
        case .array(let items):
            return "array(\(items.fingerprint))"
        case .object(let properties, let required):
            let props = properties
                .map { "\($0.name):\($0.schema.fingerprint):\($0.required)" }
                .joined(separator: "\u{1E}")
            return "object(\(props)):\(required.sorted().joined(separator: ","))"
        }
    }

    var hasPossibleValue: Bool {
        switch self {
        case .enum(let values):
            return !values.isEmpty
        case .object(let properties, let required):
            let names = Set(properties.map(\.name))
            let requiredSet = Set(required).union(properties.filter(\.required).map(\.name))
            return requiredSet.isSubset(of: names)
        case .array(let items):
            return items.hasPossibleValue
        default:
            return true
        }
    }
}
