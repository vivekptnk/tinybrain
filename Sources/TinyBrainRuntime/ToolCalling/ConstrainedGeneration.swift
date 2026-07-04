import Foundation

/// Errors raised by generation-time structured-output enforcement.
public enum GenerationError: Error, Equatable, CustomStringConvertible {
    /// A schema constraint was requested but no token lookup was provided.
    case constrainedGenerationRequiresTokenLookup

    /// The constraint has no valid continuation tokens for the current state.
    case constrainedGenerationHasNoValidTokens(state: String)

    /// A stop token was selected before the structured output was complete.
    case constrainedGenerationSelectedStopTokenBeforeCompletion(tokenId: Int, state: String)

    /// The accepted token made the constraint parser invalid.
    case constrainedGenerationAcceptedInvalidToken(tokenId: Int, token: String, state: String)

    /// The token budget ended before the structured output reached completion.
    case constrainedGenerationIncomplete

    public var description: String {
        switch self {
        case .constrainedGenerationRequiresTokenLookup:
            return "Constrained generation requires a TokenLookup for the active tokenizer."
        case .constrainedGenerationHasNoValidTokens(let state):
            return "Constrained generation has no valid tokens from state: \(state)."
        case .constrainedGenerationSelectedStopTokenBeforeCompletion(let tokenId, let state):
            return "Stop token \(tokenId) was selected before constrained generation completed from state: \(state)."
        case .constrainedGenerationAcceptedInvalidToken(let tokenId, let token, let state):
            return "Token \(tokenId) (\(token.debugDescription)) invalidated constrained generation from state: \(state)."
        case .constrainedGenerationIncomplete:
            return "Constrained generation reached maxTokens before the structured output completed."
        }
    }
}

extension GenerationConfig {
    var effectiveConstraintSchema: JSONSchema? {
        if let outputSchema {
            return outputSchema
        }

        guard let toolCallingConfig else {
            return nil
        }

        switch toolCallingConfig.toolChoice {
        case .specific(let name):
            guard let tool = toolCallingConfig.tools.first(where: { $0.name == name }) else {
                return nil
            }
            return Self.toolCallSchema(for: tool)
        case .required where toolCallingConfig.tools.count == 1:
            return Self.toolCallSchema(for: toolCallingConfig.tools[0])
        case .auto, .none, .required:
            return nil
        }
    }

    var hasActiveConstraint: Bool {
        constraintMode != .none && effectiveConstraintSchema != nil
    }

    private static func toolCallSchema(for tool: ToolDefinition) -> JSONSchema {
        .object(properties: [
            JSONSchemaProperty(name: "name", schema: .enum(values: [tool.name]), required: true),
            JSONSchemaProperty(name: "arguments", schema: tool.parameters, required: true)
        ], required: ["name", "arguments"])
    }
}

struct ConstraintSession {
    private var sampler: ConstrainedSampler
    private let tokenTable: TokenStringTable

    var isComplete: Bool {
        sampler.isComplete
    }

    var stateDescription: String {
        sampler.stateDescription
    }

    static func make(
        config: GenerationConfig,
        tokenLookup: (any TokenLookup)?
    ) throws -> ConstraintSession? {
        guard config.constraintMode != .none,
              let schema = config.effectiveConstraintSchema else {
            return nil
        }

        guard let tokenLookup else {
            throw GenerationError.constrainedGenerationRequiresTokenLookup
        }

        return ConstraintSession(
            sampler: ConstrainedSampler(schema: schema, mode: config.constraintMode),
            tokenTable: TokenStringTable(tokenLookup: tokenLookup)
        )
    }

    mutating func maskedLogits(from logits: Tensor<Float>) throws -> Tensor<Float> {
        var masked = logits
        sampler.maskLogits(&masked, tokenizer: tokenTable)

        if sampler.lastAllowedTokenCount == 0 {
            throw GenerationError.constrainedGenerationHasNoValidTokens(state: sampler.stateDescription)
        }

        return masked
    }

    mutating func accept(tokenId: Int) throws {
        let token = tokenTable.decode(tokenId: tokenId)
        let previousState = sampler.stateDescription
        sampler.advance(token: token)

        guard sampler.isValidPrefix else {
            throw GenerationError.constrainedGenerationAcceptedInvalidToken(
                tokenId: tokenId,
                token: token,
                state: previousState
            )
        }
    }
}
