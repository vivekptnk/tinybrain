/// Configuration for text generation
///
/// **TB-005:** Production-ready streaming configuration
///
/// ## Educational Overview
///
/// GenerationConfig brings together all the pieces:
/// - **maxTokens**: How many tokens to generate
/// - **sampler**: Which sampling strategy to use (from Phase 2)
/// - **stopTokens**: When to stop early (e.g., EOS token)
///
/// **Usage Example:**
/// ```swift
/// let config = GenerationConfig(
///     maxTokens: 100,
///     sampler: SamplerConfig(
///         temperature: 0.7,
///         topK: 40
///     ),
///     stopTokens: [tokenizer.eosToken]
/// )
///
/// for try await output in runner.generateStream(prompt: tokens, config: config) {
///     print(output.tokenId)
/// }
/// ```

import Foundation

// MARK: - Generation Configuration

/// Configuration for controlling text generation
///
/// **TB-005:** Combines max length, sampling strategy, and stop conditions
public struct GenerationConfig {
    /// Maximum number of tokens to generate
    ///
    /// Generation will stop after this many tokens, even if no stop token encountered.
    public var maxTokens: Int
    
    /// Sampling configuration (temperature, top-k, top-p, etc.)
    ///
    /// Controls randomness and diversity of generated text.
    /// See `SamplerConfig` for details.
    public var sampler: SamplerConfig
    
    /// Token IDs that trigger early stopping
    ///
    /// **Common use cases:**
    /// - EOS (end-of-sequence) token
    /// - Newline token (for single-line generation)
    /// - Custom termination tokens
    ///
    /// **Example:**
    /// ```swift
    /// stopTokens: [tokenizer.eosToken]  // Stop at EOS
    /// ```
    public var stopTokens: [Int]
    
    /// Initialize generation configuration
    ///
    /// - Parameters:
    ///   - maxTokens: Maximum tokens to generate (default: 100)
    ///   - sampler: Sampling configuration (default: temperature=1.0)
    ///   - stopTokens: Tokens that halt generation (default: empty)
    public init(
        maxTokens: Int = 100,
        sampler: SamplerConfig = SamplerConfig(),
        stopTokens: [Int] = []
    ) {
        self.maxTokens = maxTokens
        self.sampler = sampler
        self.stopTokens = stopTokens
    }
}

// MARK: - Token Output

/// Output from streaming token generation
///
/// **TB-005:** Rich metadata for each generated token
///
/// ## Educational: Why Include Metadata?
///
/// **tokenId** alone isn't enough for production:
/// - **probability**: Helps detect when model is "uncertain" (low prob)
/// - **timestamp**: Enables latency tracking and UX optimization
///
/// **Real-world use case:**
/// ```swift
/// for try await output in stream {
///     // Show token immediately
///     displayToken(output.tokenId)
///
///     // Log low-confidence generations for debugging
///     if output.probability < 0.1 {
///         logger.warning("Low confidence: \(output.probability)")
///     }
///
///     // Track generation speed
///     let latency = Date().timeIntervalSince(output.timestamp)
///     metrics.record(latency)
/// }
/// ```
public struct TokenOutput {
    /// Generated token ID
    ///
    /// Maps to vocabulary: `tokenizer.decode([tokenId])`
    public let tokenId: Int
    
    /// Probability/confidence of this token
    ///
    /// **Range:** 0.0 - 1.0
    /// **Interpretation:**
    /// - High (>0.5): Model is confident
    /// - Medium (0.1-0.5): Reasonable choice
    /// - Low (<0.1): Model is "guessing"
    ///
    /// **Note:** After sampling, this is the probability that was selected.
    /// For greedy, this will be the max probability.
    public let probability: Float
    
    /// Shannon entropy (nats) of the final sampling distribution
    ///
    /// Higher entropy indicates a flatter, more uncertain distribution.
    /// Lower entropy indicates a peaked, confident distribution.
    public let entropy: Float
    
    /// When this token was generated
    ///
    /// **Uses:**
    /// - Latency measurement: `Date() - output.timestamp`
    /// - UI feedback: Show "thinking..." if too slow
    /// - Performance debugging: Identify bottlenecks
    public let timestamp: Date
    
    /// Optional summary of the sampling strategy used for this token
    /// Example: "temp=0.7, topK=40, penalty=1.2"
    public let strategy: String?
    
    /// Optional energy sample for this token (Joules), if available
    public let energyJoules: Double?
    
    /// Initialize token output
    ///
    /// - Parameters:
    ///   - tokenId: Generated token ID
    ///   - probability: Confidence/probability (0.0-1.0)
    ///   - timestamp: Generation timestamp (default: now)
    public init(
        tokenId: Int,
        probability: Float,
        entropy: Float = 0.0,
        timestamp: Date = Date(),
        strategy: String? = nil,
        energyJoules: Double? = nil
    ) {
        self.tokenId = tokenId
        self.probability = probability
        self.entropy = entropy
        self.timestamp = timestamp
        self.strategy = strategy
        self.energyJoules = energyJoules
    }
}

