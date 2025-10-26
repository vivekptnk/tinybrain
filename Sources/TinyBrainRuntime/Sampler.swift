/// Sampling strategies for LLM text generation
///
/// **TB-005:** Advanced sampling to convert logits → tokens
///
/// ## Educational Overview
///
/// Sampling is how we turn raw model outputs (logits) into actual tokens:
///
/// **Problem:** Model outputs a vector of logits [0.1, 0.5, 0.2, 0.8, 0.3]
/// **Goal:** Pick one token to generate
///
/// **Strategies:**
/// - **Greedy**: Always pick highest → deterministic, boring
/// - **Temperature**: Scale randomness → control creativity
/// - **Top-K**: Limit to K best options → avoid nonsense
/// - **Top-P (Nucleus)**: Adaptive cutoff → balance quality/diversity
/// - **Repetition Penalty**: Avoid loops → more natural text
///
/// **Real-world example:**
/// ```
/// Prompt: "The cat sat on the"
/// Logits: [mat: 0.8, hat: 0.6, floor: 0.4, moon: 0.01]
///
/// Greedy → always "mat"
/// Temp=0.7 → usually "mat", sometimes "hat"
/// Top-K=2 → never "floor" or "moon"
/// Top-P=0.9 → "mat" or "hat", rarely "floor"
/// ```

import Foundation

// MARK: - Configuration

/// Configuration for sampling strategies
///
/// **Usage:**
/// ```swift
/// let config = SamplerConfig(
///     temperature: 0.8,     // Slightly creative
///     topK: 40,             // Limit to 40 best tokens
///     topP: 0.95,           // Or use nucleus sampling
///     repetitionPenalty: 1.2  // Discourage repeats
/// )
/// ```
public struct SamplerConfig {
    /// Temperature scaling (higher = more random)
    ///
    /// - 0.0: Deterministic (greedy)
    /// - 0.7: Creative but focused
    /// - 1.0: Standard softmax
    /// - 2.0: Very random/creative
    public var temperature: Float = 1.0
    
    /// Top-K: Only sample from K highest logits
    ///
    /// - nil: No top-k filtering
    /// - 1: Greedy
    /// - 40-50: Good balance
    public var topK: Int? = nil
    
    /// Top-P (nucleus sampling): Sample from smallest set with cumulative prob > P
    ///
    /// - nil: No top-p filtering
    /// - 0.9-0.95: Recommended range
    /// - 1.0: No filtering
    public var topP: Float? = nil
    
    /// Repetition penalty: Divide logits of recently used tokens
    ///
    /// - 1.0: No penalty
    /// - 1.2: Light penalty (recommended)
    /// - 2.0: Strong penalty
    public var repetitionPenalty: Float = 1.0
    
    /// Random seed for deterministic sampling
    ///
    /// - nil: Non-deterministic
    /// - 42: Reproducible results
    public var seed: UInt64? = nil
    
    public init(
        temperature: Float = 1.0,
        topK: Int? = nil,
        topP: Float? = nil,
        repetitionPenalty: Float = 1.0,
        seed: UInt64? = nil
    ) {
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
    }
}

// MARK: - Sampler

/// LLM sampling strategies
///
/// **TB-005:** Converts logits (model outputs) to token IDs
public struct Sampler {
    
    // MARK: - Greedy Sampling
    
    /// Greedy sampling: Always pick highest logit
    ///
    /// **Educational:**
    /// - Deterministic (same input → same output)
    /// - Good for reproducibility, testing
    /// - Bad for creativity, can get stuck in loops
    ///
    /// **Algorithm:** Return argmax(logits)
    ///
    /// - Parameter logits: Model output logits [vocab_size]
    /// - Returns: Token ID with highest logit
    public static func greedy(logits: Tensor<Float>) -> Int {
        var maxIndex = 0
        var maxValue = -Float.infinity
        
        for (index, value) in logits.data.enumerated() {
            if value > maxValue {
                maxValue = value
                maxIndex = index
            }
        }
        
        return maxIndex
    }
    
    // MARK: - Temperature Sampling
    
    /// Temperature sampling: Scale logits then sample from softmax
    ///
    /// **Educational:**
    ///
    /// Temperature controls randomness:
    /// - Low temp (0.1): Sharp distribution → nearly greedy
    /// - Medium temp (0.7-1.0): Balanced
    /// - High temp (2.0): Flat distribution → very random
    ///
    /// **Math:**
    /// ```
    /// scaled_logits = logits / temperature
    /// probs = softmax(scaled_logits)
    /// sample from probs
    /// ```
    ///
    /// - Parameters:
    ///   - logits: Model output logits
    ///   - temp: Temperature (higher = more random)
    ///   - seed: Optional seed for deterministic sampling
    /// - Returns: Sampled token ID
    public static func temperature(logits: Tensor<Float>, temp: Float, seed: UInt64? = nil) -> Int {
        // Handle edge case: temp ≈ 0 → greedy
        if temp < 0.01 {
            return greedy(logits: logits)
        }
        
        // Scale by temperature (element-wise division)
        let scaledData = logits.data.map { $0 / temp }
        let scaledLogits = Tensor<Float>(shape: logits.shape, data: scaledData)
        
        // Convert to probabilities via softmax
        let probs = scaledLogits.softmax()
        
        // Sample from distribution
        return sampleFromDistribution(probs.data, seed: seed)
    }
    
    // MARK: - Top-K Sampling
    
    /// Top-K sampling: Zero out all but K highest logits, then sample
    ///
    /// **Educational:**
    ///
    /// Prevents sampling very unlikely tokens:
    /// ```
    /// Original: [mat: 0.8, hat: 0.6, moon: 0.01, ...]
    /// Top-K=2:  [mat: 0.8, hat: 0.6, moon: -inf, ...]
    /// → Can only sample "mat" or "hat"
    /// ```
    ///
    /// - Parameters:
    ///   - logits: Model output logits
    ///   - k: Number of top logits to keep
    ///   - temp: Temperature for sampling
    ///   - seed: Optional seed for deterministic sampling
    /// - Returns: Sampled token ID
    public static func topK(logits: Tensor<Float>, k: Int, temp: Float, seed: UInt64? = nil) -> Int {
        let vocabSize = logits.data.count
        
        // If k >= vocab_size, just use temperature sampling
        if k >= vocabSize {
            return temperature(logits: logits, temp: temp, seed: seed)
        }
        
        // Find k-th largest value
        let sorted = logits.data.enumerated().sorted { $0.element > $1.element }
        let threshold = k < sorted.count ? sorted[k - 1].element : -Float.infinity
        
        // Zero out logits below threshold
        var filteredData = logits.data
        for i in 0..<filteredData.count {
            if filteredData[i] < threshold {
                filteredData[i] = -Float.infinity
            }
        }
        
        let filteredLogits = Tensor<Float>(shape: logits.shape, data: filteredData)
        return temperature(logits: filteredLogits, temp: temp, seed: seed)
    }
    
    // MARK: - Top-P (Nucleus) Sampling
    
    /// Top-P (nucleus) sampling: Sample from smallest set with cumulative prob > P
    ///
    /// **Educational:**
    ///
    /// Adaptive filtering based on probability mass:
    /// ```
    /// Probs (sorted): [0.5, 0.3, 0.15, 0.04, 0.01]
    /// Cumulative:     [0.5, 0.8, 0.95, 0.99, 1.0]
    ///
    /// P=0.9 → Keep first 3 tokens (cumulative = 0.95 > 0.9)
    /// ```
    ///
    /// **Advantage over Top-K:**
    /// - Adapts to confidence: sometimes needs 2 tokens, sometimes 50
    /// - Top-K is fixed: always exactly K tokens
    ///
    /// - Parameters:
    ///   - logits: Model output logits
    ///   - p: Cumulative probability threshold (0.0-1.0)
    ///   - temp: Temperature for sampling
    ///   - seed: Optional seed for deterministic sampling
    /// - Returns: Sampled token ID
    public static func topP(logits: Tensor<Float>, p: Float, temp: Float, seed: UInt64? = nil) -> Int {
        // Convert to probabilities
        let probs = logits.softmax()
        
        // Sort by probability (descending)
        let sorted = probs.data.enumerated().sorted { $0.element > $1.element }
        
        // Find cumulative probability cutoff
        var cumulative: Float = 0.0
        var cutoffIndex = sorted.count
        
        for (i, (_, prob)) in sorted.enumerated() {
            cumulative += prob
            if cumulative >= p {
                cutoffIndex = i + 1
                break
            }
        }
        
        // Zero out tokens outside nucleus
        var filteredData = logits.data
        let nucleusIndices = Set(sorted.prefix(cutoffIndex).map { $0.offset })
        
        for i in 0..<filteredData.count {
            if !nucleusIndices.contains(i) {
                filteredData[i] = -Float.infinity
            }
        }
        
        let filteredLogits = Tensor<Float>(shape: logits.shape, data: filteredData)
        return temperature(logits: filteredLogits, temp: temp, seed: seed)
    }
    
    // MARK: - Combined Sampling
    
    /// Full sampling pipeline with all strategies
    ///
    /// **Pipeline:**
    /// 1. Apply repetition penalty to history
    /// 2. Apply top-k or top-p filtering (if configured)
    /// 3. Apply temperature scaling
    /// 4. Sample from resulting distribution
    ///
    /// - Parameters:
    ///   - logits: Model output logits
    ///   - config: Sampling configuration
    ///   - history: Recently generated token IDs
    /// - Returns: Sampled token ID
    public static func sample(
        logits: Tensor<Float>,
        config: SamplerConfig,
        history: [Int]
    ) -> Int {
        var adjustedData = logits.data
        
        // Step 1: Apply repetition penalty
        if config.repetitionPenalty != 1.0 && !history.isEmpty {
            for tokenId in history {
                if tokenId >= 0 && tokenId < adjustedData.count {
                    adjustedData[tokenId] /= config.repetitionPenalty
                }
            }
        }
        
        let adjustedLogits = Tensor<Float>(shape: logits.shape, data: adjustedData)
        
        // Step 2 & 3: Apply filtering + temperature (with seed)
        if let k = config.topK {
            return topK(logits: adjustedLogits, k: k, temp: config.temperature, seed: config.seed)
        } else if let p = config.topP {
            return topP(logits: adjustedLogits, p: p, temp: config.temperature, seed: config.seed)
        } else {
            return temperature(logits: adjustedLogits, temp: config.temperature, seed: config.seed)
        }
    }
    
    // MARK: - Helper Functions
    
    /// Sample from probability distribution
    ///
    /// **Educational:**
    /// Uses cumulative distribution function (CDF):
    /// ```
    /// Probs: [0.5, 0.3, 0.2]
    /// CDF:   [0.5, 0.8, 1.0]
    /// Random: 0.65
    /// → Sample index 1 (0.5 < 0.65 <= 0.8)
    /// ```
    ///
    /// - Parameter probs: Probability distribution (should sum to ~1.0)
    /// - Returns: Sampled index
    private static func sampleFromDistribution(_ probs: [Float], seed: UInt64? = nil) -> Int {
        let threshold: Float
        if let seed = seed {
            // Use seeded random for deterministic sampling
            var generator = SeededRandomGenerator(seed: seed)
            threshold = Float(generator.next()) / Float(UInt64.max)
        } else {
            threshold = Float.random(in: 0..<1)
        }
        
        var cumulative: Float = 0.0
        
        for (index, prob) in probs.enumerated() {
            cumulative += prob
            if threshold <= cumulative {
                return index
            }
        }
        
        // Fallback (shouldn't happen if probs sum to 1)
        return probs.count - 1
    }
}

// MARK: - Seeded Random Generator

/// Simple seeded random number generator for deterministic sampling
///
/// **Educational:** Linear Congruential Generator (LCG)
/// Not cryptographically secure, but good enough for reproducible sampling.
private struct SeededRandomGenerator {
    private var state: UInt64
    
    init(seed: UInt64) {
        self.state = seed
    }
    
    mutating func next() -> UInt64 {
        // LCG parameters (from Numerical Recipes)
        state = state &* 6364136223846793005 &+ 1442695040888963407
        return state
    }
}

