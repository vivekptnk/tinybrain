/// Model runner for streaming token generation
///
/// **TB-004 Phase 5:** Efficient streaming inference with KV cache reuse
///
/// ## How It Works
///
/// Traditional (slow):
/// ```
/// Token 0: Forward pass → logits
/// Token 1: Forward pass (recompute token 0!) → logits  ← Wasteful!
/// Token 2: Forward pass (recompute 0,1!) → logits      ← Very wasteful!
/// ```
///
/// ModelRunner (fast):
/// ```
/// Token 0: Forward pass → cache K₀,V₀ → logits
/// Token 1: Reuse K₀,V₀, compute K₁,V₁ → logits  ← Fast!
/// Token 2: Reuse K₀,K₁,V₀,V₁, compute K₂,V₂ → logits  ← Fast!
/// ```
///
/// **Result:** O(n) instead of O(n²) complexity!

import Foundation

/// Configuration for model inference
public struct ModelConfig {
    /// Number of transformer layers
    public let numLayers: Int
    
    /// Hidden dimension size
    public let hiddenDim: Int
    
    /// Number of attention heads
    public let numHeads: Int
    
    /// Vocabulary size
    public let vocabSize: Int
    
    /// Maximum sequence length
    public let maxSeqLen: Int
    
    public init(numLayers: Int, hiddenDim: Int, numHeads: Int, vocabSize: Int, maxSeqLen: Int = 2048) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.numHeads = numHeads
        self.vocabSize = vocabSize
        self.maxSeqLen = maxSeqLen
    }
}

/// Model runner for streaming inference
///
/// **TB-004:** Manages KV cache and incremental token generation
public final class ModelRunner {
    /// Model configuration
    public let config: ModelConfig
    
    /// KV cache for attention
    public let kvCache: KVCache
    
    /// Current position in sequence
    public private(set) var currentPosition: Int = 0
    
    /// Initialize model runner
    public init(config: ModelConfig) {
        self.config = config
        self.kvCache = KVCache(
            numLayers: config.numLayers,
            hiddenDim: config.hiddenDim,
            maxTokens: config.maxSeqLen,
            pageSize: 16
        )
    }
    
    /// Generate next token logits using cached context
    ///
    /// **TB-004:** Core streaming API - reuses KV cache for efficiency
    ///
    /// Example:
    /// ```swift
    /// let runner = ModelRunner(config: config)
    /// let logits = runner.step(tokenId: 42)
    /// let nextToken = sample(logits)  // Sample from distribution
    /// ```
    ///
    /// - Parameter tokenId: Input token ID
    /// - Returns: Logits for next token [vocabSize]
    public func step(tokenId: Int) -> Tensor<Float> {
        // **TB-004 MVP:** Simplified forward pass for testing
        // Real implementation will load weights and run actual transformer
        
        // For now: Mock implementation that demonstrates KV cache usage
        
        // 1. Embed token (mock)
        let embedding = Tensor<Float>.random(shape: TensorShape(config.hiddenDim))
        
        // 2. For each layer: attention + MLP (mock)
        var hidden = embedding
        
        for layer in 0..<config.numLayers {
            // Compute K, V for this token
            let key = Tensor<Float>.random(shape: TensorShape(config.hiddenDim))
            let value = Tensor<Float>.random(shape: TensorShape(config.hiddenDim))
            
            // Cache them!
            kvCache.append(layer: layer, key: key, value: value, position: currentPosition)
            
            // Attention (mock - would use cached K/V here)
            hidden = Tensor<Float>.random(shape: TensorShape(config.hiddenDim))
        }
        
        // 3. Output projection (mock)
        let logits = Tensor<Float>.random(shape: TensorShape(config.vocabSize))
        
        // Increment position
        currentPosition += 1
        
        return logits
    }
    
    /// Reset state for new sequence
    ///
    /// Clears KV cache and resets position to 0
    public func reset() {
        kvCache.clear()
        currentPosition = 0
    }
    
    /// Generate stream of tokens using AsyncSequence
    ///
    /// **TB-004:** SwiftUI-friendly streaming API
    ///
    /// Example:
    /// ```swift
    /// for try await tokenId in runner.generateStream(prompt: [1, 2, 3]) {
    ///     print(tokenId, terminator: " ")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - prompt: Initial token IDs
    ///   - maxTokens: Maximum tokens to generate
    /// - Returns: AsyncThrowingStream of token IDs
    public func generateStream(prompt: [Int], maxTokens: Int = 100) -> AsyncThrowingStream<Int, Error> {
        AsyncThrowingStream { continuation in
            Task {
                // Process prompt tokens
                for tokenId in prompt {
                    let _ = self.step(tokenId: tokenId)
                }
                
                // Generate new tokens
                var generated = 0
                while generated < maxTokens {
                    // Get logits
                    let logits = self.step(tokenId: 0)  // TODO: Sample from previous logits
                    
                    // Sample next token (mock - would use proper sampling)
                    let nextToken = self.sampleToken(from: logits)
                    
                    // Yield token
                    continuation.yield(nextToken)
                    generated += 1
                    
                    // Check for end-of-sequence
                    // TODO: Check for EOS token
                }
                
                continuation.finish()
            }
        }
    }
    
    /// Sample token from logits distribution
    ///
    /// **TB-004 MVP:** Simple argmax sampling
    /// Real implementation would use top-k, top-p, temperature
    private func sampleToken(from logits: Tensor<Float>) -> Int {
        // Mock: Return random token
        // TODO: Implement proper sampling (softmax + multinomial)
        return Int.random(in: 0..<config.vocabSize)
    }
}

