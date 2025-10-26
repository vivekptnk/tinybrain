/// Quality metrics for evaluating language model outputs
///
/// **TB-004 Work Item #8:** BLEU and Perplexity metrics
///
/// ## What are these metrics?
///
/// **Perplexity:**
/// - Measures how "surprised" the model is by the actual next token
/// - Lower is better (1.0 = perfect prediction, higher = more uncertain)
/// - Formula: `exp(-mean(log(P(target_token))))`
///
/// **BLEU (Bilingual Evaluation Understudy):**
/// - Measures similarity between candidate and reference sequences
/// - Range: 0.0 to 1.0 (1.0 = perfect match)
/// - Based on n-gram precision (1-gram, 2-gram, 3-gram, 4-gram)

import Foundation

// MARK: - Perplexity

/// Calculate perplexity from logits and target tokens
///
/// **GREEN Phase:** Minimal implementation to pass tests
///
/// WHAT: Computes perplexity = exp(-mean(log(P(target))))
/// WHY: Validates model quality (lower perplexity = better predictions)
/// HOW: Convert logits → softmax → log prob → average → exp
///
/// - Parameters:
///   - logits: Array of logit tensors (one per position)
///   - targetTokens: Target token IDs (ground truth)
/// - Returns: Perplexity score (lower is better)
/// - Throws: If shapes don't match or invalid indices
public func perplexity(logits: [Tensor<Float>], targetTokens: [Int]) throws -> Float {
    precondition(logits.count == targetTokens.count,
                 "Logits count (\(logits.count)) must match target count (\(targetTokens.count))")
    
    guard !logits.isEmpty else {
        throw MetricsError.emptyInput
    }
    
    var logProbSum: Float = 0.0
    
    for (logitTensor, targetId) in zip(logits, targetTokens) {
        // Get vocab size
        let vocabSize = logitTensor.shape.count
        
        guard targetId >= 0 && targetId < vocabSize else {
            throw MetricsError.invalidTargetToken(targetId, vocabSize: vocabSize)
        }
        
        // Compute softmax (converts logits to probabilities)
        let probs = logitTensor.softmax()
        
        // Get probability of target token
        let targetProb = probs.data[targetId]
        
        // Add log probability (with small epsilon to avoid log(0))
        let epsilon: Float = 1e-10
        logProbSum += log(max(targetProb, epsilon))
    }
    
    // Average log probability
    let avgLogProb = logProbSum / Float(logits.count)
    
    // Perplexity = exp(-avgLogProb)
    return exp(-avgLogProb)
}

// MARK: - BLEU Score

/// Calculate BLEU score between candidate and reference sequences
///
/// **GREEN Phase:** Implementation with n-gram precision
///
/// WHAT: Computes BLEU = brevity_penalty × geometric_mean(n-gram_precisions)
/// WHY: Measures sequence similarity (1.0 = perfect match, 0.0 = no overlap)
/// HOW: Count n-gram matches, apply brevity penalty for short candidates
///
/// - Parameters:
///   - candidate: Predicted token sequence
///   - reference: Ground truth token sequence
///   - maxN: Maximum n-gram size (default: 4 for BLEU-4)
/// - Returns: BLEU score in range [0, 1]
public func bleuScore(candidate: [Int], reference: [Int], maxN: Int = 4) -> Float {
    guard !candidate.isEmpty && !reference.isEmpty else {
        return 0.0
    }
    
    // Brevity penalty: penalize candidates shorter than reference
    let candidateLength = Float(candidate.count)
    let referenceLength = Float(reference.count)
    let brevityPenalty: Float
    
    if candidateLength >= referenceLength {
        brevityPenalty = 1.0
    } else {
        brevityPenalty = exp(1.0 - referenceLength / candidateLength)
    }
    
    // Calculate n-gram precisions
    var precisions: [Float] = []
    
    for n in 1...min(maxN, candidate.count, reference.count) {
        let (matches, total) = nGramPrecision(candidate: candidate, reference: reference, n: n)
        
        if total > 0 {
            precisions.append(Float(matches) / Float(total))
        } else {
            precisions.append(0.0)
        }
    }
    
    // Geometric mean of precisions
    guard !precisions.isEmpty else {
        return 0.0
    }
    
    // Avoid log(0) by using max with epsilon
    let epsilon: Float = 1e-10
    let logPrecisionSum = precisions.reduce(0.0) { sum, p in
        sum + log(max(p, epsilon))
    }
    let geometricMean = exp(logPrecisionSum / Float(precisions.count))
    
    // BLEU = brevity_penalty × geometric_mean
    return brevityPenalty * geometricMean
}

/// Calculate n-gram precision
///
/// - Parameters:
///   - candidate: Predicted sequence
///   - reference: Reference sequence
///   - n: N-gram size
/// - Returns: (matches, total) where matches = count of n-grams in both,
///            total = count of candidate n-grams
private func nGramPrecision(candidate: [Int], reference: [Int], n: Int) -> (matches: Int, total: Int) {
    guard n <= candidate.count && n <= reference.count else {
        return (0, 0)
    }
    
    // Extract n-grams
    let candidateNGrams = extractNGrams(from: candidate, n: n)
    let referenceNGrams = extractNGrams(from: reference, n: n)
    
    // Count matches (with clipping - each reference n-gram can only match once)
    var referenceCounts = ngramCounts(referenceNGrams)
    var matches = 0
    
    for ngram in candidateNGrams {
        if let count = referenceCounts[ngram], count > 0 {
            matches += 1
            referenceCounts[ngram] = count - 1
        }
    }
    
    return (matches, candidateNGrams.count)
}

/// Extract n-grams from a sequence
private func extractNGrams(from sequence: [Int], n: Int) -> [[Int]] {
    guard n <= sequence.count else {
        return []
    }
    
    var ngrams: [[Int]] = []
    for i in 0...(sequence.count - n) {
        ngrams.append(Array(sequence[i..<(i + n)]))
    }
    return ngrams
}

/// Count occurrences of each n-gram
private func ngramCounts(_ ngrams: [[Int]]) -> [[Int]: Int] {
    var counts: [[Int]: Int] = [:]
    for ngram in ngrams {
        counts[ngram, default: 0] += 1
    }
    return counts
}

// MARK: - Errors

public enum MetricsError: Error {
    case emptyInput
    case invalidTargetToken(Int, vocabSize: Int)
    case shapeMismatch
}

