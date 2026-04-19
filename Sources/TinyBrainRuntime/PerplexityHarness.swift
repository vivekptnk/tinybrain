/// Perplexity regression harness for quantization quality tests.
///
/// CHA-108: Shared infrastructure used by both the
/// `QualityRegressionTests.testTinyLlamaINT4VsINT8Perplexity` regression test
/// and the `tinybrain-bench --perplexity` CLI mode so the two paths compute
/// *exactly* the same numbers.
///
/// The harness is intentionally thin — it does not own a dataset. A pinned,
/// pre-tokenized slice (see `Tests/TinyBrainRuntimeTests/Fixtures/wikitext2_slice.json`)
/// drives both flows, which keeps the test deterministic even in environments
/// that don't have the HF tokenizer or dataset libraries available.

import Foundation

/// Pinned, pre-tokenized perplexity slice (e.g. WikiText-2 validation).
public struct PerplexitySlice: Codable {
    /// Human-readable description of the raw source (dataset + selection).
    public let source: String
    /// Human-readable tokenizer identifier.
    public let tokenizer: String
    /// Beginning-of-sequence token id used to seed the first position.
    public let bosTokenId: Int
    /// Slice seed (version tag for the pinned tokens).
    public let seed: String
    /// Total number of tokens in the slice (including BOS).
    public let numTokens: Int
    /// Pre-tokenized ids. First entry is always the BOS token.
    public let tokens: [Int]
    /// Free-form notes about how the slice should be consumed.
    public let notes: String

    enum CodingKeys: String, CodingKey {
        case source
        case tokenizer
        case bosTokenId = "bos_token_id"
        case seed
        case numTokens = "num_tokens"
        case tokens
        case notes
    }

    /// Loads a slice from a JSON file on disk.
    public static func load(from url: URL) throws -> PerplexitySlice {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(PerplexitySlice.self, from: data)
    }
}

/// Shared harness operations: INT4 re-quantization + slice-driven perplexity.
public enum PerplexityHarness {

    /// Re-quantize every linear layer of an INT8 `ModelWeights` to INT4
    /// per-group.
    ///
    /// This does **not** touch the INT4 kernels or the saved TBF format — it
    /// just dequantizes each linear weight matrix to Float32 and calls the
    /// existing `quantize(mode: .int4, groupSize:)` converter. Embeddings,
    /// RMSNorm weights, and biases stay in their original representation.
    ///
    /// - Parameters:
    ///   - weights: Source model (typically loaded from an INT8 `.tbf`).
    ///   - groupSize: INT4 group size (default 32 — CHA-104 v0.2.0 knee,
    ///     matches the converter default).
    /// - Returns: New `ModelWeights` whose linear layers are INT4 per-group.
    public static func convertToINT4(
        _ weights: ModelWeights,
        groupSize: Int = 32,
        progress: ((String) -> Void)? = nil
    ) -> ModelWeights {
        func toINT4(_ layer: LinearLayerWeights, label: String) -> LinearLayerWeights {
            let start = Date()
            let floatWeights = layer.weights.dequantize()
            let int4 = floatWeights.quantize(mode: .int4, groupSize: groupSize)
            progress?(String(format: "   %@ %d→%d (%.2fs)",
                             label,
                             layer.weights.shape.count,
                             int4.data.count,
                             Date().timeIntervalSince(start)))
            return LinearLayerWeights(weights: int4, bias: layer.bias)
        }

        let newLayers = weights.layers.enumerated().map { index, layer -> TransformerLayerWeights in
            progress?("[layer \(index)/\(weights.layers.count - 1)] re-quantizing to INT4 …")
            let attention = AttentionProjectionWeights(
                query: toINT4(layer.attention.query, label: "L\(index).attn.q"),
                key: toINT4(layer.attention.key, label: "L\(index).attn.k"),
                value: toINT4(layer.attention.value, label: "L\(index).attn.v"),
                output: toINT4(layer.attention.output, label: "L\(index).attn.o")
            )

            let feedForward: FeedForwardWeights
            if let gate = layer.feedForward.gate {
                feedForward = FeedForwardWeights(
                    gate: toINT4(gate, label: "L\(index).ffn.gate"),
                    up: toINT4(layer.feedForward.up, label: "L\(index).ffn.up"),
                    down: toINT4(layer.feedForward.down, label: "L\(index).ffn.down")
                )
            } else {
                feedForward = FeedForwardWeights(
                    up: toINT4(layer.feedForward.up, label: "L\(index).ffn.up"),
                    down: toINT4(layer.feedForward.down, label: "L\(index).ffn.down")
                )
            }

            if let inNorm = layer.inputNormWeights, let postNorm = layer.postAttentionNormWeights {
                return TransformerLayerWeights(
                    attention: attention,
                    feedForward: feedForward,
                    inputNormWeights: inNorm,
                    postAttentionNormWeights: postNorm
                )
            }
            return TransformerLayerWeights(attention: attention, feedForward: feedForward)
        }

        progress?("[output projection] re-quantizing to INT4 …")
        let outputINT4 = toINT4(weights.output, label: "output")

        return ModelWeights(
            config: weights.config,
            tokenEmbeddings: weights.tokenEmbeddings,
            layers: newLayers,
            output: outputINT4,
            finalNormWeights: weights.finalNormWeights
        )
    }

    /// Result of a single perplexity evaluation.
    public struct Result {
        public let perplexity: Float
        public let numPredictions: Int
        public let elapsedSeconds: TimeInterval
    }

    /// Compute next-token perplexity for `weights` over `slice.tokens`.
    ///
    /// Teacher forcing: we feed `tokens[0..N-2]` one at a time through the
    /// runner, collect logits at each step, and measure the log-probability
    /// assigned to `tokens[i+1]` at step `i`. The returned perplexity is
    /// `exp(-mean(log(P(target))))` over the `N-1` predictions.
    public static func computePerplexity(
        weights: ModelWeights,
        slice: PerplexitySlice,
        progress: ((String) -> Void)? = nil
    ) throws -> Result {
        precondition(slice.tokens.count >= 2,
                     "Perplexity slice needs at least two tokens to have a prediction target.")

        let runner = ModelRunner(weights: weights)
        runner.reset()

        let vocabSize = weights.config.vocabSize
        var sumLogProb: Double = 0  // Float64 accumulation — INT4/INT8 ppl deltas are small
        var predictions = 0

        let start = Date()
        let progressEvery = max(32, slice.tokens.count / 8)
        for index in 0..<(slice.tokens.count - 1) {
            if let progress, index > 0, index % progressEvery == 0 {
                let elapsed = Date().timeIntervalSince(start)
                let rate = Double(index) / max(elapsed, 1e-3)
                progress(String(format: "   …%d/%d tokens (%.1f tok/s)",
                                index, slice.tokens.count - 1, rate))
            }
            let inputToken = slice.tokens[index]
            let target = slice.tokens[index + 1]
            precondition(inputToken >= 0 && inputToken < vocabSize,
                         "Input token \(inputToken) out of vocab range [0, \(vocabSize)).")
            precondition(target >= 0 && target < vocabSize,
                         "Target token \(target) out of vocab range [0, \(vocabSize)).")

            let logits = runner.step(tokenId: inputToken).data

            // Stable log-softmax at Float64: log P(target) = logits[t] − logsumexp(logits).
            // Subtract max(logits) first so exp stays bounded; accumulate the sum in Double
            // because Float32 sums lose precision over 32 000-vocab exponentials.
            let maxLogitFloat = logits.max() ?? 0
            let maxLogit = Double(maxLogitFloat)
            var sumExp: Double = 0
            for value in logits {
                sumExp += Foundation.exp(Double(value) - maxLogit)
            }
            let logSumExp = Foundation.log(sumExp) + maxLogit
            let logProbTarget = Double(logits[target]) - logSumExp
            sumLogProb += logProbTarget
            predictions += 1
        }
        let elapsed = Date().timeIntervalSince(start)

        let meanLogProb = sumLogProb / Double(predictions)
        let ppl = Float(Foundation.exp(-meanLogProb))
        return Result(perplexity: ppl, numPredictions: predictions, elapsedSeconds: elapsed)
    }
}
