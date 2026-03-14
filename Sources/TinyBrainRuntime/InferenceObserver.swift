/// Inference Observer Protocol
///
/// **TB-010: X-Ray Mode**
///
/// Allows external code to observe transformer internals during inference
/// without modifying the core inference logic. When no observer is attached,
/// the cost is a single nil-check per hook point (effectively zero).
///
/// ## Hook Points
///
/// 1. **Attention weights** — After softmax, shows which past tokens matter
/// 2. **Layer entry** — Hidden state magnitude entering each layer
/// 3. **Logits** — Full output distribution before sampling
///
/// ## Usage
///
/// ```swift
/// class MyObserver: InferenceObserver {
///     func didComputeAttention(layerIndex: Int, weights: [Float], position: Int) {
///         // weights[i] = how much attention position `position` pays to position `i`
///     }
///     func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int) {
///         // hiddenStateNorm = L2 norm of hidden state entering this layer
///     }
///     func didComputeLogits(logits: [Float], position: Int) {
///         // logits[i] = raw score for token i in vocabulary
///     }
/// }
///
/// let runner = ModelRunner(weights: weights)
/// runner.observer = MyObserver()
/// ```

import Foundation

/// Protocol for observing transformer internals during inference.
///
/// Class-only (`AnyObject`) so ModelRunner can hold a `weak` reference,
/// preventing retain cycles with view models.
public protocol InferenceObserver: AnyObject {

    /// Called after attention weights are computed for a layer.
    ///
    /// - Parameters:
    ///   - layerIndex: Which transformer layer (0-based)
    ///   - weights: Attention weight vector `[sequenceLength]` — sums to 1.0
    ///   - position: Current token position in the sequence
    func didComputeAttention(layerIndex: Int, weights: [Float], position: Int)

    /// Called at entry to each transformer layer.
    ///
    /// - Parameters:
    ///   - layerIndex: Which transformer layer (0-based)
    ///   - hiddenStateNorm: L2 norm of the hidden state vector
    ///   - position: Current token position in the sequence
    func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int)

    /// Called after logits are computed (once per `step()` call).
    ///
    /// This fires after all layers have been processed, so it can be used
    /// as a "step complete" signal to assemble per-step observations.
    ///
    /// - Parameters:
    ///   - logits: Raw logit scores `[vocabSize]` before sampling
    ///   - position: Current token position in the sequence
    func didComputeLogits(logits: [Float], position: Int)
}
