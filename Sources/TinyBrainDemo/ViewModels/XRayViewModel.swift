/// X-Ray View Model — Bridges inference observations to SwiftUI
///
/// **TB-010: X-Ray Mode**
///
/// Conforms to `InferenceObserver` to receive transformer internals,
/// accumulates per-layer data during each step, and publishes
/// `XRaySnapshot` objects to the UI.
///
/// ## Threading Model
///
/// Observer callbacks fire on the inference Task's thread (background).
/// Data is accumulated in thread-safe buffers, then dispatched to
/// `@MainActor` when a step completes (on `didComputeLogits`).

import Foundation
import Combine
import TinyBrainRuntime

/// View model for X-Ray visualizations
@MainActor
public final class XRayViewModel: ObservableObject {

    // MARK: - Published State

    /// Whether X-Ray observation is enabled
    @Published public var isEnabled: Bool = false

    /// Most recent snapshot (updates per token)
    @Published public var latestSnapshot: XRaySnapshot?

    /// Rolling history of recent snapshots
    @Published public var snapshotHistory: [XRaySnapshot] = []

    /// KV cache page allocation status (true = allocated)
    @Published public var kvCachePages: [Bool] = []

    /// Selected layer index for attention heatmap display
    @Published public var selectedLayer: Int = 0

    // MARK: - Configuration

    /// Number of transformer layers in the model
    public let numLayers: Int

    /// Number of top candidates to track
    public let topK: Int

    /// Maximum snapshots to keep in history
    public let maxHistory: Int

    // MARK: - Thread-Safe Accumulation

    /// Accumulator collects per-layer observations during a single step.
    /// Access is safe because `step()` calls observer methods sequentially.
    private let accumulator: StepAccumulator

    // MARK: - Initialization

    public init(numLayers: Int, topK: Int = 10, maxHistory: Int = 50) {
        self.numLayers = numLayers
        self.topK = topK
        self.maxHistory = maxHistory
        self.accumulator = StepAccumulator(numLayers: numLayers)
    }

    /// Reset all X-Ray state (e.g., on new conversation)
    public func reset() {
        latestSnapshot = nil
        snapshotHistory.removeAll()
        kvCachePages.removeAll()
        accumulator.reset()
    }
}

// MARK: - InferenceObserver Conformance

extension XRayViewModel: InferenceObserver {

    /// Called from background thread — accumulates layer entry norms
    nonisolated public func didEnterLayer(layerIndex: Int, hiddenStateNorm: Float, position: Int) {
        accumulator.recordLayerNorm(layerIndex: layerIndex, norm: hiddenStateNorm)
    }

    /// Called from background thread — accumulates attention weights
    nonisolated public func didComputeAttention(layerIndex: Int, weights: [Float], position: Int) {
        accumulator.recordAttention(layerIndex: layerIndex, weights: weights)
    }

    /// Called from background thread — fires last, assembles and publishes snapshot
    nonisolated public func didComputeLogits(logits: [Float], position: Int) {
        let topK = self.topK
        let snapshot = accumulator.assembleSnapshot(
            position: position,
            logits: logits,
            topK: topK
        )
        accumulator.reset()

        Task { @MainActor [weak self] in
            guard let self else { return }
            self.latestSnapshot = snapshot
            self.snapshotHistory.append(snapshot)
            if self.snapshotHistory.count > self.maxHistory {
                self.snapshotHistory.removeFirst()
            }
        }
    }
}

// MARK: - Step Accumulator (Thread-Safe Collection)

/// Collects per-layer observations during a single `step()` call.
///
/// Thread safety: `step()` calls observer methods sequentially (not concurrently),
/// so no locking is needed for the accumulation phase. The `assembleSnapshot()`
/// call also happens on the same thread before `reset()`.
private final class StepAccumulator: @unchecked Sendable {
    private let numLayers: Int
    private var attentionWeights: [[Float]]
    private var layerNorms: [Float]

    init(numLayers: Int) {
        self.numLayers = numLayers
        self.attentionWeights = Array(repeating: [], count: numLayers)
        self.layerNorms = Array(repeating: 0.0, count: numLayers)
    }

    func recordLayerNorm(layerIndex: Int, norm: Float) {
        guard layerIndex < numLayers else { return }
        layerNorms[layerIndex] = norm
    }

    func recordAttention(layerIndex: Int, weights: [Float]) {
        guard layerIndex < numLayers else { return }
        attentionWeights[layerIndex] = weights
    }

    func assembleSnapshot(position: Int, logits: [Float], topK: Int) -> XRaySnapshot {
        // Compute softmax probabilities from logits
        let maxLogit = logits.max() ?? 0
        let expLogits = logits.map { exp($0 - maxLogit) }
        let sumExp = expLogits.reduce(0, +)
        let probs = expLogits.map { $0 / sumExp }

        // Find top-K candidates
        let indexed = probs.enumerated().sorted { $0.element > $1.element }
        let topCandidates = indexed.prefix(topK).map { (offset, prob) in
            TokenCandidate(tokenId: offset, probability: prob)
        }

        // Compute entropy: -sum(p * log(p))
        let entropy = -probs.reduce(Float(0)) { sum, p in
            p > 0 ? sum + p * log(p) : sum
        }

        return XRaySnapshot(
            position: position,
            timestamp: Date(),
            attentionWeights: attentionWeights,
            layerNorms: layerNorms,
            topCandidates: Array(topCandidates),
            entropy: entropy
        )
    }

    func reset() {
        attentionWeights = Array(repeating: [], count: numLayers)
        layerNorms = Array(repeating: 0.0, count: numLayers)
    }
}
