/// Model Picker View Model
///
/// Manages scanning for available `.tbf` model files and tracks
/// which model is currently selected.
///
/// Works with `ModelPickerView` in the chat header to allow
/// switching between models at runtime.

import Foundation
import SwiftUI
import TinyBrainRuntime
import TinyBrainTokenizer

/// View model for the model picker component
@MainActor
public final class ModelPickerViewModel: ObservableObject {

    // MARK: - Published Properties

    /// All discovered model files
    @Published public private(set) var availableModels: [ModelInfo] = []

    /// Currently selected model path (nil = toy model / no file selected)
    @Published public private(set) var selectedModelPath: String?

    /// Whether a model switch is in progress
    @Published public private(set) var isSwitching: Bool = false

    /// Error message if model loading failed
    @Published public private(set) var switchError: String?

    // MARK: - Computed Properties

    /// The currently selected ModelInfo (nil if toy model active)
    public var selectedModel: ModelInfo? {
        guard let path = selectedModelPath else { return nil }
        return availableModels.first { $0.path == path }
    }

    /// Display string for the currently active model
    public var selectedDisplayName: String {
        selectedModel?.displayName ?? "Toy Model"
    }

    // MARK: - Private

    private let directoryPath: String?

    // MARK: - Init

    /// Initialize with optional directory override (defaults to project `Models/`)
    public init(directoryPath: String? = nil) {
        self.directoryPath = directoryPath
    }

    // MARK: - Public API

    /// Scan the `Models/` directory and refresh the available model list
    public func refresh() {
        availableModels = ModelScanner.scan(directoryPath: directoryPath)
    }

    /// Select a model by path.
    ///
    /// - Parameter path: Absolute path to a `.tbf` file, or nil to revert to toy model.
    public func select(path: String?) {
        selectedModelPath = path
        switchError = nil
    }

    /// Load the currently selected model as a `ModelWeights` + matching tokenizer.
    ///
    /// If no model is selected, returns toy weights and a fallback tokenizer. For
    /// real model files, tokenizer discovery is model-keyed and deliberately
    /// throws instead of falling back to a mismatched tokenizer.
    ///
    /// - Returns: Tuple of (ModelWeights, Tokenizer)
    public func loadSelected() async throws -> (weights: ModelWeights, tokenizer: any Tokenizer) {
        guard let path = selectedModelPath else {
            return (makeToyWeights(), TokenizerLoader.loadBestAvailable())
        }

        isSwitching = true
        defer { isSwitching = false }

        do {
            let weights = try await Task.detached(priority: .userInitiated) {
                try ModelLoader.load(from: path)
            }.value

            let tokenizer = try TokenizerLoader.loadTokenizer(forModelAt: path)
            try TokenizerVocabularyCompatibility.validate(
                tokenizerVocab: tokenizer.vocabularySize,
                modelVocab: weights.config.vocabSize,
                modelPath: path
            )

            return (weights, tokenizer)
        } catch {
            let message = userFacingLoadError(error, modelPath: path)
            switchError = message
            throw ModelPickerLoadError(message)
        }
    }

    // MARK: - Private Helpers

    private func makeToyWeights() -> ModelWeights {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        return ModelWeights.makeToyModel(config: config, seed: 42)
    }

    private func userFacingLoadError(_ error: Error, modelPath: String) -> String {
        if let error = error as? ModelPickerLoadError {
            return error.description
        }
        if let error = error as? TokenizerError {
            return error.description
        }

        let modelName = URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent
        return "Failed to load \(modelName): \(error.localizedDescription)"
    }
}

private struct ModelPickerLoadError: Error, CustomStringConvertible, LocalizedError {
    let description: String

    init(_ description: String) {
        self.description = description
    }

    var errorDescription: String? {
        description
    }
}

/// Validates whether a tokenizer can safely pair with a model vocabulary.
///
/// Some model files pad their embedding table above the real tokenizer entry
/// count. That is harmless when every token ID the tokenizer can emit is below
/// `tokenizerVocab <= modelVocab`: the extra model rows are unreachable padding,
/// not a decode mismatch or an out-of-bounds risk.
enum TokenizerVocabularyCompatibility: Equatable {
    case compatible
    case padded(gap: Int, allowedGap: Int)
    case tokenizerTooLarge(tokenizerVocab: Int, modelVocab: Int)
    case excessivePadding(tokenizerVocab: Int, modelVocab: Int, gap: Int, allowedGap: Int)

    var isCompatible: Bool {
        switch self {
        case .compatible, .padded:
            return true
        case .tokenizerTooLarge, .excessivePadding:
            return false
        }
    }

    static func allowedPaddingGap(for modelVocab: Int) -> Int {
        max(256, modelVocab / 1_000)
    }

    static func evaluate(tokenizerVocab: Int, modelVocab: Int) -> TokenizerVocabularyCompatibility {
        if tokenizerVocab == modelVocab {
            return .compatible
        }

        if tokenizerVocab > modelVocab {
            return .tokenizerTooLarge(tokenizerVocab: tokenizerVocab, modelVocab: modelVocab)
        }

        let gap = modelVocab - tokenizerVocab
        let allowedGap = allowedPaddingGap(for: modelVocab)
        if gap <= allowedGap {
            return .padded(gap: gap, allowedGap: allowedGap)
        }

        return .excessivePadding(
            tokenizerVocab: tokenizerVocab,
            modelVocab: modelVocab,
            gap: gap,
            allowedGap: allowedGap
        )
    }

    static func validate(tokenizerVocab: Int, modelVocab: Int, modelPath: String) throws {
        let compatibility = evaluate(tokenizerVocab: tokenizerVocab, modelVocab: modelVocab)
        guard !compatibility.isCompatible else { return }

        let modelName = URL(fileURLWithPath: modelPath).lastPathComponent
        switch compatibility {
        case .compatible, .padded:
            return
        case .tokenizerTooLarge:
            throw ModelPickerLoadError(
                "Tokenizer vocabulary mismatch for \(modelName): tokenizer has \(tokenizerVocab) tokens, model expects \(modelVocab). The tokenizer could emit token IDs the model cannot embed. Decoding with a mismatched tokenizer would produce garbage."
            )
        case .excessivePadding(_, _, let gap, let allowedGap):
            throw ModelPickerLoadError(
                "Tokenizer vocabulary mismatch for \(modelName): tokenizer has \(tokenizerVocab) tokens, model expects \(modelVocab). The \(gap)-token gap exceeds the supported padded-vocab window of \(allowedGap), which usually means the wrong tokenizer was paired with the model. Decoding with a mismatched tokenizer would produce garbage."
            )
        }
    }
}
