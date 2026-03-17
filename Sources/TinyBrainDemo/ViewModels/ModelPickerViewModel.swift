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
    /// Returns `(weights, tokenizer)`. If no model is selected, returns toy weights
    /// and a fallback tokenizer. On failure, clears the selection and surfaces an error.
    ///
    /// - Returns: Tuple of (ModelWeights, optional Tokenizer)
    public func loadSelected() async -> (weights: ModelWeights, tokenizer: (any Tokenizer)?) {
        guard let path = selectedModelPath else {
            return (makeToyWeights(), TokenizerLoader.loadBestAvailable())
        }

        isSwitching = true
        defer { isSwitching = false }

        do {
            let weights = try await Task.detached(priority: .userInitiated) {
                try ModelLoader.load(from: path)
            }.value

            // Try to load a matching tokenizer from the same directory,
            // then fall back to the global best-available search
            let dir = (path as NSString).deletingLastPathComponent
            let tokenizer = loadTokenizerFromDirectory(dir) ?? TokenizerLoader.loadBestAvailable()

            return (weights, tokenizer)
        } catch {
            switchError = "Failed to load \(selectedDisplayName): \(error.localizedDescription)"
            selectedModelPath = nil
            return (makeToyWeights(), TokenizerLoader.loadBestAvailable())
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

    /// Try to find a tokenizer file in the same directory as the model
    private func loadTokenizerFromDirectory(_ dir: String) -> (any Tokenizer)? {
        // Preference order: tokenizer.json, *.vocab, *.model, *.tiktoken
        let candidates = [
            "tokenizer.json",
            "spiece.model",
            "tokenizer.model",
        ]

        let fm = FileManager.default

        // Named candidates first
        for candidate in candidates {
            let full = (dir as NSString).appendingPathComponent(candidate)
            if fm.fileExists(atPath: full), let tok = try? TokenizerLoader.load(from: full) {
                return tok
            }
        }

        // Scan for any supported file
        let files = (try? fm.contentsOfDirectory(atPath: dir)) ?? []
        for ext in ["tiktoken", "vocab"] {
            if let match = files.first(where: { $0.hasSuffix(".\(ext)") }) {
                let full = (dir as NSString).appendingPathComponent(match)
                if let tok = try? TokenizerLoader.load(from: full) {
                    return tok
                }
            }
        }

        return nil
    }
}
