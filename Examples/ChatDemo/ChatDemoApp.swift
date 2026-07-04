/// TinyBrain Chat Demo App
///
/// **TB-006:** Production-ready chat interface
/// Full-featured SwiftUI app with streaming, telemetry, and polish

import SwiftUI
import TinyBrainDemo
import TinyBrainRuntime
import TinyBrainMetal
import TinyBrainTokenizer

#if os(macOS)
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationWillFinishLaunching(_ notification: Notification) {
        // Force the process to be a regular app with full responder chain.
        // Without this, SPM executables on macOS Tahoe don't get keyboard input.
        NSApp.setActivationPolicy(.regular)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.activate(ignoringOtherApps: true)
        DispatchQueue.main.async {
            if let window = NSApp.windows.first {
                window.makeKeyAndOrderFront(nil)
                window.orderFrontRegardless()
            }
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}
#endif

/// Screenshot automation launch arguments:
/// `--ui-demo-transcript` seeds a four-message live transcript and telemetry.
/// `--ui-xray` opens the X-Ray panel at launch.
/// `--ui-picker` opens the model picker popover at launch.
/// `--ui-error` surfaces a sample dismissible error banner.
@main
struct ChatDemoApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif

    @StateObject private var viewModel: ChatViewModel
    private let initialShowXRay: Bool
    private let initialShowPicker: Bool

    init() {
        let launchArguments = Set(CommandLine.arguments.dropFirst())
        initialShowXRay = launchArguments.contains("--ui-xray")
        initialShowPicker = launchArguments.contains("--ui-picker")

        // Initialize Metal backend
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                print("🚀 Metal GPU backend initialized")
            } catch {
                print("⚠️ Metal initialization failed: \(error)")
            }
        }

        // Load model (falls back to toy model if no real model found)
        let loadedModel = Self.loadInitialModel()
        let weights = loadedModel.weights
        let runner = ModelRunner(weights: weights)

        // Tokenizer is paired with the selected model during initial load.
        let tokenizer = loadedModel.tokenizer

        // Create view model
        let activeModelName = loadedModel.info?.displayName ?? "Toy Model"
        let activeQuant: QuantBadge = loadedModel.info.map {
            QuantBadge.derived(from: weights, fallback: QuantBadge(hint: $0.quantization))
        } ?? .toy

        let vm = ChatViewModel(
            runner: runner,
            activeWeights: loadedModel.info == nil ? nil : weights,
            tokenizer: tokenizer,
            activeModelName: activeModelName,
            activeQuant: activeQuant,
            activeModelPath: loadedModel.info?.path
        )

        if launchArguments.contains("--ui-demo-transcript") {
            vm.seedDemoTranscriptForScreenshots()
        }
        if launchArguments.contains("--ui-error") {
            vm.seedDemoErrorForScreenshots()
        }

        _viewModel = StateObject(wrappedValue: vm)

        print("✅ App initialized. Config:")
        print("   Layers: \(weights.config.numLayers)")
        print("   Hidden dim: \(weights.config.hiddenDim)")
        print("   Vocab size: \(weights.config.vocabSize)")
    }

    var body: some Scene {
        WindowGroup {
            ChatView(viewModel: viewModel, initialShowXRay: initialShowXRay, initialShowPicker: initialShowPicker)
        }
        #if os(macOS)
        .defaultSize(width: 900, height: 600)
        .commands {
            CommandGroup(replacing: .textEditing) {
                Button("Cut") {
                    NSApp.sendAction(#selector(NSText.cut(_:)), to: nil, from: nil)
                }
                .keyboardShortcut("x", modifiers: .command)

                Button("Copy") {
                    NSApp.sendAction(#selector(NSText.copy(_:)), to: nil, from: nil)
                }
                .keyboardShortcut("c", modifiers: .command)

                Button("Paste") {
                    NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: nil)
                }
                .keyboardShortcut("v", modifiers: .command)
            }
        }
        #endif
    }

    private static func loadInitialModel() -> (weights: ModelWeights, tokenizer: any Tokenizer, info: ModelInfo?) {
        let preferredPaths = [
            "Models/qwen2.5-1.5b-int8.tbf",
            "Models/tinyllama-1.1b-int8.tbf"
        ]

        for requestedPath in preferredPaths {
            let resolvedPath = resolveProjectPath(requestedPath)
            guard FileManager.default.fileExists(atPath: resolvedPath) else { continue }

            do {
                let weights = try ModelLoader.load(from: resolvedPath)
                let tokenizer = try TokenizerLoader.loadTokenizer(forModelAt: resolvedPath)
                return (weights, tokenizer, ModelInfo(path: resolvedPath))
            } catch {
                print("⚠️ Failed to load \(resolvedPath): \(error). Trying next model.")
            }
        }

        let fallbackWeights = makeToyWeights()
        return (fallbackWeights, TokenizerLoader.loadBestAvailable(), nil)
    }

    private static func makeToyWeights() -> ModelWeights {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        return ModelWeights.makeToyModel(config: config, seed: 42)
    }

    private static func resolveProjectPath(_ path: String) -> String {
        if path.hasPrefix("/") {
            return path
        }

        if FileManager.default.fileExists(atPath: path) {
            return FileManager.default.currentDirectoryPath + "/" + path
        }

        var current = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let packagePath = (current as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath) {
                return (current as NSString).appendingPathComponent(path)
            }
            let parent = (current as NSString).deletingLastPathComponent
            if parent == current || parent == "/" {
                break
            }
            current = parent
        }

        return path
    }
}
