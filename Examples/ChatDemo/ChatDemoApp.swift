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

@main
struct ChatDemoApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif

    @StateObject private var viewModel: ChatViewModel

    init() {
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
        let weights = ModelLoader.loadWithFallback(
            from: "Models/tinyllama-1.1b-int8.tbf"
        )
        let runner = ModelRunner(weights: weights)

        // Load tokenizer (auto-detects format)
        let tokenizer = TokenizerLoader.loadBestAvailable()

        // Create view model
        let vm = ChatViewModel(runner: runner, tokenizer: tokenizer)
        _viewModel = StateObject(wrappedValue: vm)

        print("✅ App initialized. Config:")
        print("   Layers: \(weights.config.numLayers)")
        print("   Hidden dim: \(weights.config.hiddenDim)")
        print("   Vocab size: \(weights.config.vocabSize)")
    }

    var body: some Scene {
        WindowGroup {
            ChatView(viewModel: viewModel)
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
}
