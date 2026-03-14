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
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Ensure app activates and window becomes key
        NSApp.activate(ignoringOtherApps: true)
        
        // Make first window key and order front
        DispatchQueue.main.async {
            if let window = NSApp.windows.first {
                window.makeKey()
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
    
    // Initialize Metal backend if available
    init() {
        if MetalBackend.isAvailable {
            do {
                TinyBrainBackend.metalBackend = try MetalBackend()
                print("🚀 Metal GPU backend initialized")
            } catch {
                print("⚠️ Metal initialization failed: \(error)")
            }
        }
    }
    
    var body: some Scene {
        WindowGroup {
            // TB-008 & TB-009: Load model and tokenizer (format-agnostic)
            let weights = ModelLoader.loadWithFallback(
                from: "Models/tinyllama-1.1b-int8.tbf"
            )
            let runner = ModelRunner(weights: weights)
            
            // Load tokenizer (auto-detects format)
            let tokenizer = TokenizerLoader.loadBestAvailable()
            
            let viewModel = ChatViewModel(runner: runner, tokenizer: tokenizer)
            
            ChatView(viewModel: viewModel)
                .frame(minWidth: 600, minHeight: 400)
                .onAppear {
                    print("✅ Model loaded. Config:")
                    print("  hiddenDim: \(weights.config.hiddenDim)")
                    print("  numHeads: \(weights.config.numHeads)")
                    print("  numKVHeads: \(weights.config.numKVHeads)")
                    print("  Calculated kvDim: \(weights.config.kvDim)")
                }
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
