/// TinyBrain Chat Demo App
///
/// A SwiftUI demonstration app showing local LLM inference with streaming output.
/// This is a standalone iOS/macOS app that can be opened in Xcode.

import SwiftUI
import TinyBrainDemo

#if os(macOS)
import AppKit

class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Ensure the app activates properly
        NSApp.activate(ignoringOtherApps: true)
        
        // Make sure the first window becomes key
        if let window = NSApp.windows.first {
            window.makeKeyAndOrderFront(nil)
        }
    }
}

// Native NSTextField wrapper for reliable input on macOS
struct MacTextField: NSViewRepresentable {
    @Binding var text: String
    var placeholder: String
    var onCommit: () -> Void
    var isDisabled: Bool = false
    
    func makeNSView(context: Context) -> NSTextField {
        let textField = NSTextField()
        textField.placeholderString = placeholder
        textField.delegate = context.coordinator
        textField.isBordered = true
        textField.bezelStyle = .roundedBezel
        textField.focusRingType = .default
        
        // CRITICAL: Enable text input
        textField.isEditable = true
        textField.isSelectable = true
        textField.allowsEditingTextAttributes = false
        textField.importsGraphics = false
        
        // Force focus on next run loop
        DispatchQueue.main.async {
            textField.window?.makeFirstResponder(textField)
            // Force the field to become active
            textField.currentEditor()?.selectedRange = NSRange(location: 0, length: 0)
        }
        
        return textField
    }
    
    func updateNSView(_ nsView: NSTextField, context: Context) {
        nsView.stringValue = text
        nsView.isEditable = !isDisabled
        nsView.isEnabled = !isDisabled
        
        // Try to grab focus if we're not disabled and text changed from outside
        if context.coordinator.shouldAutoFocus && !isDisabled {
            DispatchQueue.main.async {
                nsView.window?.makeFirstResponder(nsView)
            }
            context.coordinator.shouldAutoFocus = false
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text, onCommit: onCommit)
    }
    
    class Coordinator: NSObject, NSTextFieldDelegate {
        @Binding var text: String
        var onCommit: () -> Void
        var shouldAutoFocus = true
        
        init(text: Binding<String>, onCommit: @escaping () -> Void) {
            _text = text
            self.onCommit = onCommit
        }
        
        func controlTextDidChange(_ obj: Notification) {
            if let textField = obj.object as? NSTextField {
                text = textField.stringValue
            }
        }
        
        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                onCommit()
                return true
            }
            return false
        }
    }
}
#endif

@main
struct ChatDemoApp: App {
    #if os(macOS)
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    #endif
    
    var body: some Scene {
        WindowGroup {
            ChatView()
        }
        #if os(macOS)
        .defaultSize(width: 600, height: 500)
        #endif
        .commands {
            // Enable standard text editing commands
            CommandGroup(replacing: .textEditing) {
                Button("Cut") { NSApp.sendAction(#selector(NSText.cut(_:)), to: nil, from: nil) }
                    .keyboardShortcut("x", modifiers: .command)
                Button("Copy") { NSApp.sendAction(#selector(NSText.copy(_:)), to: nil, from: nil) }
                    .keyboardShortcut("c", modifiers: .command)
                Button("Paste") { NSApp.sendAction(#selector(NSText.paste(_:)), to: nil, from: nil) }
                    .keyboardShortcut("v", modifiers: .command)
            }
        }
    }
}

/// Main chat interface view
struct ChatView: View {
    @StateObject private var viewModel = ChatViewModel()
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text("🧠 TinyBrain Chat")
                    .font(.title2)
                    .fontWeight(.bold)
                Spacer()
                if viewModel.isGenerating {
                    ProgressView()
                        .scaleEffect(0.8)
                }
            }
            .padding()
            .background(.background)
            
            Divider()
            
            // Response area
            ScrollView {
                Text(viewModel.responseText.isEmpty ? "Response will appear here..." : viewModel.responseText)
                    .foregroundColor(viewModel.responseText.isEmpty ? .secondary : .primary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .frame(maxHeight: .infinity)
            
            Divider()
            
            // Metrics - TB-005: Now with probability!
            if viewModel.tokensPerSecond > 0 {
                HStack {
                    Label("\(String(format: "%.1f", viewModel.tokensPerSecond)) tokens/sec",
                          systemImage: "gauge.medium")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    // TB-005: Confidence indicator
                    if viewModel.averageProbability > 0 {
                        Label("\(String(format: "%.0f", viewModel.averageProbability * 100))% confidence",
                              systemImage: "chart.bar.fill")
                            .font(.caption)
                            .foregroundColor(viewModel.averageProbability > 0.5 ? .green : .orange)
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            
            Divider()
            
            // TB-005: Sampling Controls
            if !viewModel.isGenerating {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Sampling Settings")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    HStack {
                        Text("Temperature:")
                            .font(.caption)
                        Slider(value: Binding(
                            get: { Double(viewModel.temperature) },
                            set: { viewModel.temperature = Float($0) }
                        ), in: 0.1...2.0, step: 0.1)
                        Text(String(format: "%.1f", viewModel.temperature))
                            .font(.caption)
                            .frame(width: 30)
                    }
                    
                    Toggle(isOn: $viewModel.useTopK) {
                        Text("Use Top-K (\(viewModel.topK))")
                            .font(.caption)
                    }
                    
                    if viewModel.useTopK {
                        HStack {
                            Text("K:")
                                .font(.caption)
                            Slider(value: Binding(
                                get: { Double(viewModel.topK) },
                                set: { viewModel.topK = Int($0) }
                            ), in: 1...100, step: 1)
                            Text("\(viewModel.topK)")
                                .font(.caption)
                                .frame(width: 30)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.vertical, 8)
            }
            
            // Input area
            HStack(alignment: .bottom, spacing: 12) {
                #if os(macOS)
                MacTextField(
                    text: $viewModel.promptText,
                    placeholder: "Enter your prompt...",
                    onCommit: {
                        Task {
                            await viewModel.generate()
                        }
                    },
                    isDisabled: viewModel.isGenerating
                )
                .frame(minHeight: 28)  // Ensure visible height
                .background(Color.white)  // Debug: make sure field is visible
                .cornerRadius(6)
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.blue, lineWidth: 2)  // Debug: visible border
                )
                #else
                TextField("Enter your prompt...", text: $viewModel.promptText)
                    .textFieldStyle(.roundedBorder)
                    .disabled(viewModel.isGenerating)
                    .onSubmit {
                        Task {
                            await viewModel.generate()
                        }
                    }
                #endif
                
                Button(action: {
                    Task {
                        await viewModel.generate()
                    }
                }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(viewModel.promptText.isEmpty || viewModel.isGenerating)
                .keyboardShortcut(.return, modifiers: .command)
            }
            .padding()
            .background(.background)
        }
        .frame(minWidth: 400, minHeight: 300)
    }
}

// Preview disabled for command-line builds
// #Preview {
//     ChatView()
// }

