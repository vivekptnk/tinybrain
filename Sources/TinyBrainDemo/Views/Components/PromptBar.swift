/// Prompt Bar Component
///
/// Multi-line text input with send button for chat interface.
///
/// Features:
/// - Auto-resizing text input
/// - Send button with loading state
/// - Keyboard shortcuts (Cmd+Return to send)
/// - Character count indicator
/// - Platform-native input handling
/// - Accessibility support

import SwiftUI

#if os(macOS)
import AppKit

/// Custom NSTextField subclass that can always become first responder
class FocusableTextField: NSTextField {
    override var acceptsFirstResponder: Bool { true }
    override func becomeFirstResponder() -> Bool {
        let result = super.becomeFirstResponder()
        print("🎯 becomeFirstResponder called, result: \(result)")
        return result
    }
}

/// Native text field with guaranteed focus capability
struct NativeTextField: NSViewRepresentable {
    @Binding var text: String
    var isDisabled: Bool
    var onSubmit: () -> Void
    
    func makeNSView(context: Context) -> FocusableTextField {
        let textField = FocusableTextField()
        textField.placeholderString = "Type here..."
        textField.delegate = context.coordinator
        
        // Store reference in coordinator to prevent premature deallocation
        context.coordinator.textField = textField
        
        // Critical settings for keyboard input
        textField.isEditable = true
        textField.isSelectable = true
        textField.isBordered = true
        textField.bezelStyle = .roundedBezel
        textField.focusRingType = .default
        
        // Add click gesture to explicitly grab focus
        let clickGesture = NSClickGestureRecognizer(target: context.coordinator, action: #selector(Coordinator.handleClick(_:)))
        textField.addGestureRecognizer(clickGesture)
        
        print("✅ Created FocusableTextField - acceptsFirstResponder: \(textField.acceptsFirstResponder)")
        
        return textField
    }
    
    func updateNSView(_ nsView: FocusableTextField, context: Context) {
        // CRITICAL: Prevent ANY updates during editing to avoid interrupting the editing session
        if context.coordinator.isEditing {
            return
        }

        // Only update text from external changes (not from typing)
        if !context.coordinator.isTyping && nsView.stringValue != text {
            nsView.stringValue = text
        }
        
        // Update enabled state
        let shouldBeDisabled = isDisabled
        if nsView.isEditable == shouldBeDisabled {
            nsView.isEditable = !shouldBeDisabled
            nsView.isEnabled = !shouldBeDisabled
            print("🔄 Disabled state changed: \(shouldBeDisabled)")
        }
        
        // Only try to focus once, early
        if !context.coordinator.hasFocused && !isDisabled && nsView.window != nil {
            context.coordinator.hasFocused = true
            // Delay to ensure window is fully set up
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                let success = nsView.window?.makeFirstResponder(nsView) ?? false
                print("🎯 Initial focus attempt: \(success)")
            }
        }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator(text: $text, onSubmit: onSubmit)
    }
    
    class Coordinator: NSObject, NSTextFieldDelegate {
        @Binding var text: String
        var onSubmit: () -> Void
        var hasFocused = false
        var isTyping = false
        var isEditing = false  // Track active editing session
        weak var textField: FocusableTextField?  // Retain reference
        
        init(text: Binding<String>, onSubmit: @escaping () -> Void) {
            _text = text
            self.onSubmit = onSubmit
        }
        
        @objc func handleClick(_ gesture: NSClickGestureRecognizer) {
            if let textField = gesture.view as? FocusableTextField {
                let success = textField.window?.makeFirstResponder(textField) ?? false
                print("👆 Click: focus success = \(success)")
                if let firstResp = textField.window?.firstResponder {
                    print("   First responder is: \(firstResp.className)")
                    print("   Is text field? \(firstResp == textField)")
                    if let textView = firstResp as? NSTextView {
                        print("   Is field editor? \(textView.isFieldEditor)")
                        print("   Is editable? \(textView.isEditable)")
                    }
                }
            }
        }
        
        func controlTextDidBeginEditing(_ notification: Notification) {
            print("✏️ ===== STARTED EDITING =====")
            isEditing = true
            isTyping = true
        }
        
        func controlTextDidChange(_ notification: Notification) {
            guard let textField = notification.object as? NSTextField else { return }
            isTyping = true
            let newValue = textField.stringValue
            print("📝 Text changed to: '\(newValue)'")
            
            // Update binding without triggering view update
            DispatchQueue.main.async { [weak self] in
                self?.text = newValue
            }
        }
        
        func controlTextDidEndEditing(_ notification: Notification) {
            print("✅ ===== ENDED EDITING =====")
            let reason = notification.userInfo?["NSTextMovement"] as? Int ?? -1
            print("   Reason code: \(reason)")
            isEditing = false
            isTyping = false
        }
        
        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            print("⌨️ Command: \(commandSelector)")
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                onSubmit()
                return true
            }
            return false
        }
    }
}
#endif


/// Input bar for composing messages
public struct PromptBar: View {
    @Binding var text: String
    let isDisabled: Bool
    let onSend: () -> Void
    
    let theme = TinyBrainTheme.shared
    @FocusState private var isFocused: Bool
    
    public init(
        text: Binding<String>,
        isDisabled: Bool = false,
        onSend: @escaping () -> Void
    ) {
        self._text = text
        self.isDisabled = isDisabled
        self.onSend = onSend
    }
    
    public var body: some View {
        HStack(alignment: .bottom, spacing: theme.spacing.sm) {
            // Simple SwiftUI TextField - try this on both platforms
            // Sometimes simpler is better!
            TextField("Type your message...", text: $text, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .disabled(isDisabled)
                .focused($isFocused)
                .onSubmit {
                    // Only handle Return (not Cmd+Return which goes to button)
                }
            
            // Send button
            Button(action: handleSend) {
                Image(systemName: isDisabled ? "hourglass" : "arrow.up.circle.fill")
                    .font(.title2)
                    .foregroundColor(sendButtonColor)
            }
            .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isDisabled)
            .keyboardShortcut(.return, modifiers: .command)
            .accessibilityLabel("Send message")
        }
        .padding(theme.spacing.md)
        .background(.background)
        .onAppear {
            // Delayed focus to ensure window is ready
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                isFocused = true
                print("🎯 SwiftUI TextField focused")
            }
        }
    }
    
    // MARK: - Computed Properties
    
    private var sendButtonColor: Color {
        if isDisabled {
            return .secondary
        } else if text.isEmpty {
            return .secondary
        } else {
            return theme.colors.accent
        }
    }
    
    // MARK: - Actions
    
    private func handleSend() {
        #if os(iOS)
        HapticFeedback.medium.trigger()
        #endif
        
        onSend()
        
        // Re-focus after send
        Task {
            try? await Task.sleep(nanoseconds: 100_000_000) // 100ms
            isFocused = true
        }
    }
}

// MARK: - Preview

#if DEBUG
struct PromptBar_Previews: PreviewProvider {
    struct Preview: View {
        @State private var text = ""
        @State private var isGenerating = false
        
        var body: some View {
            VStack {
                Spacer()
                
                Text(text.isEmpty ? "Type something..." : text)
                    .font(.title2)
                    .padding()
                
                Spacer()
                
                PromptBar(
                    text: $text,
                    isDisabled: isGenerating,
                    onSend: {
                        print("Sending: \(text)")
                        text = ""
                    }
                )
            }
        }
    }
    
    static var previews: some View {
        Preview()
    }
}
#endif

