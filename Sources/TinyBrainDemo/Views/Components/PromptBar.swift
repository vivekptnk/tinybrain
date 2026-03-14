/// Prompt Bar Component
///
/// Native NSTextField wrapper that works on macOS Tahoe + SPM executables.
/// SwiftUI's TextField breaks in this configuration, so we use AppKit directly.

import SwiftUI

#if os(macOS)
import AppKit

/// Custom NSTextField that reliably accepts keyboard input
class FocusableTextField: NSTextField {
    override var acceptsFirstResponder: Bool { true }
}

/// Native text field with guaranteed focus and input capability
struct NativeTextField: NSViewRepresentable {
    @Binding var text: String
    var isDisabled: Bool
    var onSubmit: () -> Void

    func makeNSView(context: Context) -> FocusableTextField {
        let textField = FocusableTextField()
        textField.placeholderString = "Message TinyBrain..."
        textField.delegate = context.coordinator
        textField.isEditable = true
        textField.isSelectable = true
        textField.isBordered = false
        textField.drawsBackground = false
        textField.focusRingType = .none
        textField.font = .systemFont(ofSize: 14)
        textField.textColor = .labelColor
        textField.placeholderAttributedString = NSAttributedString(
            string: "Message TinyBrain...",
            attributes: [
                .foregroundColor: NSColor.tertiaryLabelColor,
                .font: NSFont.systemFont(ofSize: 14)
            ]
        )
        context.coordinator.textField = textField
        return textField
    }

    func updateNSView(_ nsView: FocusableTextField, context: Context) {
        if context.coordinator.isEditing { return }

        if !context.coordinator.isTyping && nsView.stringValue != text {
            nsView.stringValue = text
        }

        nsView.isEditable = !isDisabled
        nsView.isEnabled = !isDisabled

        // Auto-focus on first appear
        if !context.coordinator.hasFocused && !isDisabled && nsView.window != nil {
            context.coordinator.hasFocused = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                nsView.window?.makeFirstResponder(nsView)
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
        var isEditing = false
        weak var textField: FocusableTextField?

        init(text: Binding<String>, onSubmit: @escaping () -> Void) {
            _text = text
            self.onSubmit = onSubmit
        }

        func controlTextDidBeginEditing(_ notification: Notification) {
            isEditing = true
            isTyping = true
        }

        func controlTextDidChange(_ notification: Notification) {
            guard let textField = notification.object as? NSTextField else { return }
            isTyping = true
            let newValue = textField.stringValue
            DispatchQueue.main.async { [weak self] in
                self?.text = newValue
            }
        }

        func controlTextDidEndEditing(_ notification: Notification) {
            isEditing = false
            isTyping = false
        }

        func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
            if commandSelector == #selector(NSResponder.insertNewline(_:)) {
                onSubmit()
                return true
            }
            return false
        }
    }
}
#endif
