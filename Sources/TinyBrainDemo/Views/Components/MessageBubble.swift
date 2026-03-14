/// Message Bubble Component
///
/// Displays a single chat message with role-appropriate styling.
///
/// Features:
/// - User vs assistant styling
/// - Smooth fade-in animation
/// - Copy button on hover/long-press
/// - Accessibility labels
/// - Markdown rendering (basic)

import SwiftUI

/// A styled chat message bubble
public struct MessageBubble: View {
    let message: Message
    let theme = TinyBrainTheme.shared
    
    @State private var isHovering = false
    @State private var showCopyConfirmation = false
    
    public init(message: Message) {
        self.message = message
    }
    
    public var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.isUser {
                Spacer(minLength: 80)
            } else {
                // Assistant avatar
                Text("🧠")
                    .font(.system(size: 20))
                    .frame(width: 28, height: 28)
                    .padding(.top, 4)
                    .padding(.trailing, 8)
            }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                // Message content
                if message.content.isEmpty {
                    // Typing indicator for empty assistant messages
                    if message.isAssistant {
                        HStack(spacing: 4) {
                            ForEach(0..<3, id: \.self) { i in
                                Circle()
                                    .fill(Color.secondary.opacity(0.4))
                                    .frame(width: 6, height: 6)
                            }
                        }
                        .padding(theme.spacing.md)
                        .background(bubbleBackground)
                        .cornerRadius(theme.corners.medium)
                    }
                } else {
                    Text(message.content)
                        .font(theme.typography.body)
                        .foregroundColor(.primary)
                        .textSelection(.enabled)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(bubbleBackground)
                        .cornerRadius(theme.corners.medium)
                        .contextMenu {
                            Button(action: copyMessage) {
                                Label("Copy", systemImage: "doc.on.doc")
                            }
                        }
                }

                // Timestamp
                Text(formattedTimestamp)
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                    .padding(.horizontal, 4)
            }
            .frame(maxWidth: 500, alignment: message.isUser ? .trailing : .leading)

            if message.isAssistant {
                Spacer(minLength: 80)
            }
        }
        .padding(.horizontal, theme.spacing.md)
        .padding(.vertical, 3)
        .transition(.messageAppear)
        .accessibilityElement(children: .combine)
        .accessibilityLabel(accessibilityLabel)
    }
    
    // MARK: - Subviews
    
    private var bubbleBackground: some View {
        Group {
            if message.isUser {
                theme.colors.userMessageBackground
            } else if message.isAssistant {
                theme.colors.assistantMessageBackground
            } else {
                theme.colors.surface
            }
        }
    }
    
    private var formattedTimestamp: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: message.timestamp)
    }
    
    private var accessibilityLabel: String {
        let role = message.isUser ? "You" : "Assistant"
        return "\(role) said: \(message.content), at \(formattedTimestamp)"
    }
    
    // MARK: - Actions
    
    private func copyMessage() {
        #if os(iOS)
        UIPasteboard.general.string = message.content
        #elseif os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.content, forType: .string)
        #endif
        
        showCopyConfirmation = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            showCopyConfirmation = false
        }
    }
}

// MARK: - Preview

#if DEBUG
struct MessageBubble_Previews: PreviewProvider {
    static var previews: some View {
        VStack(spacing: 16) {
            MessageBubble(message: Message(
                role: .user,
                content: "Hello, TinyBrain! How are you today?"
            ))
            
            MessageBubble(message: Message(
                role: .assistant,
                content: "Hello! I'm doing well, thank you for asking. How can I help you today?"
            ))
            
            MessageBubble(message: Message(
                role: .user,
                content: "Can you explain how LLMs work?"
            ))
        }
        .padding()
    }
}
#endif

