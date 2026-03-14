/// Message Bubble Component
///
/// Displays a single chat message with role-appropriate styling.

import SwiftUI

/// A styled chat message bubble
public struct MessageBubble: View {
    let message: Message
    let theme = TinyBrainTheme.shared

    public init(message: Message) {
        self.message = message
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.isUser {
                Spacer(minLength: 100)
            } else {
                // Assistant avatar
                ZStack {
                    Circle()
                        .fill(Color.blue.opacity(0.1))
                        .frame(width: 28, height: 28)
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(.blue)
                }
                .padding(.top, 2)
                .padding(.trailing, 8)
            }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 3) {
                // Role label
                Text(message.isUser ? "You" : "TinyBrain")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)

                // Message content
                if message.content.isEmpty && message.isAssistant {
                    typingIndicator
                } else {
                    Text(message.content)
                        .font(.system(size: 14, weight: .regular))
                        .foregroundColor(.primary)
                        .textSelection(.enabled)
                        .lineSpacing(3)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(bubbleBackground)
                        .clipShape(RoundedRectangle(cornerRadius: 14))
                        .contextMenu {
                            Button(action: copyMessage) {
                                Label("Copy", systemImage: "doc.on.doc")
                            }
                        }
                }
            }
            .frame(maxWidth: 520, alignment: message.isUser ? .trailing : .leading)

            if message.isAssistant {
                Spacer(minLength: 100)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
    }

    // MARK: - Typing Indicator

    private var typingIndicator: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .fill(Color.secondary.opacity(0.3))
                    .frame(width: 7, height: 7)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(bubbleBackground)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }

    // MARK: - Bubble Style

    private var bubbleBackground: some ShapeStyle {
        message.isUser
            ? AnyShapeStyle(Color.blue.opacity(0.15))
            : AnyShapeStyle(Color.primary.opacity(0.06))
    }

    // MARK: - Actions

    private func copyMessage() {
        #if os(iOS)
        UIPasteboard.general.string = message.content
        #elseif os(macOS)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(message.content, forType: .string)
        #endif
    }
}
