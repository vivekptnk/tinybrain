/// Message Bubble Component
///
/// Displays role-specific chat bubbles with streaming and failure states.

import SwiftUI

#if os(macOS)
import AppKit
#elseif os(iOS)
import UIKit
#endif

/// A styled chat message bubble.
public struct MessageBubble: View {
    let message: Message
    let isStreaming: Bool
    let isFailed: Bool

    private let theme = TinyBrainTheme.shared
    @State private var isHovering = false

    public init(message: Message, isStreaming: Bool = false, isFailed: Bool = false) {
        self.message = message
        self.isStreaming = isStreaming
        self.isFailed = isFailed
    }

    public var body: some View {
        HStack(alignment: .top, spacing: 0) {
            if message.isUser {
                Spacer(minLength: 96)
            } else {
                assistantAvatar
                    .padding(.top, 2)
                    .padding(.trailing, 8)
            }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 4) {
                roleRow

                if message.content.isEmpty && message.isAssistant && isStreaming {
                    typingIndicator
                } else {
                    bubbleContent
                }

                if isFailed {
                    Text("· failed")
                        .font(theme.typography.caption)
                        .foregroundStyle(theme.colors.critical)
                }
            }
            .frame(maxWidth: message.isUser ? 520 : 580, alignment: message.isUser ? .trailing : .leading)

            if message.isAssistant {
                Spacer(minLength: 96)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 4)
        .onHover { hovering in
            isHovering = hovering
        }
    }

    // MARK: - Subviews

    private var assistantAvatar: some View {
        ZStack {
            Circle()
                .fill(theme.colors.accentQuiet)
                .frame(width: 28, height: 28)
            Image(systemName: "brain.head.profile")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(theme.colors.accent)
        }
    }

    private var roleRow: some View {
        HStack(spacing: 8) {
            if message.isUser {
                timestamp
                roleLabel
            } else {
                roleLabel
                timestamp
            }
        }
        .frame(maxWidth: .infinity, alignment: message.isUser ? .trailing : .leading)
    }

    private var roleLabel: some View {
        Text(message.isUser ? "You" : "TinyBrain")
            .font(theme.typography.label)
            .foregroundStyle(theme.colors.textSecondary)
    }

    @ViewBuilder
    private var timestamp: some View {
        if isHovering {
            Text(message.timestamp.formatted(date: .omitted, time: .shortened))
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)
                .transition(.opacity)
        }
    }

    private var bubbleContent: some View {
        HStack(alignment: .lastTextBaseline, spacing: 4) {
            Text(message.content)
                .font(theme.typography.body)
                .foregroundStyle(theme.colors.textPrimary)
                .lineSpacing(4)
                .fixedSize(horizontal: false, vertical: true)

            if isStreaming && message.isAssistant {
                Rectangle()
                    .fill(theme.colors.accent)
                    .frame(width: 2, height: 15)
                    .pulsing(minOpacity: 0.2, maxOpacity: 1.0, duration: 0.55)
            }
        }
        .textSelection(.enabled)
        .padding(.horizontal, 14)
        .padding(.vertical, 10)
        .background(bubbleBackground)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .contextMenu {
            Button(action: copyMessage) {
                Label("Copy", systemImage: "doc.on.doc")
            }
        }
    }

    private var typingIndicator: some View {
        HStack(spacing: 5) {
            ForEach(0..<3, id: \.self) { index in
                TypingDot(index: index)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(bubbleBackground)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
    }

    private var bubbleBackground: Color {
        message.isUser ? theme.colors.accentQuiet : theme.colors.fillQuaternary
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

private struct TypingDot: View {
    let index: Int

    private let theme = TinyBrainTheme.shared
    @State private var isActive = false

    var body: some View {
        Circle()
            .fill(theme.colors.textSecondary)
            .frame(width: 7, height: 7)
            .opacity(isActive ? 1.0 : 0.3)
            .animation(
                .easeInOut(duration: 0.6)
                    .delay(Double(index) * 0.2)
                    .repeatForever(autoreverses: true),
                value: isActive
            )
            .onAppear {
                isActive = true
            }
    }
}
