/// Primary in-app workbench for TinyBrain Agent.
///
/// Shows the prompt composer, bundled corpus status, suggested corpus prompts,
/// final answer reveal, and the live agent trace panel.

import SwiftUI

/// Two-column Agent screen over the bundled on-device corpus.
public struct AgentWorkbenchView: View {
    @ObservedObject private var viewModel: AgentViewModel
    @ObservedObject private var trace: AgentTraceViewModel
    private let isHostBusy: Bool
    private let theme = TinyBrainTheme.shared

    /// Creates the Agent workbench.
    public init(viewModel: AgentViewModel, isHostBusy: Bool = false) {
        self.viewModel = viewModel
        self.trace = viewModel.trace
        self.isHostBusy = isHostBusy
    }

    public var body: some View {
        GeometryReader { geometry in
            if geometry.size.width < 860 {
                ScrollView {
                    VStack(spacing: theme.spacing.sixteen) {
                        leftColumn
                        AgentTracePanel(trace: trace, preferredWidth: nil)
                            .frame(minHeight: 520)
                    }
                    .padding(theme.spacing.sixteen)
                }
                .background(theme.colors.canvas)
            } else {
                HStack(spacing: 0) {
                    ScrollView {
                        leftColumn
                            .padding(theme.spacing.sixteen)
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                    .background(theme.colors.canvas)

                    AgentTracePanel(
                        trace: trace,
                        preferredWidth: min(max(geometry.size.width * 0.36, 360), 420)
                    )
                }
                .background(theme.colors.canvas)
            }
        }
        .task {
            await viewModel.prepareIfNeeded()
        }
    }

    private var leftColumn: some View {
        VStack(alignment: .leading, spacing: theme.spacing.sixteen) {
            workbenchHeader

            if let disabledReason = viewModel.disabledReason {
                AgentDisabledState(message: disabledReason)
            }

            AgentPromptComposer(
                promptText: $viewModel.promptText,
                isRunning: viewModel.isRunning,
                isDisabled: isRunDisabled,
                disabledReason: runDisabledReason,
                onRun: { viewModel.startRun() },
                onCancel: { viewModel.cancel() }
            )

            AgentCorpusStrip(status: trace.corpusStatus)

            AgentPromptChips { prompt in
                viewModel.usePrompt(prompt)
            }
            .disabled(viewModel.disabledReason != nil)

            AgentFinalAnswerPanel(
                answer: trace.finalAnswer,
                isRunning: viewModel.isRunning || trace.isRunning,
                errorMessage: viewModel.errorMessage ?? trace.errorMessage,
                terminationReason: trace.terminationReason
            )

            Spacer(minLength: 0)
        }
        .frame(maxWidth: 760, alignment: .topLeading)
    }

    private var workbenchHeader: some View {
        HStack(spacing: theme.spacing.eight) {
            Image(systemName: "point.topleft.down.curvedto.point.bottomright.up")
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(theme.colors.accent)

            VStack(alignment: .leading, spacing: theme.spacing.two) {
                Text("TinyBrain Agent")
                    .font(theme.typography.title2)
                    .foregroundStyle(theme.colors.textPrimary)

                Text(statusLine)
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textTertiary)
            }

            Spacer()
        }
    }

    private var statusLine: String {
        if viewModel.disabledReason != nil {
            return "disabled"
        }
        if viewModel.isPreparing {
            return "indexing bundled corpus"
        }
        if viewModel.isRunning || trace.isRunning {
            return "running retrieve over local notes"
        }
        if trace.corpusStatus.isReady {
            return "ready over bundled on-device notes"
        }
        return "preparing bundled corpus"
    }

    private var isRunDisabled: Bool {
        viewModel.disabledReason != nil
            || viewModel.isPreparing
            || viewModel.isRunning
            || isHostBusy
            || !trace.corpusStatus.isReady
            || viewModel.promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var runDisabledReason: String? {
        if let disabledReason = viewModel.disabledReason {
            return disabledReason
        }
        if isHostBusy {
            return "Chat generation or model switching is active."
        }
        if viewModel.isPreparing || !trace.corpusStatus.isReady {
            return "Indexing demo corpus."
        }
        return nil
    }
}

private struct AgentPromptComposer: View {
    @Binding var promptText: String
    let isRunning: Bool
    let isDisabled: Bool
    let disabledReason: String?
    let onRun: () -> Void
    let onCancel: () -> Void

    private let theme = TinyBrainTheme.shared

    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.eight) {
            HStack(spacing: theme.spacing.eight) {
                #if os(macOS)
                NativeTextField(
                    text: $promptText,
                    isDisabled: isRunning || disabledReason != nil,
                    placeholder: "Ask TinyBrain Agent…",
                    onSubmit: {
                        if !isDisabled {
                            onRun()
                        }
                    }
                )
                .frame(height: 24)
                #else
                TextField("Ask TinyBrain Agent…", text: $promptText)
                    .font(theme.typography.body)
                    .textFieldStyle(.plain)
                    .disabled(isRunning || disabledReason != nil)
                #endif

                if isRunning {
                    Button(action: onCancel) {
                        ZStack {
                            Circle()
                                .fill(theme.colors.critical.opacity(0.16))
                                .frame(width: 26, height: 26)
                            Image(systemName: "stop.fill")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundStyle(theme.colors.critical)
                        }
                    }
                    .buttonStyle(.plain)
                    .help("Cancel agent run")
                } else {
                    Button(action: onRun) {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 25, weight: .medium))
                            .foregroundStyle(isDisabled ? theme.colors.textQuaternary : theme.colors.accent)
                    }
                    .buttonStyle(.plain)
                    .disabled(isDisabled)
                    .keyboardShortcut(.return, modifiers: .command)
                    .help(disabledReason ?? "Run agent")
                }
            }
            .padding(.horizontal, theme.spacing.twelve)
            .padding(.vertical, theme.spacing.eight)
            .frame(minHeight: 42)
            .background(theme.colors.fillQuaternary)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.large, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: theme.corners.large, style: .continuous)
                    .stroke(theme.colors.hairline, lineWidth: 0.5)
            )

            if let disabledReason {
                Text(disabledReason)
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textTertiary)
            }
        }
    }
}

private struct AgentCorpusStrip: View {
    let status: AgentCorpusStatus
    private let theme = TinyBrainTheme.shared

    var body: some View {
        HStack(spacing: theme.spacing.eight) {
            Text("Demo Corpus")
                .font(theme.typography.overline)
                .tracking(0.6)
                .foregroundStyle(theme.colors.textSecondary)

            corpusText

            Spacer()

            statusBadge
        }
        .padding(.horizontal, theme.spacing.twelve)
        .padding(.vertical, theme.spacing.eight)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(theme.colors.hairline, lineWidth: 0.5)
        )
    }

    @ViewBuilder
    private var corpusText: some View {
        switch status {
        case .idle(let noteCount), .indexing(let noteCount):
            Text("· \(noteCount) notes · on-device")
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)
        case .ready(let noteCount, let chunkCount, let embedder):
            Text("· \(noteCount) notes · \(chunkCount) chunks · \(shortEmbedder(embedder)) · on-device")
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)
                .lineLimit(1)
                .truncationMode(.middle)
        case .failed:
            Text("· indexing failed")
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.critical)
        }
    }

    private var statusBadge: some View {
        HStack(spacing: 5) {
            if case .indexing = status {
                Circle()
                    .fill(theme.colors.accent)
                    .frame(width: 5, height: 5)
                    .pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
            }
            Text(statusLabel)
                .font(theme.typography.monoSM)
                .foregroundStyle(statusColor)
        }
        .padding(.horizontal, theme.spacing.eight)
        .padding(.vertical, theme.spacing.four)
        .background(theme.colors.fillTertiary)
        .clipShape(Capsule())
    }

    private var statusLabel: String {
        switch status {
        case .idle:
            return "IDLE"
        case .indexing:
            return "INDEXING"
        case .ready:
            return "READY"
        case .failed:
            return "ERROR"
        }
    }

    private var statusColor: Color {
        switch status {
        case .ready:
            return theme.colors.positive
        case .failed:
            return theme.colors.critical
        case .indexing:
            return theme.colors.accent
        case .idle:
            return theme.colors.textTertiary
        }
    }

    private func shortEmbedder(_ embedder: String) -> String {
        embedder.replacingOccurrences(of: "NLEmbeddingProvider", with: "NL")
            .replacingOccurrences(of: "DeterministicStubEmbedder", with: "Stub")
    }
}

private struct AgentPromptChips: View {
    let onSelect: (String) -> Void
    private let theme = TinyBrainTheme.shared

    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.eight) {
            HStack(spacing: theme.spacing.eight) {
                Text("TRY")
                    .font(theme.typography.overline)
                    .tracking(0.6)
                    .foregroundStyle(theme.colors.textTertiary)

                ForEach(AgentDemoCorpus.promptChips) { chip in
                    AgentChip(chip: chip) {
                        onSelect(chip.prompt)
                    }
                }

                Spacer()
            }
        }
    }
}

private struct AgentChip: View {
    let chip: AgentPromptChip
    let action: () -> Void

    private let theme = TinyBrainTheme.shared
    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            Text(chip.label)
                .font(theme.typography.label)
                .foregroundStyle(isHovering ? theme.colors.accent : theme.colors.textSecondary)
                .padding(.horizontal, theme.spacing.eight)
                .padding(.vertical, theme.spacing.four)
                .background(isHovering ? theme.colors.accentQuiet : theme.colors.fillQuaternary)
                .clipShape(Capsule())
                .overlay(
                    Capsule()
                        .stroke(theme.colors.hairline, lineWidth: 0.5)
                )
        }
        .buttonStyle(.plain)
        .help(chip.targetFact)
        .onHover { hovering in
            isHovering = hovering
        }
    }
}

private struct AgentFinalAnswerPanel: View {
    let answer: String?
    let isRunning: Bool
    let errorMessage: String?
    let terminationReason: String?

    @StateObject private var typewriter = TypewriterEffect()
    private let theme = TinyBrainTheme.shared

    var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.twelve) {
            HStack(spacing: theme.spacing.eight) {
                Text("FINAL ANSWER")
                    .font(theme.typography.overline)
                    .tracking(0.6)
                    .foregroundStyle(theme.colors.textSecondary)

                Spacer()

                if let terminationReason {
                    Text(terminationReason.uppercased())
                        .font(theme.typography.monoSM)
                        .foregroundStyle(theme.colors.textTertiary)
                }
            }

            content
        }
        .padding(theme.spacing.twelve)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(borderColor, lineWidth: 0.5)
        )
        .onChange(of: answer) { _, newValue in
            if let newValue, !newValue.isEmpty {
                typewriter.animate(newValue)
            } else {
                typewriter.reset()
            }
        }
        .onAppear {
            if let answer, !answer.isEmpty, typewriter.displayedText.isEmpty {
                typewriter.animate(answer)
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if let errorMessage, !errorMessage.isEmpty {
            Text(errorMessage)
                .font(theme.typography.callout)
                .foregroundStyle(theme.colors.critical)
                .fixedSize(horizontal: false, vertical: true)
        } else if let answer, !answer.isEmpty {
            Text(typewriter.displayedText.isEmpty ? answer : typewriter.displayedText)
                .font(theme.typography.body)
                .foregroundStyle(theme.colors.textPrimary)
                .lineSpacing(3)
                .fixedSize(horizontal: false, vertical: true)
        } else if isRunning {
            VStack(alignment: .leading, spacing: theme.spacing.eight) {
                Text("Awaiting final answer")
                    .font(theme.typography.callout)
                    .foregroundStyle(theme.colors.textSecondary)
                shimmerLine(width: 0.92)
                shimmerLine(width: 0.74)
            }
        } else {
            Text("Final answer will appear after retrieved evidence is observed.")
                .font(theme.typography.callout)
                .foregroundStyle(theme.colors.textTertiary)
        }
    }

    private var borderColor: Color {
        if errorMessage != nil {
            return theme.colors.critical.opacity(0.30)
        }
        if answer != nil {
            return theme.colors.accentHairline
        }
        return theme.colors.hairline
    }

    private func shimmerLine(width: CGFloat) -> some View {
        GeometryReader { geometry in
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(theme.colors.fillTertiary)
                .frame(width: geometry.size.width * width, height: 8)
                .shimmer()
        }
        .frame(height: 8)
    }
}

private struct AgentDisabledState: View {
    let message: String
    private let theme = TinyBrainTheme.shared

    var body: some View {
        HStack(alignment: .top, spacing: theme.spacing.twelve) {
            ZStack {
                Circle()
                    .fill(theme.colors.warning.opacity(0.14))
                    .frame(width: 34, height: 34)
                Image(systemName: "lock.slash")
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(theme.colors.warning)
            }

            VStack(alignment: .leading, spacing: theme.spacing.four) {
                Text("Agent unavailable")
                    .font(theme.typography.label)
                    .foregroundStyle(theme.colors.textPrimary)
                Text(message)
                    .font(theme.typography.callout)
                    .foregroundStyle(theme.colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Spacer()
        }
        .padding(theme.spacing.twelve)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(theme.colors.warning.opacity(0.30), lineWidth: 0.5)
        )
    }
}
