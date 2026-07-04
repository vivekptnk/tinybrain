/// Chat View
///
/// Main TinyBrain demo interface: chat canvas, model picker, telemetry, and
/// X-Ray instrumentation.

import SwiftUI
import TinyBrainRuntime

/// Main chat interface view.
public struct ChatView: View {
    @StateObject private var viewModel: ChatViewModel
    private let theme = TinyBrainTheme.shared

    @State private var demoMode: DemoMode = .chat
    @State private var showTelemetry: Bool
    @State private var showXRay: Bool
    @State private var showErrorBanner = true
    private let initialShowPicker: Bool

    public init(viewModel: ChatViewModel, initialShowXRay: Bool = false, initialShowPicker: Bool = false) {
        _viewModel = StateObject(wrappedValue: viewModel)
        _showXRay = State(initialValue: initialShowXRay)
        _showTelemetry = State(initialValue: !initialShowXRay)
        self.initialShowPicker = initialShowPicker
    }

    public var body: some View {
        VStack(spacing: 0) {
            header

            if demoMode == .chat {
                HStack(spacing: 0) {
                    VStack(spacing: 0) {
                        if viewModel.hasError && showErrorBanner {
                            errorBanner
                                .transition(.move(edge: .top).combined(with: .opacity))
                        }

                        messagesList
                        inputBar
                    }
                    .background(theme.colors.canvas)

                    if showXRay {
                        XRayPanel(
                            xRay: viewModel.xRay,
                            isGenerating: viewModel.isGenerating,
                            tokenDecoder: { viewModel.decodeToken($0) }
                        )
                        .transition(.move(edge: .trailing))
                    } else if showTelemetry {
                        telemetrySidebar
                            .frame(width: theme.layout.sidebarWidth)
                            .transition(.move(edge: .trailing))
                    }
                }
                .transition(.opacity)
            } else {
                VStack(spacing: 0) {
                    if viewModel.hasError && showErrorBanner {
                        errorBanner
                            .transition(.move(edge: .top).combined(with: .opacity))
                    }

                    AgentWorkbenchView(
                        viewModel: viewModel.agent,
                        isHostBusy: viewModel.isGenerating || viewModel.isSwitchingModel
                    )
                }
                .transition(.opacity)
            }
        }
        .frame(minWidth: 700, minHeight: 500)
        .background(theme.colors.canvas)
        .onAppear {
            viewModel.setXRayEnabled(showXRay && demoMode == .chat)
        }
        .onChange(of: demoMode) { _, mode in
            viewModel.setXRayEnabled(showXRay && mode == .chat)
            if mode == .agent {
                Task { await viewModel.agent.prepareIfNeeded() }
            }
        }
        .onChange(of: viewModel.hasError) { _, hasError in
            if hasError {
                withAnimation(theme.animations.quick) {
                    showErrorBanner = true
                }
            }
        }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(theme.colors.accent)

            Text("TinyBrain")
                .font(theme.typography.wordmark)
                .foregroundStyle(theme.colors.textPrimary)

            ModelPickerView(
                pickerVM: viewModel.modelPicker,
                activeModelName: viewModel.activeModelName,
                activeQuant: viewModel.activeQuant,
                activeModelPath: viewModel.activeModelPath,
                switchingModelPath: viewModel.pendingModelPath,
                isSwitching: viewModel.isSwitchingModel,
                initialShowPicker: initialShowPicker
            ) { model in
                Task { await viewModel.switchModel(model) }
            }
            .disabled(viewModel.isGenerating || viewModel.agent.isRunning || viewModel.isSwitchingModel)

            statusSlot
                .frame(width: 124, alignment: .leading)

            Spacer()

            Rectangle()
                .fill(theme.colors.hairline)
                .frame(width: 0.5, height: 20)
                .padding(.leading, 4)

            HStack(spacing: 4) {
                railButton(
                    icon: demoMode == .chat ? "bubble.left.and.bubble.right.fill" : "bubble.left.and.bubble.right",
                    label: "Chat",
                    isActive: demoMode == .chat,
                    tooltip: "Show chat"
                ) {
                    withAnimation(theme.animations.quick) {
                        demoMode = .chat
                    }
                }

                railButton(
                    icon: demoMode == .agent ? "point.topleft.down.curvedto.point.bottomright.up.fill" : "point.topleft.down.curvedto.point.bottomright.up",
                    label: "Agent",
                    isActive: demoMode == .agent,
                    tooltip: "Show Agent Trace"
                ) {
                    withAnimation(theme.animations.quick) {
                        demoMode = .agent
                    }
                }

                Rectangle()
                    .fill(theme.colors.hairline)
                    .frame(width: 0.5, height: 18)
                    .padding(.horizontal, 2)

                railButton(
                    icon: showXRay ? "eye.fill" : "eye",
                    label: "X-Ray",
                    isActive: showXRay,
                    tooltip: "Toggle X-Ray Mode"
                ) {
                    withAnimation(theme.animations.quick) {
                        showXRay.toggle()
                        if showXRay {
                            showTelemetry = false
                        }
                        viewModel.setXRayEnabled(showXRay)
                    }
                }
                .disabled(demoMode != .chat)
                .opacity(demoMode == .chat ? 1.0 : 0.45)

                railButton(
                    icon: showTelemetry ? "chart.bar.fill" : "chart.bar",
                    isActive: showTelemetry,
                    tooltip: "Toggle telemetry"
                ) {
                    withAnimation(theme.animations.quick) {
                        showTelemetry.toggle()
                        if showTelemetry {
                            showXRay = false
                            viewModel.setXRayEnabled(false)
                        }
                    }
                }
                .disabled(demoMode != .chat)
                .opacity(demoMode == .chat ? 1.0 : 0.45)

                railButton(
                    icon: "square.and.pencil",
                    isActive: false,
                    tooltip: "New chat"
                ) {
                    viewModel.clearConversation()
                }
                .disabled(viewModel.messages.isEmpty && !viewModel.isGenerating)
            }
        }
        .padding(.horizontal, 16)
        .frame(height: 48)
        .background(.bar)
        .overlay(alignment: .bottom) {
            Rectangle()
                .fill(theme.colors.hairline)
                .frame(height: 0.5)
        }
    }

    @ViewBuilder
    private var statusSlot: some View {
        if viewModel.hasError {
            Button {
                withAnimation(theme.animations.quick) {
                    showErrorBanner = true
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .font(.system(size: 12, weight: .medium))
                    Text("Error")
                        .font(theme.typography.caption)
                }
                .foregroundStyle(theme.colors.critical)
            }
            .buttonStyle(.plain)
        } else if viewModel.isGenerating {
            HStack(spacing: 4) {
                Circle()
                    .fill(theme.colors.accent)
                    .frame(width: 5, height: 5)
                    .pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
                Text("Generating")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.accent)
            }
        } else if viewModel.isSwitchingModel {
            HStack(spacing: 4) {
                ProgressView()
                    .scaleEffect(0.6)
                    .frame(width: 12, height: 12)
                Text("Loading model…")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textSecondary)
            }
        } else {
            EmptyView()
        }
    }

    private func railButton(
        icon: String,
        label: String? = nil,
        isActive: Bool,
        tooltip: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                if let label {
                    Text(label)
                        .font(theme.typography.label)
                }
            }
            .foregroundStyle(isActive ? theme.colors.accent : theme.colors.textSecondary)
            .padding(.horizontal, label == nil ? 8 : 10)
            .frame(height: 30)
            .background(isActive ? theme.colors.accentQuiet : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.small, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(tooltip)
    }

    // MARK: - Error Banner

    private var errorBanner: some View {
        HStack(spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(theme.colors.critical)

            Text(viewModel.errorMessage)
                .font(theme.typography.callout)
                .foregroundStyle(theme.colors.textPrimary)
                .lineLimit(2)

            Spacer()

            if viewModel.failedModelSwitchTarget != nil {
                Button("Retry") {
                    Task { await viewModel.retryLastModelSwitch() }
                }
                .font(theme.typography.label)
                .buttonStyle(.plain)
                .foregroundStyle(theme.colors.critical)
            }

            Button {
                withAnimation(theme.animations.quick) {
                    showErrorBanner = false
                }
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(theme.colors.textSecondary)
                    .frame(width: 24, height: 24)
            }
            .buttonStyle(.plain)
            .help("Dismiss")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(theme.colors.critical.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(theme.colors.critical.opacity(0.30), lineWidth: 0.5)
        )
        .padding(.horizontal, 16)
        .padding(.top, 12)
        .padding(.bottom, 8)
        .background(theme.colors.canvas)
    }

    // MARK: - Messages List

    private var messagesList: some View {
        GeometryReader { geometry in
            ScrollViewReader { proxy in
                ScrollView {
                    if viewModel.messages.isEmpty {
                        VStack(spacing: 0) {
                            Spacer(minLength: 0)
                            emptyState
                            Spacer(minLength: 0)
                        }
                        .frame(maxWidth: .infinity)
                        .frame(minHeight: geometry.size.height)
                    } else {
                        LazyVStack(spacing: 8) {
                            ForEach(viewModel.messages) { message in
                                MessageBubble(
                                    message: message,
                                    isStreaming: streamingMessageID == message.id,
                                    isFailed: viewModel.failedMessageIDs.contains(message.id)
                                )
                                .id(message.id)
                            }
                        }
                        .padding(.top, 12)
                        .padding(.bottom, 20)
                    }
                }
                .background(theme.colors.canvas)
                .onChange(of: viewModel.messages.count) { _, _ in
                    scrollToLastMessage(proxy)
                }
                .onChange(of: viewModel.messages.last?.content) { _, _ in
                    scrollToLastMessage(proxy)
                }
            }
        }
    }

    private var streamingMessageID: UUID? {
        guard viewModel.isGenerating else { return nil }
        return viewModel.messages.last(where: { $0.isAssistant })?.id
    }

    private func scrollToLastMessage(_ proxy: ScrollViewProxy) {
        guard let last = viewModel.messages.last else { return }
        withAnimation(.easeOut(duration: 0.2)) {
            proxy.scrollTo(last.id, anchor: .bottom)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(theme.colors.accentQuiet)
                    .frame(width: 56, height: 56)
                Image(systemName: "brain.head.profile")
                    .font(.system(size: 26, weight: .medium))
                    .foregroundStyle(theme.colors.accent)
            }

            Text("TinyBrain")
                .font(theme.typography.title1)
                .foregroundStyle(theme.colors.textPrimary)

            Text("On-device LLM inference — watch it think.")
                .font(theme.typography.callout)
                .foregroundStyle(theme.colors.textSecondary)

            VStack(alignment: .leading, spacing: 8) {
                capabilityRow(
                    icon: "eye",
                    text: Text("Enable X-Ray to watch attention, layers & token odds live.")
                )
                capabilityRow(
                    icon: "bolt",
                    text: Text("Real-time tokens/sec, latency & energy telemetry.")
                )
                capabilityRow(
                    icon: "cpu",
                    text: Text("Running ")
                        + Text("\(viewModel.activeModelName) · \(viewModel.activeQuant.rawValue)")
                        .fontWeight(.semibold)
                        + Text(".")
                )
            }
            .frame(maxWidth: 360, alignment: .leading)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
    }

    private func capabilityRow(icon: String, text: Text) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(theme.colors.textSecondary)
                .frame(width: 16)
            text
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textSecondary)
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                #if os(macOS)
                NativeTextField(
                    text: $viewModel.promptText,
                    isDisabled: viewModel.isGenerating || viewModel.agent.isRunning,
                    onSubmit: { sendMessage() }
                )
                .frame(height: 22)
                #else
                TextField("Message TinyBrain…", text: $viewModel.promptText)
                    .font(theme.typography.body)
                    .textFieldStyle(.plain)
                    .disabled(viewModel.isGenerating || viewModel.agent.isRunning)
                #endif

                if viewModel.isGenerating {
                    Button {
                        viewModel.stopGeneration()
                    } label: {
                        ZStack {
                            Circle()
                                .fill(theme.colors.critical.opacity(0.16))
                                .frame(width: 24, height: 24)
                            Image(systemName: "stop.fill")
                                .font(.system(size: 9, weight: .bold))
                                .foregroundStyle(theme.colors.critical)
                        }
                    }
                    .buttonStyle(.plain)
                    .help("Stop generating")
                } else {
                    Button {
                        sendMessage()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.system(size: 24, weight: .medium))
                            .foregroundStyle(viewModel.promptText.isEmpty ? theme.colors.textQuaternary : theme.colors.accent)
                    }
                    .buttonStyle(.plain)
                    .disabled(
                        viewModel.promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || viewModel.agent.isRunning
                    )
                    .keyboardShortcut(.return, modifiers: .command)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .frame(minHeight: 40)
            .background(theme.colors.fillQuaternary)
            .clipShape(RoundedRectangle(cornerRadius: theme.corners.large, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: theme.corners.large, style: .continuous)
                    .stroke(theme.colors.hairline, lineWidth: 0.5)
            )

            if viewModel.messages.isEmpty {
                suggestionChips
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(theme.colors.hairline)
                .frame(height: 0.5)
        }
    }

    private var suggestionChips: some View {
        HStack(spacing: 8) {
            Text("Try")
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)

            ForEach(demoPrompts, id: \.label) { prompt in
                SuggestionChip(label: prompt.label) {
                    viewModel.promptText = prompt.text
                }
            }

            Spacer()
        }
    }

    private var demoPrompts: [(label: String, text: String)] {
        [
            ("Hello", "Hello, TinyBrain!"),
            ("Explain LLMs", "Explain how large language models work"),
            ("Story", "Tell me a short story about a neural network")
        ]
    }

    private func sendMessage() {
        guard !viewModel.promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard !viewModel.agent.isRunning else { return }
        Task {
            await viewModel.generate()
        }
    }

    // MARK: - Telemetry Sidebar

    private var telemetrySidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("TELEMETRY")
                    .font(theme.typography.overline)
                    .tracking(0.6)
                    .foregroundStyle(theme.colors.textSecondary)

                Spacer()

                statusPill(isLive: viewModel.isGenerating)
            }

            VStack(spacing: 10) {
                metricTile(
                    icon: "bolt",
                    label: "Tokens/sec",
                    value: telemetryHasSamples ? String(format: "%.1f", viewModel.telemetry.tokensPerSecond) : "—",
                    unit: "tok/s",
                    normalized: min(viewModel.telemetry.tokensPerSecond / 60.0, 1.0)
                )

                metricTile(
                    icon: "timer",
                    label: "Latency",
                    value: telemetryHasSamples ? String(format: "%.0f", viewModel.telemetry.millisecondsPerToken) : "—",
                    unit: "ms",
                    normalized: min(viewModel.telemetry.millisecondsPerToken / 200.0, 1.0)
                )

                metricTile(
                    icon: "flame",
                    label: "Energy",
                    value: telemetryHasSamples ? String(format: "%.2f", viewModel.telemetry.energyEstimate) : "—",
                    unit: "J",
                    normalized: min(viewModel.telemetry.energyEstimate / 4.0, 1.0),
                    meterColor: viewModel.telemetry.energyEstimate > 4.0 ? theme.colors.warning : theme.colors.accent
                )

                metricTile(
                    icon: "square.stack.3d.up",
                    label: "KV Cache",
                    value: telemetryHasSamples ? String(format: "%.0f", viewModel.telemetry.kvCacheUsagePercent) : "—",
                    unit: "%",
                    normalized: min(viewModel.telemetry.kvCacheUsagePercent / 100.0, 1.0)
                )
            }
            .animation(.easeOut(duration: 0.2), value: viewModel.telemetry.tokensPerSecond)
            .animation(.easeOut(duration: 0.2), value: viewModel.telemetry.millisecondsPerToken)
            .animation(.easeOut(duration: 0.2), value: viewModel.telemetry.energyEstimate)
            .animation(.easeOut(duration: 0.2), value: viewModel.telemetry.kvCacheUsagePercent)

            if !telemetryHasSamples {
                Text("Awaiting generation.")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textTertiary)
            }

            Spacer()
        }
        .padding(16)
        .background(.bar)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(theme.colors.hairline)
                .frame(width: 0.5)
        }
    }

    private var telemetryHasSamples: Bool {
        !viewModel.telemetry.tokenHistory.isEmpty
    }

    private func metricTile(
        icon: String,
        label: String,
        value: String,
        unit: String,
        normalized: Double,
        meterColor: Color? = nil
    ) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(theme.colors.textSecondary)
                    .frame(width: 16)

                Text(label)
                    .font(theme.typography.label)
                    .foregroundStyle(theme.colors.textSecondary)

                Spacer()
            }

            HStack(alignment: .firstTextBaseline, spacing: 4) {
                Text(value)
                    .font(theme.typography.metricValue)
                    .foregroundStyle(theme.colors.textPrimary)
                Text(unit)
                    .font(theme.typography.metricUnit)
                    .foregroundStyle(theme.colors.textSecondary)
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(theme.colors.hairline)
                    Capsule()
                        .fill(meterColor ?? theme.colors.accent)
                        .frame(width: telemetryHasSamples ? geometry.size.width * CGFloat(max(0, min(normalized, 1))) : 0)
                }
            }
            .frame(height: 3)
        }
        .padding(12)
        .background(theme.colors.fillQuaternary)
        .clipShape(RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: theme.corners.medium, style: .continuous)
                .stroke(theme.colors.hairline, lineWidth: 0.5)
        )
    }

    private func statusPill(isLive: Bool) -> some View {
        HStack(spacing: 5) {
            if isLive {
                Circle()
                    .fill(theme.colors.accent)
                    .frame(width: 5, height: 5)
                    .pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
            }

            Text(isLive ? "LIVE" : "IDLE")
                .font(theme.typography.monoSM)
                .foregroundStyle(isLive ? theme.colors.accent : theme.colors.textTertiary)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 3)
        .background(theme.colors.fillQuaternary)
        .clipShape(Capsule())
    }
}

private enum DemoMode {
    case chat
    case agent
}

private struct SuggestionChip: View {
    let label: String
    let action: () -> Void

    private let theme = TinyBrainTheme.shared
    @State private var isHovering = false

    var body: some View {
        Button(action: action) {
            Text(label)
                .font(theme.typography.label)
                .foregroundStyle(isHovering ? theme.colors.accent : theme.colors.textSecondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .background(isHovering ? theme.colors.accentQuiet : theme.colors.fillQuaternary)
                .clipShape(Capsule())
                .overlay(
                    Capsule()
                        .stroke(theme.colors.hairline, lineWidth: 0.5)
                )
        }
        .buttonStyle(.plain)
        .onHover { hovering in
            isHovering = hovering
        }
    }
}

// MARK: - Preview

#if DEBUG
struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        let viewModel = ChatViewModel(runner: runner)

        return ChatView(viewModel: viewModel)
            .frame(width: 900, height: 600)
    }
}
#endif
