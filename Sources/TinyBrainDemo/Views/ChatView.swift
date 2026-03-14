/// Chat View
///
/// **TB-006:** Main chat interface
///
/// Features:
/// - Message list with auto-scroll
/// - Streaming response display
/// - Telemetry panel
/// - X-Ray Mode visualization
/// - Platform-adaptive layout

import SwiftUI
import TinyBrainRuntime

/// Main chat interface view
public struct ChatView: View {
    @StateObject private var viewModel: ChatViewModel
    let theme = TinyBrainTheme.shared

    @State private var showTelemetry = true
    @State private var showSettings = false
    @State private var showXRay = false

    public init(viewModel: ChatViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            HStack(spacing: 0) {
                // Chat area
                VStack(spacing: 0) {
                    messagesList
                    inputBar
                }

                // Right panel
                if showXRay {
                    XRayPanel(
                        xRay: viewModel.xRay,
                        tokenDecoder: { viewModel.decodeToken($0) }
                    )
                    .transition(.move(edge: .trailing))
                } else if showTelemetry {
                    telemetrySidebar
                        .frame(width: 240)
                        .transition(.move(edge: .trailing))
                }
            }
        }
        .frame(minWidth: 700, minHeight: 500)
        #if os(macOS)
        .background(Color(nsColor: .windowBackgroundColor))
        #endif
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 12) {
            // Logo + title
            Image(systemName: "brain.head.profile")
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(.blue)

            Text("TinyBrain")
                .font(.system(size: 16, weight: .semibold, design: .rounded))

            if viewModel.isGenerating {
                HStack(spacing: 4) {
                    Circle().fill(.blue).frame(width: 5, height: 5)
                    Text("Generating")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.blue)
                }
                .pulsing()
            }

            Spacer()

            // Toolbar buttons
            HStack(spacing: 2) {
                toolbarButton(
                    icon: showXRay ? "eye.fill" : "eye",
                    color: showXRay ? .blue : .secondary,
                    tooltip: "X-Ray Mode"
                ) {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showXRay.toggle()
                        viewModel.setXRayEnabled(showXRay)
                    }
                }

                toolbarButton(
                    icon: showTelemetry ? "chart.bar.fill" : "chart.bar",
                    color: showTelemetry ? .accentColor : .secondary,
                    tooltip: "Metrics"
                ) {
                    withAnimation(.easeInOut(duration: 0.2)) { showTelemetry.toggle() }
                }

                toolbarButton(
                    icon: "arrow.counterclockwise",
                    color: .secondary,
                    tooltip: "New chat"
                ) {
                    viewModel.clearConversation()
                }
                .disabled(viewModel.messages.isEmpty && !viewModel.isGenerating)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.bar)
    }

    private func toolbarButton(icon: String, color: Color, tooltip: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(color)
                .frame(width: 30, height: 30)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(tooltip)
    }

    // MARK: - Messages List

    private var messagesList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: 2) {
                    if viewModel.messages.isEmpty {
                        emptyState
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .padding(.top, 80)
                    } else {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                        .padding(.top, 12)
                        .padding(.bottom, 20)
                    }
                }
            }
            .onChange(of: viewModel.messages.count) {
                if let last = viewModel.messages.last {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "brain.head.profile")
                .font(.system(size: 36, weight: .thin))
                .foregroundStyle(.blue.opacity(0.6))

            Text("TinyBrain")
                .font(.system(size: 24, weight: .bold, design: .rounded))

            Text("On-device LLM inference")
                .font(.system(size: 14))
                .foregroundStyle(.secondary)

            HStack(spacing: 6) {
                Image(systemName: "eye")
                    .font(.system(size: 11))
                Text("Click the eye icon to enable X-Ray Mode")
                    .font(.system(size: 12))
            }
            .foregroundStyle(.tertiary)
            .padding(.top, 8)
        }
    }

    // MARK: - Input Bar

    private var inputBar: some View {
        VStack(spacing: 0) {
            Divider()
            VStack(spacing: 8) {
                // Text input + send button
                HStack(spacing: 8) {
                    #if os(macOS)
                    NativeTextField(
                        text: $viewModel.promptText,
                        isDisabled: viewModel.isGenerating,
                        onSubmit: { sendMessage() }
                    )
                    .frame(height: 22)
                    #else
                    TextField("Message TinyBrain...", text: $viewModel.promptText)
                        .textFieldStyle(.plain)
                        .disabled(viewModel.isGenerating)
                    #endif

                    if viewModel.isGenerating {
                        Button {
                            viewModel.clearConversation()
                        } label: {
                            Image(systemName: "stop.circle.fill")
                                .font(.system(size: 22))
                                .foregroundStyle(.red.opacity(0.8))
                        }
                        .buttonStyle(.plain)
                    } else {
                        Button {
                            sendMessage()
                        } label: {
                            Image(systemName: "arrow.up.circle.fill")
                                .font(.system(size: 22))
                                .foregroundStyle(viewModel.promptText.isEmpty ? Color.secondary.opacity(0.4) : Color.blue)
                        }
                        .buttonStyle(.plain)
                        .disabled(viewModel.promptText.isEmpty)
                        .keyboardShortcut(.return, modifiers: .command)
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.primary.opacity(0.05))
                .clipShape(RoundedRectangle(cornerRadius: 18))

                // Quick prompt pills
                if viewModel.messages.isEmpty && viewModel.promptText.isEmpty {
                    HStack(spacing: 6) {
                        Text("Try:")
                            .font(.system(size: 11))
                            .foregroundStyle(.tertiary)

                        ForEach(demoPrompts, id: \.label) { prompt in
                            Button {
                                viewModel.promptText = prompt.text
                            } label: {
                                Text(prompt.label)
                                    .font(.system(size: 11, weight: .medium))
                                    .padding(.horizontal, 8)
                                    .padding(.vertical, 4)
                                    .background(Color.primary.opacity(0.05))
                                    .clipShape(Capsule())
                            }
                            .buttonStyle(.plain)
                        }

                        Spacer()
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background(.bar)
        }
    }

    private var demoPrompts: [(label: String, text: String)] {
        [
            ("Hello", "Hello, TinyBrain!"),
            ("Explain LLMs", "Explain how large language models work"),
            ("Story", "Tell me a short story about a neural network"),
        ]
    }

    private func sendMessage() {
        Task {
            await viewModel.generate()
        }
    }

    // MARK: - Telemetry Sidebar

    private var telemetrySidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("METRICS")
                .font(.system(size: 11, weight: .semibold, design: .rounded))
                .foregroundStyle(.secondary)
                .tracking(0.5)
                .padding(.bottom, 16)

            VStack(spacing: 14) {
                metricCard(
                    icon: "bolt.fill",
                    label: "Tokens/sec",
                    value: String(format: "%.1f", viewModel.telemetry.tokensPerSecond),
                    color: .blue
                )

                metricCard(
                    icon: "timer",
                    label: "Latency",
                    value: String(format: "%.0f ms", viewModel.telemetry.millisecondsPerToken),
                    color: .orange
                )

                metricCard(
                    icon: "flame.fill",
                    label: "Energy",
                    value: String(format: "%.2f J", viewModel.telemetry.energyEstimate),
                    color: .red
                )

                metricCard(
                    icon: "square.stack.3d.up.fill",
                    label: "KV Cache",
                    value: String(format: "%.0f%%", viewModel.telemetry.kvCacheUsagePercent),
                    color: .green
                )
            }

            Spacer()
        }
        .padding(16)
        .background(.bar)
    }

    private func metricCard(icon: String, label: String, value: String, color: Color) -> some View {
        HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.system(size: 12))
                .foregroundStyle(color)
                .frame(width: 20)

            VStack(alignment: .leading, spacing: 2) {
                Text(label)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.system(size: 16, weight: .semibold, design: .monospaced))
            }

            Spacer()
        }
        .padding(10)
        .background(color.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 8))
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
