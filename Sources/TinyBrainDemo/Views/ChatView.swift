/// Chat View
///
/// **TB-006:** Main chat interface
///
/// Features:
/// - Message list with auto-scroll
/// - Streaming response display
/// - Telemetry panel
/// - Sampler controls
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
            // Header
            header
            
            Divider()
            
            // Main content
            HStack(spacing: 0) {
                // Chat area
                VStack(spacing: 0) {
                    messagesList
                    
                    Divider()
                    
                    // SIMPLIFIED Input - minimal nesting
                    inputBar
                }
                
                // Right panel: X-Ray or Telemetry
                if showXRay {
                    Divider()

                    XRayPanel(
                        xRay: viewModel.xRay,
                        tokenDecoder: { viewModel.decodeToken($0) }
                    )
                    .transition(.slideFromEdge(.trailing))
                } else if showTelemetry {
                    Divider()

                    telemetrySidebar
                        .frame(width: theme.layout.sidebarWidth)
                        .transition(.slideFromEdge(.trailing))
                }
            }
        }
        .frame(minWidth: 600, minHeight: 400)
        .background(theme.gradients.background)
    }
    
    // MARK: - Input Bar (Workaround for macOS Tahoe TextField bug)
    
    private var inputBar: some View {
        VStack(spacing: 12) {
            // WORKAROUND: TextField doesn't work in SPM executables on macOS Tahoe
            // Provide demo buttons instead
            Text("⚠️ TextField broken on macOS Tahoe + SPM. Use demo prompts:")
                .font(.caption)
                .foregroundColor(.orange)
            
            HStack(spacing: 8) {
                Button("Hello") {
                    viewModel.promptText = "Hello, TinyBrain!"
                }
                
                Button("Explain LLMs") {
                    viewModel.promptText = "Can you explain how large language models work?"
                }
                
                Button("Tell a story") {
                    viewModel.promptText = "Tell me a short story about a neural network"
                }
                
                Button("Clear") {
                    viewModel.promptText = ""
                }
            }
            .disabled(viewModel.isGenerating)
            
            if !viewModel.promptText.isEmpty {
                HStack {
                    Text(viewModel.promptText)
                        .lineLimit(2)
                        .foregroundColor(.secondary)
                    
                    Spacer()
                    
                    Button("Send ➤") {
                        sendMessage()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isGenerating)
                    .keyboardShortcut(.return, modifiers: .command)
                }
            }
        }
        .padding()
        .background(Color.yellow.opacity(0.1))
    }
    
    private func sendMessage() {
        Task {
            await viewModel.generate()
        }
    }
    
    // MARK: - Header
    
    private var header: some View {
        HStack {
            // Title
            HStack(spacing: theme.spacing.xs) {
                Text("🧠")
                    .font(.title)
                
                VStack(alignment: .leading, spacing: 2) {
                    Text("TinyBrain Chat")
                        .font(theme.typography.headline)
                        .fontWeight(.semibold)
                    
                    if viewModel.isGenerating {
                        Text("Generating...")
                            .font(theme.typography.caption)
                            .foregroundColor(.secondary)
                            .pulsing()
                    }
                }
            }
            
            Spacer()
            
            // Controls
            HStack(spacing: theme.spacing.sm) {
                // X-Ray toggle
                Button(action: {
                    withAnimation {
                        showXRay.toggle()
                        viewModel.setXRayEnabled(showXRay)
                    }
                }) {
                    Image(systemName: showXRay ? "eye.fill" : "eye")
                        .foregroundColor(showXRay ? .blue : .secondary)
                }
                .help("Toggle X-Ray Mode — live transformer visualization")

                // Telemetry toggle
                Button(action: { withAnimation { showTelemetry.toggle() } }) {
                    Image(systemName: showTelemetry ? "chart.bar.fill" : "chart.bar")
                }
                .help("Toggle telemetry panel")

                // Settings
                Button(action: { showSettings.toggle() }) {
                    Image(systemName: "gear")
                }
                .help("Settings")
                
                // Clear
                Button(action: viewModel.clearConversation) {
                    Image(systemName: "trash")
                }
                .help("Clear conversation")
                .disabled(viewModel.messages.isEmpty)
            }
            .buttonStyle(.plain)
            .foregroundColor(.secondary)
        }
        .padding(theme.spacing.md)
        .background(.ultraThinMaterial)
    }
    
    // MARK: - Messages List
    
    private var messagesList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: theme.spacing.sm) {
                    if viewModel.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }
                    }
                }
                .padding(.vertical, theme.spacing.md)
            }
            .onChange(of: viewModel.messages.count) {
                // Auto-scroll to bottom
                if let last = viewModel.messages.last {
                    withAnimation(theme.animations.smooth) {
                        proxy.scrollTo(last.id, anchor: .bottom)
                    }
                }
            }
        }
    }
    
    private var emptyState: some View {
        VStack(spacing: theme.spacing.lg) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 60))
                .foregroundColor(.secondary)
            
            VStack(spacing: theme.spacing.xs) {
                Text("Welcome to TinyBrain")
                    .font(theme.typography.title)
                    .fontWeight(.semibold)
                
                Text("Start a conversation to see on-device LLM inference in action")
                    .font(theme.typography.body)
                    .foregroundColor(.secondary)
                    .multilineTextAlignment(.center)
            }
        }
        .frame(maxWidth: 400)
        .padding(theme.spacing.xl)
    }
    
    // MARK: - Telemetry Sidebar
    
    private var telemetrySidebar: some View {
        VStack(alignment: .leading, spacing: theme.spacing.md) {
            Text("Metrics")
                .font(theme.typography.headline)
                .fontWeight(.semibold)
            
            Divider()
            
            // Metrics
            VStack(alignment: .leading, spacing: theme.spacing.sm) {
                metricRow(
                    icon: "speedometer",
                    label: "Tokens/sec",
                    value: String(format: "%.1f", viewModel.telemetry.tokensPerSecond)
                )
                
                metricRow(
                    icon: "timer",
                    label: "ms/token",
                    value: String(format: "%.0f", viewModel.telemetry.millisecondsPerToken)
                )
                
                metricRow(
                    icon: "bolt.fill",
                    label: "Energy",
                    value: String(format: "%.2f J", viewModel.telemetry.energyEstimate)
                )
                
                metricRow(
                    icon: "square.stack.3d.up",
                    label: "KV Cache",
                    value: String(format: "%.0f%%", viewModel.telemetry.kvCacheUsagePercent)
                )
            }
            
            Spacer()
        }
        .padding(theme.spacing.md)
        .background(.ultraThinMaterial)
    }
    
    private func metricRow(icon: String, label: String, value: String) -> some View {
        HStack {
            Image(systemName: icon)
                .foregroundColor(theme.colors.accent)
                .frame(width: 20)
            
            Text(label)
                .font(theme.typography.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .font(theme.typography.monospace)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Preview

#if DEBUG
struct ChatView_Previews: PreviewProvider {
    static var previews: some View {
        // Create toy model for preview
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

