/// TinyBrain Chat Demo App
///
/// A SwiftUI demonstration app showing local LLM inference with streaming output.
/// This is a standalone iOS/macOS app that can be opened in Xcode.

import SwiftUI
import TinyBrainDemo

@main
struct ChatDemoApp: App {
    var body: some Scene {
        WindowGroup {
            ChatView()
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
                TextField("Enter your prompt...", text: $viewModel.promptText, axis: .vertical)
                    .textFieldStyle(.roundedBorder)
                    .lineLimit(1...5)
                    .disabled(viewModel.isGenerating)
                
                Button(action: {
                    Task {
                        await viewModel.generate()
                    }
                }) {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                }
                .disabled(viewModel.promptText.isEmpty || viewModel.isGenerating)
            }
            .padding()
            .background(.background)
        }
        .frame(minWidth: 400, minHeight: 300)
    }
}

#Preview {
    ChatView()
}

