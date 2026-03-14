/// X-Ray Panel — Container for all transformer visualizations
///
/// **TB-010: X-Ray Mode**
///
/// Assembles attention heatmap, token probability bars, layer activations,
/// and KV cache grid into a scrollable sidebar panel.

import SwiftUI
import TinyBrainRuntime

public struct XRayPanel: View {
    @ObservedObject var xRay: XRayViewModel
    let tokenDecoder: ((Int) -> String)?

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                // Header
                panelHeader

                Divider()

                // 1. Attention Heatmap (hero visualization)
                AttentionHeatmapView(
                    snapshots: xRay.snapshotHistory,
                    selectedLayer: xRay.selectedLayer,
                    numLayers: xRay.numLayers,
                    onLayerChange: { xRay.selectedLayer = $0 }
                )

                Divider()

                // 2. Token Probability Bar Chart
                TokenProbabilityBar(
                    candidates: xRay.latestSnapshot?.topCandidates ?? [],
                    tokenDecoder: tokenDecoder
                )

                Divider()

                // 3. Layer Activation Bars
                LayerActivationView(
                    layerNorms: xRay.latestSnapshot?.layerNorms ?? []
                )

                Divider()

                // 4. KV Cache Page Grid
                KVCacheGridView(pages: xRay.kvCachePages)

                // Entropy indicator
                if let snapshot = xRay.latestSnapshot {
                    Divider()
                    entropyIndicator(snapshot.entropy)
                }

                Spacer()
            }
            .padding(12)
        }
        .frame(width: 320)
        #if os(macOS)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))
        #else
        .background(Color(uiColor: .secondarySystemBackground).opacity(0.5))
        #endif
    }

    // MARK: - Panel Header

    private var panelHeader: some View {
        HStack {
            Image(systemName: "eye.trianglebadge.exclamationmark")
                .foregroundStyle(.blue)
            Text("X-Ray Mode")
                .font(.system(.headline, design: .rounded))
            Spacer()
            if let snapshot = xRay.latestSnapshot {
                Text("pos \(snapshot.position)")
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Entropy Indicator

    @ViewBuilder
    private func entropyIndicator(_ entropy: Float) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader("Output Uncertainty")
            HStack(spacing: 8) {
                // Entropy bar
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(Color.gray.opacity(0.15))
                        RoundedRectangle(cornerRadius: 3)
                            .fill(entropyColor(entropy))
                            .frame(width: geo.size.width * CGFloat(min(entropy / 10.0, 1.0)))
                    }
                }
                .frame(height: 12)

                Text(String(format: "%.2f nats", entropy))
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .frame(width: 70, alignment: .trailing)
            }
            Text(entropyLabel(entropy))
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
        }
    }

    private func entropyColor(_ entropy: Float) -> Color {
        if entropy < 2 { return .green.opacity(0.7) }
        if entropy < 5 { return .orange.opacity(0.7) }
        return .red.opacity(0.7)
    }

    private func entropyLabel(_ entropy: Float) -> String {
        if entropy < 1 { return "Very confident — model is certain about next token" }
        if entropy < 3 { return "Moderate confidence — a few likely candidates" }
        if entropy < 6 { return "Uncertain — many plausible continuations" }
        return "High uncertainty — nearly random selection"
    }
}
