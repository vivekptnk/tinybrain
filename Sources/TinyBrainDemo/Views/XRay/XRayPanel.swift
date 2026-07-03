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
    let isGenerating: Bool
    let tokenDecoder: ((Int) -> String)?
    private let theme = TinyBrainTheme.shared

    public var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                panelHeader

                separator

                vizSection(
                    title: nil,
                    tip: "Each row is a generated token. Color intensity shows how much attention the model pays to each past position."
                ) {
                    AttentionHeatmapView(
                        snapshots: xRay.snapshotHistory,
                        selectedLayer: xRay.selectedLayer,
                        numLayers: xRay.numLayers,
                        onLayerChange: { xRay.selectedLayer = $0 }
                    )
                }

                separator

                vizSection(
                    title: nil,
                    tip: "The model's top predictions for the next token. Longer bars = higher probability."
                ) {
                    TokenProbabilityBar(
                        candidates: xRay.latestSnapshot?.topCandidates ?? [],
                        tokenDecoder: tokenDecoder
                    )
                    .animation(.easeInOut(duration: 0.2), value: xRay.latestSnapshot?.position)
                }

                separator

                vizSection(
                    title: nil,
                    tip: "L2 norm of hidden state at each layer. Shows how signal magnitude evolves through the network."
                ) {
                    LayerActivationView(
                        layerNorms: xRay.latestSnapshot?.layerNorms ?? []
                    )
                    .animation(.easeInOut(duration: 0.15), value: xRay.latestSnapshot?.position)
                }

                separator

                vizSection(
                    title: nil,
                    tip: "Memory pages storing past keys and values. Filled = cached data enabling O(n) inference."
                ) {
                    KVCacheGridView(pages: xRay.kvCachePages)
                }

                if let snapshot = xRay.latestSnapshot {
                    separator
                    entropyIndicator(snapshot.entropy)
                }

                Spacer()
            }
            .padding(12)
        }
        .frame(width: 320)
        .background(.bar)
        .overlay(alignment: .leading) {
            Rectangle()
                .fill(theme.colors.hairline)
                .frame(width: 0.5)
        }
    }

    // MARK: - Panel Header

    private var panelHeader: some View {
        HStack(spacing: 8) {
            Image(systemName: "eye.fill")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(theme.colors.accent)
            Text("X-Ray")
                .font(theme.typography.title2)
                .foregroundStyle(theme.colors.textPrimary)
            Spacer()
            if let snapshot = xRay.latestSnapshot {
                Text("pos \(snapshot.position)")
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.textSecondary)
            }
            if isGenerating {
                HStack(spacing: 5) {
                    Circle()
                        .fill(theme.colors.accent)
                        .frame(width: 5, height: 5)
                        .pulsing(minOpacity: 0.35, maxOpacity: 1.0, duration: 0.8)
                    Text("LIVE")
                        .font(theme.typography.monoSM)
                        .foregroundStyle(theme.colors.accent)
                }
                .padding(.horizontal, 7)
                .padding(.vertical, 3)
                .background(theme.colors.fillQuaternary)
                .clipShape(Capsule())
            }
        }
    }

    private var separator: some View {
        Rectangle()
            .fill(theme.colors.hairline)
            .frame(height: 0.5)
    }

    // MARK: - Entropy Indicator

    @ViewBuilder
    private func entropyIndicator(_ entropy: Float) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader("Output Uncertainty")
            HStack(spacing: 8) {
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3)
                            .fill(theme.colors.hairline)
                        RoundedRectangle(cornerRadius: 3)
                            .fill(entropyColor(entropy))
                            .frame(width: geo.size.width * CGFloat(min(entropy / 10.0, 1.0)))
                    }
                }
                .frame(height: 12)

                Text(String(format: "%.2f nats", entropy))
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.textSecondary)
                    .frame(width: 70, alignment: .trailing)
            }
            Text(entropyLabel(entropy))
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)
        }
    }

    private func entropyColor(_ entropy: Float) -> Color {
        if entropy < 2 { return theme.colors.positive }
        if entropy < 5 { return theme.colors.warning }
        return theme.colors.critical
    }

    private func entropyLabel(_ entropy: Float) -> String {
        if entropy < 1 { return "Very confident — model is certain about next token" }
        if entropy < 3 { return "Moderate confidence — a few likely candidates" }
        if entropy < 6 { return "Uncertain — many plausible continuations" }
        return "High uncertainty — nearly random selection"
    }

    // MARK: - Visualization Section Wrapper

    @ViewBuilder
    private func vizSection<Content: View>(
        title: String?,
        tip: String,
        @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            if let title {
                sectionHeader(title)
            }
            content()
            Text(tip)
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
