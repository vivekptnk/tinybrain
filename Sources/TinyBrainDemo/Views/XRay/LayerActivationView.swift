/// Layer Activation View
///
/// **TB-010: X-Ray Mode**
///
/// Shows L2 norm of hidden state at each transformer layer as vertical bars.
/// Visualizes how signal magnitude evolves through the network.

import SwiftUI

struct LayerActivationView: View {
    let layerNorms: [Float]
    private let theme = TinyBrainTheme.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader("Layer Activations")

            if layerNorms.isEmpty {
                Text("Waiting for generation…")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textSecondary)
            } else {
                HStack(alignment: .bottom, spacing: 3) {
                    ForEach(Array(layerNorms.enumerated()), id: \.offset) { index, norm in
                        layerBar(index: index, norm: norm)
                    }
                }
                .frame(height: 50)
            }
        }
    }

    @ViewBuilder
    private func layerBar(index: Int, norm: Float) -> some View {
        let maxNorm = layerNorms.max() ?? 1.0
        let normalized = maxNorm > 0 ? CGFloat(norm / maxNorm) : 0

        VStack(spacing: 2) {
            RoundedRectangle(cornerRadius: 2)
                .fill(barGradient(normalized))
                .frame(width: barWidth, height: max(4, 40 * normalized))

            Text("L\(index)")
                .font(theme.typography.monoSM)
                .foregroundStyle(theme.colors.textSecondary)
        }
    }

    private var barWidth: CGFloat {
        layerNorms.count > 12 ? 8 : 14
    }

    private func barGradient(_ normalized: CGFloat) -> Color {
        theme.colors.accent.opacity(0.24 + Double(normalized) * 0.76)
    }
}
