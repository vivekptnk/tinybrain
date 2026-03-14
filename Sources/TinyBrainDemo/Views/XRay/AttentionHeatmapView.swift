/// Attention Heatmap — Hero X-Ray Visualization
///
/// **TB-010: X-Ray Mode**
///
/// Renders attention weights as a color-intensity grid using SwiftUI Canvas
/// for high performance. Shows which past tokens the model attends to.
///
/// - Columns = past token positions
/// - Rows = sequence history (recent snapshots)
/// - Color intensity = attention weight (0→transparent, 1→saturated)

import SwiftUI

struct AttentionHeatmapView: View {
    let snapshots: [XRaySnapshot]
    let selectedLayer: Int
    let numLayers: Int
    let onLayerChange: (Int) -> Void

    private let cellSize: CGFloat = 12
    private let maxVisiblePositions: Int = 40

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Header with layer picker
            HStack {
                sectionHeader("Attention Pattern")
                Spacer()
                layerPicker
            }

            if snapshots.isEmpty {
                Text("Waiting for generation...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                heatmapCanvas
                    .frame(height: heatmapHeight)
                    .clipShape(RoundedRectangle(cornerRadius: 4))

                legendBar
            }
        }
    }

    // MARK: - Layer Picker

    private var layerPicker: some View {
        HStack(spacing: 4) {
            Text("Layer")
                .font(.caption2)
                .foregroundStyle(.secondary)
            Picker("", selection: Binding(
                get: { selectedLayer },
                set: { onLayerChange($0) }
            )) {
                ForEach(0..<numLayers, id: \.self) { i in
                    Text("\(i)").tag(i)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 50)
        }
    }

    // MARK: - Heatmap Canvas

    private var heatmapHeight: CGFloat {
        let rows = min(snapshots.count, 20)
        return CGFloat(rows) * cellSize + 2
    }

    private var heatmapCanvas: some View {
        Canvas { context, size in
            let recentSnapshots = Array(snapshots.suffix(20))
            guard !recentSnapshots.isEmpty else { return }

            // Find max sequence length across visible snapshots
            let maxSeqLen = recentSnapshots.compactMap {
                $0.attentionWeights.indices.contains(selectedLayer)
                    ? $0.attentionWeights[selectedLayer].count
                    : nil
            }.max() ?? 1

            let visiblePositions = min(maxSeqLen, maxVisiblePositions)
            let cellW = min(cellSize, size.width / CGFloat(visiblePositions))
            let cellH = cellSize

            for (rowIdx, snapshot) in recentSnapshots.enumerated() {
                guard snapshot.attentionWeights.indices.contains(selectedLayer) else { continue }
                let weights = snapshot.attentionWeights[selectedLayer]

                // Show the last `visiblePositions` positions
                let startPos = max(0, weights.count - visiblePositions)
                for (colIdx, posIdx) in (startPos..<weights.count).enumerated() {
                    let weight = weights[posIdx]
                    let rect = CGRect(
                        x: CGFloat(colIdx) * cellW,
                        y: CGFloat(rowIdx) * cellH,
                        width: cellW - 1,
                        height: cellH - 1
                    )
                    let color = heatColor(weight)
                    context.fill(Path(roundedRect: rect, cornerRadius: 1), with: .color(color))
                }
            }
        }
        .background(Color.black.opacity(0.05))
    }

    // MARK: - Color Mapping

    private func heatColor(_ weight: Float) -> Color {
        // Blue (low) → Cyan → Yellow → Red (high)
        let w = min(max(weight, 0), 1)
        if w < 0.25 {
            let t = Double(w / 0.25)
            return Color(red: 0.1, green: 0.1 + 0.4 * t, blue: 0.3 + 0.4 * t).opacity(0.3 + 0.7 * t)
        } else if w < 0.5 {
            let t = Double((w - 0.25) / 0.25)
            return Color(red: 0.2 * t, green: 0.5 + 0.3 * t, blue: 0.7 - 0.3 * t)
        } else if w < 0.75 {
            let t = Double((w - 0.5) / 0.25)
            return Color(red: 0.2 + 0.6 * t, green: 0.8, blue: 0.4 - 0.3 * t)
        } else {
            let t = Double((w - 0.75) / 0.25)
            return Color(red: 0.8 + 0.2 * t, green: 0.8 - 0.5 * t, blue: 0.1)
        }
    }

    // MARK: - Legend

    private var legendBar: some View {
        HStack(spacing: 2) {
            Text("Low")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
            ForEach(0..<10, id: \.self) { i in
                Rectangle()
                    .fill(heatColor(Float(i) / 9.0))
                    .frame(width: 12, height: 6)
                    .cornerRadius(1)
            }
            Text("High")
                .font(.system(size: 9))
                .foregroundStyle(.secondary)
            Spacer()
            Text("Attention Weight")
                .font(.system(size: 9))
                .foregroundStyle(.tertiary)
        }
    }
}
