/// Retrieved passage row for the Agent trace.

import SwiftUI
import TinyBrainAgent

/// Compact rendering of one retrieved passage with rank, source, distance, and excerpt.
public struct RetrievedPassageRow: View {
    private let passage: AgentRetrievedPassage
    private let theme = TinyBrainTheme.shared

    /// Creates a retrieved passage row.
    public init(passage: AgentRetrievedPassage) {
        self.passage = passage
    }

    public var body: some View {
        VStack(alignment: .leading, spacing: theme.spacing.six) {
            HStack(spacing: theme.spacing.eight) {
                Text("[\(passage.rank + 1)]")
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.accent)
                    .padding(.horizontal, theme.spacing.four)
                    .padding(.vertical, theme.spacing.two)
                    .background(theme.colors.accentQuiet)
                    .clipShape(Capsule())

                Text(passage.source)
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.textTertiary)
                    .lineLimit(1)
                    .truncationMode(.middle)

                Spacer(minLength: 0)

                Text(String(format: "%.3f", passage.distance))
                    .font(theme.typography.monoSM)
                    .foregroundStyle(theme.colors.textSecondary)
            }

            distanceMeter

            Text(passage.excerpt)
                .font(theme.typography.caption)
                .foregroundStyle(theme.colors.textSecondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, theme.spacing.four)
    }

    private var distanceMeter: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(theme.colors.hairline)
                Capsule()
                    .fill(theme.colors.accent)
                    .frame(width: geometry.size.width * CGFloat(distanceStrength))
                    .animation(.easeOut(duration: 0.2), value: distanceStrength)
            }
        }
        .frame(height: 3)
    }

    private var distanceStrength: Double {
        max(0.08, 1.0 - min(max(passage.distance, 0) / 1.2, 1.0))
    }
}

private extension TinyBrainTheme.Spacing {
    var six: CGFloat { 6 }
}
