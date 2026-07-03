/// Token Probability Bar Chart
///
/// **TB-010: X-Ray Mode**
///
/// Shows the top-K candidate tokens with horizontal probability bars.
/// Animates smoothly as new tokens are generated.

import SwiftUI

struct TokenProbabilityBar: View {
    let candidates: [TokenCandidate]
    let tokenDecoder: ((Int) -> String)?
    private let theme = TinyBrainTheme.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader("Top Candidates")

            if candidates.isEmpty {
                Text("Waiting for generation…")
                    .font(theme.typography.caption)
                    .foregroundStyle(theme.colors.textSecondary)
            } else {
                ForEach(Array(candidates.prefix(8).enumerated()), id: \.element.id) { index, candidate in
                    candidateRow(candidate, isTopCandidate: index == 0)
                }
            }
        }
    }

    @ViewBuilder
    private func candidateRow(_ candidate: TokenCandidate, isTopCandidate: Bool) -> some View {
        HStack(spacing: 6) {
            Text(tokenLabel(candidate.tokenId))
                .font(theme.typography.monoSM)
                .foregroundStyle(theme.colors.textPrimary)
                .frame(width: 70, alignment: .trailing)
                .lineLimit(1)

            GeometryReader { geo in
                RoundedRectangle(cornerRadius: 2)
                    .fill(barColor(for: candidate.probability, isTopCandidate: isTopCandidate))
                    .frame(width: max(2, geo.size.width * CGFloat(candidate.probability)))
            }
            .frame(height: 14)

            Text(String(format: "%.1f%%", candidate.probability * 100))
                .font(theme.typography.monoSM)
                .foregroundStyle(theme.colors.textSecondary)
                .frame(width: 42, alignment: .trailing)
        }
        .frame(height: 18)
    }

    private func tokenLabel(_ tokenId: Int) -> String {
        if let decode = tokenDecoder {
            let text = decode(tokenId)
            // Clean up whitespace tokens for display
            let display = text.replacingOccurrences(of: "\n", with: "\\n")
                              .replacingOccurrences(of: "\t", with: "\\t")
            return display.isEmpty ? "<\(tokenId)>" : display
        }
        return "[\(tokenId)]"
    }

    private func barColor(for probability: Float, isTopCandidate: Bool) -> Color {
        if isTopCandidate {
            return theme.colors.accent
        }
        let p = min(max(Double(probability), 0), 1)
        return theme.colors.accent.opacity(0.24 + p * 0.56)
    }
}

// MARK: - Shared Section Header

func sectionHeader(_ title: String) -> some View {
    let theme = TinyBrainTheme.shared
    return Text(title.uppercased())
        .font(theme.typography.overline)
        .foregroundStyle(theme.colors.textSecondary)
        .tracking(0.6)
}
