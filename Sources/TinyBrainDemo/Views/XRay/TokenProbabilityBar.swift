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

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            sectionHeader("Top Candidates")

            if candidates.isEmpty {
                Text("Waiting for generation...")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(candidates.prefix(8)) { candidate in
                    candidateRow(candidate)
                }
            }
        }
    }

    @ViewBuilder
    private func candidateRow(_ candidate: TokenCandidate) -> some View {
        HStack(spacing: 6) {
            // Token text
            Text(tokenLabel(candidate.tokenId))
                .font(.system(.caption, design: .monospaced))
                .frame(width: 70, alignment: .trailing)
                .lineLimit(1)

            // Probability bar
            GeometryReader { geo in
                RoundedRectangle(cornerRadius: 2)
                    .fill(barColor(for: candidate.probability))
                    .frame(width: max(2, geo.size.width * CGFloat(candidate.probability)))
            }
            .frame(height: 14)

            // Probability value
            Text(String(format: "%.1f%%", candidate.probability * 100))
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.secondary)
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

    private func barColor(for probability: Float) -> Color {
        if probability > 0.5 {
            return .green.opacity(0.8)
        } else if probability > 0.2 {
            return .blue.opacity(0.7)
        } else if probability > 0.05 {
            return .orange.opacity(0.6)
        } else {
            return .gray.opacity(0.4)
        }
    }
}

// MARK: - Shared Section Header

func sectionHeader(_ title: String) -> some View {
    Text(title.uppercased())
        .font(.system(.caption2, design: .rounded, weight: .semibold))
        .foregroundStyle(.secondary)
        .tracking(0.5)
}
