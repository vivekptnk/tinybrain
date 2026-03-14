/// KV Cache Grid Visualization
///
/// **TB-010: X-Ray Mode**
///
/// Shows cache page allocation as a grid of small squares.
/// Filled squares = allocated pages with cached K/V data.
/// Empty squares = free pages available for new tokens.

import SwiftUI

struct KVCacheGridView: View {
    let pages: [Bool]

    private let columns = Array(repeating: GridItem(.fixed(10), spacing: 2), count: 16)

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                sectionHeader("KV Cache Pages")
                Spacer()
                Text(utilizationText)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.secondary)
            }

            if pages.isEmpty {
                Text("Cache empty")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(columns: columns, spacing: 2) {
                    ForEach(Array(pages.enumerated()), id: \.offset) { _, allocated in
                        RoundedRectangle(cornerRadius: 1)
                            .fill(allocated ? Color.blue.opacity(0.7) : Color.gray.opacity(0.15))
                            .frame(width: 10, height: 10)
                    }
                }
            }
        }
    }

    private var utilizationText: String {
        guard !pages.isEmpty else { return "0%" }
        let used = pages.filter { $0 }.count
        let pct = Int(Double(used) / Double(pages.count) * 100)
        return "\(used)/\(pages.count) (\(pct)%)"
    }
}
