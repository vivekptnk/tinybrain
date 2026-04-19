// ResponseParser — extracts a JSON array of indices from the model's
// search output and maps it to `[EntityID]` via the candidate corpus.
// Tolerant by design: the model may wrap the array in chatter, emit
// trailing commas, or stop early. We take what we can parse and clamp
// the rest.

import Cartographer
import Foundation

enum ResponseParser {

    /// Parse a search response into ordered, de-duplicated annotation IDs.
    static func parseSearchIndices(_ raw: String, corpus: [Annotation]) -> [EntityID] {
        guard !corpus.isEmpty else { return [] }

        let indices = extractIndices(from: raw, upperBound: corpus.count)
        var seen = Set<Int>()
        var ordered: [EntityID] = []
        ordered.reserveCapacity(indices.count)
        for index in indices {
            if seen.insert(index).inserted {
                ordered.append(corpus[index].id)
            }
        }
        return ordered
    }

    // MARK: - Private

    /// Extract all non-negative integers inside the first pair of square
    /// brackets in `raw`. Works whether the model returns `[0, 2, 1]`,
    /// `Answer: [0,2,1]`, or `[0, 2, 1,]`. If no brackets are present we
    /// scan the whole string for bare integers as a fallback.
    private static func extractIndices(from raw: String, upperBound: Int) -> [Int] {
        let body = firstBracketedSubstring(in: raw) ?? raw
        var result: [Int] = []
        var current = ""
        for scalar in body.unicodeScalars {
            if CharacterSet.decimalDigits.contains(scalar) {
                current.unicodeScalars.append(scalar)
                continue
            }
            if let parsed = Int(current), parsed >= 0 && parsed < upperBound {
                result.append(parsed)
            }
            current.removeAll(keepingCapacity: true)
        }
        if let parsed = Int(current), parsed >= 0 && parsed < upperBound {
            result.append(parsed)
        }
        return result
    }

    private static func firstBracketedSubstring(in raw: String) -> String? {
        guard
            let open = raw.firstIndex(of: "["),
            let close = raw[open...].firstIndex(of: "]")
        else {
            return nil
        }
        let start = raw.index(after: open)
        guard start <= close else { return nil }
        return String(raw[start..<close])
    }
}
