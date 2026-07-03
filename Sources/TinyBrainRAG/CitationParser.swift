import Foundation

/// A citation marker found in generated answer text.
public struct Citation: Equatable, Sendable {
    /// The numeric marker inside `[n]`.
    public let marker: Int

    /// The resolved passage, or `nil` when the marker is out of range.
    public let passage: RetrievedPassage?

    /// The marker range in the original answer string.
    public let range: Range<String.Index>

    /// Creates a citation value.
    public init(marker: Int, passage: RetrievedPassage?, range: Range<String.Index>) {
        self.marker = marker
        self.passage = passage
        self.range = range
    }
}

/// Extracts `[n]` citation markers and resolves them to included passages.
public struct CitationParser {
    private let passages: [RetrievedPassage]
    private let regex = try! NSRegularExpression(pattern: #"^\[(\d+)\]$"#)
    private let scanningRegex = try! NSRegularExpression(pattern: #"\[(\d+)\]"#)

    /// Creates a parser whose marker `[1]` maps to `passages[0]`.
    public init(passages: [RetrievedPassage]) {
        self.passages = passages
    }

    /// Parses all well-formed numeric citation markers in `answer`.
    public func parse(_ answer: String) -> [Citation] {
        let fullRange = NSRange(answer.startIndex..<answer.endIndex, in: answer)
        return scanningRegex.matches(in: answer, range: fullRange).compactMap { match in
            guard match.numberOfRanges == 2,
                  let markerRange = Range(match.range(at: 1), in: answer),
                  let fullMarkerRange = Range(match.range(at: 0), in: answer),
                  regex.firstMatch(
                    in: String(answer[fullMarkerRange]),
                    range: NSRange(location: 0, length: match.range(at: 0).length)
                  ) != nil,
                  let marker = Int(answer[markerRange]) else {
                return nil
            }

            let passage: RetrievedPassage?
            if marker > 0 && marker <= passages.count {
                passage = passages[marker - 1]
            } else {
                passage = nil
            }
            return Citation(marker: marker, passage: passage, range: fullMarkerRange)
        }
    }
}
