import Foundation
import TinyBrainTokenizer

/// Controls token-aware document chunking for retrieval.
public struct ChunkingConfig: Equatable, Sendable {
    /// Target token count for each emitted chunk.
    public var targetTokens: Int

    /// Number of tokens to carry from one chunk into the next.
    public var overlapTokens: Int

    /// Whether paragraph boundaries should be preferred before sentence boundaries.
    public var respectParagraphs: Bool

    /// Creates a chunking configuration.
    ///
    /// - Precondition: `targetTokens` must be positive, `overlapTokens` must be
    ///   non-negative, and `overlapTokens < targetTokens`.
    public init(
        targetTokens: Int = 128,
        overlapTokens: Int = 24,
        respectParagraphs: Bool = true
    ) {
        precondition(targetTokens > 0, "targetTokens must be positive")
        precondition(overlapTokens >= 0, "overlapTokens must be non-negative")
        precondition(overlapTokens < targetTokens, "overlapTokens must be smaller than targetTokens")
        self.targetTokens = targetTokens
        self.overlapTokens = overlapTokens
        self.respectParagraphs = respectParagraphs
    }
}

/// A token-bounded passage extracted from a source document.
public struct DocumentChunk: Equatable, Codable, Sendable {
    /// Chunk text stored with the retrieval index metadata.
    public let text: String

    /// Original source path supplied by the caller.
    public let sourcePath: String

    /// Token window represented by this chunk, including overlap with neighbors.
    public let tokenRange: Range<Int>

    /// Zero-based chunk index within the source document.
    public let ordinal: Int

    /// Creates a document chunk.
    public init(text: String, sourcePath: String, tokenRange: Range<Int>, ordinal: Int) {
        self.text = text
        self.sourcePath = sourcePath
        self.tokenRange = tokenRange
        self.ordinal = ordinal
    }
}

/// Splits documents into token-measured chunks for retrieval indexing.
public struct DocumentChunker {
    private let tokenizer: any Tokenizer
    private let config: ChunkingConfig

    /// Creates a chunker backed by the same tokenizer used for prompt budgeting.
    public init(tokenizer: any Tokenizer, config: ChunkingConfig = .init()) {
        precondition(config.overlapTokens < config.targetTokens, "overlapTokens must be smaller than targetTokens")
        self.tokenizer = tokenizer
        self.config = config
    }

    /// Splits `text` into token-budgeted chunks.
    ///
    /// Empty and whitespace-only documents return no chunks. Boundaries are
    /// chosen at paragraphs first, then sentences, and finally by hard token
    /// split when no suitable boundary exists.
    public func chunk(_ text: String, sourcePath: String) -> [DocumentChunk] {
        guard let documentRange = trimmedRange(in: text, range: text.startIndex..<text.endIndex) else {
            return []
        }

        var chunks: [DocumentChunk] = []
        var start = documentRange.lowerBound
        var nextTokenStart = 0
        var ordinal = 0

        while start < documentRange.upperBound {
            let remainingTokens = tokenCount(text, in: start..<documentRange.upperBound)
            let rawEnd: String.Index
            if remainingTokens <= config.targetTokens {
                rawEnd = documentRange.upperBound
            } else {
                let hardEnd = indexAtOrBeforeTokenOffset(
                    config.targetTokens,
                    in: text,
                    from: start,
                    to: documentRange.upperBound
                )
                rawEnd = bestBoundaryEnd(in: text, from: start, to: hardEnd) ?? hardEnd
            }

            guard let chunkRange = trimmedRange(in: text, range: start..<rawEnd) else {
                break
            }

            let chunkText = String(text[chunkRange])
            let chunkTokenCount = tokenizer.encode(chunkText).count
            let tokenEnd = nextTokenStart + chunkTokenCount
            chunks.append(DocumentChunk(
                text: chunkText,
                sourcePath: sourcePath,
                tokenRange: nextTokenStart..<tokenEnd,
                ordinal: ordinal
            ))
            ordinal += 1

            if rawEnd >= documentRange.upperBound {
                break
            }

            let overlap = min(config.overlapTokens, max(0, chunkTokenCount - 1))
            if overlap == 0 {
                start = rawEnd
                nextTokenStart = tokenEnd
                continue
            }

            let overlapStart = indexAtOrBeforeTokenOffset(
                max(1, chunkTokenCount - overlap),
                in: text,
                from: chunkRange.lowerBound,
                to: chunkRange.upperBound
            )
            let nextStart = skipLeadingWhitespace(in: text, from: overlapStart, to: documentRange.upperBound)
            if nextStart > start && nextStart < rawEnd {
                start = nextStart
                nextTokenStart = tokenEnd - overlap
            } else {
                start = rawEnd
                nextTokenStart = tokenEnd
            }
        }

        return chunks
    }

    /// Reads a UTF-8 text file and chunks its contents.
    public func chunk(fileAt url: URL) throws -> [DocumentChunk] {
        let text = try String(contentsOf: url, encoding: .utf8)
        return chunk(text, sourcePath: url.path)
    }

    private func tokenCount(_ text: String, in range: Range<String.Index>) -> Int {
        tokenizer.encode(String(text[range])).count
    }

    private func indexAtOrBeforeTokenOffset(
        _ offset: Int,
        in text: String,
        from start: String.Index,
        to upperBound: String.Index
    ) -> String.Index {
        guard offset > 0 else { return start }

        var cursor = start
        var best = start
        while cursor < upperBound {
            let next = text.index(after: cursor)
            let count = tokenCount(text, in: start..<next)
            if count > offset {
                return best == start ? next : best
            }
            best = next
            if count == offset {
                return next
            }
            cursor = next
        }
        return upperBound
    }

    private func bestBoundaryEnd(
        in text: String,
        from start: String.Index,
        to hardEnd: String.Index
    ) -> String.Index? {
        guard start < hardEnd else { return nil }
        let minimumBoundaryTokens = max(1, config.targetTokens / 2)

        if config.respectParagraphs,
           let paragraph = paragraphBoundaries(in: text, from: start, to: hardEnd)
            .last(where: { tokenCount(text, in: start..<$0) >= minimumBoundaryTokens }) {
            return paragraph
        }

        return sentenceBoundaries(in: text, from: start, to: hardEnd)
            .last(where: { tokenCount(text, in: start..<$0) >= minimumBoundaryTokens })
    }

    private func paragraphBoundaries(
        in text: String,
        from start: String.Index,
        to end: String.Index
    ) -> [String.Index] {
        var boundaries: [String.Index] = []
        var cursor = start
        while cursor < end {
            if text[cursor].isNewline {
                var lookahead = text.index(after: cursor)
                while lookahead < end && text[lookahead].isWhitespace {
                    let next = text.index(after: lookahead)
                    if text[lookahead].isNewline {
                        boundaries.append(next)
                        break
                    }
                    lookahead = next
                }
            }
            cursor = text.index(after: cursor)
        }
        return boundaries
    }

    private func sentenceBoundaries(
        in text: String,
        from start: String.Index,
        to end: String.Index
    ) -> [String.Index] {
        var boundaries: [String.Index] = []
        var cursor = start
        while cursor < end {
            let character = text[cursor]
            if character == "." || character == "!" || character == "?" {
                let next = text.index(after: cursor)
                if next == end || text[next].isWhitespace {
                    boundaries.append(next)
                }
            }
            cursor = text.index(after: cursor)
        }
        return boundaries
    }

    private func trimmedRange(in text: String, range: Range<String.Index>) -> Range<String.Index>? {
        var lower = range.lowerBound
        var upper = range.upperBound

        while lower < upper && text[lower].isWhitespace {
            lower = text.index(after: lower)
        }

        while lower < upper {
            let previous = text.index(before: upper)
            guard text[previous].isWhitespace else { break }
            upper = previous
        }

        return lower < upper ? lower..<upper : nil
    }

    private func skipLeadingWhitespace(
        in text: String,
        from start: String.Index,
        to upperBound: String.Index
    ) -> String.Index {
        var cursor = start
        while cursor < upperBound && text[cursor].isWhitespace {
            cursor = text.index(after: cursor)
        }
        return cursor
    }
}
