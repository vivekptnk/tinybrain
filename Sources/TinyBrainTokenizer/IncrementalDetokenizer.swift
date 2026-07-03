import Foundation

/// Incrementally decodes generated token IDs without losing tokenizer-owned
/// word-boundary context.
///
/// Some tokenizers, especially SentencePiece-style BPE tokenizers, only reveal
/// spaces when a sequence is decoded together. Decoding `[token]` one ID at a
/// time can turn "Hello world" into "Helloworld". This helper accumulates the
/// IDs, decodes the full sequence, and emits only the stable suffix delta that
/// has not already been shown.
///
/// Known edges: trailing U+FFFD holdback and explicit end-of-stream flushing
/// are still caller-owned. The current implementation intentionally preserves
/// the existing RAG streaming behavior.
public struct IncrementalDetokenizer {
    private let tokenizer: any Tokenizer
    private var tokenIDs: [Int] = []
    private var emittedText = ""

    /// Full decoded text for all token IDs appended so far.
    public private(set) var decodedText = ""

    /// Creates an incremental detokenizer backed by `tokenizer`.
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Appends one token ID and returns the new decoded text delta, if stable.
    ///
    /// If the tokenizer rewrites previously emitted text, `nil` is returned and
    /// `decodedText` still reflects the latest full decode for final consumers.
    public mutating func append(_ tokenID: Int) -> String? {
        tokenIDs.append(tokenID)
        decodedText = tokenizer.decode(tokenIDs)

        guard decodedText.hasPrefix(emittedText) else {
            return nil
        }

        let delta = String(decodedText.dropFirst(emittedText.count))
        emittedText = decodedText
        return delta.isEmpty ? nil : delta
    }
}
