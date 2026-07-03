import Foundation
import TinyBrainTokenizer

/// Token budget controls for retrieval-augmented prompts.
public struct PromptBudget: Equatable, Sendable {
    /// Maximum model context window in tokens.
    public var contextWindow: Int

    /// Tokens reserved for generation after the prompt.
    public var generationHeadroom: Int

    /// Creates a prompt budget.
    ///
    /// - Precondition: `contextWindow` must be positive and
    ///   `generationHeadroom` must be non-negative.
    public init(contextWindow: Int = 2_048, generationHeadroom: Int = 256) {
        precondition(contextWindow > 0, "contextWindow must be positive")
        precondition(generationHeadroom >= 0, "generationHeadroom must be non-negative")
        self.contextWindow = contextWindow
        self.generationHeadroom = generationHeadroom
    }
}

/// Prompt envelope used when assembling retrieval-augmented instructions.
public enum PromptTemplate: String, CaseIterable, Equatable, Sendable {
    /// Preserve the original raw RAG prompt format.
    case none

    /// TinyLlama/Zephyr chat prompt format with a BOS token for generation.
    case zephyr

    var generationPrefixTokenIDs: [Int] {
        switch self {
        case .none:
            return []
        case .zephyr:
            return [1]
        }
    }
}

/// Builds deterministic numbered-passage prompts under a token budget.
public struct RAGPromptBuilder {
    private let tokenizer: any Tokenizer
    private let budget: PromptBudget
    private let template: PromptTemplate

    /// Prompt template used for rendered prompts and budget accounting.
    public var promptTemplate: PromptTemplate {
        template
    }

    /// Maximum answer tokens reserved by this prompt budget.
    public var generationHeadroom: Int {
        budget.generationHeadroom
    }

    /// Creates a prompt builder using the model tokenizer for all token counts.
    public init(
        tokenizer: any Tokenizer,
        budget: PromptBudget = .init(),
        template: PromptTemplate = .none
    ) {
        self.tokenizer = tokenizer
        self.budget = budget
        self.template = template
    }

    /// Builds a prompt and returns the passages that fit inside the budget.
    ///
    /// Passages are sorted by retrieval rank, numbered from `[1]`, and never
    /// truncated. If the next passage would exceed the available prompt window,
    /// it and all lower-ranked passages are dropped.
    public func build(
        question: String,
        passages: [RetrievedPassage]
    ) -> (prompt: String, included: [RetrievedPassage]) {
        let ordered = passages.sorted {
            if $0.rank == $1.rank {
                return $0.chunk.ordinal < $1.chunk.ordinal
            }
            return $0.rank < $1.rank
        }
        let promptLimit = max(0, budget.contextWindow - budget.generationHeadroom)

        var included: [RetrievedPassage] = []
        for passage in ordered {
            let candidate = included + [passage]
            let prompt = renderPrompt(question: question, passages: candidate)
            if tokenIDs(for: prompt).count <= promptLimit {
                included = candidate
            } else {
                break
            }
        }

        return (renderPrompt(question: question, passages: included), included)
    }

    func tokenIDs(for prompt: String) -> [Int] {
        template.generationPrefixTokenIDs + tokenizer.encode(prompt)
    }

    private func renderPrompt(question: String, passages: [RetrievedPassage]) -> String {
        switch template {
        case .none:
            return renderRawPrompt(question: question, passages: passages)
        case .zephyr:
            return renderZephyrPrompt(question: question, passages: passages)
        }
    }

    private func renderPassageBlock(passages: [RetrievedPassage]) -> String {
        passages.enumerated()
            .map { index, passage in "[\(index + 1)] \(passage.chunk.text)" }
            .joined(separator: "\n\n")
    }

    private func renderRawPrompt(question: String, passages: [RetrievedPassage]) -> String {
        let passageBlock = renderPassageBlock(passages: passages)

        return """
        You are TinyBrain Chat. Answer only from the numbered passages below. Do not use outside knowledge. If the passages do not contain the answer, say you do not know.

        Question:
        \(question)

        Passages:
        \(passageBlock)

        Cite every claim with its passage marker. For example, cite like: The steep time is 16 hours [1].

        Answer:
        """
    }

    private func renderZephyrPrompt(question: String, passages: [RetrievedPassage]) -> String {
        let passageBlock = renderPassageBlock(passages: passages)

        return """
        <|system|>
        You are TinyBrain Chat. Answer only from the numbered passages below. Do not use outside knowledge. If the passages do not contain the answer, say you do not know.

        Passages:
        \(passageBlock)

        Cite every claim with its passage marker. For example, cite like: The steep time is 16 hours [1].</s>
        <|user|>
        \(question)</s>
        <|assistant|>
        """ + "\n"
    }
}
