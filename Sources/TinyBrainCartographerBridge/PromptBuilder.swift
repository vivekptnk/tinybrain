// PromptBuilder — shapes the prompts that `TinyBrainSmartService` sends to
// its `InferenceBackend`. Deliberately deterministic and plain-text so tests
// can compare prompts byte-for-byte.
//
// Search returns a JSON array of zero-based corpus indices: small, easy to
// parse, and caps output tokens tightly. Summarize asks for three short
// sentences of prose.

import Cartographer
import Foundation

enum PromptBuilder {

    // MARK: - Search

    static func search(query: String, corpus: [Annotation]) -> String {
        var output = """
        You are a relevance ranker for geographic map annotations. \
        Return ONLY a JSON array of the indices of the annotations that match \
        the query, ordered most-relevant first. Omit indices that do not \
        match. Do not write anything besides the JSON array.

        Query: \(query)

        Annotations:

        """
        for (index, annotation) in corpus.enumerated() {
            output.append("\(index): \(render(annotation))\n")
        }
        output.append("\nAnswer:")
        return output
    }

    // MARK: - Summarize

    static func summarize(annotations: [Annotation]) -> String {
        var output = """
        Summarize the following map annotations in no more than three short \
        sentences. Do not use lists or markdown. Focus on what a user would \
        want to know about this region.

        Annotations:

        """
        for annotation in annotations {
            output.append("- \(render(annotation))\n")
        }
        output.append("\nSummary:")
        return output
    }

    // MARK: - Helpers

    private static func render(_ annotation: Annotation) -> String {
        var parts: [String] = []
        let title = annotation.title.trimmingCharacters(in: .whitespacesAndNewlines)
        if !title.isEmpty { parts.append(title) }
        let body = annotation.body.trimmingCharacters(in: .whitespacesAndNewlines)
        if !body.isEmpty { parts.append(body) }
        if !annotation.metadata.isEmpty {
            let sortedValues = annotation.metadata
                .sorted(by: { $0.key < $1.key })
                .map { "\($0.key)=\($0.value)" }
                .joined(separator: ", ")
            parts.append(sortedValues)
        }
        let rendered = parts.joined(separator: " — ")
        return rendered.isEmpty ? "[untitled \(annotation.type.rawValue)]" : rendered
    }
}
