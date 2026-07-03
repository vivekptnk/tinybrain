import ArgumentParser
import Foundation
import NaturalLanguage
import ProximaEmbeddings
import TinyBrainRAG
import TinyBrainRuntime
import TinyBrainTokenizer

enum EmbedderChoice: String, ExpressibleByArgument {
    case nl
    case stub
}

@main
struct RAGDemo: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "tinybrain-rag",
        abstract: "Private on-device retrieval-augmented answers over local notes.",
        version: "0.3.0"
    )

    @Option(name: .long, help: "Folder of .txt/.md files to index recursively.")
    var dir: String?

    @Option(name: .long, help: "Path to a TinyBrain .tbf model.")
    var model: String = "Models/tinyllama-1.1b-int8.tbf"

    @Option(name: .long, help: "Embedding provider: nl or stub.")
    var embedder: EmbedderChoice = .nl

    @Option(name: [.customLong("question", withSingleDash: true)], help: "Question to answer. Repeat for scripted mode.")
    var questions: [String] = []

    @Option(name: .long, help: "Number of passages to retrieve.")
    var k: Int = 4

    @Option(name: .long, help: "Maximum answer tokens.")
    var tokens: Int = 32

    @Flag(name: .long, help: "Run retrieval only, without loading or calling a model.")
    var noGenerate: Bool = false

    func run() async throws {
        printBanner()

        let resolvedModel = resolvePath(model)
        if !noGenerate && !FileManager.default.fileExists(atPath: resolvedModel) {
            printStartup(
                embedderSummary: embedderStartupSummary,
                modelSummary: "missing: \(resolvedModel)",
                quantizationSummary: "unavailable"
            )
            printMissingModel(path: resolvedModel)
            return
        }

        let tokenizer = TokenizerLoader.loadBestAvailable()
        let (index, embedderSummary) = try makeIndex()
        let (generator, modelSummary, quantizationSummary) = try makeGenerator(modelPath: resolvedModel)
        printStartup(
            embedderSummary: embedderSummary,
            modelSummary: modelSummary,
            quantizationSummary: quantizationSummary
        )

        let boundedK = max(1, k)
        let engine = RAGEngine(
            index: index,
            generator: generator,
            tokenizer: tokenizer,
            retrievalK: boundedK,
            generationConfig: GenerationConfig(
                maxTokens: max(1, tokens),
                sampler: SamplerConfig(temperature: 0.7, topK: 40, seed: 42)
            )
        )

        let chunks = try await indexDocuments(with: engine)
        let fileCount = Set(chunks.map(\.sourcePath)).count
        print("Indexed \(chunks.count) chunks from \(fileCount) files.\n")

        if !questions.isEmpty {
            for question in questions {
                try await answer(question, engine: engine, k: boundedK)
            }
            return
        }

        print("Ask a question about your notes. Type \"quit\" or \"exit\" to leave.\n")
        while true {
            print("Question: ", terminator: "")
            guard let line = readLine() else {
                print("\nBye.")
                break
            }

            let question = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if question.isEmpty {
                continue
            }
            let lowered = question.lowercased()
            if lowered == "quit" || lowered == "exit" {
                print("\nBye.")
                break
            }

            try await answer(question, engine: engine, k: boundedK)
        }
    }

    private var embedderStartupSummary: String {
        switch embedder {
        case .nl:
            return "NLEmbeddingProvider (.english)"
        case .stub:
            return "DeterministicStubEmbedder"
        }
    }

    private func makeIndex() throws -> (RAGIndex, String) {
        switch embedder {
        case .nl:
            let provider = try NLEmbeddingProvider(language: .english)
            return (
                RAGIndex(embedder: provider),
                "NLEmbeddingProvider (.english, \(provider.dimension)d)"
            )
        case .stub:
            let provider = DeterministicStubEmbedder(dimension: 64, seed: 42)
            return (
                RAGIndex(embedder: provider),
                "DeterministicStubEmbedder (64d, seed 42)"
            )
        }
    }

    private func makeGenerator(modelPath: String) throws -> (any AnswerGenerator, String, String) {
        if noGenerate {
            return (
                EmptyAnswerGenerator(),
                "retrieval only (--no-generate)",
                "not loaded"
            )
        }

        let weights = try ModelWeights.load(from: modelPath)
        let runner = ModelRunner(weights: weights)
        let modelName = "\(URL(fileURLWithPath: modelPath).lastPathComponent) (\(weights.config.numLayers) layers, \(weights.config.hiddenDim)d, context \(weights.config.maxSeqLen))"
        return (
            ModelRunnerGenerator(runner: runner),
            "TinyBrain ModelRunner: \(modelName)",
            quantizationSummary(for: weights)
        )
    }

    private func indexDocuments(with engine: RAGEngine) async throws -> [DocumentChunk] {
        if let dir {
            let folder = URL(fileURLWithPath: resolvePath(dir))
            return try await engine.index(
                folderAt: folder,
                chunkingConfig: ChunkingConfig(targetTokens: 128, overlapTokens: 24)
            )
        }

        return try await engine.index(
            documents: sampleNotes,
            chunkingConfig: ChunkingConfig(targetTokens: 128, overlapTokens: 24)
        )
    }

    private func answer(_ question: String, engine: RAGEngine, k: Int) async throws {
        print("Question: \(question)")

        if noGenerate {
            let start = DispatchTime.now()
            let passages = try await engine.retrieve(question, k: k)
            printRetrieved(passages, elapsedMilliseconds(from: start))
            print("\nAnswer generation disabled (--no-generate).\n")
            return
        }

        let start = DispatchTime.now()
        var answerStarted = false
        for try await event in engine.answerStream(question) {
            switch event {
            case .passages(let passages):
                printRetrieved(passages, elapsedMilliseconds(from: start))
            case .token(let token):
                if !answerStarted {
                    print("\nAnswer: ", terminator: "")
                    answerStarted = true
                }
                print(token, terminator: "")
            case .done(let citations):
                if !answerStarted {
                    print("\nAnswer: ", terminator: "")
                }
                print("\n")
                printSourceMapping(citations)
            }
        }
    }

    private func printRetrieved(_ passages: [RetrievedPassage], _ retrievalMs: Double) {
        print("\nRetrieved passages (lower distance = more relevant, \(String(format: "%.1f", retrievalMs)) ms):")
        if passages.isEmpty {
            print("  none")
            return
        }

        for (index, passage) in passages.enumerated() {
            let distance = String(format: "%.3f", passage.distance)
            let text = passage.chunk.text.replacingOccurrences(of: "\n", with: " ")
            print("  [\(index + 1)] \(distance)  \(passage.chunk.sourcePath)#\(passage.chunk.ordinal)")
            print("      \(text)")
        }
    }

    private func printSourceMapping(_ citations: [Citation]) {
        print("Sources:")
        guard !citations.isEmpty else {
            print("  No citations emitted.")
            print("")
            return
        }

        for citation in citations {
            if let passage = citation.passage {
                print("  [\(citation.marker)] \(passage.chunk.sourcePath)#\(passage.chunk.ordinal)")
            } else {
                print("  [\(citation.marker)] unresolved")
            }
        }
        print("")
    }

    private func printStartup(
        embedderSummary: String,
        modelSummary: String,
        quantizationSummary: String
    ) {
        print("Embedder: \(embedderSummary)")
        print("Model: \(modelSummary)")
        print("Quantization: \(quantizationSummary)\n")
    }

    private func printBanner() {
        print("""

        +--------------------------------------------------+
        | TinyBrain RAG                                    |
        | Private retrieval and generation over your notes. |
        +--------------------------------------------------+

        """)
    }

    private func printMissingModel(path: String) {
        print("""
        Model file not found: \(path)
        Expected path: Models/tinyllama-1.1b-int8.tbf

        Convert TinyLlama to TinyBrain format with:
        python Scripts/convert_model.py \\
          --input Models/tinyllama-raw/model.safetensors \\
          --output Models/tinyllama-1.1b-int8.tbf \\
          --auto-config

        Exiting without generation.
        """)
    }

    private func quantizationSummary(for weights: ModelWeights) -> String {
        var labels: [String] = []
        for layer in weights.layers {
            labels.append(precisionLabel(layer.attention.query.weights.precision))
            labels.append(precisionLabel(layer.attention.key.weights.precision))
            labels.append(precisionLabel(layer.attention.value.weights.precision))
            labels.append(precisionLabel(layer.attention.output.weights.precision))
            if let gate = layer.feedForward.gate {
                labels.append(precisionLabel(gate.weights.precision))
            }
            labels.append(precisionLabel(layer.feedForward.up.weights.precision))
            labels.append(precisionLabel(layer.feedForward.down.weights.precision))
        }
        labels.append(precisionLabel(weights.output.weights.precision))

        let unique = Array(Set(labels)).sorted()
        return unique.joined(separator: "+")
    }

    private func precisionLabel(_ precision: QuantizationPrecision) -> String {
        switch precision {
        case .int8:
            return "INT8"
        case .int4:
            return "INT4"
        }
    }

    private func elapsedMilliseconds(from start: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000
    }

    private func resolvePath(_ path: String) -> String {
        if path.hasPrefix("/") {
            return path
        }

        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: path) {
            return URL(fileURLWithPath: path).path
        }

        if let root = projectRoot() {
            return URL(fileURLWithPath: root).appendingPathComponent(path).path
        }

        return URL(fileURLWithPath: path).path
    }

    private func projectRoot() -> String? {
        var current = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let packagePath = URL(fileURLWithPath: current).appendingPathComponent("Package.swift").path
            if FileManager.default.fileExists(atPath: packagePath) {
                return current
            }

            let parent = URL(fileURLWithPath: current).deletingLastPathComponent().path
            if parent == current || parent == "/" {
                break
            }
            current = parent
        }
        return nil
    }
}

private struct EmptyAnswerGenerator: AnswerGenerator {
    func generateStream(
        prompt: [Int],
        config: GenerationConfig
    ) -> AsyncThrowingStream<TokenOutput, Error> {
        AsyncThrowingStream { continuation in
            continuation.finish()
        }
    }
}
