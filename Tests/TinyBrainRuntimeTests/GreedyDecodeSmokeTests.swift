import Foundation
import XCTest
@testable import TinyBrainRuntime
import TinyBrainTokenizer
import TinyBrainMetal

/// Env-gated real-model decode smoke for perf refactors.
///
/// Enable with `TINYBRAIN_RUN_QWEN_SMOKE=1`. The test prints exact generated
/// token ids and decode throughput for Qwen, TinyLlama, and Gemma INT8 models.
final class GreedyDecodeSmokeTests: XCTestCase {
    func testGreedyDecodeTokenIdsAndThroughput() async throws {
        guard ProcessInfo.processInfo.environment["TINYBRAIN_RUN_QWEN_SMOKE"] == "1" else {
            throw XCTSkip("Set TINYBRAIN_RUN_QWEN_SMOKE=1 to run real-model decode smoke")
        }

        if MetalBackend.isAvailable {
            TinyBrainBackend.metalBackend = try? MetalBackend()
        }

        let cases = [
            SmokeModel(
                name: "qwen2.5-1.5b-int8",
                modelPath: "Models/qwen2.5-1.5b-int8.tbf",
                tokenizerPath: "Models/qwen2.5-1.5b-raw/tokenizer.json",
                promptStyle: .qwenChatML
            ),
            SmokeModel(
                name: "tinyllama-1.1b-int8",
                modelPath: "Models/tinyllama-1.1b-int8.tbf",
                tokenizerPath: "Models/tinyllama-raw/tokenizer.json",
                promptStyle: .prependBOS
            ),
            SmokeModel(
                name: "gemma-2b-int8",
                modelPath: "Models/gemma-2b-int8.tbf",
                tokenizerPath: "Models/gemma-2b-raw/tokenizer.json",
                promptStyle: .prependBOS
            )
        ]

        let prompts = [
            "The capital of France is",
            "1 + 1 =",
            "Once upon a time"
        ]

        for model in cases {
            guard FileManager.default.fileExists(atPath: resolveProjectPath(model.modelPath)) else {
                throw XCTSkip("\(model.modelPath) not available")
            }
            guard FileManager.default.fileExists(atPath: resolveProjectPath(model.tokenizerPath)) else {
                throw XCTSkip("\(model.tokenizerPath) not available")
            }

            let weights = try ModelLoader.load(from: model.modelPath)
            let tokenizer = try TokenizerLoader.loadHuggingFace(from: resolveProjectPath(model.tokenizerPath))
            let runner = ModelRunner(weights: weights)

            for prompt in prompts {
                runner.reset()
                let promptIds = model.promptStyle.encode(prompt: prompt, tokenizer: tokenizer)
                let generation = GenerationConfig(
                    maxTokens: 48,
                    sampler: SamplerConfig(temperature: 0.0, topK: 1),
                    stopTokens: []
                )

                var generated: [Int] = []
                let start = Date()
                for try await output in runner.generateStream(prompt: promptIds, config: generation) {
                    generated.append(output.tokenId)
                }
                let elapsed = Date().timeIntervalSince(start)
                let tokensPerSecond = elapsed > 0 ? Double(generated.count) / elapsed : 0

                XCTAssertEqual(generated.count, 48)
                print("""
                TINYBRAIN_GREEDY_SMOKE model=\(model.name) prompt=\(prompt.debugDescription) tokens_per_second=\(String(format: "%.3f", tokensPerSecond)) token_ids=\(generated)
                """)
            }
        }
    }

    private struct SmokeModel {
        let name: String
        let modelPath: String
        let tokenizerPath: String
        let promptStyle: PromptStyle
    }

    private enum PromptStyle {
        case prependBOS
        case qwenChatML

        func encode(prompt: String, tokenizer: BPETokenizer) -> [Int] {
            switch self {
            case .prependBOS:
                return [tokenizer.bosToken] + tokenizer.encode(prompt)
            case .qwenChatML:
                let chatPrompt = """
                <|im_start|>system
                You are a concise, helpful assistant.<|im_end|>
                <|im_start|>user
                \(prompt)<|im_end|>
                <|im_start|>assistant

                """
                return tokenizer.encode(chatPrompt)
            }
        }
    }

    private func resolveProjectPath(_ path: String) -> String {
        if FileManager.default.fileExists(atPath: path) { return path }

        var dir = FileManager.default.currentDirectoryPath
        for _ in 0..<10 {
            let packagePath = (dir as NSString).appendingPathComponent("Package.swift")
            if FileManager.default.fileExists(atPath: packagePath) {
                let fullPath = (dir as NSString).appendingPathComponent(path)
                if FileManager.default.fileExists(atPath: fullPath) { return fullPath }
            }
            dir = (dir as NSString).deletingLastPathComponent
            if dir == "/" { break }
        }
        return path
    }
}
