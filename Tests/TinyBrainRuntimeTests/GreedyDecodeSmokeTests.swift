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
    func testBatchedPrefillMatchesSerialGreedyAndReducesPromptMatmuls() async throws {
        // WHAT: Batched prefill must be an additive fast path for prompt processing.
        // WHY: The decode loop is already verified; prefill may only change how the
        // prompt is processed before the first generated token.
        // HOW: Compare greedy token IDs against the serial env fallback, then prove
        // the default path performs fewer quantized matmul calls for a multi-token
        // prompt by using the CPU streaming counters.

        let previousBackend = TinyBrainBackend.metalBackend
        TinyBrainBackend.metalBackend = nil
        defer {
            TinyBrainBackend.metalBackend = previousBackend
            unsetenv("TINYBRAIN_DISABLE_BATCHED_PREFILL")
        }

        let modelConfig = ModelConfig(
            numLayers: 2,
            hiddenDim: 32,
            numHeads: 4,
            vocabSize: 128,
            maxSeqLen: 64,
            numKVHeads: 2,
            intermediateDim: 64,
            ropeTheta: 10000.0
        )
        let weights = ModelWeights.makeToyModel(config: modelConfig, seed: 0xBADC0DE)
        let prompt = [1, 7, 11, 13, 17, 19]
        let generation = GenerationConfig(
            maxTokens: 6,
            sampler: SamplerConfig(temperature: 0.0, topK: 1),
            stopTokens: []
        )

        setenv("TINYBRAIN_DISABLE_BATCHED_PREFILL", "1", 1)
        QuantizedMatmulStats.reset()
        let serialRunner = ModelRunner(weights: weights)
        let serialTokens = try await collectTokens(runner: serialRunner, prompt: prompt, config: generation)
        let serialMatmuls = QuantizedMatmulStats.streamingINT8Count + QuantizedMatmulStats.streamingINT4Count

        unsetenv("TINYBRAIN_DISABLE_BATCHED_PREFILL")
        QuantizedMatmulStats.reset()
        let batchedRunner = ModelRunner(weights: weights)
        let batchedTokens = try await collectTokens(runner: batchedRunner, prompt: prompt, config: generation)
        let batchedMatmuls = QuantizedMatmulStats.streamingINT8Count + QuantizedMatmulStats.streamingINT4Count

        XCTAssertEqual(batchedTokens, serialTokens,
                       "Batched prefill must preserve greedy token-id sequence")
        XCTAssertEqual(batchedRunner.currentPosition, prompt.count + generation.maxTokens - 1,
                       "Prefill should leave decode positioned exactly after the prompt")
        XCTAssertLessThan(batchedMatmuls, serialMatmuls,
                          "Default prefill should batch prompt matmuls instead of replaying serial step()")
    }

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

            for prompt in prompts {
                let promptIds = model.promptStyle.encode(prompt: prompt, tokenizer: tokenizer)
                let generation = GenerationConfig(
                    maxTokens: 48,
                    sampler: SamplerConfig(temperature: 0.0, topK: 1),
                    stopTokens: []
                )

                setenv("TINYBRAIN_DISABLE_BATCHED_PREFILL", "1", 1)
                let serial = try await timedGeneration(
                    runner: ModelRunner(weights: weights),
                    prompt: promptIds,
                    config: generation
                )

                unsetenv("TINYBRAIN_DISABLE_BATCHED_PREFILL")
                let batched = try await timedGeneration(
                    runner: ModelRunner(weights: weights),
                    prompt: promptIds,
                    config: generation
                )

                XCTAssertEqual(serial.tokens.count, 48)
                XCTAssertEqual(batched.tokens.count, 48)
                XCTAssertEqual(batched.tokens, serial.tokens,
                               "Batched prefill changed greedy output for \(model.name), prompt \(prompt.debugDescription)")
                print("""
                TINYBRAIN_GREEDY_SMOKE model=\(model.name) prompt=\(prompt.debugDescription) serial_tps=\(String(format: "%.3f", serial.tokensPerSecond)) batched_tps=\(String(format: "%.3f", batched.tokensPerSecond)) serial_token_ids=\(serial.tokens) batched_token_ids=\(batched.tokens)
                """)
            }
        }

        unsetenv("TINYBRAIN_DISABLE_BATCHED_PREFILL")
    }

    func testBatchedPrefillLongPromptTimeToFirstToken() async throws {
        guard ProcessInfo.processInfo.environment["TINYBRAIN_RUN_QWEN_SMOKE"] == "1" else {
            throw XCTSkip("Set TINYBRAIN_RUN_QWEN_SMOKE=1 to run real-model prefill benchmark")
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

        let basePrompt = """
        You are a concise assistant. Read the following context and answer briefly.
        Context: TinyBrain is a Swift-native on-device language model runtime for Apple Silicon.
        It is designed to be transparent, educational, and fast. The benchmark should make the
        prompt long enough that prefill dominates time to first token while decode remains unchanged.
        """

        for model in cases {
            guard FileManager.default.fileExists(atPath: resolveProjectPath(model.modelPath)) else {
                throw XCTSkip("\(model.modelPath) not available")
            }
            guard FileManager.default.fileExists(atPath: resolveProjectPath(model.tokenizerPath)) else {
                throw XCTSkip("\(model.tokenizerPath) not available")
            }

            let weights = try ModelLoader.load(from: model.modelPath)
            let tokenizer = try TokenizerLoader.loadHuggingFace(from: resolveProjectPath(model.tokenizerPath))
            var prompt = basePrompt
            var promptIds = model.promptStyle.encode(prompt: prompt, tokenizer: tokenizer)
            while promptIds.count < 200 {
                prompt += "\n" + basePrompt
                promptIds = model.promptStyle.encode(prompt: prompt, tokenizer: tokenizer)
            }

            let generation = GenerationConfig(
                maxTokens: 48,
                sampler: SamplerConfig(temperature: 0.0, topK: 1),
                stopTokens: []
            )

            setenv("TINYBRAIN_DISABLE_BATCHED_PREFILL", "1", 1)
            let serial = try await timedGeneration(
                runner: ModelRunner(weights: weights),
                prompt: promptIds,
                config: generation,
                captureFirstToken: true
            )

            unsetenv("TINYBRAIN_DISABLE_BATCHED_PREFILL")
            let batched = try await timedGeneration(
                runner: ModelRunner(weights: weights),
                prompt: promptIds,
                config: generation,
                captureFirstToken: true
            )

            XCTAssertEqual(batched.tokens, serial.tokens,
                           "Long-prompt batched prefill changed greedy output for \(model.name)")
            print("""
            TINYBRAIN_PREFILL_BENCH model=\(model.name) prompt_tokens=\(promptIds.count) serial_ttft_ms=\(String(format: "%.3f", serial.firstTokenMilliseconds ?? -1)) batched_ttft_ms=\(String(format: "%.3f", batched.firstTokenMilliseconds ?? -1)) serial_tps=\(String(format: "%.3f", serial.tokensPerSecond)) batched_tps=\(String(format: "%.3f", batched.tokensPerSecond)) token_ids=\(batched.tokens)
            """)
        }

        unsetenv("TINYBRAIN_DISABLE_BATCHED_PREFILL")
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

    private struct TimedGeneration {
        let tokens: [Int]
        let elapsed: TimeInterval
        let firstTokenMilliseconds: Double?

        var tokensPerSecond: Double {
            elapsed > 0 ? Double(tokens.count) / elapsed : 0
        }
    }

    private func collectTokens(
        runner: ModelRunner,
        prompt: [Int],
        config: GenerationConfig
    ) async throws -> [Int] {
        var generated: [Int] = []
        for try await output in runner.generateStream(prompt: prompt, config: config) {
            generated.append(output.tokenId)
        }
        return generated
    }

    private func timedGeneration(
        runner: ModelRunner,
        prompt: [Int],
        config: GenerationConfig,
        captureFirstToken: Bool = false
    ) async throws -> TimedGeneration {
        var generated: [Int] = []
        let start = Date()
        var firstTokenMilliseconds: Double?
        for try await output in runner.generateStream(prompt: prompt, config: config) {
            if captureFirstToken && firstTokenMilliseconds == nil {
                firstTokenMilliseconds = Date().timeIntervalSince(start) * 1000
            }
            generated.append(output.tokenId)
        }
        return TimedGeneration(
            tokens: generated,
            elapsed: Date().timeIntervalSince(start),
            firstTokenMilliseconds: firstTokenMilliseconds
        )
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
