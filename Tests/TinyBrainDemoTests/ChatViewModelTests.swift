/// Chat View Model Tests
///
/// **TDD Phase: RED**
/// Tests define requirements for chat view model orchestration.
///
/// Tests cover:
/// - Message history management
/// - Streaming generation integration
/// - Telemetry integration
/// - Sampler configuration
/// - Error handling

import XCTest
@testable import TinyBrainDemo
@testable import TinyBrainRuntime
@testable import TinyBrainTokenizer

// Note: These tests may fail to link on Xcode 26 beta due to a SwiftUICore.tbd
// linker bug. Run with: swift test --skip TinyBrainDemoTests if affected.
// Tracked: FB15847293 (Apple Feedback)

@MainActor
final class ChatViewModelTests: XCTestCase {
    
    var viewModel: ChatViewModel!
    
    override func setUp() async throws {
        // Create toy model for testing
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        let weights = ModelWeights.makeToyModel(config: config, seed: 42)
        let runner = ModelRunner(weights: weights)
        viewModel = ChatViewModel(runner: runner)
    }
    
    override func tearDown() async throws {
        viewModel = nil
    }
    
    // MARK: - Initialization Tests
    
    func testInitialState() {
        XCTAssertEqual(viewModel.messages.count, 0, "Should start with no messages")
        XCTAssertEqual(viewModel.promptText, "", "Prompt should be empty")
        XCTAssertFalse(viewModel.isGenerating, "Should not be generating initially")
        XCTAssertNotNil(viewModel.telemetry, "Telemetry should be initialized")
        XCTAssertEqual(viewModel.activeModelName, "Toy Model", "Default runner identity should be toy")
        XCTAssertEqual(viewModel.activeQuant, .toy, "Toy runner should not be labeled INT8")
    }

    func testActiveModelIdentityCanBeSeededFromLoadedWeights() {
        let config = ModelConfig(
            numLayers: 2,
            hiddenDim: 128,
            numHeads: 4,
            vocabSize: 100,
            maxSeqLen: 256
        )
        let weights = ModelWeights.makeToyModel(config: config, seed: 7)
        let runner = ModelRunner(weights: weights)
        let seeded = ChatViewModel(
            runner: runner,
            activeModelName: "tinyllama-1.1b-int8",
            activeQuant: QuantBadge.derived(from: weights),
            activeModelPath: "/tmp/tinyllama-1.1b-int8.tbf"
        )

        XCTAssertEqual(seeded.activeModelName, "tinyllama-1.1b-int8")
        XCTAssertEqual(seeded.activeQuant, .int8)
        XCTAssertEqual(seeded.activeModelPath, "/tmp/tinyllama-1.1b-int8.tbf")
    }
    
    // MARK: - Message History Tests
    
    func testAddUserMessage() {
        viewModel.promptText = "Hello"
        viewModel.addUserMessage()
        
        XCTAssertEqual(viewModel.messages.count, 1, "Should have one message")
        XCTAssertEqual(viewModel.messages[0].role, .user, "Message should be from user")
        XCTAssertEqual(viewModel.messages[0].content, "Hello", "Content should match prompt")
        XCTAssertEqual(viewModel.promptText, "", "Prompt should be cleared after sending")
    }
    
    func testAddEmptyUserMessageDoesNothing() {
        viewModel.promptText = ""
        viewModel.addUserMessage()
        
        XCTAssertEqual(viewModel.messages.count, 0, "Empty message should not be added")
    }
    
    func testAddAssistantMessage() {
        viewModel.addAssistantMessage(content: "AI response")
        
        XCTAssertEqual(viewModel.messages.count, 1, "Should have one message")
        XCTAssertEqual(viewModel.messages[0].role, .assistant, "Message should be from assistant")
        XCTAssertEqual(viewModel.messages[0].content, "AI response", "Content should match")
    }
    
    func testMessageHistoryOrder() {
        viewModel.promptText = "First"
        viewModel.addUserMessage()
        viewModel.addAssistantMessage(content: "Response 1")
        viewModel.promptText = "Second"
        viewModel.addUserMessage()
        viewModel.addAssistantMessage(content: "Response 2")
        
        XCTAssertEqual(viewModel.messages.count, 4)
        XCTAssertEqual(viewModel.messages[0].content, "First")
        XCTAssertEqual(viewModel.messages[1].content, "Response 1")
        XCTAssertEqual(viewModel.messages[2].content, "Second")
        XCTAssertEqual(viewModel.messages[3].content, "Response 2")
    }
    
    // MARK: - Clear Conversation Tests
    
    func testClearConversation() {
        viewModel.promptText = "Test"
        viewModel.addUserMessage()
        viewModel.addAssistantMessage(content: "Response")
        
        XCTAssertEqual(viewModel.messages.count, 2)
        
        viewModel.clearConversation()
        
        XCTAssertEqual(viewModel.messages.count, 0, "Messages should be cleared")
        XCTAssertFalse(viewModel.isGenerating, "Should not be generating")
    }
    
    // MARK: - Sampler Configuration Tests
    
    func testDefaultSamplerConfig() {
        let config = viewModel.currentSamplerConfig
        
        XCTAssertGreaterThan(config.temperature, 0, "Temperature should be positive")
        XCTAssertEqual(config.temperature, 0.4, accuracy: 0.001, "Default chat sampling should be focused enough to reduce open-prompt rambling")
        XCTAssertEqual(config.repetitionPenalty, 1.2, accuracy: 0.001)
    }

    func testQwenSamplerDefaultsUseGenerationConfigValues() {
        let qwen = ChatViewModel(
            runner: ModelRunner(weights: makeArgmaxModel(argmaxToken: 1)),
            activeModelPath: "/tmp/qwen2.5-1.5b-int8.tbf"
        )

        let config = qwen.currentSamplerConfig

        XCTAssertEqual(config.temperature, 0.7, accuracy: 0.001)
        XCTAssertEqual(config.topP ?? -1, 0.8, accuracy: 0.001)
        XCTAssertEqual(config.topK, 20)
        XCTAssertEqual(config.repetitionPenalty, 1.05, accuracy: 0.001)
    }

    func testTinyLlamaSamplerDefaultsStayChatTuned() {
        let tinyLlama = ChatViewModel(
            runner: ModelRunner(weights: makeArgmaxModel(argmaxToken: 1)),
            activeModelPath: "/tmp/tinyllama-1.1b-int8.tbf"
        )

        let config = tinyLlama.currentSamplerConfig

        XCTAssertEqual(config.temperature, 0.4, accuracy: 0.001)
        XCTAssertEqual(config.topK, 40)
        XCTAssertNil(config.topP)
        XCTAssertEqual(config.repetitionPenalty, 1.2, accuracy: 0.001)
    }
    
    func testSamplerPresets() {
        // Balanced preset
        viewModel.applySamplerPreset(.balanced)
        let balanced = viewModel.currentSamplerConfig
        XCTAssertEqual(balanced.temperature, 0.7, accuracy: 0.01)
        
        // Creative preset
        viewModel.applySamplerPreset(.creative)
        let creative = viewModel.currentSamplerConfig
        XCTAssertGreaterThan(creative.temperature, 0.7, "Creative should have higher temperature")
        
        // Precise preset
        viewModel.applySamplerPreset(.precise)
        let precise = viewModel.currentSamplerConfig
        XCTAssertLessThan(precise.temperature, 0.7, "Precise should have lower temperature")
    }
    
    func testCustomSamplerSettings() {
        viewModel.temperature = 1.5
        viewModel.topK = 25
        viewModel.useTopK = true
        
        let config = viewModel.currentSamplerConfig
        
        XCTAssertEqual(config.temperature, 1.5, accuracy: 0.01)
        XCTAssertEqual(config.topK, 25)
        XCTAssertNil(config.topP, "Top-P should be nil when using Top-K")
    }
    
    // MARK: - Generation State Tests
    
    func testGenerationStateToggle() {
        XCTAssertFalse(viewModel.isGenerating)
        
        viewModel.setGenerating(true)
        XCTAssertTrue(viewModel.isGenerating)
        
        viewModel.setGenerating(false)
        XCTAssertFalse(viewModel.isGenerating)
    }

    func testGenerateStopsBeforeDisplayingEOSToken() async {
        let eosToken = 2
        let runner = ModelRunner(weights: makeArgmaxModel(argmaxToken: eosToken))
        let tokenizer = makeBoundaryTokenizer()
        viewModel = ChatViewModel(runner: runner, tokenizer: tokenizer)
        viewModel.temperature = 0
        viewModel.topK = 1
        viewModel.promptText = "hello"

        await viewModel.generate()

        XCTAssertFalse(viewModel.isGenerating, "EOS should end generation cleanly")
        XCTAssertEqual(viewModel.messages.count, 2)
        XCTAssertEqual(viewModel.messages.last?.role, .assistant)
        XCTAssertEqual(viewModel.messages.last?.content, "", "EOS must not leak into visible assistant text")
    }

    func testGenerateDetokenizesAccumulatedSentencePieceBoundaries() async {
        let tokenizer = makeBoundaryTokenizer()
        let runner = ModelRunner(
            weights: makeTransitionModel(
                defaultNextToken: 10,
                transitions: [
                    10: 11,
                    11: 12,
                    12: 2
                ]
            )
        )
        viewModel = ChatViewModel(runner: runner, tokenizer: tokenizer)
        viewModel.temperature = 0
        viewModel.topK = 1
        viewModel.promptText = "hello"

        await viewModel.generate()

        XCTAssertEqual(viewModel.messages.last?.content, "Hello world.")
        XCTAssertFalse(viewModel.messages.last?.content.contains("Helloworld") ?? true)
        XCTAssertFalse(viewModel.isGenerating)
    }

    func testZephyrChatTemplateMatchesTinyLlamaFormatExactly() {
        let messages = [Message(role: .user, content: "hi")]

        let formatted = TinyBrainChatTemplate.format(messages: messages)
        let expected = """
        <|system|>
        \(TinyBrainChatDefaults.systemPrompt)</s>
        <|user|>
        hi</s>
        <|assistant|>

        """

        XCTAssertEqual(formatted, expected)
        XCTAssertEqual(Array(formatted.utf8), Array(expected.utf8), "Template bytes must not gain stray spaces, missing newlines, or misplaced </s> markers")
    }

    func testQwenChatMLTemplateMatchesExpectedFormatExactly() {
        let messages = [Message(role: .user, content: "hi")]

        let formatted = TinyBrainQwenChatTemplate.format(
            messages: messages,
            systemPrompt: "You are TinyBrain."
        )
        let expected = """
        <|im_start|>system
        You are TinyBrain.<|im_end|>
        <|im_start|>user
        hi<|im_end|>
        <|im_start|>assistant

        """

        XCTAssertEqual(formatted, expected)
        XCTAssertEqual(Array(formatted.utf8), Array(expected.utf8), "ChatML bytes must preserve Qwen's exact role markers and newlines")
    }

    func testPromptTokenizationDoesNotPrependBOSForQwenButDoesForTinyLlama() {
        let qwenTokenizer = makeQwenChatTokenizer()
        let qwenPrompt = TinyBrainQwenChatTemplate.format(
            messages: [Message(role: .user, content: "hi")],
            systemPrompt: "You are TinyBrain."
        )

        let qwenTokens = TinyBrainPromptTokenizer.encode(
            prompt: qwenPrompt,
            tokenizer: qwenTokenizer,
            fallbackVocabSize: 200_000
        )

        XCTAssertFalse(qwenTokenizer.addsBosToken)
        XCTAssertEqual(qwenTokens.first, 151_644, "Qwen ChatML should begin with the encoded <|im_start|> token, not a prepended BOS")
        XCTAssertNotEqual(qwenTokens.first, qwenTokenizer.bosToken)

        let tinyLlamaTokenizer = makeTinyLlamaChatTokenizer()
        let tinyPrompt = TinyBrainChatTemplate.format(messages: [Message(role: .user, content: "hi")])

        let tinyTokens = TinyBrainPromptTokenizer.encode(
            prompt: tinyPrompt,
            tokenizer: tinyLlamaTokenizer,
            fallbackVocabSize: 32_000
        )

        XCTAssertTrue(tinyLlamaTokenizer.addsBosToken)
        XCTAssertEqual(tinyTokens.first, tinyLlamaTokenizer.bosToken)
    }

    func testMultiTokenStopMatcherStopsAtEncodedUserTurnBoundary() {
        let tokenizer = MarkerTokenizer()
        let userBoundary = tokenizer.encode("<|user|>")
        let stopSequences = TinyBrainChatStops.stopSequences(
            for: tokenizer,
            promptStyle: .zephyrChat,
            eosTokens: [2]
        )
        XCTAssertTrue(stopSequences.contains(userBoundary), "Chat stop sequences must be encoded with the active tokenizer")

        var matcher = StopSequenceMatcher<Int>(
            stopSequences: [userBoundary],
            tokenID: { $0 }
        )
        var emitted: [Int] = []
        var stopped = false

        tokenLoop: for token in [101] + userBoundary + [102] {
            switch matcher.append(token) {
            case .emit(let safeTokens):
                emitted.append(contentsOf: safeTokens)
            case .stop(let safeTokens):
                emitted.append(contentsOf: safeTokens)
                stopped = true
                break tokenLoop
            }
        }

        XCTAssertTrue(stopped)
        XCTAssertEqual(emitted, [101], "Boundary tokens should be consumed as the stop signal, not emitted to the visible answer")
        XCTAssertFalse(emitted.contains(userBoundary[0]))
        XCTAssertFalse(emitted.contains(userBoundary[1]))
    }

    func testQwenStopMatcherStopsAtEncodedImEndBoundary() {
        let tokenizer = QwenMarkerTokenizer()
        let imEnd = tokenizer.encode("<|im_end|>")
        let stopSequences = TinyBrainChatStops.stopSequences(
            for: tokenizer,
            promptStyle: .qwenChatML,
            eosTokens: [151_645, 151_643]
        )

        XCTAssertTrue(stopSequences.contains([151_645]))
        XCTAssertTrue(stopSequences.contains([151_643]))
        XCTAssertTrue(stopSequences.contains(imEnd), "Qwen chat stops must include the encoded <|im_end|> turn boundary")

        var matcher = StopSequenceMatcher<Int>(
            stopSequences: stopSequences,
            tokenID: { $0 }
        )
        var emitted: [Int] = []
        var stopped = false

        tokenLoop: for token in [101] + imEnd + [102] {
            switch matcher.append(token) {
            case .emit(let safeTokens):
                emitted.append(contentsOf: safeTokens)
            case .stop(let safeTokens):
                emitted.append(contentsOf: safeTokens)
                stopped = true
                break tokenLoop
            }
        }

        XCTAssertTrue(stopped)
        XCTAssertEqual(emitted, [101], "<|im_end|> should stop generation without leaking into visible assistant text")
    }
    
    // MARK: - Error Handling Tests
    
    func testHandleError() {
        viewModel.handleError(message: "Test error")
        
        // Should add an error message or system message
        XCTAssertTrue(viewModel.hasError, "Should have error flag set")
        XCTAssertEqual(viewModel.errorMessage, "Test error", "Error message should match")
    }
    
    func testClearError() {
        viewModel.handleError(message: "Test error")
        XCTAssertTrue(viewModel.hasError)
        
        viewModel.clearError()
        XCTAssertFalse(viewModel.hasError, "Error should be cleared")
    }
    
    // MARK: - Telemetry Integration Tests
    
    func testTelemetryIsIntegrated() {
        XCTAssertNotNil(viewModel.telemetry, "Telemetry should be available")
        
        // Telemetry should update during generation
        viewModel.telemetry.recordTokenWithProbability(tokenId: 1, probability: 0.8, at: Date())
        
        XCTAssertEqual(viewModel.telemetry.tokenHistory.count, 1)
    }
    
    func testTelemetryResetWithConversation() {
        // Need at least 2 tokens to calculate rate
        viewModel.telemetry.recordToken(at: Date())
        viewModel.telemetry.recordToken(at: Date().addingTimeInterval(0.1))
        viewModel.telemetry.calculateMetrics()
        
        XCTAssertGreaterThan(viewModel.telemetry.tokensPerSecond, 0, "Should have positive rate with 2 tokens")
        
        viewModel.clearConversation()
        
        // Telemetry should also be reset
        XCTAssertEqual(viewModel.telemetry.tokensPerSecond, 0, accuracy: 0.01, 
                      "Telemetry should reset with conversation")
    }

    func testTokenizerVocabCompatibilityAcceptsExactTinyLlamaVocab() {
        let result = TokenizerVocabularyCompatibility.evaluate(
            tokenizerVocab: 32_000,
            modelVocab: 32_000
        )

        XCTAssertEqual(result, .compatible)
        XCTAssertTrue(result.isCompatible)
    }

    func testTokenizerVocabCompatibilityAcceptsGemmaPaddedEmbeddingRows() {
        let result = TokenizerVocabularyCompatibility.evaluate(
            tokenizerVocab: 255_933,
            modelVocab: 256_000
        )

        XCTAssertEqual(result, .padded(gap: 67, allowedGap: 512))
        XCTAssertTrue(result.isCompatible)
    }

    func testTokenizerVocabCompatibilityAcceptsQwenReservedRows() {
        let result = TokenizerVocabularyCompatibility.evaluate(
            tokenizerVocab: 151_665,
            modelVocab: 151_936
        )

        XCTAssertEqual(result, .padded(gap: 271, allowedGap: 512))
        XCTAssertTrue(result.isCompatible)
    }

    func testTokenizerVocabCompatibilityRejectsExcessivePaddingGap() {
        let result = TokenizerVocabularyCompatibility.evaluate(
            tokenizerVocab: 50_000,
            modelVocab: 256_000
        )

        XCTAssertEqual(result, .excessivePadding(
            tokenizerVocab: 50_000,
            modelVocab: 256_000,
            gap: 206_000,
            allowedGap: 512
        ))
        XCTAssertFalse(result.isCompatible)
    }

    func testTokenizerVocabCompatibilityRejectsTokenizerLargerThanModel() {
        let result = TokenizerVocabularyCompatibility.evaluate(
            tokenizerVocab: 32_001,
            modelVocab: 32_000
        )

        XCTAssertEqual(result, .tokenizerTooLarge(tokenizerVocab: 32_001, modelVocab: 32_000))
        XCTAssertFalse(result.isCompatible)
    }

    func testSwitchModelRejectsTokenizerVocabMismatchAndKeepsPreviousModel() async throws {
        viewModel.promptText = "keep this"
        viewModel.addUserMessage()
        let originalMessages = viewModel.messages

        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainChatVocabMismatch-\(UUID().uuidString)")
        let modelsDirectory = tempRoot.appendingPathComponent("Models")
        let gemmaRawDirectory = modelsDirectory.appendingPathComponent("gemma-2b-raw")
        try FileManager.default.createDirectory(at: gemmaRawDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let modelURL = modelsDirectory.appendingPathComponent("gemma-2b-int8.tbf")
        try makeArgmaxModel(argmaxToken: 1, vocabSize: 768).save(to: modelURL.path)
        try writeTokenizer(vocabSize: 4, to: gemmaRawDirectory.appendingPathComponent("tokenizer.json"))

        await viewModel.switchModel(ModelInfo(path: modelURL.path))

        XCTAssertEqual(viewModel.activeModelName, "Toy Model")
        XCTAssertNil(viewModel.activeModelPath)
        XCTAssertEqual(viewModel.messages, originalMessages, "A failed switch should not clear the active conversation")
        XCTAssertTrue(viewModel.hasError)
        XCTAssertTrue(viewModel.errorMessage.contains("Tokenizer vocabulary mismatch"))
        XCTAssertTrue(viewModel.errorMessage.contains("exceeds the supported padded-vocab window"))
        XCTAssertTrue(viewModel.errorMessage.contains("Decoding with a mismatched tokenizer would produce garbage"))
    }

    func testSwitchModelRejectsMissingTokenizerAndKeepsPreviousModel() async throws {
        viewModel.promptText = "still here"
        viewModel.addUserMessage()
        let originalMessages = viewModel.messages

        let tempRoot = FileManager.default.temporaryDirectory
            .appendingPathComponent("TinyBrainChatMissingTokenizer-\(UUID().uuidString)")
        let modelsDirectory = tempRoot.appendingPathComponent("Models")
        try FileManager.default.createDirectory(at: modelsDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempRoot) }

        let modelURL = modelsDirectory.appendingPathComponent("gemma-2b-int8.tbf")
        try makeArgmaxModel(argmaxToken: 1, vocabSize: 8).save(to: modelURL.path)

        await viewModel.switchModel(ModelInfo(path: modelURL.path))

        XCTAssertEqual(viewModel.activeModelName, "Toy Model")
        XCTAssertNil(viewModel.activeModelPath)
        XCTAssertEqual(viewModel.messages, originalMessages)
        XCTAssertTrue(viewModel.hasError)
        XCTAssertTrue(viewModel.errorMessage.contains("No tokenizer found for gemma-2b-int8.tbf"))
        XCTAssertTrue(viewModel.errorMessage.contains("gemma-2b-raw/tokenizer.json"))
    }

    // MARK: - Test Helpers

    private func makeBoundaryTokenizer() -> BPETokenizer {
        BPETokenizer(
            vocab: [
                "<BOS>": 1,
                "<EOS>": 2,
                "<UNK>": 3,
                "<PAD>": 4,
                "▁": 5,
                "▁Hello": 10,
                "▁world": 11,
                ".": 12
            ],
            merges: [],
            specialTokens: BPEVocabulary.SpecialTokens(
                bos_token: "<BOS>",
                eos_token: "<EOS>",
                unk_token: "<UNK>",
                pad_token: "<PAD>"
            ),
            usesSentencePieceWhitespace: true
        )
    }

    private func makeTinyLlamaChatTokenizer() -> BPETokenizer {
        BPETokenizer(
            vocab: [
                "<BOS>": 1,
                "</s>": 2,
                "<UNK>": 3,
                "<PAD>": 4,
                "<|system|>": 5,
                "<|user|>": 6,
                "<|assistant|>": 7,
                "h": 8,
                "i": 9
            ],
            merges: [],
            specialTokens: BPEVocabulary.SpecialTokens(
                bos_token: "<BOS>",
                eos_token: "</s>",
                unk_token: "<UNK>",
                pad_token: "<PAD>"
            ),
            byteFallback: false,
            preTokenizedTokens: ["<|system|>", "<|user|>", "<|assistant|>", "</s>"],
            usesSentencePieceWhitespace: false,
            byteLevel: false,
            byteLevelPattern: nil,
            addsBosToken: true
        )
    }

    private func makeQwenChatTokenizer() -> BPETokenizer {
        BPETokenizer(
            vocab: [
                "<UNK>": 0,
                "system": 1,
                "user": 2,
                "assistant": 3,
                "hi": 4,
                "\n": 198,
                "<|endoftext|>": 151_643,
                "<|im_start|>": 151_644,
                "<|im_end|>": 151_645
            ],
            merges: [],
            specialTokens: BPEVocabulary.SpecialTokens(
                bos_token: nil,
                eos_token: "<|im_end|>",
                unk_token: "<UNK>",
                pad_token: "<|endoftext|>"
            ),
            byteFallback: false,
            preTokenizedTokens: ["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
            usesSentencePieceWhitespace: false,
            byteLevel: false,
            byteLevelPattern: nil,
            addsBosToken: false,
            appliesNFC: true
        )
    }

    private func makeArgmaxModel(argmaxToken: Int, vocabSize: Int = 16) -> ModelWeights {
        let hiddenDim = 2
        let config = ModelConfig(
            numLayers: 0,
            hiddenDim: hiddenDim,
            numHeads: 1,
            vocabSize: vocabSize,
            maxSeqLen: 32
        )
        let tokenEmbeddings = Tensor<Float>(
            shape: TensorShape(vocabSize, hiddenDim),
            data: [Float](repeating: 0, count: vocabSize * hiddenDim)
        )
        let outputWeights = Tensor<Float>(
            shape: TensorShape(hiddenDim, vocabSize),
            data: [Float](repeating: 0, count: hiddenDim * vocabSize)
        )
        let outputBias = Tensor<Float>(
            shape: TensorShape(vocabSize),
            data: (0..<vocabSize).map { $0 == argmaxToken ? 20 : -20 }
        )

        return ModelWeights(
            config: config,
            tokenEmbeddings: tokenEmbeddings,
            layers: [],
            output: LinearLayerWeights(floatWeights: outputWeights, bias: outputBias)
        )
    }

    private func makeTransitionModel(
        defaultNextToken: Int,
        transitions: [Int: Int],
        vocabSize: Int = 16
    ) -> ModelWeights {
        let config = ModelConfig(
            numLayers: 0,
            hiddenDim: vocabSize,
            numHeads: 1,
            vocabSize: vocabSize,
            maxSeqLen: 32
        )

        var embeddings = [Float](repeating: 0, count: vocabSize * vocabSize)
        for token in 0..<vocabSize {
            embeddings[token * vocabSize + token] = 1
        }

        var outputData = [Float](repeating: 0, count: vocabSize * vocabSize)
        for token in 0..<vocabSize {
            let next = transitions[token] ?? defaultNextToken
            outputData[token * vocabSize + next] = 20
        }

        return ModelWeights(
            config: config,
            tokenEmbeddings: Tensor<Float>(shape: TensorShape(vocabSize, vocabSize), data: embeddings),
            layers: [],
            output: LinearLayerWeights(
                floatWeights: Tensor<Float>(shape: TensorShape(vocabSize, vocabSize), data: outputData),
                bias: Tensor<Float>.zeros(shape: TensorShape(vocabSize))
            )
        )
    }

    private func writeTokenizer(vocabSize: Int, to url: URL) throws {
        var entries: [String] = [
            "\"<BOS>\": 0",
            "\"<EOS>\": 1",
            "\"<UNK>\": 2",
            "\"<PAD>\": 3"
        ]
        if vocabSize > 4 {
            for id in 4..<vocabSize {
                entries.append("\"\(id)\": \(id)")
            }
        }

        let tokenizerJSON = """
        {
          "vocab": {
            \(entries.joined(separator: ",\n            "))
          },
          "merges": [],
          "special_tokens": {
            "bos_token": "<BOS>",
            "eos_token": "<EOS>",
            "unk_token": "<UNK>",
            "pad_token": "<PAD>"
          }
        }
        """
        try tokenizerJSON.write(to: url, atomically: true, encoding: .utf8)
    }

    private struct MarkerTokenizer: Tokenizer {
        let vocabularySize = 128

        func encode(_ text: String) -> [Int] {
            switch text {
            case "<|user|>":
                return [40, 41]
            case "<|system|>":
                return [42, 43, 44]
            case "</s>":
                return [2]
            default:
                return [10]
            }
        }

        func decode(_ tokens: [Int]) -> String {
            tokens.map(String.init).joined(separator: " ")
        }
    }

    private struct QwenMarkerTokenizer: Tokenizer {
        let vocabularySize = 151_665

        func encode(_ text: String) -> [Int] {
            switch text {
            case "<|im_end|>":
                return [151_645]
            case "</s>":
                return [2]
            default:
                return [10]
            }
        }

        func decode(_ tokens: [Int]) -> String {
            tokens.map(String.init).joined(separator: " ")
        }
    }
}
