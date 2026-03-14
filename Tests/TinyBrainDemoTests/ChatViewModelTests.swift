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
}

