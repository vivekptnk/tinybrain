/// Telemetry View Model Tests
///
/// **TDD Phase: RED**
/// Tests define requirements for telemetry tracking and aggregation.
///
/// Tests cover:
/// - Tokens/sec calculation
/// - ms/token averaging
/// - Energy estimation
/// - KV-cache usage tracking
/// - History buffer management

import XCTest
@testable import TinyBrainDemo
@testable import TinyBrainRuntime

@MainActor
final class TelemetryViewModelTests: XCTestCase {
    
    var viewModel: TelemetryViewModel!
    
    override func setUp() async throws {
        viewModel = TelemetryViewModel()
    }
    
    override func tearDown() async throws {
        viewModel = nil
    }
    
    // MARK: - Initialization Tests
    
    func testInitialState() {
        XCTAssertEqual(viewModel.tokensPerSecond, 0.0, accuracy: 0.01, "Initial tokens/sec should be 0")
        XCTAssertEqual(viewModel.millisecondsPerToken, 0.0, accuracy: 0.01, "Initial ms/token should be 0")
        XCTAssertEqual(viewModel.energyEstimate, 0.0, accuracy: 0.01, "Initial energy should be 0")
        XCTAssertEqual(viewModel.kvCacheUsagePercent, 0.0, accuracy: 0.01, "Initial cache usage should be 0")
        XCTAssertEqual(viewModel.tokenHistory.count, 0, "Token history should start empty")
    }
    
    // MARK: - Tokens Per Second Tests
    
    func testTokensPerSecondCalculation() {
        // Simulate 10 tokens generated over 1.8 seconds (9 intervals)
        let baseTime = Date()
        viewModel.recordToken(at: baseTime)
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.2))
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.4))
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.6))
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.8))
        viewModel.recordToken(at: baseTime.addingTimeInterval(1.0))
        viewModel.recordToken(at: baseTime.addingTimeInterval(1.2))
        viewModel.recordToken(at: baseTime.addingTimeInterval(1.4))
        viewModel.recordToken(at: baseTime.addingTimeInterval(1.6))
        viewModel.recordToken(at: baseTime.addingTimeInterval(1.8))
        
        viewModel.calculateMetrics()
        
        // 9 intervals in 1.8 seconds = 5.0 tokens/sec
        XCTAssertEqual(viewModel.tokensPerSecond, 5.0, accuracy: 0.1, "Should calculate 5.0 tokens/sec")
    }
    
    func testTokensPerSecondWithSingleToken() {
        viewModel.recordToken(at: Date())
        viewModel.calculateMetrics()
        
        // Single token: edge case, should not crash
        XCTAssertGreaterThanOrEqual(viewModel.tokensPerSecond, 0, "Single token should not produce negative rate")
    }
    
    func testTokensPerSecondReset() {
        viewModel.recordToken(at: Date())
        viewModel.recordToken(at: Date().addingTimeInterval(1.0))
        viewModel.calculateMetrics()
        
        XCTAssertGreaterThan(viewModel.tokensPerSecond, 0, "Should have non-zero rate")
        
        viewModel.reset()
        
        XCTAssertEqual(viewModel.tokensPerSecond, 0.0, accuracy: 0.01, "Reset should clear tokens/sec")
    }
    
    // MARK: - Milliseconds Per Token Tests
    
    func testMillisecondsPerTokenCalculation() {
        // Simulate tokens with known intervals
        let baseTime = Date()
        viewModel.recordToken(at: baseTime)
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.1))  // 100ms
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.2))  // 100ms
        viewModel.recordToken(at: baseTime.addingTimeInterval(0.3))  // 100ms
        
        viewModel.calculateMetrics()
        
        // Average should be 100ms per token
        XCTAssertEqual(viewModel.millisecondsPerToken, 100.0, accuracy: 5.0, "Should calculate ~100ms/token")
    }
    
    func testMillisecondsPerTokenInverse() {
        // ms/token should be inverse of tokens/sec
        let baseTime = Date()
        for i in 0..<10 {
            viewModel.recordToken(at: baseTime.addingTimeInterval(Double(i) * 0.1))
        }
        
        viewModel.calculateMetrics()
        
        let expectedMs = 1000.0 / viewModel.tokensPerSecond
        XCTAssertEqual(viewModel.millisecondsPerToken, expectedMs, accuracy: 1.0, 
                      "ms/token should be 1000 / tokens_per_sec")
    }
    
    // MARK: - Energy Estimation Tests
    
    func testEnergyEstimation() {
        // Energy should be estimated based on token count and duration
        viewModel.recordToken(at: Date())
        viewModel.recordToken(at: Date().addingTimeInterval(1.0))
        viewModel.recordToken(at: Date().addingTimeInterval(2.0))
        
        viewModel.calculateMetrics()
        
        // Should have some positive energy estimate
        XCTAssertGreaterThan(viewModel.energyEstimate, 0, "Energy estimate should be positive")
        
        // Energy should scale with token count
        let energyForThree = viewModel.energyEstimate
        
        viewModel.recordToken(at: Date().addingTimeInterval(3.0))
        viewModel.calculateMetrics()
        
        XCTAssertGreaterThan(viewModel.energyEstimate, energyForThree, 
                           "Energy should increase with more tokens")
    }
    
    func testEnergyEstimateFormula() {
        // Energy ≈ tokens * estimated_watts_per_token * time_per_token
        // Placeholder formula: ~2 watts during inference, so 2W * seconds
        viewModel.recordToken(at: Date())
        viewModel.recordToken(at: Date().addingTimeInterval(1.0))
        
        viewModel.calculateMetrics()
        
        // Very rough estimate, just verify it's in reasonable range (millijoules to joules)
        XCTAssertGreaterThan(viewModel.energyEstimate, 0.001, "Energy should be > 1mJ")
        XCTAssertLessThan(viewModel.energyEstimate, 100, "Energy should be < 100J for demo")
    }
    
    // MARK: - KV-Cache Usage Tests
    
    func testKVCacheUsagePercent() {
        viewModel.updateKVCacheUsage(used: 512, total: 2048)
        
        XCTAssertEqual(viewModel.kvCacheUsagePercent, 25.0, accuracy: 0.1, "512/2048 should be 25%")
    }
    
    func testKVCacheUsageZeroTotal() {
        viewModel.updateKVCacheUsage(used: 0, total: 0)
        
        // Should handle division by zero gracefully
        XCTAssertEqual(viewModel.kvCacheUsagePercent, 0.0, accuracy: 0.01, "Zero total should be 0%")
    }
    
    func testKVCacheUsageFull() {
        viewModel.updateKVCacheUsage(used: 2048, total: 2048)
        
        XCTAssertEqual(viewModel.kvCacheUsagePercent, 100.0, accuracy: 0.1, "Full cache should be 100%")
    }
    
    // MARK: - Token History Tests
    
    func testTokenHistoryBuffering() {
        // Add more tokens than the max history size
        let maxHistory = 50
        for i in 0..<100 {
            viewModel.recordTokenWithProbability(
                tokenId: i,
                probability: Float(i) / 100.0,
                at: Date().addingTimeInterval(Double(i) * 0.01)
            )
        }
        
        // History should be capped
        XCTAssertLessThanOrEqual(viewModel.tokenHistory.count, maxHistory, 
                                "History should be capped at max size")
        
        // Should keep most recent tokens
        if let last = viewModel.tokenHistory.last {
            XCTAssertGreaterThan(last.tokenId, 50, "Should keep most recent tokens")
        }
    }
    
    func testTokenHistoryRetainsMetadata() {
        viewModel.recordTokenWithProbability(tokenId: 42, probability: 0.85, at: Date())
        
        XCTAssertEqual(viewModel.tokenHistory.count, 1)
        
        let entry = viewModel.tokenHistory[0]
        XCTAssertEqual(entry.tokenId, 42)
        XCTAssertEqual(entry.probability, 0.85, accuracy: 0.01)
    }
    
    func testTokenHistoryReset() {
        viewModel.recordTokenWithProbability(tokenId: 1, probability: 0.5, at: Date())
        viewModel.recordTokenWithProbability(tokenId: 2, probability: 0.6, at: Date())
        
        XCTAssertEqual(viewModel.tokenHistory.count, 2)
        
        viewModel.reset()
        
        XCTAssertEqual(viewModel.tokenHistory.count, 0, "Reset should clear history")
    }
    
    // MARK: - Metric Calculation Edge Cases
    
    func testCalculateMetricsWithNoData() {
        viewModel.calculateMetrics()
        
        // Should not crash and should return zero metrics
        XCTAssertEqual(viewModel.tokensPerSecond, 0.0, accuracy: 0.01)
        XCTAssertEqual(viewModel.millisecondsPerToken, 0.0, accuracy: 0.01)
    }
    
    func testCalculateMetricsIdempotent() {
        viewModel.recordToken(at: Date())
        viewModel.recordToken(at: Date().addingTimeInterval(1.0))
        
        viewModel.calculateMetrics()
        let firstResult = viewModel.tokensPerSecond
        
        viewModel.calculateMetrics()
        let secondResult = viewModel.tokensPerSecond
        
        XCTAssertEqual(firstResult, secondResult, accuracy: 0.01, 
                      "Calculating metrics multiple times should give same result")
    }
    
    // MARK: - Running Average Tests
    
    func testRunningAverageProbability() {
        viewModel.recordTokenWithProbability(tokenId: 1, probability: 0.8, at: Date())
        viewModel.recordTokenWithProbability(tokenId: 2, probability: 0.6, at: Date())
        viewModel.recordTokenWithProbability(tokenId: 3, probability: 0.9, at: Date())
        
        let average = viewModel.averageProbability
        
        // (0.8 + 0.6 + 0.9) / 3 = 0.767
        XCTAssertEqual(average, 0.767, accuracy: 0.01, "Should calculate average probability")
    }
    
    func testRunningAverageProbabilityEmpty() {
        let average = viewModel.averageProbability
        
        XCTAssertEqual(average, 0.0, accuracy: 0.01, "Empty history should have 0 average")
    }
}

