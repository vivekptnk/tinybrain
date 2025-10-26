/// Telemetry View Model
///
/// **TDD Phase: GREEN**
/// Implementation to satisfy telemetry tests.
///
/// Tracks real-time metrics for model inference:
/// - Tokens per second
/// - Milliseconds per token
/// - Energy consumption estimate
/// - KV-cache utilization
/// - Token probability history
///
/// **Performance:** Designed for 2Hz update rate without blocking UI

import Foundation
import SwiftUI
import Combine

/// Tracks and aggregates telemetry metrics for model inference
@MainActor
public final class TelemetryViewModel: ObservableObject {
    
    // MARK: - Published Properties
    
    /// Current tokens per second rate
    @Published public private(set) var tokensPerSecond: Double = 0.0
    
    /// Average milliseconds per token
    @Published public private(set) var millisecondsPerToken: Double = 0.0
    
    /// Estimated energy consumption in joules
    @Published public private(set) var energyEstimate: Double = 0.0
    
    /// KV-cache usage percentage (0-100)
    @Published public private(set) var kvCacheUsagePercent: Double = 0.0
    
    /// Recent token history with probabilities (for charts)
    @Published public private(set) var tokenHistory: [TokenHistoryEntry] = []
    
    /// Average probability of recent tokens (for confidence indicator)
    @Published public private(set) var averageProbability: Double = 0.0
    
    // MARK: - Private State
    
    /// Timestamps of token generation events
    private var tokenTimestamps: [Date] = []
    
    /// Maximum history entries to retain
    private let maxHistorySize: Int = 50
    
    /// Estimated watts during inference (placeholder for energy calculation)
    private let estimatedWattsPerToken: Double = 2.0 // ~2W for M-series chips during inference
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Public Methods
    
    /// Record a token generation event
    public func recordToken(at timestamp: Date = Date()) {
        tokenTimestamps.append(timestamp)
    }
    
    /// Record a token with its probability (for chart visualization)
    public func recordTokenWithProbability(tokenId: Int, probability: Float, at timestamp: Date = Date()) {
        let entry = TokenHistoryEntry(
            tokenId: tokenId,
            probability: probability,
            timestamp: timestamp
        )
        
        tokenHistory.append(entry)
        
        // Cap history size (keep most recent)
        if tokenHistory.count > maxHistorySize {
            tokenHistory.removeFirst(tokenHistory.count - maxHistorySize)
        }
        
        // Also record timestamp for rate calculations
        tokenTimestamps.append(timestamp)
        
        // Update average probability
        updateAverageProbability()
    }
    
    /// Update KV-cache usage metrics
    public func updateKVCacheUsage(used: Int, total: Int) {
        if total > 0 {
            kvCacheUsagePercent = (Double(used) / Double(total)) * 100.0
        } else {
            kvCacheUsagePercent = 0.0
        }
    }
    
    /// Calculate all metrics from recorded data
    public func calculateMetrics() {
        calculateTokensPerSecond()
        calculateMillisecondsPerToken()
        calculateEnergyEstimate()
    }
    
    /// Reset all metrics and history
    public func reset() {
        tokenTimestamps.removeAll()
        tokenHistory.removeAll()
        tokensPerSecond = 0.0
        millisecondsPerToken = 0.0
        energyEstimate = 0.0
        kvCacheUsagePercent = 0.0
        averageProbability = 0.0
    }
    
    // MARK: - Private Calculations
    
    private func calculateTokensPerSecond() {
        guard tokenTimestamps.count >= 2 else {
            tokensPerSecond = 0.0
            return
        }
        
        // Calculate rate from first to last token
        let firstTime = tokenTimestamps.first!
        let lastTime = tokenTimestamps.last!
        let duration = lastTime.timeIntervalSince(firstTime)
        
        if duration > 0 {
            // tokens - 1 because we count intervals, not tokens
            let intervalCount = Double(tokenTimestamps.count - 1)
            tokensPerSecond = intervalCount / duration
        } else {
            tokensPerSecond = 0.0
        }
    }
    
    private func calculateMillisecondsPerToken() {
        guard tokenTimestamps.count >= 2 else {
            millisecondsPerToken = 0.0
            return
        }
        
        // Calculate average interval between consecutive tokens
        var totalInterval: TimeInterval = 0
        for i in 1..<tokenTimestamps.count {
            totalInterval += tokenTimestamps[i].timeIntervalSince(tokenTimestamps[i - 1])
        }
        
        let averageInterval = totalInterval / Double(tokenTimestamps.count - 1)
        millisecondsPerToken = averageInterval * 1000.0 // convert to milliseconds
    }
    
    private func calculateEnergyEstimate() {
        guard tokenTimestamps.count >= 2 else {
            energyEstimate = 0.0
            return
        }
        
        // Simplified energy model:
        // Energy (Joules) = Power (Watts) × Time (seconds)
        // We estimate ~2W during inference on Apple Silicon
        
        let firstTime = tokenTimestamps.first!
        let lastTime = tokenTimestamps.last!
        let duration = lastTime.timeIntervalSince(firstTime)
        
        // E = P * t
        energyEstimate = estimatedWattsPerToken * duration
    }
    
    private func updateAverageProbability() {
        guard !tokenHistory.isEmpty else {
            averageProbability = 0.0
            return
        }
        
        let sum = tokenHistory.reduce(0.0) { $0 + Double($1.probability) }
        averageProbability = sum / Double(tokenHistory.count)
    }
}

// MARK: - Supporting Types

/// Entry in token history for chart visualization
public struct TokenHistoryEntry: Identifiable {
    public let id = UUID()
    public let tokenId: Int
    public let probability: Float
    public let timestamp: Date
    
    public init(tokenId: Int, probability: Float, timestamp: Date) {
        self.tokenId = tokenId
        self.probability = probability
        self.timestamp = timestamp
    }
}

