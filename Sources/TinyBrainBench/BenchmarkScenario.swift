/// Benchmark scenario configuration (loaded from YAML)
///
/// TB-007 Phase 3: Support for scripted benchmark scenarios

import Foundation

struct BenchmarkScenario: Codable {
    let name: String
    let model: String
    let prompts: [String]
    let maxTokens: Int
    let backend: String?  // "cpu", "metal", or "auto"
    let warmup: Int?
    let sampler: SamplerSettings?
    
    enum CodingKeys: String, CodingKey {
        case name
        case model
        case prompts
        case maxTokens = "max_tokens"
        case backend
        case warmup
        case sampler
    }
}

struct SamplerSettings: Codable {
    let temperature: Float?
    let topK: Int?
    let topP: Float?
    let repetitionPenalty: Float?
    
    enum CodingKeys: String, CodingKey {
        case temperature
        case topK = "top_k"
        case topP = "top_p"
        case repetitionPenalty = "repetition_penalty"
    }
}

struct ScenarioFile: Codable {
    let scenarios: [BenchmarkScenario]
}

/// Benchmark results for JSON output
struct BenchmarkResult: Codable {
    let device: DeviceInfo
    let scenario: String?
    let metrics: Metrics
    let timestamp: String
    
    struct DeviceInfo: Codable {
        let name: String
        let os: String
        let metalAvailable: Bool
        
        enum CodingKeys: String, CodingKey {
            case name
            case os
            case metalAvailable = "metal_available"
        }
    }
    
    struct Metrics: Codable {
        let tokensPerSec: Double
        let msPerToken: Double
        let memoryPeakMB: Double
        let totalTokens: Int
        let elapsedSeconds: Double
        
        enum CodingKeys: String, CodingKey {
            case tokensPerSec = "tokens_per_sec"
            case msPerToken = "ms_per_token"
            case memoryPeakMB = "memory_peak_mb"
            case totalTokens = "total_tokens"
            case elapsedSeconds = "elapsed_seconds"
        }
    }
}

/// Memory tracking utilities
enum MemoryTracker {
    static func currentMemoryUsageMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(
                    mach_task_self_,
                    task_flavor_t(MACH_TASK_BASIC_INFO),
                    $0,
                    &count
                )
            }
        }
        
        guard kerr == KERN_SUCCESS else {
            return 0
        }
        
        return Double(info.resident_size) / (1024 * 1024)
    }
}

#if canImport(Darwin)
import Darwin
#endif

