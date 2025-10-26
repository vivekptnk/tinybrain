/// TDD Tests for Benchmark Harness CLI (TB-007 Phase 3)
///
/// RED phase: These tests define expected behavior before implementation.

import XCTest
import Foundation

/// Note: We can't directly test an executable target in Swift, so these tests
/// verify the benchmark harness behavior via subprocess execution.
final class BenchmarkHarnessTests: XCTestCase {
    
    let benchPath = ".build/debug/tinybrain-bench"
    
    // MARK: - YAML Scenario Loading
    
    func testYAMLScenarioLoading() throws {
        // Test that --scenario flag loads YAML scenarios correctly
        guard let fixturesURL = Bundle.module.url(forResource: "test_scenario", withExtension: "yml") else {
            XCTFail("Could not find test_scenario.yml in bundle")
            return
        }
        let scenarioPath = fixturesURL.path
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--scenario", scenarioPath, "--dry-run"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)!
        
        // Should parse and show scenarios
        XCTAssertEqual(process.terminationStatus, 0, "Should succeed")
        XCTAssertTrue(output.contains("Toy Model - Short Prompt"), "Should load scenario name")
        XCTAssertTrue(output.contains("2 scenarios"), "Should count scenarios")
    }
    
    func testYAMLScenarioMissingFile() throws {
        // Should fail gracefully with missing file
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--scenario", "nonexistent.yml"]
        
        let pipe = Pipe()
        process.standardError = pipe
        
        try process.run()
        process.waitUntilExit()
        
        XCTAssertNotEqual(process.terminationStatus, 0, "Should fail")
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)!
        XCTAssertTrue(output.contains("not found") || output.contains("Error"), "Should show error")
    }
    
    // MARK: - JSON Output Format
    
    func testJSONOutputFormat() throws {
        // Test that --output json produces valid JSON
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--demo", "--output", "json", "--tokens", "5"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        
        // Should be valid JSON
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        XCTAssertNotNil(json, "Should produce valid JSON")
        
        // Validate schema
        XCTAssertNotNil(json?["device"], "Should include device info")
        XCTAssertNotNil(json?["metrics"], "Should include metrics")
        
        if let metrics = json?["metrics"] as? [String: Any] {
            XCTAssertNotNil(metrics["tokens_per_sec"], "Should include tokens/sec")
            XCTAssertNotNil(metrics["ms_per_token"], "Should include ms/token")
        }
    }
    
    func testMarkdownOutputFormat() throws {
        // Test that --output markdown produces markdown tables
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--demo", "--output", "markdown", "--tokens", "5"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)!
        
        // Should contain markdown table syntax
        XCTAssertTrue(output.contains("|"), "Should have markdown table")
        XCTAssertTrue(output.contains("tokens/sec") || output.contains("ms/token"), "Should have metrics")
    }
    
    // MARK: - Memory Tracking
    
    func testMemoryTracking() throws {
        // Verify memory metrics are reported
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--demo", "--output", "json", "--tokens", "10"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        if let metrics = json?["metrics"] as? [String: Any] {
            XCTAssertNotNil(metrics["memory_peak_mb"], "Should track peak memory")
            
            if let memoryPeak = metrics["memory_peak_mb"] as? Double {
                XCTAssertGreaterThan(memoryPeak, 0, "Memory should be > 0")
                XCTAssertLessThan(memoryPeak, 10000, "Memory should be reasonable (< 10GB)")
            }
        }
    }
    
    // MARK: - Device Info Reporting
    
    func testDeviceInfoReporting() throws {
        // Test that --device-info shows system information
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--device-info"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)!
        
        XCTAssertEqual(process.terminationStatus, 0)
        XCTAssertTrue(output.contains("Device") || output.contains("CPU") || output.contains("GPU"), 
                     "Should show device info")
    }
    
    // MARK: - Warmup Iterations
    
    func testWarmupIterations() throws {
        // Ensure warmup runs before timing
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--demo", "--warmup", "3", "--tokens", "5", "--verbose"]
        
        let pipe = Pipe()
        process.standardOutput = pipe
        
        try process.run()
        process.waitUntilExit()
        
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8)!
        
        // Should mention warmup in verbose mode
        XCTAssertTrue(output.contains("warmup") || output.contains("Warmup") || output.contains("warm"), 
                     "Should perform warmup")
    }
    
    // MARK: - Edge Cases
    
    func testInvalidArguments() throws {
        // Should fail gracefully with invalid arguments
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--invalid-flag"]
        
        try process.run()
        process.waitUntilExit()
        
        XCTAssertNotEqual(process.terminationStatus, 0, "Should fail with invalid arguments")
    }
    
    func testZeroTokens() throws {
        // Should handle edge case of zero tokens
        let process = Process()
        process.executableURL = URL(fileURLWithPath: benchPath)
        process.arguments = ["--demo", "--tokens", "0"]
        
        try process.run()
        process.waitUntilExit()
        
        // Should either succeed with warning or fail gracefully
        XCTAssertTrue(process.terminationStatus == 0 || process.terminationStatus == 1)
    }
}

