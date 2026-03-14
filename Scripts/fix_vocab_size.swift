#!/usr/bin/env swift

import Foundation

// Fix vocabulary size in TBF model file
let modelPath = "Models/tinyllama-1.1b-int8.tbf"

print("🔧 Fixing vocabulary size in model...")

do {
    var data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    print("📁 Model size: \(data.count) bytes")
    
    // TBF format: [4 bytes magic][4 bytes version][4 bytes config_length][config_json]
    guard data.count >= 12 else {
        print("❌ File too small")
        exit(1)
    }
    
    // Check magic
    let magic = String(data: data[0..<4], encoding: .utf8)
    guard magic == "TBFM" else {
        print("❌ Invalid magic: \(magic ?? "nil")")
        exit(1)
    }
    
    // Read version
    let version = data[4..<8].withUnsafeBytes { $0.load(as: UInt32.self) }
    print("📋 Version: \(version)")
    
    // Read config length
    let configLength = data[8..<12].withUnsafeBytes { $0.load(as: UInt32.self) }
    print("📋 Config length: \(configLength)")
    
    // Read config JSON
    let configData = data[12..<(12 + Int(configLength))]
    let configString = String(data: configData, encoding: .utf8)!
    print("📋 Original config: \(configString)")
    
    // Parse and update config
    var config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
    config["vocabSize"] = 31994
    
    // Create new config JSON
    let newConfigData = try JSONSerialization.data(withJSONObject: config)
    let newConfigString = String(data: newConfigData, encoding: .utf8)!
    print("🔧 Updated config: \(newConfigString)")
    
    // Update the data
    let newConfigLength = UInt32(newConfigData.count)
    
    // Write new config length (little-endian)
    withUnsafeBytes(of: newConfigLength.littleEndian) { bytes in
        data.replaceSubrange(8..<12, with: bytes)
    }
    
    // Replace config data
    data.replaceSubrange(12..<(12 + Int(configLength)), with: newConfigData)
    
    // Write back to file
    try data.write(to: URL(fileURLWithPath: modelPath))
    print("✅ Model vocabulary size updated to 31,994")
    
} catch {
    print("❌ Error: \(error)")
    exit(1)
}
