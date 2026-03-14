#!/usr/bin/env swift

import Foundation

// Test model loading to see where the crash occurs
print("🧪 Testing model loading...")

do {
    // Try to load the model
    let modelPath = "Models/tinyllama-1.1b-int8.tbf"
    print("📁 Loading model from: \(modelPath)")
    
    // This will help us see where exactly the crash occurs
    print("✅ Model file exists: \(FileManager.default.fileExists(atPath: modelPath))")
    
    // Try to read the file
    let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    print("✅ Model file read: \(data.count) bytes")
    
    // Check TBF header
    let magic = String(data: data[0..<4], encoding: .utf8)
    print("✅ Magic: \(magic ?? "nil")")
    
    let version = data[4..<8].withUnsafeBytes { $0.load(as: UInt32.self) }
    print("✅ Version: \(version)")
    
    let configLength = data[8..<12].withUnsafeBytes { $0.load(as: UInt32.self) }
    print("✅ Config length: \(configLength)")
    
    let configData = data[12..<(12 + Int(configLength))]
    let config = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
    print("✅ Config: \(config)")
    
    print("🎉 Model loading test completed successfully!")
    
} catch {
    print("❌ Error during model loading: \(error)")
}
