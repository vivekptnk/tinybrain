#!/usr/bin/env swift

import Foundation

// Simple model inspector to check dimensions
let modelPath = "Models/tinyllama-1.1b-int8.tbf"

print("🔍 Inspecting model: \(modelPath)")

do {
    let data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    print("📁 Model file size: \(data.count) bytes")
    
    // Read TBF header (first 4KB)
    let headerSize = 4096
    let headerData = data.prefix(headerSize)
    
    // Try to find JSON config in header
    if let headerString = String(data: headerData, encoding: .utf8),
       let jsonStart = headerString.range(of: "{"),
       let jsonEnd = headerString.range(of: "}", options: .backwards) {
        
        let jsonString = String(headerString[jsonStart.lowerBound...jsonEnd.upperBound])
        print("📋 Found JSON config in header:")
        print(jsonString)
        
        // Try to parse as JSON
        if let jsonData = jsonString.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
            
            print("\n🔧 Model Configuration:")
            for (key, value) in json {
                print("  \(key): \(value)")
            }
        }
    } else {
        print("❌ No JSON config found in header")
    }
    
} catch {
    print("❌ Error reading model: \(error)")
}
