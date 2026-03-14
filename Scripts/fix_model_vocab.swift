#!/usr/bin/env swift

import Foundation

// Fix model vocabulary size to match actual tokenizer
let modelPath = "Models/tinyllama-1.1b-int8.tbf"

print("🔧 Fixing model vocabulary size...")

do {
    var data = try Data(contentsOf: URL(fileURLWithPath: modelPath))
    print("📁 Original model size: \(data.count) bytes")
    
    // Read first 4KB to find JSON config
    let headerSize = 4096
    let headerData = data.prefix(headerSize)
    
    if let headerString = String(data: headerData, encoding: .utf8),
       let jsonStart = headerString.range(of: "{"),
       let jsonEnd = headerString.range(of: "}", options: .backwards) {
        
        let jsonString = String(headerString[jsonStart.lowerBound...jsonEnd.upperBound])
        
        if let jsonData = jsonString.data(using: .utf8),
           var json = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any] {
            
            print("📋 Original config: \(json)")
            
            // Update vocabulary size to match actual tokenizer (31,994)
            json["vocabSize"] = 31994
            
            print("🔧 Updated config: \(json)")
            
            // Create new JSON string
            let newJsonData = try JSONSerialization.data(withJSONObject: json)
            let newJsonString = String(data: newJsonData, encoding: .utf8)!
            
            // Create new header with updated JSON
            var newHeader = String(repeating: "\0", count: headerSize)
            let jsonBytes = Array(newJsonString.utf8)
            let startIndex = newHeader.startIndex
            let endIndex = newHeader.index(startIndex, offsetBy: jsonBytes.count)
            newHeader.replaceSubrange(startIndex..<endIndex, with: newJsonString)
            
            // Replace header in data
            let newHeaderData = newHeader.data(using: .utf8)!
            data.replaceSubrange(0..<headerSize, with: newHeaderData)
            
            // Write back to file
            try data.write(to: URL(fileURLWithPath: modelPath))
            print("✅ Model vocabulary size updated to 31,994")
            
        } else {
            print("❌ Failed to parse JSON config")
        }
    } else {
        print("❌ No JSON config found in header")
    }
    
} catch {
    print("❌ Error fixing model: \(error)")
}
