# Getting Started with TinyBrain

Learn how to integrate TinyBrain into your iOS or macOS app.

## Overview

TinyBrain makes it easy to run large language models on-device with just a few lines of code. This guide walks you through your first integration.

## Prerequisites

Before you begin, ensure you have:

- **macOS 14 Sonoma** or later
- **Xcode 16** or later
- **Apple Silicon** (M1, M2, M3, or M4) for best performance
- A compatible model file in `.tbf` format

## Installation

### Swift Package Manager

Add TinyBrain to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/vivekp/tinybrain.git", from: "0.1.0")
]
```

Or in Xcode:
1. File → Add Package Dependencies
2. Enter: `https://github.com/vivekp/tinybrain.git`
3. Select version and add to your target

## Basic Usage

### Loading a Model

```swift
import TinyBrain

// Load a quantized model
let model = try await TinyBrain.load("path/to/model.tbf")
```

### Generating Text

```swift
// Generate text with streaming output
let stream = try await model.generateStream(prompt: "Explain neural networks:")

for try await token in stream {
    print(token, terminator: "")
}
```

### SwiftUI Integration

```swift
import SwiftUI
import TinyBrain

struct ChatView: View {
    @State private var prompt = ""
    @State private var response = ""
    @State private var isGenerating = false
    
    var body: some View {
        VStack {
            TextEditor(text: $response)
            
            HStack {
                TextField("Enter prompt", text: $prompt)
                
                Button("Generate") {
                    Task {
                        await generate()
                    }
                }
                .disabled(isGenerating)
            }
        }
    }
    
    func generate() async {
        isGenerating = true
        response = ""
        
        do {
            let model = try await TinyBrain.load("model.tbf")
            let stream = try await model.generateStream(prompt: prompt)
            
            for try await token in stream {
                response += token
            }
        } catch {
            response = "Error: \(error)"
        }
        
        isGenerating = false
    }
}
```

## Next Steps

- Learn about <doc:Architecture>
- Explore the Example app in `Examples/ChatDemo`
- Read the full API documentation: ``TinyBrain``

