#!/usr/bin/env swift
/// Quick diagnostic to check model weight dimensions

import Foundation

// Change to your workspace root
let projectRoot = "/Users/vivekesque/Desktop/CreativeSpace/CodingProjects/tinybrain"
let modelPath = "\(projectRoot)/Models/tinyllama-1.1b-int8.tbf"

// Add the Sources directory to the module search path
#if canImport(TinyBrainRuntime)
import TinyBrainRuntime

print("🔍 Diagnosing model dimensions...")
print("Loading model from: \(modelPath)")

do {
    let weights = try ModelWeights.load(from: modelPath)
    let config = weights.config
    
    print("\n📊 Model Configuration:")
    print("  Hidden dim: \(config.hiddenDim)")
    print("  Vocab size: \(config.vocabSize)")
    print("  Num layers: \(config.numLayers)")
    print("  Num heads: \(config.numHeads)")
    
    print("\n🔢 Embeddings shape: \(weights.tokenEmbeddings.shape)")
    print("  Expected: [\(config.vocabSize), \(config.hiddenDim)]")
    
    if let firstLayer = weights.layers.first {
        print("\n🧩 Layer 0 Attention Projections:")
        print("  Query weights: \(firstLayer.attention.query.weights.shape)")
        print("  Key weights: \(firstLayer.attention.key.weights.shape)")
        print("  Value weights: \(firstLayer.attention.value.weights.shape)")
        print("  Output weights: \(firstLayer.attention.output.weights.shape)")
        
        print("\n  Expected for matmul(input: [1, \(config.hiddenDim)]):")
        print("    Weight shape should be: [\(config.hiddenDim), \(config.hiddenDim)]")
        
        print("\n🍔 Layer 0 Feed-Forward:")
        print("  Up weights: \(firstLayer.feedForward.up.weights.shape)")
        print("  Down weights: \(firstLayer.feedForward.down.weights.shape)")
        
        let ffnHidden = config.hiddenDim * 4
        print("\n  Expected:")
        print("    Up should be: [\(config.hiddenDim), \(ffnHidden)]")
        print("    Down should be: [\(ffnHidden), \(config.hiddenDim)]")
    }
    
    print("\n🎯 Output projection:")
    print("  Weights: \(weights.output.weights.shape)")
    print("  Expected: [\(config.hiddenDim), \(config.vocabSize)]")
    
    // Test a single forward pass to trigger the error
    print("\n🧪 Testing forward pass...")
    let runner = ModelRunner(weights: weights)
    let logits = runner.step(tokenId: 0)
    print("✅ Forward pass succeeded! Logits shape: \(logits.shape)")
    
} catch {
    print("❌ Error: \(error)")
}

#else
print("❌ TinyBrainRuntime module not found")
print("Run with: swift -I .build/debug -L .build/debug -lTinyBrainRuntime diagnose_model_dims.swift")
#endif


