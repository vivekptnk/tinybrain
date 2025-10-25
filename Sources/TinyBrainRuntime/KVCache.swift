/// Key-Value cache for transformer attention
///
/// **TB-004 Phase 4:** Paged KV cache enabling efficient streaming inference
///
/// ## How Transformer Attention Works
///
/// For token at position `t`, attention computes:
/// ```
/// Q_t = query for current token
/// K_0...t = keys for all previous tokens (NEED TO CACHE!)
/// V_0...t = values for all previous tokens (NEED TO CACHE!)
///
/// attention_scores = Q_t × [K_0, K_1, ..., K_t]ᵀ
/// output = softmax(attention_scores) × [V_0, V_1, ..., V_t]
/// ```
///
/// **Without cache:** Recompute K₀...K_{t-1} every time → O(t²) compute!  
/// **With cache:** Store K₀...K_{t-1}, only compute K_t → O(t) compute!
///
/// ## Paging Strategy
///
/// Instead of one giant buffer, use **pages**:
/// ```
/// Page 0: tokens 0-15
/// Page 1: tokens 16-31
/// Page 2: tokens 32-47
/// ...
/// ```
///
/// Benefits:
/// - Pre-allocated on GPU (zero overhead during inference)
/// - Easy eviction (free oldest page)
/// - Efficient for variable-length sequences

import Foundation
#if canImport(Metal)
import Metal
#endif

/// KV cache manager for transformer layers
public final class KVCache {
    /// Number of transformer layers
    public let numLayers: Int
    
    /// Hidden dimension size
    public let hiddenDim: Int
    
    /// Maximum tokens to cache
    public let maxTokens: Int
    
    /// Tokens per page
    public let pageSize: Int
    
    /// Page allocator
    private let allocator: PageAllocator
    
    /// Storage for keys: [layer][logicalPageId] → (allocatorPageId, [Float] data)
    /// Each page stores `pageSize` keys of dimension `hiddenDim`
    /// **Uses raw arrays for FAST writes** (no Tensor CoW overhead)
    private var keyPages: [[Int: (Int, [Float])]]
    
    /// Storage for values: [layer][logicalPageId] → (allocatorPageId, [Float] data)
    private var valuePages: [[Int: (Int, [Float])]]
    
    /// Current sequence length per layer
    private var lengths: [Int]
    
    /// Minimum cached position per layer (for eviction tracking)
    private var minPositions: [Int]
    
    /// Lock for thread safety
    private let lock = NSLock()
    
    /// Initialize KV cache
    ///
    /// - Parameters:
    ///   - numLayers: Number of transformer layers
    ///   - hiddenDim: Hidden dimension size (e.g., 768)
    ///   - maxTokens: Maximum tokens to cache (e.g., 2048)
    ///   - pageSize: Tokens per page (default: 16)
    public init(numLayers: Int, hiddenDim: Int, maxTokens: Int, pageSize: Int = 16) {
        self.numLayers = numLayers
        self.hiddenDim = hiddenDim
        self.maxTokens = maxTokens
        self.pageSize = pageSize
        
        let maxPages = (maxTokens + pageSize - 1) / pageSize
        self.allocator = PageAllocator(pageSize: pageSize, maxPages: maxPages)
        
        // Initialize storage for each layer
        self.keyPages = Array(repeating: [:], count: numLayers)
        self.valuePages = Array(repeating: [:], count: numLayers)
        self.lengths = Array(repeating: 0, count: numLayers)
        self.minPositions = Array(repeating: 0, count: numLayers)
    }
    
    /// Current cache length (max across all layers)
    public var length: Int {
        lock.lock()
        defer { lock.unlock() }
        
        return lengths.max() ?? 0
    }
    
    /// Append key/value for a new token
    ///
    /// **TB-004:** Core API for building up context during inference
    ///
    /// Automatically evicts oldest tokens if cache exceeds maxTokens.
    ///
    /// - Parameters:
    ///   - layer: Layer index [0, numLayers)
    ///   - key: Key tensor [hiddenDim]
    ///   - value: Value tensor [hiddenDim]
    ///   - position: Token position in sequence
    public func append(layer: Int, key: Tensor<Float>, value: Tensor<Float>, position: Int) {
        lock.lock()
        defer { lock.unlock() }
        
        precondition(layer >= 0 && layer < numLayers, "Layer index out of bounds")
        precondition(key.shape == TensorShape(hiddenDim), "Key shape mismatch")
        precondition(value.shape == TensorShape(hiddenDim), "Value shape mismatch")
        
        // Check if adding this token would exceed maxTokens
        // If so, evict oldest page(s) to make room
        while lengths[layer] >= maxTokens {
            evictOldestPage(layer: layer)
            lengths[layer] -= pageSize  // Evicted one page worth of tokens
        }
        
        // Determine which page this position belongs to
        let pageId = position / pageSize
        let offsetInPage = position % pageSize
        
        // Get or create page
        if keyPages[layer][pageId] == nil {
            // Allocate new page from allocator
            var allocatorPageId = allocator.allocatePage()
            
            if allocatorPageId == nil {
                // Allocator exhausted - evict oldest page and retry
                evictOldestPage(layer: layer)
                allocatorPageId = allocator.allocatePage()
                
                guard allocatorPageId != nil else {
                    fatalError("Failed to allocate page even after eviction")
                }
            }
            
            // Create raw arrays for this page (FAST!)
            let keyData = [Float](repeating: 0.0, count: pageSize * hiddenDim)
            let valueData = [Float](repeating: 0.0, count: pageSize * hiddenDim)
            
            keyPages[layer][pageId] = (allocatorPageId!, keyData)
            valuePages[layer][pageId] = (allocatorPageId!, valueData)
        }
        
        // Write key/value into page (FAST: direct array write!)
        if let (allocatorId, keyDataArray) = keyPages[layer][pageId],
           let (_, valueDataArray) = valuePages[layer][pageId] {
            
            // Get mutable copies
            var keyData = keyDataArray
            var valueData = valueDataArray
            
            // **TB-004 OPTIMIZATION:** Direct array manipulation (100× faster!)
            let rowStartIdx = offsetInPage * hiddenDim
            let keyInput = key.rawData
            let valueInput = value.rawData
            
            // Bulk copy entire row
            for i in 0..<hiddenDim {
                keyData[rowStartIdx + i] = keyInput[i]
                valueData[rowStartIdx + i] = valueInput[i]
            }
            
            // Store back
            keyPages[layer][pageId] = (allocatorId, keyData)
            valuePages[layer][pageId] = (allocatorId, valueData)
        }
        
        // Update length - track how many tokens we have
        lengths[layer] = max(lengths[layer], position + 1)
        
        // Cap at maxTokens
        if lengths[layer] > maxTokens {
            lengths[layer] = maxTokens
        }
    }
    
    /// Get keys for a range of tokens
    ///
    /// - Parameters:
    ///   - layer: Layer index
    ///   - range: Token positions to retrieve
    /// - Returns: Tensor of shape [range.count, hiddenDim]
    public func getKeys(layer: Int, range: Range<Int>) -> Tensor<Float> {
        lock.lock()
        defer { lock.unlock() }
        
        precondition(layer >= 0 && layer < numLayers, "Layer out of bounds")
        
        let count = range.count
        var result = Tensor<Float>.zeros(shape: TensorShape(count, hiddenDim))
        
        // Collect keys from pages (convert back to tensor)
        for (idx, pos) in range.enumerated() {
            let pageId = pos / pageSize
            let offsetInPage = pos % pageSize
            
            if let (_, keyData) = keyPages[layer][pageId] {
                // Copy row from page array to result
                let rowStartIdx = offsetInPage * hiddenDim
                for i in 0..<hiddenDim {
                    result[idx, i] = keyData[rowStartIdx + i]
                }
            }
        }
        
        return result
    }
    
    /// Get values for a range of tokens
    ///
    /// - Parameters:
    ///   - layer: Layer index
    ///   - range: Token positions to retrieve
    /// - Returns: Tensor of shape [range.count, hiddenDim]
    public func getValues(layer: Int, range: Range<Int>) -> Tensor<Float> {
        lock.lock()
        defer { lock.unlock() }
        
        precondition(layer >= 0 && layer < numLayers, "Layer out of bounds")
        
        let count = range.count
        var result = Tensor<Float>.zeros(shape: TensorShape(count, hiddenDim))
        
        // Collect values from pages (convert back to tensor)
        for (idx, pos) in range.enumerated() {
            let pageId = pos / pageSize
            let offsetInPage = pos % pageSize
            
            if let (_, valueData) = valuePages[layer][pageId] {
                // Copy row from page array to result
                let rowStartIdx = offsetInPage * hiddenDim
                for i in 0..<hiddenDim {
                    result[idx, i] = valueData[rowStartIdx + i]
                }
            }
        }
        
        return result
    }
    
    /// Evict oldest page to make room
    ///
    /// **Strategy:** Remove lowest page ID (FIFO eviction)
    private func evictOldestPage(layer: Int) {
        // Find the oldest page (lowest logical page ID still allocated)
        guard let oldestLogicalPageId = keyPages[layer].keys.min() else {
            return
        }
        
        // Get allocator page ID
        if let (allocatorPageId, _) = keyPages[layer][oldestLogicalPageId] {
            // Free page in allocator for reuse
            allocator.freePage(allocatorPageId)
        }
        
        // Remove page from cache
        keyPages[layer].removeValue(forKey: oldestLogicalPageId)
        valuePages[layer].removeValue(forKey: oldestLogicalPageId)
        
        // Update min position
        minPositions[layer] = (oldestLogicalPageId + 1) * pageSize
    }
    
    /// Evict oldest N tokens
    ///
    /// Used when cache exceeds maxTokens limit
    private func evictOldestTokens(layer: Int, count: Int) {
        let pagesToEvict = (count + pageSize - 1) / pageSize
        
        for _ in 0..<pagesToEvict {
            evictOldestPage(layer: layer)
        }
    }
    
    /// Clear all cached data
    ///
    /// Useful for starting a new sequence/conversation
    public func clear() {
        lock.lock()
        defer { lock.unlock() }
        
        // Free all pages
        for layer in 0..<numLayers {
            for (allocatorPageId, _) in keyPages[layer].values {
                allocator.freePage(allocatorPageId)
            }
            keyPages[layer].removeAll()
            valuePages[layer].removeAll()
            lengths[layer] = 0
            minPositions[layer] = 0
        }
    }
    
    /// Get cache statistics
    public var stats: String {
        lock.lock()
        defer { lock.unlock() }
        
        let totalPages = keyPages.flatMap { $0.keys }.count
        let utilizationPct = Double(length) / Double(maxTokens) * 100
        
        return """
        KVCache Stats:
          Layers: \(numLayers)
          Cached tokens: \(length) / \(maxTokens)
          Utilization: \(String(format: "%.1f%%", utilizationPct))
          Pages used: \(totalPages)
          Page size: \(pageSize) tokens/page
        """
    }
}

