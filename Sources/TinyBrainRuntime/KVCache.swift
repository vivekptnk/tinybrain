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

/// **REVIEW HITLER FIX:** Page storage using class (reference semantics)
///
/// This avoids copying entire pages on every dictionary access!
private final class KVPage {
    /// Raw Float array for fast access
    var data: [Float]
    
    /// Allocator page ID for cleanup
    let allocatorPageId: Int
    
    init(allocatorPageId: Int, size: Int) {
        self.allocatorPageId = allocatorPageId
        self.data = [Float](repeating: 0.0, count: size)
    }
}

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
    
    /// Storage for keys: [layer][logicalPageId] → KVPage
    /// **REVIEW HITLER FIX:** Uses class (reference semantics) to avoid copying!
    private var keyPages: [[Int: KVPage]]
    
    /// Storage for values: [layer][logicalPageId] → KVPage
    private var valuePages: [[Int: KVPage]]
    
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
            
            // **REVIEW HITLER FIX:** Create KVPage (class = no copying!)
            let keyPage = KVPage(allocatorPageId: allocatorPageId!, size: pageSize * hiddenDim)
            let valuePage = KVPage(allocatorPageId: allocatorPageId!, size: pageSize * hiddenDim)
            
            keyPages[layer][pageId] = keyPage
            valuePages[layer][pageId] = valuePage
        }
        
        // Write key/value into page (FAST: direct mutation, no copying!)
        if let keyPage = keyPages[layer][pageId],
           let valuePage = valuePages[layer][pageId] {
            
            // **REVIEW HITLER FIX:** Direct mutation (reference semantics)
            // No copying! We mutate the class's internal array directly
            let rowStartIdx = offsetInPage * hiddenDim
            let keyInput = key.rawData
            let valueInput = value.rawData
            
            // Bulk write directly to page's data array
            for i in 0..<hiddenDim {
                keyPage.data[rowStartIdx + i] = keyInput[i]
                valuePage.data[rowStartIdx + i] = valueInput[i]
            }
            // No need to store back - it's a class reference!
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
            
            if let keyPage = keyPages[layer][pageId] {
                // Copy row from page to result
                let rowStartIdx = offsetInPage * hiddenDim
                for i in 0..<hiddenDim {
                    result[idx, i] = keyPage.data[rowStartIdx + i]
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
            
            if let valuePage = valuePages[layer][pageId] {
                // Copy row from page to result
                let rowStartIdx = offsetInPage * hiddenDim
                for i in 0..<hiddenDim {
                    result[idx, i] = valuePage.data[rowStartIdx + i]
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
        
        // Free allocator pages
        if let keyPage = keyPages[layer][oldestLogicalPageId] {
            allocator.freePage(keyPage.allocatorPageId)
        }
        if let valuePage = valuePages[layer][oldestLogicalPageId] {
            allocator.freePage(valuePage.allocatorPageId)
        }
        
        // Remove pages from cache
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
            for keyPage in keyPages[layer].values {
                allocator.freePage(keyPage.allocatorPageId)
            }
            for valuePage in valuePages[layer].values {
                allocator.freePage(valuePage.allocatorPageId)
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

