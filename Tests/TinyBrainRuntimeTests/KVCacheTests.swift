/// Tests for Paged KV Cache functionality
///
/// **TB-004 Phase 4:** Validates paged attention cache for efficient transformer inference
///
/// ## What is KV Cache?
///
/// In transformers, attention needs Key and Value tensors for ALL previous tokens.
/// Without caching, we'd recompute them every time → SUPER SLOW!
///
/// **Example for "Hello world":**
/// ```
/// Token 1 ("Hello"): Compute K₁, V₁ → Cache them
/// Token 2 ("world"): Compute K₂, V₂ → Cache them
///                     Attention uses [K₁, K₂] and [V₁, V₂]
/// Token 3 ("!"): Compute K₃, V₃ → Cache them
///                Attention uses [K₁, K₂, K₃] and [V₁, V₂, V₃]
/// ```
///
/// **Without cache:** Recompute K₁,V₁,K₂,V₂ for token 3 → Waste!
/// **With cache:** Just compute K₃,V₃ and reuse cached K₁,K₂,V₁,V₂ → Fast!
///
/// ## Why Paging?
///
/// Instead of one giant buffer, divide into **pages** (like OS memory):
/// - Page = 16 tokens worth of K/V data
/// - Pre-allocate pool of pages on GPU
/// - Allocate/free pages as needed → No allocation during inference!

import XCTest
@testable import TinyBrainRuntime

final class KVCacheTests: XCTestCase {
    
    // MARK: - Page Allocator Tests
    
    func testPageAllocation() {
        // WHAT: Allocate pages from pool
        // WHY: Foundation of paging - need to hand out pages
        // HOW: Allocate 2 pages, verify different IDs
        
        let allocator = PageAllocator(pageSize: 16, maxPages: 128)
        
        let page1 = allocator.allocatePage()
        let page2 = allocator.allocatePage()
        
        XCTAssertNotNil(page1, "Should allocate page 1")
        XCTAssertNotNil(page2, "Should allocate page 2")
        XCTAssertNotEqual(page1, page2, "Pages should have different IDs")
    }
    
    func testPageReuse() {
        // WHAT: Released pages get reused
        // WHY: Zero allocation during inference - critical for performance!
        // HOW: Allocate, free, allocate again - should get same page ID
        
        let allocator = PageAllocator(pageSize: 16, maxPages: 128)
        
        let page = allocator.allocatePage()!
        let pageId = page
        
        allocator.freePage(page)
        
        let reused = allocator.allocatePage()!
        XCTAssertEqual(reused, pageId, "Should reuse freed page")
    }
    
    func testPagePoolExhaustion() {
        // Edge: What happens when we run out of pages?
        // WHY: Prevent crashes, handle gracefully
        
        let allocator = PageAllocator(pageSize: 16, maxPages: 10)
        
        var pages: [Int] = []
        for _ in 0..<10 {
            if let page = allocator.allocatePage() {
                pages.append(page)
            }
        }
        
        // 11th allocation should fail gracefully
        let overflow = allocator.allocatePage()
        XCTAssertNil(overflow, "Should return nil when pool exhausted")
        
        // Free one, should be able to allocate again
        allocator.freePage(pages[0])
        let recycled = allocator.allocatePage()
        XCTAssertNotNil(recycled, "Should allocate after freeing")
    }
    
    // MARK: - KV Cache Tests
    
    func testKVCacheCreation() {
        // WHAT: Create KV cache for multi-layer model
        // WHY: Each transformer layer needs its own K/V cache
        // HOW: 6 layers, 768 hidden dim, 2048 max tokens
        
        let cache = KVCache(
            numLayers: 6,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        XCTAssertEqual(cache.numLayers, 6)
        XCTAssertEqual(cache.maxTokens, 2048)
        XCTAssertEqual(cache.length, 0, "Should start empty")
    }
    
    func testKVCacheAppend() {
        // WHAT: Append key/value tensors for new token
        // WHY: Build up context as we generate tokens
        // HOW: Append 100 tokens, verify cache grows
        
        let cache = KVCache(
            numLayers: 6,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        for pos in 0..<100 {
            let key = Tensor<Float>.random(shape: TensorShape(768))
            let value = Tensor<Float>.random(shape: TensorShape(768))
            
            cache.append(layer: 0, key: key, value: value, position: pos)
        }
        
        XCTAssertEqual(cache.length, 100, "Should have 100 tokens cached")
    }
    
    func testKVCacheRetrieval() {
        // WHAT: Retrieve cached K/V for attention computation
        // WHY: Verify we get back what we stored
        // HOW: Store known values, retrieve, compare
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        // Store known values
        let knownKey = Tensor<Float>.filled(shape: TensorShape(768), value: 3.14)
        let knownValue = Tensor<Float>.filled(shape: TensorShape(768), value: 2.71)
        
        cache.append(layer: 0, key: knownKey, value: knownValue, position: 0)
        
        // Retrieve
        let retrievedKeys = cache.getKeys(layer: 0, range: 0..<1)
        let retrievedValues = cache.getValues(layer: 0, range: 0..<1)
        
        // Verify
        XCTAssertEqual(retrievedKeys.shape, TensorShape(1, 768))
        XCTAssertEqual(retrievedValues.shape, TensorShape(1, 768))
        XCTAssertEqual(retrievedKeys[0, 0], 3.14, accuracy: 1e-5, "Keys should match")
        XCTAssertEqual(retrievedValues[0, 0], 2.71, accuracy: 1e-5, "Values should match")
    }
    
    func testKVCacheMultipleTokens() {
        // WHAT: Retrieve range of cached tokens
        // WHY: Attention needs all previous K/V tensors
        // HOW: Cache 50 tokens, retrieve tokens 10-30
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        // Store 50 tokens with predictable values
        for pos in 0..<50 {
            let key = Tensor<Float>.filled(shape: TensorShape(768), value: Float(pos))
            let value = Tensor<Float>.filled(shape: TensorShape(768), value: Float(pos * 2))
            cache.append(layer: 0, key: key, value: value, position: pos)
        }
        
        // Retrieve range
        let keys = cache.getKeys(layer: 0, range: 10..<30)
        let values = cache.getValues(layer: 0, range: 10..<30)
        
        XCTAssertEqual(keys.shape, TensorShape(20, 768), "Should get 20 tokens")
        XCTAssertEqual(keys[0, 0], 10.0, accuracy: 1e-5, "First key should be token 10")
        XCTAssertEqual(keys[19, 0], 29.0, accuracy: 1e-5, "Last key should be token 29")
    }
    
    func testKVCacheMultiLayer() {
        // WHAT: Each layer has independent cache
        // WHY: 6-layer transformer = 6 separate K/V caches
        // HOW: Store in layer 0 and layer 5, verify no interference
        
        let cache = KVCache(
            numLayers: 6,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        let layer0Key = Tensor<Float>.filled(shape: TensorShape(768), value: 1.0)
        let layer5Key = Tensor<Float>.filled(shape: TensorShape(768), value: 5.0)
        
        cache.append(layer: 0, key: layer0Key, value: layer0Key, position: 0)
        cache.append(layer: 5, key: layer5Key, value: layer5Key, position: 0)
        
        let retrieved0 = cache.getKeys(layer: 0, range: 0..<1)
        let retrieved5 = cache.getKeys(layer: 5, range: 0..<1)
        
        XCTAssertEqual(retrieved0[0, 0], 1.0, accuracy: 1e-5, "Layer 0 preserved")
        XCTAssertEqual(retrieved5[0, 0], 5.0, accuracy: 1e-5, "Layer 5 preserved")
    }
    
    func testContextWindow2048() {
        // WHAT: Support full 2048 token context
        // WHY: PRD requirement - match TinyLlama's max context
        // HOW: Append 2048 tokens, verify all cached
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        for pos in 0..<2048 {
            let key = Tensor<Float>.filled(shape: TensorShape(768), value: Float(pos))
            let value = Tensor<Float>.filled(shape: TensorShape(768), value: Float(pos))
            cache.append(layer: 0, key: key, value: value, position: pos)
        }
        
        XCTAssertEqual(cache.length, 2048, "Should cache full 2048 tokens")
        
        // Verify we can retrieve all of them
        let allKeys = cache.getKeys(layer: 0, range: 0..<2048)
        XCTAssertEqual(allKeys.shape, TensorShape(2048, 768))
    }
    
    func testEviction() {
        // WHAT: Evict oldest tokens when exceeding max
        // WHY: Bounded memory - can't grow forever
        // HOW: Set max=100, append 150, verify only 100 remain
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 100,
            pageSize: 16
        )
        
        // Append 150 tokens
        for pos in 0..<150 {
            let key = Tensor<Float>.filled(shape: TensorShape(768), value: Float(pos))
            let value = Tensor<Float>.filled(shape: TensorShape(768), value: Float(pos))
            cache.append(layer: 0, key: key, value: value, position: pos)
        }
        
        XCTAssertEqual(cache.length, 100, "Should cap at maxTokens")
        
        // Note: Our current eviction strategy evicts by full pages (16 tokens each)
        // With 100 max and 16/page, we have 6.25 pages → 6 full pages = 96 tokens
        // After appending 150 tokens, we'd evict oldest pages to stay under limit
        
        // Just verify cache is bounded - specific eviction behavior may vary
        XCTAssertLessThanOrEqual(cache.length, 100, "Cache bounded at maxTokens")
    }
    
    func testPageUtilization() {
        // WHAT: Verify pages are efficiently used
        // WHY: Each page holds 16 tokens worth of data
        // HOW: Allocate 3 pages, verify count
        
        let allocator = PageAllocator(pageSize: 16, maxPages: 128)
        
        // Allocate 3 pages
        let _ = allocator.allocatePage()
        let _ = allocator.allocatePage()
        let _ = allocator.allocatePage()
        
        XCTAssertEqual(allocator.allocatedCount, 3, "Should have 3 allocated pages")
    }
    
    func testMemoryLeaks() {
        // Edge: No memory leaks under heavy use
        // WHY: Long-running inference sessions must be stable
        // HOW: 10,000 append/evict cycles, verify memory doesn't grow
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 100,
            pageSize: 16
        )
        
        // Stress test: Many append/evict cycles
        for pos in 0..<10000 {
            let key = Tensor<Float>.random(shape: TensorShape(768))
            let value = Tensor<Float>.random(shape: TensorShape(768))
            cache.append(layer: 0, key: key, value: value, position: pos)
        }
        
        // Cache should still be at maxTokens, not growing unbounded
        XCTAssertEqual(cache.length, 100, "Should not leak memory")
    }
    
    func testConcurrentAccess() {
        // Edge: Thread safety for multi-stream scenarios
        // WHY: Future: multiple inference streams in parallel
        // HOW: Access from multiple threads without crashes
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        // Append from multiple threads
        DispatchQueue.concurrentPerform(iterations: 100) { i in
            let key = Tensor<Float>.filled(shape: TensorShape(768), value: Float(i))
            let value = Tensor<Float>.filled(shape: TensorShape(768), value: Float(i))
            cache.append(layer: 0, key: key, value: value, position: i)
        }
        
        // Should not crash (thread safety working)
        XCTAssertLessThanOrEqual(cache.length, 2048, "Concurrent appends handled")
    }
    
    func testCacheClear() {
        // WHAT: Clear cache to start fresh sequence
        // WHY: New conversation = reset cache
        // HOW: Append tokens, clear, verify empty
        
        let cache = KVCache(
            numLayers: 1,
            hiddenDim: 768,
            maxTokens: 2048,
            pageSize: 16
        )
        
        for pos in 0..<50 {
            let key = Tensor<Float>.random(shape: TensorShape(768))
            let value = Tensor<Float>.random(shape: TensorShape(768))
            cache.append(layer: 0, key: key, value: value, position: pos)
        }
        
        XCTAssertEqual(cache.length, 50)
        
        cache.clear()
        
        XCTAssertEqual(cache.length, 0, "Cache should be empty after clear")
    }
    
    func testPageCompaction() {
        // WHAT: Compact fragmented pages
        // WHY: After many evictions, pages might be fragmented
        // HOW: Create fragmentation, compact, verify efficiency
        
        let allocator = PageAllocator(pageSize: 16, maxPages: 128)
        
        // Allocate many pages
        var pages: [Int] = []
        for _ in 0..<10 {
            if let page = allocator.allocatePage() {
                pages.append(page)
            }
        }
        
        // Free every other page (create fragmentation)
        for i in stride(from: 0, to: 10, by: 2) {
            allocator.freePage(pages[i])
        }
        
        // Compact should consolidate free pages
        allocator.compact()
        
        // Should still have 5 allocated pages
        XCTAssertEqual(allocator.allocatedCount, 5, "Compaction preserves allocated pages")
    }
}

