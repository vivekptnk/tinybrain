/// FlashAttention kernel — fused QK^T + softmax + V in a single GPU pass
///
/// **Algorithm (online softmax):**
/// For each query row, iterate over key/value tiles (blocks of Bc columns).
/// Within each tile:
///   1. Compute S_tile = Q_tile @ K_tile^T / sqrt(headDim)
///   2. Apply causal mask (if enabled)
///   3. Online softmax: track running max & sum across tiles
///   4. Accumulate output: O += diag(correction) @ O_prev + P_tile @ V_tile
///
/// This avoids materializing the full [seqLen × seqLen] attention matrix,
/// keeping memory usage O(Br × Bc) per threadgroup instead of O(seqLen²).
///
/// **GQA support:** numHeads can differ from numKVHeads via head-index mapping.
/// Each query head maps to kvHead = queryHead / (numHeads / numKVHeads).
///
/// **Dispatch shape:**
///   - threadgroupsPerGrid: (numHeads, ceilDiv(seqLen, Br), batch)
///   - threadsPerThreadgroup: (Bc, Br, 1)
///
/// **Buffers:**
///   [0] query    — float [batch, seqLen, numHeads * headDim]  (or [seqLen, numHeads * headDim] if batch=1)
///   [1] keys     — float [batch, kvSeqLen, numKVHeads * headDim]
///   [2] values   — float [batch, kvSeqLen, numKVHeads * headDim]
///   [3] output   — float [batch, seqLen, numHeads * headDim]
///   [4] mask     — float [kvSeqLen] or nullptr (causal mask: 0.0 = attend, -inf = ignore)
///   [5] params   — AttentionParams struct

#include <metal_stdlib>
using namespace metal;

/// Attention parameters passed from Swift dispatch
struct AttentionParams {
    uint seqLen;       // Query sequence length
    uint kvSeqLen;     // Key/Value sequence length (may differ from seqLen for cached KV)
    uint headDim;      // Dimension per head
    uint numHeads;     // Number of query heads
    uint numKVHeads;   // Number of KV heads (GQA: may be < numHeads)
    uint batch;        // Batch size
    float scale;       // 1.0 / sqrt(headDim)
    uint useMask;      // 1 if mask buffer is valid, 0 otherwise
};

// Tile sizes — tuned for Apple Silicon threadgroup memory limits (32 KB max)
// Br = rows of Q processed per threadgroup, Bc = columns of K processed per tile
// For Br=Bc=32, headDim=128: smem = 32*32 + 32 + 32 + 32*128 = 5184 floats = ~20 KB
constant uint TILE_Br = 32;
constant uint TILE_Bc = 32;

/// FlashAttention kernel with online softmax
///
/// Each threadgroup handles one query head for one block of Br query positions.
/// It iterates over all KV positions in tiles of Bc, maintaining running
/// softmax statistics (max, sum) for numerical stability.
kernel void flash_attention(
    device const float* query   [[buffer(0)]],
    device const float* keys    [[buffer(1)]],
    device const float* values  [[buffer(2)]],
    device float*       output  [[buffer(3)]],
    device const float* mask    [[buffer(4)]],
    constant AttentionParams& params [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],     // (head, qBlock, batch)
    uint3 tid  [[thread_position_in_threadgroup]],    // (tc, tr, 0) within tile
    threadgroup float* smem [[threadgroup(0)]]
) {
    const uint head     = tgid.x;
    const uint qBlock   = tgid.y;
    const uint batchIdx = tgid.z;

    const uint headDim   = params.headDim;
    const uint seqLen    = params.seqLen;
    const uint kvSeqLen  = params.kvSeqLen;
    const uint numHeads  = params.numHeads;
    const uint numKVHeads = params.numKVHeads;
    const float scaleFactor = params.scale;

    // GQA head mapping: which KV head does this query head use?
    const uint kvHead = head / (numHeads / numKVHeads);

    // Thread indices within the tile
    const uint tr = tid.y;  // row within Br (query position)
    const uint tc = tid.x;  // col within Bc (used for KV iteration)

    // Global query row this thread is responsible for
    const uint qRow = qBlock * TILE_Br + tr;
    // NOTE: Do NOT early-return here — all threads must reach every
    // threadgroup_barrier.  Use `validRow` to gate reads/writes instead.
    const bool validRow = (qRow < seqLen);

    // Strides for indexing into [batch, seqLen, numHeads * headDim]
    const uint qStride = numHeads * headDim;     // stride per position in Q
    const uint kvStride = numKVHeads * headDim;   // stride per position in K/V

    // Base pointers for this batch element
    device const float* Q_base = query  + batchIdx * seqLen * qStride;
    device const float* K_base = keys   + batchIdx * kvSeqLen * kvStride;
    device const float* V_base = values + batchIdx * kvSeqLen * kvStride;
    device float*       O_base = output + batchIdx * seqLen * qStride;

    // Pointer to this query row's head slice: Q[qRow, head*headDim .. (head+1)*headDim]
    device const float* q_row = Q_base + qRow * qStride + head * headDim;

    // Output pointer for this query row's head slice
    device float* o_row = O_base + qRow * qStride + head * headDim;

    // ── Threadgroup memory layout ───────────────────────────────────────
    // We need space for:
    //   1. S tile: Br × Bc floats (attention scores for current KV tile)
    //   2. Per-row running max: Br floats
    //   3. Per-row running sum: Br floats
    //   4. Per-row output accumulator: Br × headDim floats
    //
    // Total smem: Br*Bc + Br + Br + Br*headDim floats
    // For Br=Bc=32, headDim=64: 32*32 + 32 + 32 + 32*64 = 1024+64+2048 = 3136 floats = 12.25 KB

    threadgroup float* S_tile  = smem;                              // [Br × Bc]
    threadgroup float* row_max = smem + TILE_Br * TILE_Bc;          // [Br]
    threadgroup float* row_sum = row_max + TILE_Br;                 // [Br]
    threadgroup float* O_acc   = row_sum + TILE_Br;                 // [Br × headDim]

    // ── Initialize running statistics ───────────────────────────────────
    // Each thread handles its own row (tr) — initialize if tc == 0
    if (tc == 0) {
        row_max[tr] = -INFINITY;
        row_sum[tr] = 0.0;
    }

    // Initialize output accumulator to zero
    for (uint d = tc; d < headDim; d += TILE_Bc) {
        O_acc[tr * headDim + d] = 0.0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Iterate over KV tiles ───────────────────────────────────────────
    const uint numKVTiles = (kvSeqLen + TILE_Bc - 1) / TILE_Bc;

    for (uint kvTile = 0; kvTile < numKVTiles; kvTile++) {
        const uint kvStart = kvTile * TILE_Bc;
        const uint kvCol = kvStart + tc;  // Global KV position for this thread

        // ── Step 1: Compute S = Q @ K^T / sqrt(d) for this tile ─────────
        // Each thread computes one element: S[tr, tc] = dot(q[qRow], k[kvCol]) * scale
        float score = -INFINITY;
        if (validRow && kvCol < kvSeqLen) {
            score = 0.0;
            device const float* k_col = K_base + kvCol * kvStride + kvHead * headDim;
            for (uint d = 0; d < headDim; d++) {
                score += q_row[d] * k_col[d];
            }
            score *= scaleFactor;

            // ── Step 2: Apply mask if provided ──────────────────────────
            if (params.useMask != 0 && mask != nullptr) {
                score += mask[kvCol];  // mask contains 0.0 or -inf
            }
        }

        S_tile[tr * TILE_Bc + tc] = score;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Step 3: Online softmax update ───────────────────────────────
        // Find tile max for this row
        // Thread tc=0 computes the max across all Bc columns for row tr
        if (tc == 0 && validRow) {
            float tile_max = -INFINITY;
            for (uint c = 0; c < TILE_Bc; c++) {
                tile_max = max(tile_max, S_tile[tr * TILE_Bc + c]);
            }

            float prev_max = row_max[tr];
            float new_max = max(prev_max, tile_max);

            // Correction factor for previous accumulations
            float prev_correction = exp(prev_max - new_max);
            float new_sum = row_sum[tr] * prev_correction;

            // Exponentiate scores with new max and accumulate sum
            for (uint c = 0; c < TILE_Bc; c++) {
                float s = S_tile[tr * TILE_Bc + c];
                float p = exp(s - new_max);
                S_tile[tr * TILE_Bc + c] = p;  // Overwrite with softmax numerator
                new_sum += p;
            }

            // Correct previous output accumulator
            for (uint d = 0; d < headDim; d++) {
                O_acc[tr * headDim + d] *= prev_correction;
            }

            row_max[tr] = new_max;
            row_sum[tr] = new_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── Step 4: Accumulate P @ V for this tile ──────────────────────
        // Each thread accumulates across headDim dimensions
        // O_acc[tr, d] += sum_c P[tr, c] * V[kvStart+c, d]
        if (validRow) for (uint d = tc; d < headDim; d += TILE_Bc) {
            float acc = 0.0;
            for (uint c = 0; c < TILE_Bc; c++) {
                uint kvPos = kvStart + c;
                if (kvPos < kvSeqLen) {
                    float p = S_tile[tr * TILE_Bc + c];
                    device const float* v_row = V_base + kvPos * kvStride + kvHead * headDim;
                    acc += p * v_row[d];
                }
            }
            O_acc[tr * headDim + d] += acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Normalize by sum and write output ───────────────────────────────
    if (validRow) {
        float inv_sum = 1.0 / (row_sum[tr] + 1e-7);
        for (uint d = tc; d < headDim; d += TILE_Bc) {
            o_row[d] = O_acc[tr * headDim + d] * inv_sum;
        }
    }
}
