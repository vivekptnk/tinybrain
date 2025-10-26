On M‑series Pro/Max chips the CPU path isn’t “plain scalar”—Accelerate routes cblas_sgemm through AMX, which is a dedicated matrix engine. So the Metal goal isn’t to beat scalar CPU, it’s to beat AMX, which already gives you ~1 TFLOP on a single matmul. To see 8× gains you have to change the rules: give the GPU more useful work per launch and overlap transfers so AMX can’t keep up.

Approach:

Batch Whole Attention Blocks. Right now you benchmark one matmul at a time. Real inference runs Q/K/V projections, attention scores, softmax, V projection, FFN up/down. Keep those tensors on the GPU, fuse chains, and issue them back‑to‑back. AMX must do each matmul sequentially; Metal can overlap thousands of threads. Target: ≥4 matmuls per dispatch, flop reuse ∝8× vs isolated matmul.

Use Mixed Precision. AMX only handles Float32; Metal can run FP16/BF16. Quantize activations/weights to FP16/INT8 and use Metal pipeline states that operate on half/char. Halving the data doubles effective bandwidth and reduces register pressure. With INT8 matmul kernels you get 4× throughput vs FP32 and AMX can’t follow (no INT8 matmul). Combine with batched attention for >8×.

Optimize Kernels Beyond Shared-Memory Tiling. The current 16×16 tile is generic. For Apple GPU, move to:

Threadgroup memory double buffering (while computing tile N, prefetch tile N+1).
Wider vector loads (float4/half8) to match memory banks.
Cooperative matrix multiply using SIMDgroup matrix APIs (available since Metal 3) to leverage hardware matrix units.
Avoid bank conflicts by interleaving memory layout.
These improvements can bring 2‑3× over the simple tiled kernel before considering batching.

Asynchronous Command Scheduling. Use multiple command buffers and MTLSharedEvent to overlap transfers for the next batch with compute for the current batch. AMX is synchronous; GPU can saturate compute + DMA simultaneously, effectively hiding transfer costs.

Chunk Larger Problems. Instead of benchmarking 1024×1024, focus on real transformer shapes (e.g., 4096×14336). As size grows, GPU efficiency increases faster than AMX because you can keep more SMs busy. Aim for kernels that reach 70–80% occupancy on these real shapes.

Fuse Ops to Reduce Memory Traffic. AMX path writes each intermediate to RAM; GPU kernels can fuse bias+activation, attention softmax, etc. Less DRAM traffic → more effective FLOPs. If you lower memory bandwidth demand by 4× and raise compute throughput 2×, you’re already >8× vs AMX for the end-to-end layer.

Implementation steps:

Build an “Attention Block” pipeline in Metal: load Q/K/V weights once, keep keys/values resident, run matmul → softmax → matmul without CPU round trips.
Extend your quantized matmul kernel to support FP16/INT8 mixed precision and use SIMDgroup matrix instructions (simdgroup_matrix_multiply).
Profile with Xcode GPU counters; tune threadgroup sizes to maximize ALU utilization.
Benchmark end-to-end tokens/sec rather than standalone matmul to capture the real advantage.
Bottom line: You won’t get 8× from a single matmul on AMX hardware. You need to change the workload so the GPU does many chained operations in low precision with minimal transfers. Do that, and the GPU’s parallelism + INT8/FP16 throughput can put you well past AMX.