# Universal GPU optimization techniques (backend-neutral theory)

Terminology: "warp" = 32 threads (NVIDIA, RDNA wave32 default); "wave" = 32 or 64 lanes (AMD; CDNA is wave64). "smem" = CUDA shared memory, "LDS" = AMD local data share. Each backend file applies these ideas with concrete instructions.

## Contents
1. Memory hierarchy and what to optimize at each level
2. Coalescing and vectorization
3. Bytes in flight (Little's law)
4. Shared memory / LDS and bank conflicts
5. Registers, occupancy, ILP
6. Warp/wave-level primitives and reductions
7. Tiling math
8. Software pipelining and async movement
9. Work decomposition: persistent kernels, split-K, stream-K, rasterization
10. Atomics
11. Fusion
12. Launch overhead
13. Instruction-level tricks
14. Mixed precision and numerics

---

## 1. Memory hierarchy

Fastest to slowest: registers -> shared memory/LDS -> L1 (often shared with smem on NVIDIA) -> L2 -> (AMD: Infinity Cache/MALL) -> HBM/GDDR/LPDDR. Each level up buys roughly an order of magnitude in bandwidth. Optimization is mostly about doing each unit of data movement at the highest level possible:

- Register reuse: outer-product tiles (each loaded element used for many FMAs/MMAs).
- smem/LDS reuse: cooperative tile loads, each element reused by all warps in the block.
- L2 reuse: block/tile scheduling order (rasterization) so concurrently-running blocks share operands.
- DRAM: touch each byte once; fuse ops; compress (FP8/FP4/INT4) when the op is memory-bound.

## 2. Coalescing and vectorization

- Consecutive lanes should touch consecutive addresses. A warp's 32 x 16 B = 512 B request becomes four 128 B lines, ideal.
- Use 128-bit loads/stores (`float4`, `uint4`, `int4`, or `__nv_bfloat162`x4 etc.). Check the generated instruction (`LDG.E.128`, `global_load_dwordx4`); the compiler will not vectorize through possible aliasing or misalignment. Use `__restrict__`, `__builtin_assume_aligned`, `alignas`, or reinterpret pointers after proving alignment.
- Handle heads/tails separately: peel an unaligned prefix scalar, run the aligned body vectorized, finish with a scalar tail (or predicated vector store). A runtime alignment check once per row/tensor is cheap.
- Prefer SoA / row-contiguous layouts for the dimension threads map to. If the problem is strided, transpose through smem/LDS rather than issuing strided global accesses.
- Use 32-bit offsets inside kernels when extents allow; promote to 64-bit only on the base pointer arithmetic. 64-bit indices double address VGPRs on AMD and add IMAD/IADD3 pairs on NVIDIA.
- Streaming data that will not be reused: use non-temporal / streaming cache hints (`__ldcs/__stcs`, `ld.global.cs`; `__builtin_nontemporal_load/store` on AMD) to avoid evicting reusable data from L2.

## 3. Bytes in flight

To saturate DRAM you need `in_flight_bytes >= bandwidth * latency`. For ~3 TB/s and ~600-800 ns that is several MB across the GPU, i.e. tens of KB per SM. Ways to get it: more resident warps, more independent loads per thread before the first use (unroll and batch loads up front), 128-bit vectors, async copies (`cp.async`, TMA, global->LDS) that do not consume registers while in flight.

Rule: in a streaming kernel, issue all loads for the unroll block first, then compute, then store; do not interleave load->use->load->use.

## 4. Shared memory / LDS and bank conflicts

- NVIDIA: 32 banks x 4 B. A warp access is conflict-free if each bank is hit by at most one distinct 32-bit word (broadcast of the same word is free). 128-bit accesses are processed in quarter-warp phases.
- AMD (CDNA/RDNA): LDS also banked at 4 B (32 banks on most gfx9/gfx10+ parts; verify wider-LDS parts like gfx950 in the profile). Conflict behaviour of `ds_read_b64/b128` is per sub-group of lanes; measure with `SQ_LDS_BANK_CONFLICT`.
- Fixes, in order of preference when you control layout: XOR swizzle (zero wasted space, composes with `ldmatrix`/MFMA operand loads), padding (`+1`/`+4` words per row; simple, wastes space and breaks 16 B alignment if careless), transposed storage.
- Check both the **write** pattern and the **read** pattern; a layout that is conflict-free on write is often 4-8-way on the MMA operand read.
- smem/LDS capacity directly limits occupancy and pipeline depth: budget `stages * (A_tile + B_tile)` bytes plus epilogue staging (reuse the same buffer when the mainloop is finished).

## 5. Registers, occupancy, ILP

- Occupancy is a means, not a goal. Memory-bound streaming kernels need enough warps for latency hiding; compute-bound tiled kernels (GEMM/attention) usually want **few warps with big register tiles** (1-2 CTAs/SM with 255 regs/thread on NVIDIA; 1-2 waves/SIMD with up to 512 VGPR+AGPR on CDNA).
- Dynamic indexing of a register array puts it in local memory (spill). Fully unroll loops that index register arrays and use compile-time indices (`#pragma unroll`, `constexpr` loops, `std::index_sequence`/`if constexpr` fold).
- Use `__launch_bounds__(maxThreads, minBlocksPerSM)` (CUDA) or `__launch_bounds__(maxThreads)` + `amdgpu_waves_per_eu` (HIP) instead of global register caps. CUDA also has `__maxnreg__(N)` per kernel.
- ILP: independent accumulator chains hide FMA latency (4-8 cycles NVIDIA; ~4-8 cycles VALU on AMD). A reduction over one accumulator is latency-bound; use >= 4 independent accumulators and combine at the end.
- Register pressure from address computation: hoist invariant address math, use one base pointer plus compile-time immediate offsets (unrolled access patterns fold into `[reg+imm]`).

## 6. Warp/wave primitives and reductions

- Reduction hierarchy: thread-local (vector loads + multiple accumulators) -> intra-warp/wave (shuffles; DPP on AMD) -> inter-warp via smem/LDS (one partial per warp) -> inter-block via second kernel, atomics, or a "last block done" pattern with a threadfence and ticket counter.
- Fewer barriers is better: warp-level reductions need no block barrier; one smem round-trip for the cross-warp step.
- Butterfly (`xor`) shuffles leave the result on all lanes (useful for softmax max/sum broadcast); `down` shuffles leave it on lane 0 only.
- Votes/ballots (`__ballot_sync`, `__any_sync`; `__ballot`, `__any` on HIP with 64-bit masks on wave64) enable compaction, early exit, histogramming via match.
- Always be explicit about the lane mask type: 32-bit on NVIDIA, **64-bit on wave64 AMD**.

## 7. Tiling math

For a GEMM-like tile `Bm x Bn x Bk` computed by one block: global traffic per block per k-step = `(Bm+Bn)*Bk*sizeof(T)`, FLOPs = `2*Bm*Bn*Bk`, so AI_global = `Bm*Bn / (Bm+Bn) / sizeof(T)` FLOP/B (x2). Bigger square-ish tiles raise AI; limits are registers (accumulators `Bm*Bn*4 B / threads`) and smem (stages). Same math one level down for warp tiles vs smem bandwidth: a warp tile `Wm x Wn` needs `(Wm+Wn)*k` smem bytes per `Wm*Wn*k` MACs; if smem bandwidth is the limiter, grow the warp tile or use register-level operand reuse.

Choose the *instruction tile* from the matrix unit (e.g. mma m16n8k16, wgmma m64nNk16, tcgen05 M128 N<=256 K16, MFMA 32x32x8 / 16x16x16), the *warp tile* from register budget, the *block tile* from L2/smem budget and wave quantization.

## 8. Software pipelining and async movement

Pattern: `prologue: issue loads for stages 0..S-2` -> `mainloop: wait stage i; issue load for stage i+S-1; compute stage i` -> `epilogue`. Stage count S: 2 (double buffer) when smem is scarce, 3-5 on Ampere+, 4-8 with TMA. More stages hide more latency but cost smem and reduce CTAs/SM. Prefer **async** copy paths that do not occupy registers while in flight (cp.async, TMA, global->LDS direct loads). Register-staged prefetch (load into regs, store to smem after compute) is the portable fallback and works on every AMD part.

## 9. Work decomposition

- **Wave quantization:** with `T` tiles and `S` SMs/CUs, efficiency = `T / (ceil(T/S)*S)`. Fix with smaller tiles at the tail, split-K, stream-K, or a persistent kernel with a tile scheduler.
- **Split-K / stream-K:** partition the K loop to fill idle SMs; reduce partials through workspace (deterministic fixed-order reduce) or atomics (non-deterministic). Essential for skinny/decode shapes.
- **Persistent kernels:** launch `#SMs x CTAs/SM` blocks that loop over a tile queue. Removes launch latency/tail between tiles, lets the epilogue of tile *i* overlap the mainloop of tile *i+1* (producer/consumer roles), and allows custom rasterization.
- **Rasterization / swizzled tile order:** map consecutive scheduled CTAs to a 2-D patch of the output (e.g. groups of 8-16 tile-rows) so concurrently-resident CTAs share A/B panels in L2. On multi-die AMD parts, also account for XCD assignment (see `hip-techniques.md`).

## 10. Atomics

- Global float/int atomics are fast in L2, slow under contention. Warp-aggregate first (`match`/ballot + leader or shuffle-reduce) so one lane per distinct address issues one atomic.
- Prefer a reduction tree + single write; use atomics for histogram-like irregular targets or when the ordering is irrelevant.
- Shared-memory atomics are cheap on modern parts (native on NVIDIA; native LDS atomics on AMD) and are the right place to privatize histograms before one global flush per block.
- 16-bit float atomics (`atomicAdd` on `__half2`/`__nv_bfloat162`) exist but have lower throughput and precision hazards; accumulate in fp32 where possible.

## 11. Fusion

- Highest-ROI optimization for memory-bound chains: every removed intermediate tensor saves one write + one read.
- Fuse into the **epilogue** of the producer (GEMM + bias + activation + residual + quantize/convert + store), into the **prologue** of the consumer (dequantize-on-load, normalize-on-load), or horizontally (multi-tensor apply, grouped GEMM).
- Cost: register pressure and code complexity. Template on epilogue functors/`constexpr` flags so the compiler specializes; do not branch on runtime flags inside hot loops.

## 12. Launch overhead

Each launch costs ~2-5 us of host/driver time and leaves a GPU gap of ~1-3 us. For sequences of small kernels: CUDA Graphs / HIP Graphs (capture once, replay), Programmatic Dependent Launch on sm_90+ to overlap the next kernel's prologue (weight prefetch, smem setup) with the previous kernel's tail, fusion, or one persistent megakernel. Avoid `cudaDeviceSynchronize`/`hipDeviceSynchronize` and implicit syncs (pageable memcpy, `cudaMalloc`/`hipMalloc` in steady state; use stream-ordered allocators: `cudaMallocAsync`/`hipMallocAsync` or a caching pool).

## 13. Instruction-level tricks

- Integer divide/modulo by runtime constants: precompute a magic multiplier (fastdiv) per launch; by powers of two use shifts/masks. Division is the most common hidden ALU hog in index math (NHWC/NCHW decomposition, attention head indexing).
- `exp(x)` -> `exp2(x * log2e)` with the multiply folded into an earlier scale (softmax scale); hardware `ex2.approx` (NVIDIA) / `v_exp_f32` (AMD) are native one-instruction ops.
- Reciprocal/sqrt: `rsqrtf`, `__frcp_rn`/`__fdividef`, `v_rcp_f32`, `v_rsq_f32`. State the error bound.
- Packed math: `half2`/`bfloat162` FMAs (NVIDIA), `v_pk_fma_f16`, `v_pk_add_f32`/`v_pk_mul_f32`/`v_pk_fma_f32` (gfx90a+ packed FP32, 2x FP32 throughput on CDNA2/3 for the right patterns).
- Bit manipulation: LOP3 (arbitrary 3-input logic, NVIDIA; `v_bitop3` on newest AMD, `v_and_or`/`v_xad` earlier), `PRMT`/`v_perm_b32` for byte shuffles, `BFE/BFI`. Essential for int4/fp4 dequant.
- Predication beats branches for short divergent sections; branches beat predication when a whole warp skips heavy work uniformly.
- Use `const`/`__restrict__` and the read-only path (`__ldg`, `ld.global.nc`, scalar loads `s_load` on AMD for wave-uniform data) to let the compiler pick the cheapest load path.

## 14. Mixed precision and numerics

- Accumulate in FP32 (or higher) for FP16/BF16/FP8/INT8 inputs; INT32 for INT8.
- Hopper FP8 tensor-core accumulation has reduced effective precision before being added to the FP32 accumulator: for long K, periodically promote partial sums into a separate FP32 accumulator (interval ~ 128 K-elements is the common choice; validate on your data).
- Scale handling: per-tensor scales fold into the epilogue; per-block/group scales (MX, NVFP4, 1x128 / 128x128 block scaling) must be applied in-mainloop or via hardware block-scaled MMA where available (sm_100/120 on NVIDIA; gfx950 scaled MFMA on AMD).
- Online algorithms (online softmax, Welford variance) keep a running max/mean and rescale accumulators; they trade a few extra FMAs for one fewer pass over memory.
- Stochastic rounding / round-to-nearest-even choices matter for training: pick explicitly rather than inheriting the default conversion.
