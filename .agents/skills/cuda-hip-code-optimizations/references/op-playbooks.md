# Operation playbooks (CUDA and HIP, designed separately)

Each recipe lists: bound, plan, per-backend specifics, and "what to measure". Start from the vendor-library baseline; deviate only for fusion, shapes, or dtypes the library does not cover.

## Contents
1. Streaming / elementwise
2. Reductions (sum/max/mean, argmax, norms)
3. Softmax, LayerNorm, RMSNorm
4. GEMM (dense, batched, grouped, skinny/GEMV, FP8/FP4)
5. Attention (prefill, decode, paged KV)
6. Convolution
7. Scan, sort, top-k, histogram
8. Gather / scatter / embedding / index ops
9. Transpose / permute / layout conversion
10. Quantization and dequantization
11. Optimizers and multi-tensor ops
12. MoE routing and grouped GEMM
13. Tiny-op chains (launch-bound)

---

## 1. Streaming / elementwise (memory-bound)

Floor: `bytes / measured_copy_bw`.
- 16 B vectors per thread (32 B on sm_100+), unroll 2-4 independent vector loads before compute, grid-stride or one tile per block with `grid = min(ceil(n/tile), waves * #SM)`.
- Peel head/tail; check alignment once. 32-bit index math.
- Fuse chains of elementwise ops and dtype conversions into one pass; broadcast operands via scalar/uniform loads (AMD: `s_load`; NVIDIA: constant/`__ldg` + broadcast).
- Packed conversions: `cvt.rn.bf16x2.f32`, FP8 `cvt.rn.satfinite.e4m3x2.f32` (CUDA); `v_cvt_pk_bf16_f32` (gfx950), `v_cvt_pk_fp8_f32` (gfx942/950) (HIP).
- Activation `exp/tanh/erf`: use `exp2` form; GELU-tanh via `tanh.approx` on NVIDIA when accuracy allows; AMD `v_exp_f32`, `v_rcp_f32`.
- Measure: achieved GB/s vs copy kernel; if < 90 %, look at load width, tail waves, TLB (huge tensors), and instruction-bound math (SASS/GCN instruction count per element).

## 2. Reductions

Bound: memory for large inputs; latency for many small rows.
- Large 1-D: per-thread vector accumulation with 4-8 accumulators -> warp reduce -> block reduce -> second-pass (or last-block-done with `__threadfence()`/release-acquire). Use 2-4 CTAs/SM worth of blocks; avoid atomics unless nondeterminism is allowed.
- Row-wise (N rows x C cols): choose mapping by C:
  - C <= ~256: one warp (wave) per row or several rows per warp (shuffle-only, no smem, no barriers).
  - 256 < C <= ~8-16K: one block per row, vectorized, register-resident row if it fits (avoid re-reading).
  - C larger: cluster/DSMEM (sm_90+) or two-pass/split row across blocks.
- Argmax/argmin: carry (value,index) pairs with a stable tie rule (lowest index); packed 64-bit compare tricks (value bits in high half) allow single-instruction `max` on integers after monotone float->uint mapping.
- AMD: wave64 reduces one more step; use DPP-based wave reduce, cross-wave via LDS; for tiny rows two rows per wave64 can match NVIDIA's warp-per-row density.
- Matrix-unit reduction trick: when ALU-bound (e.g. very low-precision inputs), reduce via MMA against a ones vector (tensor core / MFMA).

## 3. Softmax, LayerNorm, RMSNorm

Bound: memory (read x once, write y once) unless C is tiny.
- **Softmax:** online algorithm (running max `m`, running sum `l` rescaled by `exp(m_old - m_new)`), single read pass + write pass; keep the row in registers when `C/threads*sizeof` fits; otherwise two passes with L2/smem reuse. Use `exp2` with `x*log2e`. Mask (causal/padding) folded into the load predicate. Output in the consumer's dtype.
- **LayerNorm/RMSNorm:** one block per row; Welford (or two-moment in fp32 when C moderate) for variance; `rsqrt`; fuse residual add, bias, scale/shift, dropout, and output quantization (FP8/INT8 with row/block `amax` -> scale) in the same kernel; save `mean/rstd` (fp32) for backward.
- **Backward:** dx needs two row reductions (`sum(dy*gamma)`, `sum(dy*gamma*xhat)`): do in one pass over the row with both sums, second pass for output; `dgamma/dbeta` = column reductions across rows: accumulate per-block partials in smem then a deterministic second kernel (or atomics if allowed).
- Small C (<= 1024): multiple rows per block; use warp-per-row to eliminate barriers.
- Wide C (>= 16K), Hopper+: cluster of 2-8 CTAs per row with DSMEM reductions.
- AMD: C a multiple of 8/16 for 128-bit loads; waves-per-row mapping tuned for wave64; block 256-1024 threads; compare with MIOpen/aiter/rocm "norm" kernels.

## 4. GEMM

Bound: compute for large M,N,K; memory for skinny shapes (decode: M <= 64).
Shape classes and strategies:
| Class | Strategy |
|---|---|
| Large square-ish | Max-size tiles, persistent, rasterized, async pipeline, matrix-unit saturating |
| Medium | Smaller tiles / more CTAs, stream-K to fix wave quantization |
| Skinny M (1-64) | **Memory-bound weight streaming**: split-K or split-N across SMs, vectorized weight loads, dequant in registers, tiny M tile (m16 MMA with padding or CUDA-core FMA for M <= 4), activations broadcast/in registers |
| Batched | Batch index in grid z; ensure each batch gets enough CTAs; grouped GEMM for variable shapes |
| Weight-only quant (INT4/FP4) | Dequant fused in mainloop prologue; LOP3/PRMT (NVIDIA) or `v_perm`/`v_and_or` + bit tricks (AMD); keep scales per group in registers/LDS |

CUDA tiers:
- sm_75: 2-stage register-prefetch mainloop, `mma.sync.m16n8k8`, smem 64 KB.
- sm_80/86/89: cp.async 3-5 stages, ldmatrix with swizzle, `mma.sync.m16n8k16`; sm_86/89 smaller smem tile configs; sm_89 FP8 `m16n8k32`.
- sm_90: persistent warp-specialized TMA + wgmma, cluster multicast, 2 consumer warpgroups pingpong/cooperative, `setmaxnreg`, TMA-store epilogue; FP8 with fp32 promotion.
- sm_100/103: tcgen05 2-CTA 256xN tiles, TMEM double-buffered accumulators, dedicated epilogue warps, block-scaled MXFP8/NVFP4, CLC/stream-K.
- sm_110: as sm_100 if tcgen05 is supported (verify); expect bandwidth-limited decode.
- sm_120/121: Ampere-style `mma.sync` (incl. block-scaled FP4/FP6/FP8) + TMA/cp.async; no tcgen05.
HIP tiers:
- gfx908: MFMA with AGPR accumulators, 2-stage LDS, 64 KB LDS.
- gfx90a: MFMA unified registers, BF16_1k, packed FP32 epilogue math.
- gfx942: 256-thread WG, 256x256 tiles, ping-pong LDS, `sched_group_barrier`, XCD remap, FP8 FNUZ, `s_setprio` pingpong between wave pairs.
- gfx950: + direct-to-LDS b128 loads, LDS transpose reads, 160 KB LDS, bigger-K MFMA, scaled MFMA (MXFP8/6/4).
- RDNA2: VALU SGEMM with dot2/dot4; RDNA3/3.5: WMMA wave32 (replicated layout); RDNA4: WMMA v2 + FP8 + swmmac.

Epilogue fusion: bias, activation (relu/gelu/silu), residual, scaling/amax output, quantization, transposed store. Make it a functor template; do not branch at runtime in the mainloop.
Heuristics: build a tile-config table keyed by (arch, dtype, M,N,K buckets) and autotune offline; cache by problem shape. Tune `splitK` and tile order. Always offer a fallback to the vendor library for shapes with no tuned config.
Measure: tensor/MFMA pipe utilization, smem/LDS bank conflicts, L2 hit rate, SM/CU occupancy (should be 1-2 CTAs with full registers), achieved TFLOP/s vs library at the same clocks.

## 5. Attention

Bound: prefill compute (tensor) / MUFU (Blackwell); decode memory (KV-cache read).
- **Prefill (FlashAttention-style):** tile Q (block rows) vs stream K/V tiles; S = QK^T in registers/TMEM, online softmax, O accumulate with rescale; never materialize S. Causal: skip fully-masked tiles, mask only diagonal tiles. GQA/MQA: process all Q heads that share a KV head in the same CTA to amortize K/V loads. Variable length: persistent scheduler over (batch, head, q-tile) with longest-first ordering.
- Hopper (sm_90): FA3-style: producer TMA warp, 2 consumer warpgroups pingpong (softmax of one overlaps wgmma of the other), intra-warpgroup pipelining of QK and PV, FP8 with block-wise quantization / incoherent processing for accuracy.
- Blackwell DC (sm_100/103): FA4-style: S and O in TMEM, MMA warp (single thread) issuing QK and PV tcgen05, softmax warps, correction warps, **partial software exp2 on FMA units** because MUFU is the limiter; conditional rescaling (skip rescale unless max rises by a threshold); 2-CTA mode for larger tiles.
- Ampere/Ada (sm_80/86/89): FA2-style with mma.sync, cp.async double buffer, 128-row Q tiles, 4 warps, head dim 64/128 specializations; sm_89 FP8 variants.
- sm_120/121: FA2-style mma.sync + TMA, FP8/FP4 optional; smem limit 99 KB constrains head-dim 128 tile sizes.
- CDNA3/4: CK/aiter FMHA-style: MFMA 32x32x8/16x16x16 (gfx942) or 16x16x32 (gfx950), K/V tiles in LDS with swizzle, direct-to-LDS on gfx950, LDS budget 64 KB (gfx942) forces smaller tiles/Q in registers; XCD remap over (batch,head).
- RDNA3/4: WMMA flash attention with wave32, Q tile in registers, K/V through LDS; limited bandwidth: head-dim 64/128 with small tiles.
- **Decode (q_len=1) / split-KV:** partition KV length across CTAs (flash-decoding), each writes partial (O, m, l) then a combine kernel; memory-bound: KV in FP8/INT4 with in-register dequant, vector loads of 16 B along head_dim, multiple KV heads per CTA.
- **Paged KV:** block table lookup per tile (scalar loads), blocks of 16-128 tokens; prefetch next block index early; TMA gather (Hopper+) or per-lane pointer arrays; keep page size >= 16 for coalescing.
- Numerics: fp32 softmax statistics; log-sum-exp saved for backward; masks use `-inf` safely (guard `exp(-inf - -inf)` NaNs when an entire tile row is masked).

## 6. Convolution

- Dense 3x3/1x1 on tensor cores: implicit GEMM (NHWC / channels-last), im2col done on the fly by the loader (index math via fastdiv; precompute per-tile offsets). 1x1 = GEMM.
- Winograd F(2x2,3x3)/F(4x4,3x3) for stride-1 3x3 with enough channels; FFT for large kernels; direct for tiny channels.
- Depthwise: memory-bound; tile the spatial plane in smem/LDS (halo) with vectorized channel-contiguous access (NHWC, 8 channels/thread).
- Use cuDNN/MIOpen as the baseline; custom only for fusion (conv+bias+act+residual+quant) or unusual layouts.
- Measure: tensor utilization (implicit GEMM), index-math instruction share (should be small).

## 7. Scan, sort, top-k, histogram

- **Scan:** decoupled look-back single-pass (CUB `DeviceScan`, rocPRIM `inclusive_scan` use it): per-tile aggregate publish + look-back with flags; requires forward-progress guarantees (CUDA: fine in practice with tile-ordered launch; AMD: same pattern in rocPRIM). Block scan via warp shuffles (`shfl_up`) + smem partials.
- **Radix sort:** onesweep (CUB), 8-bit digits, key-value; segmented sort for many small segments (warp/block-level sort via bitonic/merge in registers). Radix select for top-k with large N; for small k (<= 32/64) and large N: per-thread partial top-k in registers -> warp merge (bitonic/sorting network) -> block merge.
- **Histogram:** privatize in smem/LDS (per-warp copies reduce atomic contention), flush with global atomics; for few bins use warp-level match/ballot counting; bin count > smem -> multi-pass or sort-based.
- **Compaction/select:** ballot + popcount prefix per warp, scan across warps, scatter.

## 8. Gather / scatter / embedding / index ops

- Row gather (embedding lookup): one warp/wave (or block) per row, 16 B vectors along the embedding dim; prefetch index with scalar load; rows of 128-512 B ideal. Sort indices (or use batch dedup) for cache locality in training.
- Scatter-add / embedding backward: sort indices and do segmented reduction (deterministic) vs atomics with warp aggregation (fast, nondeterministic); for fp16/bf16 accumulate in fp32 workspace.
- `index_select`, `take`, masked ops: fuse with consumers; avoid materializing index tensors in int64 when int32 suffices.
- Hopper+: TMA gather4/scatter variants for row-indexed tiles (verify availability in PTX ISA / CuTe); AMD: buffer loads with per-lane offsets give free bounds checking.

## 9. Transpose / permute / layout conversion

- 2-D transpose: 32x32 (or 64x64) tile through smem with +1 word padding or XOR swizzle; 16 B vector loads/stores, each thread handling 4x4 sub-tiles; achieves ~copy bandwidth.
- Hopper+: TMA load tile, transpose in smem via `ldmatrix.trans`/`stmatrix`, TMA store; eliminates register shuffles.
- AMD: LDS tile with padding; on gfx950 `ds_read_b64_tr_b16` performs the transpose on read.
- N-D permute: collapse dims to a 2-D or 3-D tiled transpose problem; precompute strides with fastdiv; fuse into producers/consumers whenever possible (the best transpose is the one you do not launch).

## 10. Quantization and dequantization

- Quantize kernel: read activations once, compute per-row/block `amax` (warp/wave reduce), scale = amax / fmt_max, convert with saturating packed converts, write quantized values + scales **in the exact layout the consumer MMA expects** (swizzled scale tensors for sm_100/sm_120 block-scaled MMA; MX/NVFP4 scale layouts; gfx950 scale operand layout).
- FP8 maxima: E4M3FN 448, E5M2 57344, E4M3FNUZ (gfx942) 240. Do not hard-code one for both vendors.
- Dequant in registers (weight-only): INT4 -> FP16 via LOP3 with magic bias (`0x6400` pattern) then sub/fma; INT8 -> FP16 via PRMT/`v_perm`; FP4 (E2M1) via 16-entry LUT in registers/`prmt`/shuffles or arithmetic; fuse scale multiply into the MMA operand or epilogue (per-group scales need mainloop application).
- Fused activation quant: RMSNorm/SwiGLU + quant in one kernel to avoid a bf16 round trip.
- Stochastic rounding for training: hardware instructions where present (verify per arch) else add random bits before truncation (Philox/xorshift state in registers).
- Validate: dequantized reference computation, per-block error histograms, saturation counts.

## 11. Optimizers and multi-tensor ops

- Fused Adam/AdamW/Lion/LAMB: pure streaming (read p,g,m,v; write p,m,v): 16 B vectors, fp32 states/master weights, bf16 params/grads with in-register upcast; one multi-tensor kernel handles a list of tensors via a chunk table (tensor id + offset per block) to avoid per-tensor launches.
- Gradient clipping: global norm = block partial sums -> single final reduction (deterministic), then scale folded into the optimizer kernel.
- Overlap with communication: launch per-bucket kernels on the compute stream as soon as the all-reduce bucket completes.

## 12. MoE routing and grouped GEMM

- Routing: top-k over experts per token (warp-level top-k for E <= 256), softmax/sigmoid gating fused; produce expert-sorted token index via counting sort (histogram + scan + scatter) - single kernel for small E.
- Grouped GEMM: one persistent kernel with a device-side tile scheduler over (expert, m-tile, n-tile); pad per-expert M to the tile granularity or use predicated tails; TMA descriptors per expert (Hopper+: pointer-array via tensor map in global memory + `tensormap.replace`) or gather loads with sorted indices; fuse gate-up + SwiGLU + down projection where possible.
- AMD: CK grouped GEMM/aiter MoE kernels as baseline; XCD remap across expert groups.
- Load imbalance: sort experts by token count descending, use stream-K-like splitting for hot experts.

## 13. Tiny-op chains (launch-bound)

- Capture in CUDA Graph / HIP Graph; enable PDL (sm_90+) with `griddepcontrol.wait` after prologue work.
- Fuse residual + norm + quant + small GEMV-like ops into one kernel; persistent megakernel with a software work queue and intra-kernel dependencies (flag counters in global memory with acquire/release) when dozens of sub-10 us kernels run back-to-back.
- Pre-allocate workspaces; avoid dynamic allocation and host syncs; keep launch parameters in `__grid_constant__`/kernel args to avoid constant-memory updates.
