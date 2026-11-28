# CUDA techniques (CUDA 13.x, C++20, sm_75 .. sm_121)

## Contents
1. Kernel scaffolding: bounds, attributes, launch
2. Vectorized and cache-controlled memory access
3. cp.async multistage pipeline (sm_80/86/89/120)
4. ldmatrix/mma.sync mainloop (sm_75..sm_89, sm_120)
5. Hopper: mbarrier, TMA, wgmma, clusters, setmaxnreg
6. Blackwell DC: tcgen05, TMEM, CTA pairs, CLC
7. Warp specialization blueprint
8. Warp-level primitives
9. PDL, Graphs, streams
10. Libraries to build on (CUTLASS/CuTe, CUB, cuBLASLt)
11. C++20 usage in device code
12. Pitfalls

---

## 1. Kernel scaffolding

```cpp
template <int kThreads, int kMinBlocks>
__global__ void __launch_bounds__(kThreads, kMinBlocks)
my_kernel(const float* __restrict__ in, float* __restrict__ out, int n);
```
- `__launch_bounds__(maxThreads, minBlocksPerSM)` lets the compiler cap registers to hit the occupancy you chose. Prefer it to `-maxrregcount`. `__maxnreg__(N)` caps registers for one kernel; they are mutually exclusive on a kernel.
- Dynamic smem > 48 KB: `cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` once, before launching (outside timed region). Clusters: launch with `cudaLaunchKernelEx` and `cudaLaunchAttributeClusterDimension`; non-portable cluster size 16: `cudaFuncSetAttribute(k, cudaFuncAttributeNonPortableClusterSizeAllowed, 1)`.
- Choose block size as a multiple of 32 (and of warp-group 128 for wgmma kernels). Use `cudaOccupancyMaxPotentialBlockSize` only as a starting point.
- Arch-specific code: `#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)`; the host pass must still compile (use `__CUDA_ARCH__` guards only inside the kernel bodies). Check `__CUDA_ARCH_FEAT_SM90_ALL` / `__CUDA_ARCH_FEAT_SM100_ALL` style macros for the `a` targets.

## 2. Memory access

```cpp
// 128-bit load/store; check alignment (16 B) before using.
uint4 v = *reinterpret_cast<const uint4*>(in + i);
*reinterpret_cast<uint4*>(out + i) = v;
```
- Read-only/non-coherent path: `__ldg(ptr)` or mark `const __restrict__`; streaming hints: `__ldcs`, `__stcs` (evict-first), `__ldlu`; PTX `ld.global.L1::no_allocate`, `ld.global.L2::128B` / `.L2::256B` prefetch sizing, `prefetch.global.L2`, `createpolicy.fractional.L2::evict_last` + `.L2::cache_hint` operands for pinning hot data (e.g. weights re-read every step).
- sm_100+: 256-bit accesses with `_32a` vector types (e.g. `ulonglong4_32a`); ensure 32 B alignment. Measure: some paths only reach full rate with the wide form.
- L2 persistence (sm_80+): `cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, ...)` with `hitRatio`; reset after use, since persisting lines pollute other kernels.
- Avoid local-memory traffic: look for `STL/LDL` in SASS.
- Shared-memory vector access: `ld.shared.v4.b32`; align smem arrays to 16 B (`alignas(16)`), or 128 B for TMA/wgmma swizzle patterns (1024 B alignment for 128B-swizzle atoms).

## 3. cp.async multistage pipeline (Ampere-style)

```cpp
__device__ __forceinline__ void cp_async_16B_zfill(void* smem, const void* gmem, bool pred) {
  unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  int src_size = pred ? 16 : 0;                       // 0 -> zero-fill the 16 B
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(gmem), "r"(src_size));
}
__device__ __forceinline__ void cp_async_commit()   { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }
```
Mainloop skeleton (S stages):
```
for s in 0..S-2: issue_loads(stage s, k=s); cp_async_commit();
for k in 0..K_TILES:
    cp_async_wait<S-2>(); __syncthreads();         // stage k ready, previous consumers done
    if (k+S-1 < K_TILES) issue_loads(stage (k+S-1)%S, k+S-1);
    cp_async_commit();                              // always commit to keep group counting uniform
    mma_over_stage(k % S);                          // ldmatrix + mma.sync, fragments double-buffered in regs
```
- `.cg` bypasses L1 (16 B only) and is the right default for streaming tile loads; `.ca` for small (4/8 B) copies you want in L1.
- Double-buffer the *register fragments* (load k-slice i+1 from smem while mma on slice i).
- Overlap epilogue: write C via smem staging for fully coalesced 128-bit stores; reuse pipeline smem for it after the last mainloop barrier.
- C++ alternative: `cuda::pipeline` + `cuda::memcpy_async` (libcu++); compiles to cp.async on sm_80+ and to TMA bulk copies in the `cuda::device::experimental` APIs on sm_90+.

## 4. ldmatrix + mma.sync mainloop

- `ldmatrix.sync.aligned.m8n8.x4.shared.b16` loads four 8x8 b16 tiles to a warp's fragments in one instruction; `.trans` variant to transpose on the fly (B operand from row-major).
- Make smem layout conflict-free for ldmatrix: each 8x8 tile row is 16 B; the 8 row addresses of a phase must fall in different bank groups: 128B XOR swizzle `chunk ^= (row & 7)`.
- mma shapes: `m16n8k16` (FP16/BF16, sm_80+), `m16n8k8` (FP16 sm_75; TF32 sm_80), `m16n8k32` (INT8; FP8 on sm_89/sm_120), `m8n8k4` FP64. Accumulate FP32. Use the `.row.col` form and arrange warp tiles as multiples (e.g. 64x64 warp tile = 4x8 mma tiles x k-slices).
- For sm_120 low-precision: `mma.sync.aligned.kind::f8f6f4...` and block-scaled forms; operands are packed (FP4 two per byte; FP6 padded containers per spec) - follow the PTX ISA layouts exactly.
- Epilogue conversion: pack with `cvt.rn.bf16x2.f32` / `__floats2bfloat162_rn`; FP8 with `cvt.rn.satfinite.e4m3x2.f32`.

## 5. Hopper (sm_90a)

### mbarrier + TMA
```
__shared__ alignas(8) uint64_t bar;                 // one per stage (full) + one per stage (empty)
init:   mbarrier.init.shared.b64 [bar], arrive_count;
        fence.mbarrier_init / cluster sync before use
producer (one elected thread):
        mbarrier.arrive.expect_tx.shared.b64 _, [bar], bytes;
        cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [smem], [tmap, {x, y}], [bar];
consumer:
        mbarrier.try_wait.parity.shared.b64 p, [bar], phase;   // spin on phase bit; flip phase each wrap
```
- Create tensor maps on host with `cuTensorMapEncodeTiled` (driver API; link `-lcuda` or fetch via `cudaGetDriverEntryPoint`); pass as `const __grid_constant__ CUtensorMap`. Box dims <= 256 per dimension; inner box bytes must be a multiple of 16 B and <= swizzle span (128 B for 128B swizzle) - split wider tiles into several TMA ops.
- Use the swizzle mode matching the wgmma smem descriptor (typically `CU_TENSOR_MAP_SWIZZLE_128B` for K-major FP16 64-element rows).
- TMA multicast: `.multicast::cluster` with a CTA mask loads the same tile into several CTAs' smem; each CTA's barrier is credited.
- After generic-proxy writes to smem that async-proxy units will read (e.g. epilogue staging before a TMA store): `fence.proxy.async.shared::cta`, then `cp.async.bulk.tensor ... global.shared::cta`, `cp.async.bulk.commit_group`, `cp.async.bulk.wait_group.read`.

### wgmma
```
wgmma.fence.sync.aligned;
wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 {d...}, descA|{a regs}, descB, scaleD, immScaleA, immScaleB, tnspA, tnspB;
wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;      // or N to keep N groups in flight
```
- 128 threads (warpgroup) cooperate; accumulators occupy `N/2` FP32 registers per thread for m64nNk16 (N=256 -> 128 regs): two consumer warpgroups with a 128x256 CTA tile use 128 accumulator regs/thread each; this is why `setmaxnreg` (e.g. producer `dec 40`, consumers `inc 232`) matters.
- Keep >= 1 wgmma group in flight while issuing the next: `wait_group 1` in steady state, release the smem stage consumed by the *previous* group.
- Smem descriptors (64-bit): start address >> 4, leading-dimension byte offset, stride-dimension byte offset, swizzle mode; get them wrong and results are silently wrong - validate against CUTLASS/CuTe `make_gmma_desc`.
- Pingpong schedule: two consumer warpgroups alternate between MMA and epilogue/softmax using named barriers (`bar.sync id, count`) so one's math overlaps the other's non-tensor work (FlashAttention-3 uses this to hide softmax).
- FP8: operands K-major only for the smem-sourced form (transpose in the producer or via `ldmatrix`/in-kernel transposes if the layout is MN-major); apply fp32 promotion every ~4 k-blocks into a second accumulator for long K.

### Clusters / DSMEM
- Use clusters (2x1x1 typical) for TMA multicast of the shared operand; `cluster.sync()`, `cluster.map_shared_rank(ptr, rank)` for DSMEM reads/atomics (e.g. cross-CTA reductions for very wide softmax/norm rows), `cudaLaunchKernelEx` with cluster attribute. Size must divide the grid; account for cluster placement limiting resident CTAs/SM.

## 6. Blackwell data center (sm_100a/103a)

Structure of a tcgen05 GEMM CTA (CUTLASS 3.x/4.x sm100 collectives are the reference implementation; read them before writing raw PTX):
1. **Setup:** `tcgen05.alloc` TMEM columns (e.g. 512 for two 128x256 FP32 accumulator stages), write the TMEM base address to smem, `tcgen05.relinquish_alloc_permit`.
2. **TMA warp:** loop over k-tiles: wait empty barrier -> `expect_tx` -> TMA load A and B (multicast across the CTA pair as configured).
3. **MMA warp (one elected thread issues):** wait full barrier -> `tcgen05.mma.cta_group::2.kind::f16 [tmem_d], descA, descB, idesc, enable_input_d` for each k-slice (K=16 per instruction for 16-bit; many per stage) -> `tcgen05.commit.cta_group::2.mbarrier::arrive::one` to release the smem stage and to signal accumulator-ready.
4. **Epilogue warps (4 warps = 128 lanes of TMEM):** wait accumulator-ready -> `tcgen05.ld.sync.aligned.32x32b.x*` TMEM->registers (`tcgen05.wait::ld`) -> apply scale/bias/activation/convert -> st.shared -> TMA store. Warp *w* of a warpgroup may only access TMEM lane quadrant `w % 4`.
5. **Teardown:** `tcgen05.dealloc`; ensure all `tcgen05.ld/mma` complete before.
- Tile choices: 2-CTA `256x128x64`/`256x256x64` (16-bit) so each SM owns 128 rows; 1-CTA `128x256` when clusters are impractical. `tcgen05.mma` throughput is limited by N: use N=256 whenever registers/TMEM allow.
- Scheduling: CLC (`clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes`) or static persistent scheduler with stream-K.
- Block-scaled: scale-factor tensors are loaded to smem by TMA and copied to TMEM with `tcgen05.cp`; pass TMEM addresses of SFA/SFB in the instruction. Layout is a fixed swizzle (blocks of 128 rows x 4 scale columns); your quantize kernel must emit exactly that layout.
- Precision options: FP16/BF16 (`kind::f16`), TF32 (`kind::tf32`), FP8 (`kind::f8f6f4`), MXFP8/MXFP4/NVFP4 (block-scaled kinds), INT8 (`kind::i8`, reduced on sm_103).
- If the kernel is attention: softmax in a separate set of warps reading S from TMEM, writing P back (to TMEM as the A operand of the PV MMA or to smem), correction warps rescaling O in TMEM; MUFU-bound -> partial polynomial exp.

## 7. Warp specialization blueprint

| Role | sm_90 | sm_100 |
|---|---|---|
| Producer | 1 warp (1 thread issues TMA), few regs | 1 warp issues TMA |
| MMA issuer | consumer warpgroups issue wgmma | 1 warp, 1 thread issues tcgen05 |
| Epilogue | consumer warpgroups (registers) | dedicated epilogue warps (TMEM -> regs) |
| Sync primitives | full/empty mbarrier pairs, named barriers | full/empty mbarriers, tcgen05.commit, TMEM full/empty barriers |

Rules: separate barrier arrays per stage and per direction; track the phase bit per stage; the last k-tile must release resources so the next persistent tile can start; do not let producers run ahead of the epilogue's smem reuse.

## 8. Warp-level primitives

```cpp
template <typename T, typename Op>
__device__ __forceinline__ T warp_reduce(T v, Op op) {
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}
```
- `__reduce_add_sync/min/max/and/or/xor_sync` (sm_80+, 32-bit ints) compile to `REDUX`: single instruction, faster than 5 shuffles.
- `__match_any_sync` for warp-aggregated atomics; `__ballot_sync` + `__popc` for compaction/prefix counts; `__shfl_up_sync` for warp scans.
- `elect.sync` (sm_90+) selects one lane without divergence tricks (`cute::elect_one_sync()`).
- Named barriers (`bar.sync id, nthreads` / `barrier.cluster`) for subsets of warps; avoid `__syncthreads()` in warp-specialized code since not all warps reach it.

## 9. PDL, Graphs, streams

- **Programmatic Dependent Launch (sm_90+):** launch with `cudaLaunchAttributeProgrammaticStreamSerialization=1`; in the *primary* kernel call `cudaTriggerProgrammaticLaunchCompletion()` (`griddepcontrol.launch_dependents`) as soon as dependent kernels may begin; in the *secondary* call `cudaGridDependencySynchronize()` (`griddepcontrol.wait`) before reading the primary's outputs. Do prologue work (weight prefetch to L2/smem, barrier init, TMEM alloc) before the wait. Works inside CUDA Graphs (programmatic dependency edges).
- CUDA Graphs for repeated sequences; update kernel params with `cudaGraphExecKernelNodeSetParams` instead of re-instantiating. Avoid host callbacks in the hot path.
- Overlap copy and compute with separate streams + events; use pinned memory; prefer `cudaMallocAsync` pools.
- Green contexts (CUDA 12.4+/13) to partition SMs between latency-critical and background work.

## 10. Libraries to build on

- **CUTLASS 3.x/4.x + CuTe** (header-only C++, plus CuTe DSL in Python in 4.x): reference-quality collectives for sm_80/90/100/120; use as the baseline and as a source for descriptor/layout code. Epilogue Visitor Tree (EVT) for fused epilogues.
- **CUB / cuda::std / Thrust**: block/warp reductions, scans, radix sort (`DeviceRadixSort` onesweep), histograms, segmented ops, `cub::BlockLoad/Store` with vectorized transposes.
- **cuBLASLt**: autotuned GEMM with epilogues; `cublasLtMatmulAlgoGetHeuristic` and workspace sizing; FP8/FP4 block-scaled matmul APIs on Hopper/Blackwell.
- **cuDNN** (conv, SDPA/flash attention frontends), **NCCL** for collectives, **cuSPARSELt** for 2:4.

## 11. C++20 in device code

Useful: `concepts`/`requires` to constrain kernels and tile configs; `consteval`/`constexpr` tables (swizzle maps, tile schedules); NTTP structs for tile configs (`template <TileConfig C>`); `std::bit_cast` through `cuda::std::bit_cast`; `cuda::std::span`, `cuda::std::array`, `cuda::std::atomic_ref`; designated initializers for launch configs; `[[likely]]`/`[[unlikely]]`.
Avoid in device code: `<ranges>`, `<format>`, exceptions, RTTI, `std::` containers that allocate, coroutines. Use `cuda::std::` equivalents (libcu++) rather than `std::`. `if consteval`, `std::mdspan` (C++23) are unavailable on the nvcc side; use `cute::Layout`/`cuda::std::mdspan` where available.

## 12. Pitfalls

- Illegal `wgmma`/`tcgen05` use on a non-`a` target compiles to a trap or fails to assemble; verify `-arch=sm_90a`, `sm_100a|f`, `sm_120a`.
- Missing `fence.proxy.async` / `fence.mbarrier_init` / `cluster.sync()` after barrier init cause rare hangs.
- Wrong mbarrier phase bookkeeping hangs kernels; run `compute-sanitizer --tool synccheck` and test with small K-tile counts (1, 2, S-1, S, S+1).
- TMA descriptors must live in constant/param/global memory with 64 B alignment; copy to `__grid_constant__` kernel parameter, not a local copy.
- TMEM column allocation must be a power of two >= 32 and total per SM <= 512 columns across co-resident CTAs: using all 512 forces 1 CTA/SM.
- `__syncthreads()` inside divergent code or in producer/consumer-split warps deadlocks.
- fp16 `atomicAdd` on `half2` is not deterministic and has lower precision: reduce in fp32.
- Beware ptxas register allocation changes between CUDA versions: re-check spills after every toolkit upgrade.
