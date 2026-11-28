# CUDA architectures: sm_75 .. sm_121 (CUDA 13.x)

All numbers below are **design-time defaults**. Query the real values at runtime (`cudaDeviceGetAttribute`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`) and confirm against the PTX ISA / CUDA Programming Guide "Compute Capabilities" tables when something decisive depends on one. Items marked (verify) are ones to double-check before relying on them.

## Contents
1. Common facts
2. Per-architecture sheets (75, 80, 86, 89, 90, 100/103, 110, 120/121)
3. Arch-target suffixes (`a`, `f`) and which instructions need them
4. Feature matrix
5. CUDA 13 specifics that affect kernels

---

## 1. Common facts (all of sm_75+)

- 32-thread warps, 4 SM sub-partitions (SMSPs), each issues 1 warp-instruction/cycle.
- 64K x 32-bit registers per SM, max 255 registers/thread, allocation granularity per warp.
- Independent thread scheduling: always use `*_sync` warp intrinsics with correct masks; do not rely on implicit warp-synchronous execution.
- 128-bit global loads/stores, 32-byte sector granularity, 128-byte cache lines.
- Shared memory: 32 banks x 4 B; carve-out configurable via `cudaFuncAttributePreferredSharedMemoryCarveout` and opt-in `cudaFuncAttributeMaxDynamicSharedMemorySize` for > 48 KB.
- `ldmatrix`/`stmatrix`(sm_90+) move 8x8 b16 matrices between smem and mma fragments.

## 2. Per-architecture sheets

### sm_75 - Turing (RTX 20, T4)
- Max 1024 threads/SM, 16 blocks/SM, 32 warps/SM. smem up to 64 KB/SM (64 KB max per block with opt-in).
- Tensor cores: `mma.sync.aligned.m16n8k8` FP16 (FP16 or FP32 accumulate), INT8 (m8n8k16), INT4/binary. **No BF16 MMA, no TF32, no FP64 MMA.**
- No `cp.async`: global->smem must stage through registers (`LDG` -> `STS`). Hide latency with register double-buffering.
- Separate INT32 pipe (concurrent INT + FP32 issue) - index math is cheap relative to Pascal but still compete for issue slots.
- Design guidance: 2-stage register-prefetch pipeline; small smem budget limits tiles (e.g. 128x128x32 FP16 with 2 stages ~ 32 KB + epilogue reuse). BF16 workloads must run on CUDA cores or convert to FP16.

### sm_80 - Ampere data center (A100/A30)
- Max 2048 threads/SM, 32 blocks/SM; smem 164 KB/SM, up to 163 KB per block (opt-in); large L2 (40 MB on A100) with **L2 persistence/residency controls** (`cudaAccessPolicyWindow`, `cudaLimitPersistingL2CacheSize`) and `createpolicy`/`cp.async...L2::cache_hint`.
- Tensor cores: `mma.sync.m16n8k16` FP16/BF16, `m16n8k8` TF32, `m8n8k4` FP64, INT8 `m16n8k32`, INT4, 2:4 structured sparsity `mma.sp`.
- `cp.async` (`LDGSTS`): 4/8/16 B global->smem without registers; `.cg` (L2 only, 16 B) vs `.ca`; zero-fill form (`src-size` operand) for predicated edges. `cp.async.commit_group/wait_group` pipelines; `cuda::pipeline` / `cuda::memcpy_async` wrappers.
- Asynchronous barriers in smem (`mbarrier`, `cuda::barrier`) with arrive-on-cp.async.
- `redux.sync` warp reductions for 32-bit integers (`__reduce_add_sync` etc.).
- Design guidance: 3-5 stage cp.async pipelines; ldmatrix with XOR-swizzled smem; 128x128x32 or 128x256x32 block tiles, 4-8 warps; split-K for small problems; MIG partitions change SM count.

### sm_86 - Ampere GA10x (RTX 30, A10, A40, A2)
- Same ISA as sm_80 for SASS purposes but different resources: 100 KB smem/SM (99 KB/block), 1536 threads/SM, 24 blocks/SM, 48 warps/SM.
- Doubled FP32 throughput (half of the 128 lanes/SM can also execute INT32 -> FP32-heavy code gets up to 2x vs INT mix); FP64 at 1/64 rate.
- Tensor-core throughput per SM lower than sm_80 for dense FP16 on consumer parts, and FP32-accumulate rates can be throttled vs FP16-accumulate on GeForce (verify per SKU by benchmark).
- Design guidance: smaller smem budget -> 3-stage pipelines with 128x128x32 tiles at most 48-64 KB/stage set; benchmark dedicated tile configs, do not reuse sm_80 configs.

### sm_89 - Ada Lovelace (RTX 40, L4, L40)
- Resources as sm_86 (99 KB smem/block, 1536 thr/SM) with a much larger L2 (up to ~96 MB on AD102): keep working sets L2-resident; tune tile order for L2 reuse.
- **FP8 `mma.sync`** (E4M3/E5M2, `m16n8k32`) with FP32 accumulate. No wgmma, no TMA, no clusters.
- Design guidance: Ampere-style mainloop (cp.async + ldmatrix + mma.sync) with FP8 operands; keep scaling factors in the epilogue; GeForce FP32-accumulate throttling applies (verify).

### sm_90 / sm_90a - Hopper (H100/H200, GH200)
- 228 KB smem/SM, up to 227 KB/block; 2048 threads/SM; 64K regs/SM; **setmaxnreg** lets producer warps give registers to consumer warps (warp-specialized kernels: e.g. 40 regs producer / 232 regs consumer warpgroups).
- **TMA** (Tensor Memory Accelerator): `cp.async.bulk.tensor` 1-5D tiles global<->smem driven by a tensor map (`CUtensorMap` via `cuTensorMapEncodeTiled`), hardware swizzle (32/64/128 B), multicast across a cluster, OOB zero-fill, reduction stores (`cp.reduce.async.bulk`). Single thread issues; completion via mbarrier transaction count (`mbarrier.expect_tx`).
- **wgmma** (`wgmma.mma_async`, sm_90a only): warpgroup (4 warps, 128 threads) MMA `m64nNk16` (FP16/BF16; k32 FP8; k8 TF32; N = 8..256 step 8); A from registers or smem, B from smem, accumulators in registers; asynchronous: `wgmma.fence`, `wgmma.commit_group`, `wgmma.wait_group N` allow overlapping MMA with other work. Needs smem operand descriptors with matching swizzle mode.
- **Thread block clusters** (up to 8 portable / 16 non-portable CTAs), **distributed shared memory** (`cluster.map_shared_rank`, `ld/st.shared::cluster`), cluster barriers, TMA multicast (one global read feeds many CTAs' smem).
- **PDL** (programmatic dependent launch): `griddepcontrol.wait/launch_dependents`.
- FP8 E4M3/E5M2 MMA; limited-precision accumulation (see universal-techniques section 14).
- DPX instructions (dynamic programming min/max fused ops); `stmatrix`; `fence.proxy.async` required when generic-proxy smem writes feed async-proxy (TMA/wgmma) consumers.
- Design guidance: persistent warp-specialized kernel (1 producer warp issuing TMA + 2 consumer warpgroups issuing wgmma, ping-pong or cooperative), 128x128x64 / 128x256x64 tiles, 4-7 stages, cluster 2x1 with multicast, TMA store epilogue, stream-K scheduling.

### sm_100 / sm_100a / sm_100f and sm_103 - Blackwell data center (B100/B200/GB200; B300/GB300)
- 228 KB smem/SM (227 KB/block), 2048 thr/SM; **Tensor Memory (TMEM)**: 256 KB/SM (128 lanes x 512 columns x 32-bit) dedicated to MMA accumulators (and A operand optionally). Allocated in CTA with `tcgen05.alloc` (power-of-two columns >= 32), freed with `tcgen05.dealloc`, plus `relinquish_alloc_permit`.
- **tcgen05.mma** is issued by **one thread** (not a warp/warpgroup); operands: A from smem or TMEM, B from smem, D in TMEM; instruction descriptor + smem descriptors encode layouts/swizzle. Shapes: M 64/128 (1-CTA) or 128/256 (2-CTA `cta_group::2`), N up to 256 (multiples of 8/16 by dtype), K=16 for FP16/BF16, 32 for FP8, 64 for 4-bit kinds (verify per kind). Completion signalled via `tcgen05.commit` -> mbarrier.
- **2-SM MMA (CTA pair)**: two CTAs in a cluster cooperatively compute a 256-row tile; each SM holds half the A tile; B is shared via DSMEM/multicast. Halves B smem traffic per SM.
- Epilogue reads TMEM with `tcgen05.ld` (e.g. 32x32b shapes) by dedicated epilogue warps; mainloop and epilogue overlap by double-buffering TMEM accumulator stages (256 KB TMEM supports 2 stages of 128x256 FP32).
- **Block-scaled MMA**: MXFP8, MXFP6, MXFP4 (E8M0 scales, block 32) and **NVFP4** (E4M3 scales, block 16) via `kind::mxf8f6f4`, `kind::mxf4`, `kind::mxf4nvf4` (verify exact kind names in PTX ISA). Scale factors live in TMEM/smem with a specific swizzled layout (CUTLASS exposes `Sm1xxBlockScaledConfig`); produce them in that layout from the quantization kernel.
- **No wgmma.** Hopper wgmma kernels do not run; port to tcgen05.
- **Cluster Launch Control (CLC):** dynamic tile scheduling where running CTAs cancel/steal unlaunched cluster work (`clusterlaunchcontrol.try_cancel`), giving load balancing without a global atomic queue.
- **256-bit global loads/stores** on sm_100+ via new 32-byte-aligned vector types (`double4_32a`, `longlong4_32a`, `ulonglong4_32a`, ...); older `double4`/`long4` types are deprecated in CUDA 13 in favour of `_16a`/`_32a` variants.
- Decompression engine, FP64/INT8 balance differs between sm_100 (B200: full FP64) and sm_103 (B300/GB300 Ultra: higher dense NVFP4 throughput, reduced FP64/INT8 tensor throughput) - confirm per-SKU rates before choosing a precision path.
- Special function (MUFU) throughput did not scale with tensor throughput: attention softmax `exp` can become the limiter. Mitigation (FlashAttention-4-style): compute part of the exponentials with a polynomial on FMA units (Cody-Waite range reduction + degree-3 polynomial) and conditionally rescale only when the running max changes meaningfully.
- Targets: `sm_100a`/`sm_103a` expose tcgen05 etc.; family-specific `sm_100f` runs on sm_100 and sm_103 (a single binary for both DC parts) at the cost of excluding instructions unique to one member.
- Design guidance: warp-specialized persistent kernel - 1 TMA-producer warp, 1 MMA-issuer warp (single elected thread issues tcgen05), 4+ epilogue warps (TMEM -> regs -> smem -> TMA store), CTA-pair 256x256 tiles for FP16/FP8; use block-scaled kinds for FP4/FP8 inference; 4-8 smem stages; stream-K or CLC scheduler.

### sm_110 - Jetson Thor (renamed from sm_101 in CUDA 13)
- Blackwell-class automotive/robotics SoC; GPU shares **LPDDR5X unified memory** with the CPU: arithmetic intensity thresholds are far lower than HBM parts, so most LLM decode/elementwise work is DRAM-bound at a few hundred GB/s.
- Tensor-core ISA is Blackwell-family (verify the exact tcgen05/TMEM support and `sm_110a` feature set in the PTX ISA release notes before writing tcgen05 code); smem and cluster limits may differ from sm_100.
- Design guidance: optimize for bytes (FP8/FP4 weights, fusion), exploit zero-copy/unified memory properly (avoid redundant memcpy), watch power modes (`nvpmodel`), and benchmark under the deployed power profile.

### sm_120 / sm_121 - Blackwell consumer / workstation (RTX 50, RTX PRO Blackwell) and GB10 (DGX Spark)
- 99 KB smem/block (100 KB/SM class), 1536 threads/SM. **No tcgen05, no TMEM, no wgmma, no 2-SM MMA.** Treat it as "Ada/Ampere-style `mma.sync` plus TMA, plus new low-precision types".
- `mma.sync` extended with FP4/FP6/FP8 (`kind::f8f6f4`) and **block-scaled** variants (`.block_scale` with MXFP8/MXFP4/NVFP4) - build with `sm_120a`/`sm_121a` to enable. Warp-level MMA means register-resident accumulators, ldmatrix-style operand loads and cp.async/TMA for global->smem (TMA exists; cluster multicast is generally not exploited on these parts - CUTLASS uses 1x1x1 clusters).
- sm_121 (GB10) shares sm_120's ISA but is an SoC with 128 GB **LPDDR5X unified memory** (~270 GB/s class): bandwidth-bound for most inference; optimize bytes and fusion before FLOPs. Do not assume sm_120 SM counts or clocks.
- Design guidance: Ampere/Ada-style multistage cp.async or TMA mainloop with warp-level block-scaled `mma.sync`, 128x128x128-ish tiles for FP4/FP8, smaller smem stage budgets; **do not** reuse sm_100 tcgen05 kernels - write a separate sm_120 variant.

## 3. Arch-target suffixes

- Plain `sm_XY`: baseline ISA features, forward-compatible with later architectures via PTX JIT.
- `sm_XYa` (architecture-specific): unlocks features exclusive to that exact architecture (`sm_90a`: wgmma, setmaxnreg, some TMA variants; `sm_100a`/`sm_103a`: tcgen05; `sm_120a`/`sm_121a`: block-scaled/low-precision mma.sync). **Binary is not forward-compatible.**
- `sm_XYf` (family-specific, CUDA 12.9+): features common to a family (e.g. `sm_100f` covers sm_100 and sm_103), forward-compatible within the family.
- Ship: SASS for every supported sm (`-gencode arch=compute_XY,code=sm_XY`), plus PTX for the highest baseline (`code=compute_XY`) for forward compatibility where allowed. Never ship only `sm_90a` and expect Blackwell to run it.
- Runtime dispatch: query `cudaDeviceProp::major/minor` (or `cudaDevAttrComputeCapabilityMajor/Minor`) and pick the kernel tier; do not select by `__CUDA_ARCH__` alone because multiple device binaries are compiled from the same source - `__CUDA_ARCH__` is a *compile-pass* macro (use `__CUDA_ARCH_SPECIFIC__` / `__CUDA_ARCH_FAMILY_SPECIFIC__` for suffix-sensitive code).
- Use `#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900` guards around arch-specific kernels so the other passes compile to an empty/trap stub, and assert on dispatch at runtime.

## 4. Feature matrix

| Feature | 75 | 80 | 86 | 89 | 90 | 100/103 | 110 | 120/121 |
|---|---|---|---|---|---|---|---|---|
| mma.sync FP16 | y | y | y | y | y | y | y | y |
| BF16 / TF32 MMA | - | y | y | y | y | y | y | y |
| FP8 MMA | - | - | - | mma.sync | wgmma | tcgen05 | tcgen05 (verify) | mma.sync |
| FP4/FP6 + block scaling | - | - | - | - | - | tcgen05 | verify | mma.sync |
| cp.async | - | y | y | y | y | y | y | y |
| TMA | - | - | - | - | y | y | y (verify) | y |
| wgmma | - | - | - | - | y (90a) | - | - | - |
| tcgen05 / TMEM | - | - | - | - | - | y | verify | - |
| Clusters / DSMEM | - | - | - | - | y | y | y (verify) | limited |
| FP64 tensor | - | y | - | - | y | y (103 reduced) | - | - |
| 2:4 sparsity | - | y | y | y | y | y | y | y |

## 5. CUDA 13 specifics

- Support for Maxwell/Pascal/Volta targets removed from the offline toolchain; sm_75 is the minimum.
- `nvcc -arch=native`, `-arch=all-major`, and `--threads N` to parallelize multi-arch compilation; large template-heavy builds (CUTLASS 4.x) are dominated by per-arch compile time: build only needed archs in dev.
- Deprecated 16-byte-aligned-by-accident vector types (`double4`, `long4`, ...) -> use explicit `_16a`/`_32a` types so the alignment (and 256-bit load eligibility) is deliberate.
- Several legacy fields of `cudaDeviceProp` (e.g. clock rates) were removed in CUDA 13: use `cudaDeviceGetAttribute`.
- Host compiler floor GCC 15/MSVC: C++20 modules/concepts in host code are fine; device code is limited to what nvcc's C++20 mode supports.
- CUDA Graphs, green contexts (SM partitioning), `cudaMemcpyBatchAsync`, stream-ordered memory pools are the supported tools for multi-kernel latency/throughput work.
