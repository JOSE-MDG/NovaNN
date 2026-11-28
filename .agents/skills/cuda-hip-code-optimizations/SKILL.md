---
name: cuda-hip-code-optimizations
description: Expert-level GPU kernel optimization for CUDA 13.x (sm_75 to sm_121, incl. Hopper/Blackwell) and HIP/ROCm 7.x (gfx908 to gfx1201, CDNA1-4 and RDNA2-4). Use to write, review, port, profile or aggressively tune GPU kernels/operations - GEMM, attention, reductions, softmax, layernorm/rmsnorm, elementwise, convolution, scan, sort, top-k, histogram, gather/scatter, transpose, quantization, fused optimizers, deep-learning backend ops. Make sure to use this skill whenever the user mentions CUDA, HIP, ROCm, kernels, tensor cores, MFMA, WMMA, wgmma, tcgen05, TMA, shared memory/LDS, occupancy, coalescing, bank conflicts, warp/wave primitives, Nsight/ncu/rocprof, roofline, or "squeeze more performance" from a GPU op, even if they do not say "optimize". Also apply it to any GPU work in the NovaNN native core: ncore/native/cuda, ncore/native/hip, kernel launch heuristics (resolveLaunchConfig, LaunchParamsBase), the per-device engine stream, GPU dispatch, or benchmarking/tuning a NovaNN kernel. Treats CUDA and HIP as two fully separate, individually-tuned backends.
---

# CUDA / HIP Code Optimizations

Act as a senior GPU performance engineer whose goal is to extract the last few
percent of hardware capability from an operation, either by writing it from
scratch or by tearing apart an existing implementation. Every recommendation
must be tied to a hardware limit (bandwidth, issue rate, tensor throughput,
latency, occupancy, launch overhead) and be verifiable with a measurement.

## Fixed target matrix (do not suggest anything outside it unless asked)

| | CUDA backend | HIP backend |
|---|---|---|
| SDK | CUDA 13.x | ROCm 7.x |
| Host compiler (minimum) | GCC 15.x (Linux), MSVC (Windows) | clang 21.x (Linux only) |
| Language standard | C++20 (nvcc's ceiling) | C++23 |
| Architectures | sm_75, 80, 86, 89, 90, 100, 103, 110, 120, 121 | gfx908, 90a, 942, 950, gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200-1201 |

Consequences worth internalizing:
- CUDA 13 dropped Maxwell/Pascal/Volta: sm_75 (Turing) is the floor, so no
  Volta-era assumptions (independent thread scheduling is the baseline, `_sync`
  warp intrinsics are mandatory, no legacy `__shfl` without `_sync`).
- Code in `.cu` files must stay within C++20 (concepts, `consteval`,
  `std::bit_cast` via `cuda::std`, NTTP class types are fine; no `std::mdspan`,
  `if consteval`/deducing-this are C++23 and are HIP-only).
- HIP is Linux-only here: Windows tooling concerns apply to CUDA only
  (MSVC flags, `/Zc:preprocessor`, `__cplusplus` vs `_MSVC_LANG`).
- Windows MSVC builds cannot use some GCC-isms (`__builtin_expect`,
  `__attribute__`) in host code of shared headers; use `[[likely]]`/`[[unlikely]]`
  or wrapper macros.

## Core principle: two independent backends

CUDA and HIP kernels are written, tuned and reviewed independently. Do NOT
produce a single kernel with `#ifdef __HIP_PLATFORM_AMD__` islands and call it
"portable": warp size (32 vs 64 on CDNA), matrix instructions (mma/wgmma/tcgen05
vs MFMA/WMMA), shared-memory sizes and banking, async copy mechanisms (cp.async/
TMA vs buffer/global-to-LDS loads), cache hierarchy, and scheduler behavior are
different enough that each backend wants a different algorithm structure.

Shared between backends (keep these common):
- op specification (shapes, dtypes, layouts, epilogues), reference
  implementation (fp64 / CPU), correctness tests, benchmark harness, host-side
  dispatch/heuristics interface.

Separate per backend (never share):
- tile shapes, pipeline depth, thread/wave count, data layouts in LDS/smem,
  instruction selection, vector widths, scheduling hints, register budgets.

## NovaNN integration (ncore)

NovaNN is not a standalone ops project: kernels live inside the ncore
runtime, and the conventions below are load-bearing. They describe how the
generic guidance above maps onto this repository.

Real layout:

```
ncore/native/kernels/<Op>DispatchImpl.cpp    # device-agnostic dispatch, resolves backend
ncore/native/cuda/kernels/<domain>/<Op>Kernel.cu   # CUDA impl (+ .h C-linkage header)
ncore/native/hip/kernels/<domain>/<Op>Kernel.hip   # HIP impl (+ .h C-linkage header)
ncore/include/ncore/headeronly/heuristics/kernels/ # launch heuristics: pattern structs + resolver
ncore/tests/<feature>/                       # GTest pins (correctness, engine stream, heuristics)
```

Entry points and dispatch:
- The public entry point of each backend kernel is a C-linkage function
  `launchCuda<Op>Kernel` / `launchHip<Op>Kernel`, declared in the `.h`
  next to the `.cu`/`.hip`, taking `const Tensor *restrict` arguments and
  returning `novaStatus_t`. The device-agnostic dispatch layer under
  `ncore/native/kernels/` resolves the active device and calls it.
- Every error path reports through `novaStatus_t`. Never `abort()`,
  never swallow, never print-and-continue; Cython will translate statuses
  to Python exceptions later.

Launch heuristics are the selector:
- Grid/block/smem sizing is not hand-rolled per kernel. Fill the matching
  pattern struct (nine exist, all inheriting `LaunchParamsBase`:
  `ElementWiseParams`, `ReductionParams`, `LayoutTransformParams`,
  `ParallelScanParams`, `GemmParams`, `GatherScatterParams`,
  `SortHistogramParams`, `SpatialNeighborhoodParams`, `FusedOperatorParams`)
  and call `resolveLaunchConfig()` from
  `ncore/include/ncore/headeronly/heuristics/kernels/launch_config.hh`.
- A new execution pattern means a new struct inheriting `LaunchParamsBase`
  plus a `fullKeyFor` overload, and tests under `ncore/tests/heuristics/`.
- Manually tuned per-arch variants stay legal, but they are picked as
  configuration the resolver can select, not as hard-coded launches.

Engine stream (non-negotiable):
- Each device owns one non-blocking engine stream reused by every
  allocation, transfer and launch. Launchers resolve it internally; the
  null stream is never used for device work. Never synchronize the
  engine stream from a launcher; draining is explicit and happens only
  before a device pool is destroyed.
- Never run the suites with `HIP_LAUNCH_BLOCKING=1`: it serializes every
  launch, hides the null-stream drop, and makes the gates meaningless.

Memory:
- Buffers come from the Rust-managed allocator (`ncore_memory` via the
  `memorycsrc` bridge: reserve/release/resize/empty_cache). Kernels never
  call `cudaMalloc`/`hipMalloc` directly. Every buffer remembers the
  stream its allocation ran on; releasing or resizing it on a different
  stream is rejected (crossing streams requires an explicit event
  dependency).

Editing discipline (short version of the repo rules):
- Headers shared with C++ cast through `STATIC_CAST`
  (`ncore/include/ncore/headeronly/macros.h`); designated initializers
  for struct/table aggregates, one member per line; comments state
  rationale, never narration, and contain no em-dash separators;
  documentation follows the doxygen voice (one-line briefs).
- Never hand-edit files whose banner reads DO NOT EDIT GENERATED CODE
  (the CPU codegen under `tools/codegen/` owns them).

Deliverables when asked to add or optimize an op:
- The `<Op>Kernel.cu/.h` and `<Op>Kernel.hip/.h` pair with its C entry
  point, dispatch wiring, heuristic params, and GTest coverage under
  `ncore/tests/` that pins the semantics. A changed behavior without a
  pin will drift back.
- Kernel defects are reported and pinned by tests; kernel semantics are
  never changed unilaterally. Propose, await the owner's decision.
- Verify through the preset scripts:
  `scripts/build-presets.sh <preset>` then `scripts/compile-presets.sh`
  then `scripts/run-tests.sh <preset>` (cuda-* and hip-* presets).

## Workflow (follow in order; skipping steps is how time gets wasted)

1. **Characterize the op.** Compute FLOPs, minimum bytes moved (compulsory
   traffic), arithmetic intensity (FLOP/byte), and compare against the target
   device's ridge point. Decide: memory-bound, compute-bound, latency-bound
   (low parallelism / dependent chains), or launch-bound (tiny kernels). State
   the *theoretical floor* in microseconds before touching code.
2. **Pick the target tier(s)** using the architecture dispatch table below and
   read the matching architecture reference file.
3. **Establish a trustworthy baseline** (see `references/methodology.md`):
   warmup, L2/Infinity-Cache flush for cold-cache numbers, median of many runs,
   clocks pinned when comparing, and a vendor-library baseline
   (cuBLAS/cuBLASLt/CUTLASS/CUB/cuDNN or hipBLASLt/rocBLAS/CK/rocPRIM/MIOpen).
   Use `assets/bench_harness_cuda.cuh` / `assets/bench_harness_hip.hpp`.
4. **Profile before editing.** Nsight Compute / Nsight Systems on NVIDIA;
   rocprofv3 / rocprof-compute on AMD. Read the resource usage (registers,
   spills, smem/LDS) and the ISA (SASS / GCN assembly) for the hot loop.
5. **Change one thing, re-measure, keep a log** (hypothesis -> change -> metric
   delta). Revert regressions immediately.
6. **Validate numerics every iteration**: compare to an fp64 reference with a
   tolerance scaled to the accumulation length and dtype; check NaN/Inf, edge
   shapes (non-multiples of tile sizes, M/N/K = 1, huge strides), alignment
   (misaligned base pointers), and determinism requirements.
7. **Stop at the roofline.** Typical "done" thresholds: streaming kernels reach
   >= 90-95 % of *measured* copy bandwidth (not datasheet); large GEMM reaches
   >= 75-90 % of tensor peak depending on arch/dtype; if you are there, further
   effort should go to fusion or removing the op, not tuning it.

## Architecture dispatch (which reference to read, which kernel tier to write)

CUDA (read `references/cuda-architectures.md` then `references/cuda-techniques.md`):

| sm | Tier | Defining features for kernel design |
|---|---|---|
| 75 | Legacy tensor | `mma.sync` m16n8k8 FP16, INT8/INT4 MMA; no cp.async, no BF16/TF32 MMA; 64 KB smem |
| 80 | Ampere DC | `cp.async`, BF16/TF32/FP64 MMA, 2:4 sparsity, `ldmatrix`, 163 KB smem/block, L2 persistence |
| 86 | Ampere consumer | Same ISA as 80; 99 KB smem/block, 1536 thr/SM, doubled FP32 issue (FP32 heavy kernels differ from 80) |
| 89 | Ada | Ampere ISA + FP8 `mma.sync`; 99 KB smem; large L2; no wgmma/TMA |
| 90 | Hopper | TMA, `wgmma` (needs sm_90a), clusters + DSMEM, mbarrier tx-count, `setmaxnreg`, PDL, FP8 |
| 100 / 103 | Blackwell DC | `tcgen05` + Tensor Memory (TMEM), 2-SM MMA (CTA pairs), block-scaled MXFP8/6/4 + NVFP4, CLC, 256-bit loads; no wgmma |
| 110 | Jetson Thor | Blackwell-class, LPDDR unified memory: bandwidth-limited; verify feature set in PTX ISA |
| 120 / 121 | Blackwell consumer / GB10 | Extended `mma.sync` incl. FP4/FP6/FP8 and block-scaled variants; TMA; NO tcgen05/TMEM/wgmma; 99 KB smem |

HIP (read `references/hip-architectures.md` then `references/hip-techniques.md`):

| gfx | Tier | Defining features for kernel design |
|---|---|---|
| 908 | CDNA1 | Wave64, MFMA (results in AGPRs), 64 KB LDS, no FP64 matrix |
| 90a | CDNA2 | Unified 512 VGPR file, FP64 MFMA, packed FP32 (`v_pk_*`), HW FP64 atomics |
| 942 | CDNA3 | 8 XCDs (workgroup round-robin!), FP8/BF8 (FNUZ) MFMA, xf32, sparse MFMA, Infinity Cache |
| 950 | CDNA4 | 160 KB LDS, larger-K MFMA, FP4/FP6/FP8 (OCP) + scaled MFMA, wider direct-to-LDS loads, LDS transpose reads |
| 1030-1036 | RDNA2 | Wave32, WGP/CU mode, no matrix cores; dot instructions + packed math only |
| 1100-1103, 1150-1153 | RDNA3 / 3.5 | WMMA (wave32, FP16/BF16/IU8/IU4), VOPD dual issue; 3.5 parts are mostly iGPUs (LPDDR bound) |
| 1200-1201 | RDNA4 | WMMA v2 (new register layout), FP8/BF8 WMMA, sparse `swmmac` |

When the user targets several archs, produce a separate variant per tier and a
dispatch table. Never lower a Hopper or CDNA3 kernel to "work everywhere".

## Universal optimization checklist (roughly ordered by typical payoff)

1. Eliminate redundant global traffic: fuse producers/consumers, keep
   intermediates on-chip, fuse epilogues (bias, activation, residual, quant,
   dropout), avoid layout-conversion kernels.
2. Make every global access coalesced and vectorized (128-bit minimum;
   256-bit on sm_100+ via the `_32a` types where applicable), aligned, and with
   enough bytes in flight (Little's law: in-flight bytes >= bandwidth x latency).
3. Remove local-memory spills and dynamic register indexing; check `-Xptxas -v`
   / `-Rpass-analysis=kernel-resource-usage`.
4. Tile for reuse: registers > shared memory/LDS > L1 > L2. Choose tile sizes
   from arithmetic-intensity math, not from folklore.
5. Remove shared-memory/LDS bank conflicts (swizzle > padding when you control
   both write and read side; padding is fine for simple transposes).
6. Overlap data movement with compute: multistage pipelines (cp.async/TMA on
   NVIDIA; double-buffered LDS and global->LDS loads on AMD), warp/wave
   specialization where the hardware rewards it (Hopper+, CDNA3/4 with
   `s_setprio`/sched barriers).
7. Use the matrix units whenever the math allows (even for "non-GEMM" work such
   as reductions via ones-vector MMA when the ALU is the bottleneck).
8. Balance occupancy vs ILP: more registers per thread with fewer resident warps
   is often *faster* for compute-bound kernels; verify rather than maximize
   occupancy.
9. Fix tail effects and wave quantization: persistent kernels, stream-K/split-K,
   tile-order rasterization for L2/XCD locality.
10. Cut launch overhead: CUDA Graphs / HIP Graphs, Programmatic Dependent Launch
    (sm_90+), fusion, persistent megakernels for chains of tiny ops.
11. Replace slow instructions: int div/mod -> magic multiply or shifts,
    `exp` -> `exp2` with pre-scaled log2(e), `1/sqrt` -> `rsqrt`, fp32->bf16x2
    packed converts, LOP3/PRMT (NVIDIA) and `v_perm`/`v_bfe`/DPP (AMD) tricks.
12. Micro-schedule last: instruction ordering, `sched_group_barrier` (AMD),
    register bank/reuse hints, SASS-level inspection.

## How to respond to a request

- For "write this op": deliver per-backend kernels with the chosen tile/
  pipeline parameters, a short rationale tied to the roofline (what bound, what
  expected efficiency), the host launch code (grid/block/smem attributes such
  as `cudaFuncAttributeMaxDynamicSharedMemorySize` / `hipFuncSetAttribute`),
  a correctness test and a bench entry.
- For "optimize this kernel": first state the diagnosis (bound + evidence
  from code reading or profile), then ordered fixes with expected gain, then
  the patched code. If no profile is provided, ask for the specific
  `ncu`/`rocprof-compute` sections listed in `references/methodology.md`
  only when the diagnosis genuinely cannot be made from the code.
- Be concise: no tutorials the reader did not ask for. Name the instruction,
  the flag or the number. Flag uncertainty explicitly (e.g. "verify with PTX
  ISA / LLVM AMDGPU docs") instead of inventing details; hardware limits must
  be queried at runtime (`cudaDeviceGetAttribute`, `hipDeviceGetAttribute`,
  occupancy APIs), not hard-coded, except as design-time defaults.
- Prefer building on CUTLASS/CuTe/CUB (NVIDIA) and Composable Kernel/CK-Tile/
  rocPRIM/hipCUB (AMD) primitives when they already hit the roofline; write raw
  PTX/GCN-intrinsic kernels when fusion, odd shapes, or custom dtypes demand it.

## Anti-patterns to flag on sight

- One kernel, `#ifdef`'d for both vendors, with a hard-coded warp size of 32.
- Using `warpSize` as a compile-time constant in HIP code (CDNA is wave64,
  RDNA is wave32; derive from `__AMDGCN_WAVEFRONT_SIZE__` or template on it).
- `atomicAdd` on floats in a hot path without warp aggregation; on AMD, forgetting
  that CDNA2+/3 need `-munsafe-fp-atomics` or `unsafeAtomicAdd` for HW float atomics.
- Benchmarks with a single launch, warm L2 for a "cold" workload, or wall-clock
  timing without synchronization.
- Maximizing occupancy blindly; `-maxrregcount` instead of `__launch_bounds__`.
- Accumulating FP8/FP16 sums in low precision; ignoring Hopper FP8 accumulator
  promotion when K is large.
- Passing 64-bit indices everywhere when 32-bit offsets suffice (register and
  ALU pressure, worse on AMD where it doubles VGPR use).
- Writing tcgen05/wgmma code for the wrong arch (sm_120 has no tcgen05; sm_100
  has no wgmma; wgmma/tcgen05 need the `a`/`f` arch-suffixed targets).
- Calling `cudaMalloc`/`hipMalloc` inside a kernel path instead of the
  Rust-managed pool (`ncore_memory`: reserve/release/resize), or
  synchronizing the device engine stream mid-pipeline.
- Launching device work on the null stream; ncore routes every launch
  through the per-device engine stream.
- Hand-rolling grid/block sizing in a kernel path instead of filling the
  heuristics pattern struct and calling `resolveLaunchConfig()`.

## Reference index (read only what the task needs)

- `references/methodology.md` - roofline, benchmarking protocol, profiler
  workflows (Nsight, rocprofv3/rocprof-compute), numerics validation.
- `references/universal-techniques.md` - memory hierarchy, coalescing,
  vectorization, bank conflicts, occupancy/ILP, reductions, scheduling,
  persistent kernels, atomics, fusion, launch overhead. Backend-neutral theory.
- `references/cuda-architectures.md` - per-sm capabilities and limits.
- `references/cuda-techniques.md` - cp.async, TMA, wgmma, tcgen05/TMEM,
  clusters, mbarrier, PDL, warp specialization, PTX/inline asm, CUDA 13 notes.
- `references/hip-architectures.md` - per-gfx capabilities and limits.
- `references/hip-techniques.md` - MFMA/WMMA, LDS, DPP, buffer loads,
  direct-to-LDS, XCD remapping, sched barriers, waves-per-EU, ROCm 7 notes.
- `references/op-playbooks.md` - concrete design recipes per operation for both
  backends (GEMM, attention, reductions, softmax/norms, elementwise, conv, scan,
  sort/top-k, histogram, gather/scatter, transpose, quantization, optimizers).
- `references/build-and-toolchain.md` - nvcc/hipcc flags, arch targets,
  CMake, fatbins, ISA inspection scripts.
- `assets/bench_harness_cuda.cuh`, `assets/bench_harness_hip.hpp` - drop-in
  timing harnesses (cold/hot cache, median/min/p90, GB/s and TFLOP/s helpers).
- `scripts/inspect_kernel.sh` - one-shot resource-usage + ISA dump for either
  backend.
