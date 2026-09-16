---
name: cuda-hip-kernels
description: Advanced reference for creating, refactoring, and optimizing CUDA/HIP GPU kernels (grid-stride loops, coalescing and vectorization, shared-memory banking, warp/wavefront shuffle reductions, register pressure vs occupancy, tensor/matrix cores, launch sizing). Use this skill whenever writing a new GPU kernel, reviewing or refactoring an existing .cu/.hip kernel, tuning block/grid dimensions, porting a kernel between CUDA and HIP, diagnosing bank conflicts, spills, or low occupancy, or validating a kernel recommendation against the actual build toolchain before suggesting it.
license: MIT
---

# CUDA/HIP Kernels

A working reference for creating, refactoring, and optimizing GPU
kernels at an advanced level. Use it for all three jobs; the workflow
below tells you which reference file to open for each.

## How to use this skill

1. **Classify first.** Map the kernel to one `ExecutionPattern`
   (element-wise, reduction, layout change, scan, sort/histogram,
   stencil, GEMM, gather/scatter, fused) using
   `references/patterns.md`. The pattern decides which optimizations
   are legal: what works for a streaming map (vectorize everything)
   is wrong for a reduction (shuffle before shared) or a transpose
   (tile before vectorize).
2. **Check the toolchain before recommending anything.** A feature
   being standard CUDA/HIP does not mean it compiles or pays off on
   every target in this project — see Toolchain support below. Device
   code has a much smaller usable library than host code even with the
   same `-std=` flag.
3. **Size the launch from facts, not magic numbers.** Block/grid
   geometry comes from `references/launch-sizing.md`: closed-form
   sizing for small launches, measured analysis only when the shape
   proves it worthwhile. Never hardcode `<<<256, 1024>>>` because it
   "looks right".
4. **Port by invariant, not by rename.** CUDA-to-HIP is not
   `s/cuda/hip/`. Wavefront width, lane-mask width, ballot width,
   `__launch_bounds__` conversion, and shuffle availability all change
   meaning — see `references/backend-deltas.md`.

## Toolchain support (project floor, verified from CMake)

| Toolchain | Status |
|---|---|
| **CUDA toolkit** | `>= 13.0`, minimum SM 75 (Turing). Arch list includes 75/80/86/89/90/100/103/110/120/121. Occupancy APIs (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaOccupancyMaxPotentialBlockSize`) available; they are host-side driver calls, never free, and can surface prior async errors. CUDA 13 adds shared-memory register spilling (opt-in via pragma). |
| **ROCm/HIP** | `>= 7.0`, Linux only. CDNA wavefront is 64 lanes (wave64-only); RDNA supports wave32/64. Occupancy API mirrors CUDA (`hipOccupancyMaxActiveBlocksPerMultiprocessor`). Matrix cores via MFMA builtins; prefer Composable Kernel over hand-rolled builtins. |
| **Host compilers** | CUDA host code via GCC (device via nvcc); HIP via Clang/hipcc. `.cu` translation units build at C++20, HIP/host at C++23 — headers shared by both stay in the C++20 subset. C++23 library features (`std::expected`, `std::stacktrace`) are host-only. |
| **Warnings** | `-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow -Wold-style-cast` (+ `-Wuseless-cast` on GNU; clang adds its stricter set on HIP). New kernel code must be warning-clean under all three compilers: explicit casts, `U`/`ULL` suffixes, no shadowing, no C-style casts. |

## Universal rules (every kernel, every pattern)

These come from the CUDA C++ Best Practices Guide (§10–13) and apply
before any pattern-specific work. Higher priority first; do not tune
instruction trivia while memory access is broken.

1. **Grid-stride loops.** Every kernel iterates `i = tid; i < n;
   i += gridStride`. Any grid stays correct; only performance varies.
   This is what makes launch heuristics safe to be approximate.
2. **Coalesce first.** Adjacent threads touch adjacent addresses.
   Prove 8/16-byte vector transactions from pointer alignment and
   contiguity; misaligned scalar access can halve effective bandwidth.
   Prefer `float4`/16B transactions over wider custom widths.
3. **Vectorize loads/stores, scalarize addresses.** One 16B
   transaction beats four scalar ones; 64-bit indexing only past 4 GB.
4. **Minimize host traffic.** Data lives on-device across launches;
   transfers are pinned + async + overlapped, never synchronous in a
   hot path. Kernel time under ~5–20 us is launch-dominated — fuse or
   batch instead of tuning the kernel.
5. **Branch on uniformity, not hope.** Intra-warp divergence serializes
   paths; on wave64 it idles up to 64 lanes. Hoist uniform tests above
   the loop, predicate tails with masks, never branch per element on
   data.
6. **Registers are a step function, not a slope.** 32→33 regs/thread
   can drop resident warps 64→48 on a 64K-register SM. Read `ptxas -v`
   output; shrink live ranges (tight scopes, recompute cheap values)
   before touching `__launch_bounds__` or `-maxrregcount`, both of
   which force spilling to buy occupancy — usually a net loss unless
   measured ≥5%.
7. **Occupancy hides latency; it is not performance.** Past enough
   warps to saturate the bottleneck (bandwidth-bound: many;
   compute-bound: fewer, fatter threads), extra occupancy buys nothing.
   Size for the bottleneck, verified by roofline, not by the
   occupancy percentage.

## Workflow per job

### Creating a kernel

1. Classify the pattern (`references/patterns.md`) and copy the
   closest existing launcher shape (host params struct → resolver →
   `dim3` → grid-stride kernel).
2. Write the memory access pattern before the compute: who loads what,
   in which order, with what width. If you cannot draw the coalescing,
   stop and re-tile.
3. Apply the pattern's required techniques in priority order
   (e.g. reduction: vectorized load → per-thread coarsen → warp
   shuffle → one shared sweep → single combine).
4. Size the launch (`references/launch-sizing.md`). Small problem:
   fixed closed form. Large problem: prove the choice (waves,
   occupancy clamp, tile budget).
5. Build all three presets warning-clean; add a GTest pinning the
   geometry decisions (see `references/launch-sizing.md` for what is
   asserted).

### Refactoring a kernel

1. Read it against the universal rules above; list violations in
   priority order (access pattern → divergence → occupancy → trivia).
2. Fix one level at a time, keeping grid-stride correctness so every
   intermediate step runs.
3. Migrate launch sizing off magic numbers onto the pattern resolver;
   delete per-kernel constants.
4. Mirror the change to the sibling backend the same day (CUDA↔HIP
   drift is how the old kernels rotted). Checklist in
   `references/backend-deltas.md`.

### Optimizing a kernel

1. Roofline it first: bytes moved vs peak bandwidth, FLOPs vs peak
   compute. Optimizing the non-bottleneck is theater.
2. Work the pattern checklist top-down; stop at the first level that
   moves the roofline number. Common order: layout/coalescing →
   shared staging and bank conflicts → occupancy/register balance →
   instruction mix (`__ldg`, FMA contraction, fast math where valid).
3. Only then consider structural moves: persistent CTAs, split-K /
   Stream-K (GEMM), fusion with neighbors, CUDA Graphs for
   launch-dominated sequences.
4. Re-verify numerically (bitwise where required; tolerance where
   reassociation is legal) and re-run the full test preset.

## Sources

- CUDA C++ Best Practices Guide 13.3 (APOD cycle, coalescing, shared
  banks, occupancy, launch bounds):
  https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- CUDA Runtime Occupancy API:
  https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__OCCUPANCY.html
- HIP porting guide 7.0 (wavefront, lane masks, launch-bounds
  conversion): https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.1/how-to/hip_porting_guide.html
- CUB `BlockReduce` (raking vs warp-reductions, one barrier, zero
  bank conflicts): https://nvidia.github.io/cccl/cub/api/classcub_1_1BlockReduce.html
- CUTLASS Stream-K heuristic (fallback without wave quantization):
  https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/
- Shared-memory register spilling (CUDA 13, QUDA 5–10%):
  https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling
