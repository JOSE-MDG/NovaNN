# HIP techniques (ROCm 7.x, clang 21, C++23, Linux)

## Contents
1. Kernel scaffolding: wave size, launch bounds, waves-per-EU
2. Memory access: vector, scalar, buffer, non-temporal
3. LDS and bank conflicts
4. MFMA (CDNA) mainloop
5. WMMA (RDNA3/3.5/4) mainloop
6. Direct-to-LDS loads and pipelining (gfx942/gfx950)
7. Instruction scheduling: sched_group_barrier, s_setprio, iglp_opt
8. XCD-aware workgroup remapping (gfx942/gfx950)
9. Wave-level primitives: DPP, permlane, readlane
10. Atomics
11. Graphs, streams, multi-GPU
12. Libraries to build on
13. C++23 in device code
14. Pitfalls

---

## 1. Kernel scaffolding

```cpp
constexpr int kWave =
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
    __AMDGCN_WAVEFRONT_SIZE__;   // 64 on gfx9, 32 on gfx10+ (device pass only)
#else
    64;                          // host pass placeholder; choose at runtime from device props
#endif

template <int kThreads>
__global__ __launch_bounds__(kThreads) __attribute__((amdgpu_waves_per_eu(1, 2)))
void k(...);
```
- Never assume `warpSize == 32`. Template kernels on wave size or derive from the macro; on the host choose a variant from `hipDeviceProp_t::warpSize`/`gcnArchName`.
- `__launch_bounds__(MAX_THREADS)` is the primary register-budget control. In HIP the optional second argument means *minimum waves per execution unit*, not blocks per SM as in CUDA - prefer the explicit attributes: `__attribute__((amdgpu_flat_work_group_size(256, 256)))` and `__attribute__((amdgpu_waves_per_eu(min, max)))`.
- Register budget per wave (CDNA, 512 regs/lane/SIMD): 8 waves/SIMD -> <= 64 regs; 4 -> <= 128; 2 -> <= 256; 1 -> <= 512 (VGPR + AGPR). MFMA GEMMs typically run 1-2 waves/SIMD with 256-512 regs; verify with `-Rpass-analysis=kernel-resource-usage`.
- Workgroup size: multiples of the wave size; 256 threads (4 waves, one per SIMD) is the standard GEMM choice on CDNA; 512 for 2 waves/SIMD.
- Dynamic LDS beyond 64 KB (gfx950): `hipFuncSetAttribute(k, hipFuncAttributeMaxDynamicSharedMemorySize, bytes)`.
- Prefer `hipLaunchKernelGGL` or `<<<>>>`; `hipExtLaunchKernelGGL` for events/flags around a launch. Cooperative launch `hipLaunchCooperativeKernel` for grid sync (check support per device).

## 2. Memory access

- **Vector loads**: `global_load_dwordx4` via `float4`/`uint4` aligned 16 B pointer casts, or `typedef float __attribute__((ext_vector_type(4))) f32x4;` (generated code is identical, ext_vector_type is more convenient for arithmetic).
- **Buffer loads/stores**: base in SGPRs (4-dword resource descriptor), per-lane 32-bit offset in a VGPR, hardware bounds checking (OOB returns 0 / ignores stores - free predication of tile edges) and fewer VGPRs than 64-bit global addressing. Use `__builtin_amdgcn_raw_buffer_load_b128` / `__builtin_amdgcn_make_buffer_rsrc` (name/availability changes between LLVM versions - verify with clang 21) or CK's `amd_buffer_load`/`amd_buffer_store` wrappers.
- **Scalar loads** (`s_load_dwordx*`, `s_buffer_load`): wave-uniform, read-only data (scales, biases, small tables). Make address uniform and pointer `const __restrict__`; avoid per-lane divergence in the address expression. `__builtin_amdgcn_readfirstlane(x)` forces a value into an SGPR when you know it is uniform (saves VGPRs, enables SALU math).
- **Cache policy**: `__builtin_nontemporal_load/store` -> `nt` bit streaming; cache-control bits (`glc`/`slc`/`dlc` on gfx9/10, `sc0/sc1/nt` on gfx940+, `th`/`scope` on gfx12) via inline asm or `__builtin_amdgcn_raw_buffer_load` aux argument.
- **Coalescing**: a wave64 `dwordx4` load covers 1024 B; align row starts to 16 B (ideally 128 B) and pad leading dimensions.
- **Index width**: keep per-lane offsets 32-bit; 64-bit VGPR pairs double address registers and use `v_add_co/v_addc` chains.
- **Large strides / TLB**: on MI300-class parts, very large strided accesses across multi-GB tensors can hit TLB limits; tile so each workgroup's working set stays within a few hundred MB.

## 3. LDS and bank conflicts

- Declare with `__shared__ T buf[...]` (static) or `extern __shared__` (dynamic). Align to 16 B.
- `ds_read_b128` / `ds_write_b128` are the widest ops; they are serviced in sub-wave groups - the conflict-free condition is about each group of lanes, not the whole wave. Instead of analyzing by hand: write the layout, **measure `SQ_LDS_BANK_CONFLICT` and `SQ_LDS_IDX_ACTIVE`** with rocprofv3/rocprof-compute and iterate (XOR swizzle on the 16 B chunk index within a row; or +16 B padding per N rows).
- MFMA operand loading from LDS: each lane reads K-contiguous elements for its row/column (e.g. 4 x f16 for 32x32x8 per lane = 8 B `ds_read_b64`; 8 x f16 for 16x16x32 on gfx950 = 16 B `ds_read_b128`). Pick the LDS layout so that these reads are K-contiguous and conflict-free; write side then needs the transposed pattern - use the swizzle to satisfy both.
- gfx950: use `ds_read_b64_tr_b16` family when the global layout is MN-major but the MFMA wants K-contiguous.
- Epilogue: stage C through LDS to convert the MFMA output layout (each lane holds strided elements of the 16x16/32x32 tile) to row-contiguous 16 B stores; reuse the mainloop LDS allocation.
- LDS occupancy: workgroups/CU = floor(LDS_per_CU / LDS_per_WG); a 64 KB workgroup on a 64 KB-LDS part leaves exactly 1 WG/CU.

## 4. MFMA mainloop (gfx908/90a/942/950)

Builtins (examples; signatures use clang vector types):
```cpp
typedef _Float16 half4  __attribute__((ext_vector_type(4)));
typedef float    float16 __attribute__((ext_vector_type(16)));
// 32x32x8 f16: A,B: 4 halves per lane; C/D: 16 floats per lane (wave64)
d = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, c, /*cbsz*/0, /*abid*/0, /*blgp*/0);
// 16x16x16 f16:  A,B: 4 halves/lane; C/D: 4 floats/lane
d = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
// gfx950: larger K per instruction (8 halves per lane)
d = __builtin_amdgcn_mfma_f32_16x16x32_f16(a8, b8, c, 0, 0, 0);
// gfx942 FP8: 8 fp8 packed in a 64-bit operand per lane
d = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a64, b64, c, 0, 0, 0);
```
- Pick the shape by register pressure vs reuse: `32x32` reads fewer operand bytes per FLOP (better LDS/register bandwidth), takes 16 accumulator registers per lane and has longer latency (16 passes on gfx90a/942 for f16 32x32x8); `16x16` uses 4 accum regs and shorter latency; mix to balance. Verify cycle counts per target in the ISA guide / `llvm-mca`-style tables.
- Issue model: MFMA runs on the matrix core while the VALU/LDS/VMEM units can issue other instructions from *other* waves - or the same wave if independent. Interleave `ds_read`, `global_load`, address increments and MFMAs: **dependent MFMA chains need independent accumulators**; a back-to-back dependent MFMA (same accumulator) incurs a stall of the previous instruction's pass count.
- Wave tile: a 256-thread workgroup (4 waves) computing 256x256: 2x2 wave grid, each wave 128x128 = 4x4 tiles of 32x32 (16 accumulators of 16 regs = 256 regs) on gfx942 - at the register limit, hence 1 wave/SIMD; or 128x128 workgroup tiles with smaller wave tiles for 2 waves/SIMD.
- Epilogue converts packed accumulators (e.g. `v_cvt_pk_bf16_f32` on gfx950, software round on gfx942) before store.
- Use the matrix instruction calculator to derive lane->(row,col) maps for each `v_mfma_*`.
- Sparse: `__builtin_amdgcn_smfmac_f32_16x16x32_f16` style (compressed A + index) on gfx942/950.
- FP64 GEMM (gfx90a+/942/950): `__builtin_amdgcn_mfma_f64_16x16x4f64`.

## 5. WMMA mainloop (gfx1100-1103, 1150-1153, 1200-1201)

```cpp
typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef _Float16 half8  __attribute__((ext_vector_type(8)));
typedef float    float8 __attribute__((ext_vector_type(8)));
// gfx11: A,B = 16 halves per lane (replicated across lane halves), C = 8 floats
c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a16, b16, c);
// gfx12: A,B = 8 halves per lane, C = 8 floats
c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a8, b8, c);
// gfx12 FP8: __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(...)
```
- Compile separate code paths with `#if defined(__GFX11__)` / `__GFX12__`; layouts and operand counts differ.
- Throughput is modest compared to CDNA; the usual limiter on consumer parts is LDS/VGPR bandwidth and clocks - use large per-wave tiles (e.g. 64x64 per wave = 4x4 WMMA tiles) and register double-buffer fragments.
- rocWMMA provides portable fragment abstractions across gfx11/gfx12 (and MFMA); use as baseline or reference for layouts.
- RDNA2 has no WMMA: implement SGEMM-like kernels with `v_fma_f32`/`v_dot2_f32_f16`; ensure wave32 compile (default) and consider `-mcumode` only if workgroup-level LDS sharing across WGP halves hurts.

## 6. Direct-to-LDS loads and pipelining

- gfx942: `global_load_lds_dword` (4 B/lane; needs M0 for LDS base, address pattern is wave-contiguous): usable but narrow; many production kernels still use register-staged prefetch on gfx942.
- gfx950: `global_load_lds_dwordx3/x4` (12/16 B per lane): `__builtin_amdgcn_global_load_lds(ptr_g, ptr_lds, size, offset, aux)` with size 16 (verify builtin signature in clang 21; CK-Tile exposes wrappers). LDS destination is **wave-linear** (consecutive lanes write consecutive 16 B chunks): choose the global->LDS lane mapping so that the resulting LDS layout is what MFMA reads want (swizzle on the *global* address side, since the LDS side is fixed).
- Completion tracking with `s_waitcnt vmcnt(N)` (loads) then `s_barrier` for cross-wave visibility; the compiler's automatic waitcnt insertion is conservative - check the ISA for unnecessary `vmcnt(0)` inside the loop and restructure with explicit `__builtin_amdgcn_s_waitcnt` only if needed.
- Typical loop (2-3 stage): issue global->LDS for tile k+2 -> `ds_read` operands of tile k+1 into registers -> MFMA on tile k -> `s_waitcnt` / barrier -> advance. Keep MFMA density high between waits.

## 7. Instruction scheduling controls

The AMDGPU backend scheduler often clusters loads or MFMAs poorly in hand-written pipelines. Tools:
- `__builtin_amdgcn_sched_group_barrier(mask, size, syncID)`: build an explicit instruction mix pattern, e.g. interleave 1 MFMA : 1 `ds_read` : 1 `global_load`. Masks (verify values in clang docs): `0x008` MFMA/WMMA, `0x100` DS read, `0x200` DS write, `0x020` VMEM read, `0x040` VMEM write, `0x002` VALU, `0x004` SALU.
- `__builtin_amdgcn_sched_barrier(mask)`: prevent the scheduler from moving selected instruction classes across this point (mask 0 = block everything).
- `__builtin_amdgcn_iglp_opt(n)`: predefined interleave patterns (n=0/1 for GEMM-style DS/MFMA mixes); quick win to try before manual groups.
- `__builtin_amdgcn_s_setprio(p)` (0-3): raise priority of a wave during the MFMA burst and lower it elsewhere to produce natural ping-pong between waves sharing a SIMD (used in 8-wave ping-pong GEMMs on gfx942/950).
- `__builtin_amdgcn_s_barrier()` and `__builtin_amdgcn_wave_barrier()`; `__builtin_amdgcn_s_waitcnt(imm)`; `__builtin_amdgcn_s_sleep(n)` for polling loops.
- LLVM flags (only when profiling justifies): `-mllvm -amdgpu-schedule-...`, `-mllvm -amdgpu-igrouplp-exact-solver-max-branches=...`.

## 8. XCD-aware workgroup remapping (gfx942, gfx950)

Workgroups are distributed round-robin across XCDs: WG linear id `i` -> XCD `i % NUM_XCD` (NUM_XCD = 8 for MI300X/MI355X). Each XCD has its own L2, so consecutive tile ids land on different L2s and share nothing. Remap so that the contiguous block of tiles processed by one XCD is a compact 2-D patch:
```cpp
__device__ int remap_xcd(int wg, int num_wgs, int num_xcd = 8) {
  // Requires num_wgs % num_xcd == 0 for the simple form; otherwise handle the remainder.
  int xcd = wg % num_xcd;            // XCD this WG physically runs on
  int slot = wg / num_xcd;           // position within that XCD
  int per_xcd = num_wgs / num_xcd;
  return xcd * per_xcd + slot;       // tiles [xcd*per_xcd, ...) now run on XCD xcd
}
```
Then apply the usual L2 rasterization (grouped column-major over M-groups) to the remapped id. Make `num_xcd` a runtime/template parameter (1 in CPX/partitioned modes, 8 in SPX). Gains of 5-20 % on GEMM/attention for large problems are typical; always verify L2 hit rate in rocprof-compute.

## 9. Wave-level primitives

- `__shfl_xor(var, mask)` / `__shfl_down` etc. compile to `ds_bpermute_b32` (goes through LDS hardware, ~LDS latency). `__shfl_*_sync(mask, ...)` variants exist (mask is 64-bit on wave64: use `0xffffffffffffffffull`).
- **DPP** modifiers do intra-row (16 lanes) shuffles inside a VALU op with no extra instruction: `__builtin_amdgcn_update_dpp(old, src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)`. Reduction on wave64: 4 DPP steps (`quad_perm`, `row_shr`/`row_xmask`), then `row_bcast:15`/`:31` (gfx9) or `v_permlane*`/`readlane` + combine for cross-row; rocPRIM/hipCUB already implement these - copy their wave reduce.
- `v_readlane` / `__builtin_amdgcn_readlane(v, lane)` and `readfirstlane` move a lane to an SGPR (uniform broadcast); `v_writelane`.
- Wave votes: `__ballot(pred)` returns 64-bit mask on wave64 (32 on wave32 compile); `__popcll`/`__builtin_popcountll`; `__builtin_amdgcn_ballot_w64/w32`.
- gfx950: `__builtin_amdgcn_permlane32_swap`, `permlane16_swap` exchange halves without LDS.
- `__builtin_amdgcn_ds_swizzle` for constant patterns (older, still useful).

## 10. Atomics

- Float `atomicAdd` default semantics on gfx90a/942/950: implemented as a CAS loop unless you opt into hardware RMW: compile with `-munsafe-fp-atomics` or call `unsafeAtomicAdd(float*, float)` (also `unsafeAtomicMax/Min`). Safe only for coarse-grained (device) memory; fine-grained host-visible memory may require the safe path.
- Prefer wave-aggregation (ballot + DPP reduce + leader atomic), then block-level LDS atomics, then one global atomic per block.
- Packed `atomicAdd` on `__half2`/`__hip_bfloat162` exists on newer targets (verify per gfx); fp32 accumulation preferred.
- `__hip_atomic_*` builtins with explicit scope (`__HIP_MEMORY_SCOPE_AGENT`, `_WORKGROUP`, `_WAVEFRONT`) and ordering allow weaker/cheaper semantics than default `atomicAdd` (system scope) - choose the narrowest correct scope.

## 11. Graphs, streams, multi-GPU

- `hipGraph` capture/replay for sequences of small kernels; `hipStreamBeginCapture`. Graph launch latency is generally lower than loose launches on ROCm 7.
- MI250/MI210 multi-GCD and MI300/MI350 partitioning change device enumeration: use `HIP_VISIBLE_DEVICES`, query topology with `amd-smi`/`rocm-smi --showtopo`, and use RCCL for collectives (xGMI/Infinity Fabric).
- Allocation: `hipMallocAsync`/pools, `hipExtMallocWithFlags(hipDeviceMallocUncached/...)` for fine-grained needs; coarse-grained device memory is fastest.
- Overlap compute/copy with separate streams and `hipEvent`; pinned host memory via `hipHostMalloc`.

## 12. Libraries to build on

- **Composable Kernel (CK) / CK-Tile**: templated tile-programming library (GEMM, FMHA, norms, MoE) for gfx9/gfx11/gfx12; the reference for MFMA/WMMA pipelines, direct-to-LDS, XCD remap, scheduling.
- **hipBLASLt** (Tensile/TensileLite + autotuning, FP8/FP4 on gfx942/950), **rocBLAS**, **rocWMMA**, **rocPRIM / hipCUB** (scan, sort, reduce, histogram), **MIOpen** (conv), **aiter** (AMD's tuned AI ops: FMHA, MoE, GEMM), **RCCL**.
- Always benchmark against hipBLASLt for GEMM-shaped work and aiter/CK for attention; beat them through fusion, odd shapes or novel dtypes.

## 13. C++23 in device code

- clang treats `constexpr` functions as implicitly `__host__ __device__`, so many `<algorithm>`/`<bit>`/`<array>` utilities work on device if they are constexpr and free of host-only features. `std::bit_cast`, `std::array`, `std::span`, `if consteval`, `static operator()`, deducing `this`, `std::unreachable`, `[[assume(expr)]]` are all usable and helpful (`[[assume]]` maps to `llvm.assume` - useful for alignment/divisibility facts).
- `std::mdspan` can be used for host/device views but check each accessor/layout for `__device__` viability; CK-Tile and rocWMMA ship their own tensor abstractions.
- Avoid on device: exceptions, RTTI, `<iostream>`, allocating containers, `<format>`/`<print>`, `std::function`, virtual dispatch on hot paths.
- HIP builds are `-std=c++23` with clang 21; ROCm headers expect clang (do not build HIP with GCC).

## 14. Pitfalls

- gfx942/950 binaries are not forward-compatible; missing target in `--offload-arch` -> runtime "invalid device function".
- FP8 FNUZ vs OCP mismatch silently produces wrong magnitudes (factor of 2 / different NaN encoding).
- Over-unrolled MFMA loops blow the instruction cache (shared between CUs in a WGP-like group): limit unroll to what fits (check `.text` size, `s_nop`/stalls with I-cache counters).
- LDS bank-conflict analysis from NVIDIA experience does not transfer exactly; measure.
- `__syncthreads()` on a workgroup split into different wave roles deadlocks; use `s_barrier` only when all waves participate, otherwise use LDS flags + `s_sleep`/`s_waitcnt` or split roles across workgroups.
- `hipcc` with `-fgpu-rdc` disables many optimizations across TU boundaries and slows builds: avoid unless device linking is required.
- Large kernels with `scratch` > 0: check `.private_segment_fixed_size` in the code object; fix before anything else.
- Forgetting `-O3` plus `-mllvm -amdgpu-early-inline-all=true` style flags are rarely needed; the biggest gains come from `__launch_bounds__`/`waves_per_eu`, `-munsafe-fp-atomics`, and correct vector widths.
