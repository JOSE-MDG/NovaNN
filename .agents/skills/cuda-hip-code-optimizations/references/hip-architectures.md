# HIP / ROCm architectures: gfx908 .. gfx1201 (ROCm 7.x, clang 21)

Numbers here are design-time defaults. Query at runtime (`hipDeviceGetAttribute`, `hipGetDeviceProperties`, `hipOccupancyMaxActiveBlocksPerMultiprocessor`) and confirm instruction availability against the LLVM AMDGPU backend docs (`AMDGPUUsage`) and AMD's CDNA/RDNA ISA guides. Items marked (verify) deserve a double-check before you depend on them. For matrix instruction register layouts, use AMD's **matrix instruction calculator** (ROCm/amd_matrix_instruction_calculator) instead of guessing.

## Contents
1. Execution model recap
2. CDNA sheets: gfx908, gfx90a, gfx942, gfx950
3. RDNA sheets: RDNA2, RDNA3/3.5, RDNA4
4. Feature matrix
5. FP8 formats across targets
6. Compile targets and feature flags

---

## 1. Execution model recap

- **CDNA (gfx9xx): wave64 only.** 4 SIMD16 units per CU; a wave64 VALU op takes 4 cycles on a SIMD, 4 SIMDs issue in parallel; max 8 waves/SIMD (gfx90a+: 8; occupancy bound by VGPR+AGPR file of 512 regs/lane per SIMD). Scalar unit (SALU) + SGPRs handle wave-uniform work: exploit uniformity (scalar loads, scalar address math).
- **RDNA (gfx10+): wave32 default** (wave64 optional via `-mwavefrontsize64`, rarely better). Work-group processors (WGP = 2 CUs sharing LDS) in **WGP mode** (default for HIP on gfx10+; workgroups can span the full WGP's LDS) vs **CU mode** (`-mcumode`): affects LDS sharing and barrier behaviour.
- LDS: banked 4 B; capacity per workgroup 64 KB on gfx908/90a/942, **160 KB on gfx950**, 64 KB per workgroup on RDNA2/3 (WGP has 128 KB total; check attribute), more on RDNA4 (verify). 
- Registers: unified VGPR/AGPR file on gfx90a+ (arch VGPR 0-255 + AGPR 256-511 allocated together, 512 max per wave); gfx908 has separate 256 VGPR + 256 AGPR files, and MFMA accumulators must be in AGPRs on gfx908.
- Cache line size 128 B (L2); L1/vector cache 64 B lines on older parts (verify per target); coalescing granularity per wave is 64 B-ish chunks - prefer 16 B per lane, consecutive lanes consecutive addresses, i.e. a wave64 `dwordx4` load covers 1024 B.
- Scalar memory (`s_load_dwordx*`) is the cheapest path for wave-uniform, read-only data (kernel args, scales, bias vectors read by every lane): mark pointers `const __restrict__` and keep the address wave-uniform so the compiler emits SMEM loads.

## 2. CDNA sheets

### gfx908 - CDNA1 (MI100)
- MFMA: FP32/FP16/BF16 (original BF16 `bf16`, not `_1k`)/INT8. **No FP64 matrix, no packed FP32, no HW FP32/FP64 global atomic add (CAS loops).**
- 64 KB LDS, wave64, 120 CUs on MI100, HBM2 ~1.2 TB/s.
- Accumulators must live in **AGPRs** (`v_accvgpr_read/write` to move); compiler inserts copies - minimize epilogue traffic by converting directly from AGPR sets.
- Design guidance: 256x128 / 128x128 block tiles, 4 waves/workgroup, MFMA `32x32x8f16` (4 AGPR groups per tile) or `16x16x16f16`, double-buffered LDS (64 KB budget), software prefetch via registers (no direct-to-LDS).

### gfx90a - CDNA2 (MI210/MI250X; MI250 is 2 GCDs, each a separate device)
- Unified 512-entry VGPR file (AGPRs and VGPRs allocate together), MFMA reads/writes VGPRs directly (no mandatory AGPR copies).
- **FP64 MFMA** (`v_mfma_f64_16x16x4f64`, `4x4x4f64`), **packed FP32** ops (`v_pk_fma_f32`, `v_pk_add_f32`, `v_pk_mul_f32`, `v_pk_mov_b32`) for 2x FP32 throughput, BF16 `_1k` MFMA variants (use these; the original BF16 forms are deprecated).
- HW FP64/FP32 atomic add on global memory (enabled by `-munsafe-fp-atomics` or `unsafeAtomicAdd`; without it `atomicAdd(float*)` becomes a CAS loop to preserve fine-grained semantics).
- 64 KB LDS; L2 8 MB per GCD; each GCD is its own HIP device: multi-GCD needs peer copies/RCCL even within one package.
- Design guidance: as gfx908 but keep accumulators in VGPR namespace; use packed FP32 in non-MFMA-bound math (softmax, norms) when pattern fits.

### gfx942 - CDNA3 (MI300X / MI300A / MI325X)
- **Multi-die:** 8 XCDs per MI300X (38 CUs each, 304 CUs total; MI300A: 6 XCDs, 228 CUs), each XCD has a private 4 MB L2; shared 256 MB Infinity Cache (MALL) across the IODs, HBM3 ~5.3 TB/s (MI300X; MI325X HBM3E ~6 TB/s). **Workgroup dispatch is round-robin across XCDs** (WG id i lands on XCD i mod 8): tiles that share operands must be remapped (see `hip-techniques.md`) or L2 hit rates collapse.
- MFMA: FP16/BF16/TF32-ish **xf32**, INT8, **FP8/BF8 (FNUZ: E4M3FNUZ, E5M2FNUZ)**, FP64; structured **sparse MFMA** (`v_smfmac_*`); examples `v_mfma_f32_32x32x8_f16`, `v_mfma_f32_16x16x16_f16`, `v_mfma_f32_16x16x32_fp8_fp8`, `v_mfma_f32_32x32x16_fp8_fp8`.
- 64 KB LDS/CU; direct global->LDS loads exist but only dword (4 B) width (`global_load_lds_dword`, `buffer_load_dword ... lds`).
- 128-byte L2 line; large unified HBM; "unsafe" FP atomics as gfx90a.
- Partition modes (SPX/DPX/QPX/CPX) change what a "device" is (all XCDs vs subsets): confirm `hipGetDeviceProperties().multiProcessorCount` at runtime and keep XCD remap parametric.
- Design guidance: 256x256x64 / 256x128x64 FP16/BF16 tiles, 4-8 waves per workgroup, MFMA 16x16x16 or 32x32x8, **ping-pong LDS double buffering with explicit `sched_group_barrier` interleaving of `ds_read`/`global_load`/MFMA**, XCD-aware tile remap, FP8 with FNUZ conversion constants (scale accounting: FNUZ max = 240 for E4M3FNUZ, bias differs from OCP E4M3FN).

### gfx950 - CDNA4 (MI350X / MI355X)
- 8 XCDs, 32 CUs each (256 CUs), 4 MB L2 per XCD, 256 MB Infinity Cache, HBM3E ~8 TB/s; same XCD round-robin caveat.
- **LDS 160 KB per CU** (double gfx942): deeper stages/bigger tiles possible; LDS read/write bandwidth doubled (verify bank behaviour for b128 accesses in the profile).
- New/larger-K MFMA: e.g. `v_mfma_f32_16x16x32_f16/bf16`, `v_mfma_f32_32x32x16_f16/bf16`; FP8/BF8 now **OCP** formats (E4M3FN, E5M2) plus FP6/FP4 (`f8f6f4` family) and **block-scaled MFMA** (`v_mfma_scale_f32_16x16x128_f8f6f4`, `v_mfma_scale_f32_32x32x64_f8f6f4`, MX E8M0 scales); dedicated conversions (`v_cvt_scalef32_*`, `v_cvt_pk_*` for FP8/FP4, bf16 packed converts).
- **Direct-to-LDS global loads up to 96/128 bits** (`global_load_lds_dwordx3/x4`, `buffer_load_dwordx4 ... lds`): lets a pipeline load tiles without bouncing through VGPRs (frees registers, enables deeper prefetch).
- **LDS transpose reads** (`ds_read_b64_tr_b16`, `_b8`, `_b4`, `_b6`): load MFMA operands transposed straight from LDS (e.g. to feed MN-major B or FP4/FP8 layouts without extra permutes).
- `v_permlane16_swap_b32`, `v_permlane32_swap_b32` for cross-half-wave exchange; improved packed conversion and exp throughput (attention).
- Design guidance: same ping-pong skeleton as gfx942 but use direct-to-LDS loads, bigger K per MFMA, transposed LDS reads, 160 KB LDS budgets (e.g. 256x256x64 BF16 with 3 stages), scaled MFMA for MXFP8/MXFP4.

## 3. RDNA sheets

### RDNA2 - gfx1030 (Navi21), gfx1031 (Navi22), gfx1032 (Navi23), gfx1033 (Van Gogh), gfx1034 (Navi24), gfx1035 (Rembrandt iGPU), gfx1036 (Raphael iGPU)
- Wave32, WGP mode, **no matrix cores/WMMA** - GEMM-like work runs on VALU with packed dot instructions (`v_dot2_f32_f16`, `v_dot4_i32_i8` / `v_dot4c_i32_i8` availability varies per chip: confirm via `-mattr` and the target ID features, e.g. `+dot1-insts,+dot2-insts,...`).
- Infinity Cache (L3) 16-128 MB depending on the chip; GDDR6 (or DDR for iGPUs: gfx1035/1036 and Van Gogh are bandwidth-starved, optimize bytes).
- 128 KB LDS per WGP (up to 64 KB per workgroup; verify), up to 1024 VGPR/lane/SIMD, max 256 VGPRs/wave (wave32).
- Design guidance: register-tiled SGEMM-style kernels (8x8 thread tiles), `half2` packed math with FP32 accumulation via `v_dot2_f32_f16`, int8 dot products for quantized work, 128-bit global/LDS accesses, careful occupancy (16 waves/SIMD feasible with <= 64 VGPRs).

### RDNA3 - gfx1100 (Navi31), gfx1101 (Navi32), gfx1102 (Navi33), gfx1103 (Phoenix iGPU)
- Wave32, **WMMA** instructions (wave32): `v_wmma_f32_16x16x16_f16`, `_bf16`, `v_wmma_f16_16x16x16_f16`, `v_wmma_i32_16x16x16_iu8`, `_iu4`. HIP builtin form: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v16h a, v16h b, v8f c)`. **A and B fragments are replicated across the two 16-lane halves** (lanes 0-15 and 16-31 hold the same A/B data); accumulators are distributed 8 values/lane with interleaved row mapping - use the matrix calculator; rocWMMA wraps this.
- No FP8 matrix; `v_dot2_f32_bf16` etc. exist for BF16 dot paths (verify per chip).
- Dual-issue VALU (VOPD) for pairs of independent ops: write code with independent operand streams and let the compiler pair them (check `v_dual_*` in ISA); register-bank constraints apply.
- Larger VGPR file (192 KB/SIMD on Navi31 class) -> bigger register tiles viable (verify per chip); L2 6 MB (Navi31) + Infinity Cache 96 MB; LDS 128 KB per WGP, 64 KB per workgroup (verify).
- Design guidance: WMMA 16x16x16 tiles composed into 64x64 or 128x64 wave tiles, LDS-staged operands with padding to avoid conflicts, wave32 with 2-4 waves per workgroup per WGP, register double-buffered global loads (no direct-to-LDS), conversions packed.

### RDNA3.5 - gfx1150 (Strix Point), gfx1151 (Strix Halo), gfx1152 (Krackan Point), gfx1153 (verify target mapping on the exact SKU)
- Same WMMA ISA family as RDNA3, mostly **iGPUs with shared LPDDR5X** (Strix Halo: up to 256 GB/s, large unified memory, 40 CUs). Everything memory-bound: prioritize quantized weights, fusion and minimal DRAM traffic; WMMA throughput is rarely the limiter for LLM-decode shapes.
- Strix Halo can address huge unified pools (tens of GB) - validate SVM/`hipMallocManaged`/coarse- vs fine-grained behaviour and prefer device-local allocations for hot tensors (`hipExtMallocWithFlags` and `HSA_XNACK` considerations).

### RDNA4 - gfx1200 (Navi44), gfx1201 (Navi48)
- **WMMA v2** (gfx12 encoding): `v_wmma_f32_16x16x16_f16/bf16`, **FP8/BF8** (`v_wmma_f32_16x16x16_fp8_fp8`, `_bf8_*`), `iu8`, `iu4`; **new, smaller register layouts** (A/B no longer replicated across half-waves: 4 VGPR/lane for 16 f16 per half... use `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(v8h a, v8h b, v8f c)`); layouts differ from gfx11, so gfx11 and gfx12 kernels cannot share fragment code.
- **Sparse WMMA** `v_swmmac_*` (2:4 structured sparsity) with index operands.
- Improved caches/bandwidth, better ray-tracing blocks (irrelevant for compute); GDDR6 on Navi48/44; FP8 formats are OCP-style (verify).
- Design guidance: as RDNA3 with gfx12 fragment layouts, FP8 WMMA for inference GEMMs (scale in epilogue), swmmac only if sparsity is structured and worth the pruning cost.

## 4. Feature matrix

| Feature | 908 | 90a | 942 | 950 | RDNA2 | RDNA3/3.5 | RDNA4 |
|---|---|---|---|---|---|---|---|
| Wave size | 64 | 64 | 64 | 64 | 32 | 32 | 32 |
| Matrix instr | MFMA | MFMA | MFMA | MFMA | - | WMMA | WMMA v2 |
| FP16/BF16 matrix | y | y (BF16_1k) | y | y | - | y | y |
| FP8 matrix | - | - | FNUZ | OCP (+FP6/FP4) | - | - | y |
| Block-scaled MX | - | - | - | y | - | - | - |
| FP64 matrix | - | y | y | y (verify rate) | - | - | - |
| Packed FP32 | - | y | y | y | - | - | - |
| HW float atomics | - | y | y | y | verify | verify | verify |
| Direct global->LDS | dword | dword | dword | up to b128 | - | - | - |
| LDS per workgroup | 64 KB | 64 KB | 64 KB | 160 KB | 64 KB | 64 KB | verify |
| Multi-die XCD dispatch | - | - (2 GCD devices) | y | y | - | - | - |
| Sparse matrix | - | - | smfmac | smfmac | - | - | swmmac |

## 5. FP8 formats

- gfx942: E4M3**FNUZ** / E5M2**FNUZ** (no negative zero, no inf, single NaN pattern, different bias): values and max range differ from NVIDIA/OCP E4M3FN (max 448) - FNUZ E4M3 max is 240. Converting a model trained with OCP FP8 requires re-scaling/bias adjustment (halve scales or multiply bits accordingly).
- gfx950, RDNA4 (verify), NVIDIA: OCP E4M3FN / E5M2.
- Use HIP types `__hip_fp8_e4m3_fnuz`, `__hip_fp8_e4m3` (OCP) from `<hip/hip_fp8.h>` and select by arch at compile time (`__gfx942__`, `__gfx950__` macros).

## 6. Compile targets and feature flags

- Target IDs can carry features: `gfx90a:xnack+`, `gfx942:sramecc+:xnack-`. Defaults are usually right; mismatched xnack/sramecc between code object and runtime produces "no compatible code" errors.
- Macros: `__gfx908__`, `__gfx90a__`, `__gfx942__`, `__gfx950__`, `__gfx1100__`, `__gfx1200__`, family macros `__GFX9__`, `__GFX10__`, `__GFX11__`, `__GFX12__` (compile-pass specific). Use them to choose MFMA vs WMMA paths and wave-size-specific code; `__AMDGCN_WAVEFRONT_SIZE__` gives the pass's wave size.
- Build one code object per target with `--offload-arch=gfx942 --offload-arch=gfx950 ...`; each is a separate compile of the same source, so `#if defined(__gfx950__)` guards select per-target kernels. Host code cannot rely on these macros (host pass defines none of them): choose by `hipDeviceProp_t::gcnArchName` at runtime.
- No PTX-like forward-compat: a gfx942 binary does **not** run on gfx950. Ship every arch you support.
