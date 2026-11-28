# Build, flags, and toolchain (CUDA 13.x / ROCm 7.x)

## Contents
1. Fixed toolchain
2. CUDA: nvcc flags and arch targets
3. HIP: hipcc/clang flags and offload targets
4. CMake
5. Inspecting compiled kernels
6. Compile-time and binary-size control
7. Runtime dispatch checklist
8. NovaNN build wiring

---

## 1. Fixed toolchain

| | CUDA | HIP |
|---|---|---|
| SDK | CUDA 13.x | ROCm 7.x |
| Host compiler | GCC >= 15 (Linux), MSVC (Windows) | clang >= 21 (Linux only, `hipcc` wraps AMD clang) |
| Standard | `-std=c++20` | `-std=c++23` |
| Device compile | `nvcc` (NVVM -> PTX -> `ptxas` -> SASS) | clang -> LLVM AMDGPU -> code object (per gfx) |

Never mix: do not compile HIP code with GCC; do not rely on C++23 features in `.cu` files.

## 2. CUDA flags

Baseline release flags:
```
nvcc -std=c++20 -O3 --expt-relaxed-constexpr --expt-extended-lambda \
     -lineinfo -Xptxas=-v --threads 0 \
     -gencode arch=compute_80,code=sm_80  -gencode arch=compute_86,code=sm_86 \
     -gencode arch=compute_89,code=sm_89  -gencode arch=compute_90a,code=sm_90a \
     -gencode arch=compute_100f,code=sm_100f -gencode arch=compute_120a,code=sm_120a ...
```
- `-lineinfo` is free at runtime and required for source-level Nsight correlation. Do not use `-G` (disables optimization) except for debugging.
- Arch targets (from the supported set 75, 80, 86, 89, 90, 100, 103, 110, 120, 121):
  - Plain targets (`sm_75/80/86/89`): portable SASS; add one PTX target (highest plain arch) for forward-compat where wanted.
  - `sm_90a`: wgmma, setmaxnreg; **not forward-compatible**.
  - `sm_100a` / `sm_103a` (exact arch) or `sm_100f` (family: runs on 100 and 103): tcgen05/TMEM, block-scaled kinds.
  - `sm_110a` (Thor): check the PTX ISA release notes for the exact feature set before enabling arch-specific code.
  - `sm_120a` / `sm_121a`: block-scaled/low-precision `mma.sync`.
  - Shorthands: `-arch=native` (dev box), `-arch=all-major`, `-arch=sm_90a` (shorthand for matching compute+sm).
- Math flags: `--use_fast_math` = `-ftz=true -prec-div=false -prec-sqrt=false -fmad=true` plus intrinsics; apply per-file or use explicit intrinsics (`__expf`, `__fdividef`, `__fsqrt_rn` etc.) instead of global fast-math.
- Register/occupancy control: prefer `__launch_bounds__`/`__maxnreg__`; global `-maxrregcount` is blunt.
- Compile-time: `--threads N` parallelizes per-arch compilation; `-split-compile N` helps big translation units; `-t0`; avoid heavy header-only template trees in every TU (explicit instantiation in separate `.cu` files, e.g. one per tile config).
- Fatbin size: CUDA 13 compresses by default; `-Xfatbin=-compress-all` for older flows. Limit arch list in dev builds.
- Windows/MSVC: `-Xcompiler "/Zc:preprocessor /Zc:__cplusplus"`, `/permissive-` needed for modern CUTLASS/CuTe; avoid GCC attributes in shared headers.
- Useful defines: `-DNDEBUG`, `-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1`, `-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED` (as the library requires).
- Linking driver API (TMA `cuTensorMapEncodeTiled`): link `-lcuda`, or obtain via `cudaGetDriverEntryPointByVersion` to avoid a link-time dependency.

## 3. HIP flags

```
hipcc -std=c++23 -O3 -munsafe-fp-atomics \
      --offload-arch=gfx90a --offload-arch=gfx942 --offload-arch=gfx950 \
      --offload-arch=gfx1100 --offload-arch=gfx1201 \
      -Rpass-analysis=kernel-resource-usage k.hip -o k
```
- `--offload-arch=<gfx>` once per target (repeat flag); do not use deprecated `--amdgpu-target`.
- `-munsafe-fp-atomics`: hardware FP atomics on gfx90a/942/950 (coarse-grained memory only). Set per target file if mixed with fine-grained host-shared memory.
- `-ffp-contract=fast` is already the default for HIP (FMA contraction); `-ffast-math` only per-op after an accuracy audit; `-fgpu-flush-denormals-to-zero` as needed.
- `-fno-gpu-rdc` (default) is fastest; `-fgpu-rdc` only for device-side linking, at an optimization cost.
- Code size / parallel compile: `-parallel-jobs=N` for multi-arch device compilation; `-mllvm -amdgpu-...` flags are experimental - use only with measurement.
- Wave size override (rare): `-mwavefrontsize64` on gfx10+/gfx11/gfx12; default wave32 is usually right. `-mcumode` for CU mode on gfx10+.
- Features: `--offload-arch=gfx942:sramecc+:xnack-` to pin features when required.
- Debug info for profilers: `-g1`/`-gline-tables-only` keeps optimization and gives line tables for rocprof-compute source views.
- Env at runtime: `HIP_VISIBLE_DEVICES`, `HSA_XNACK`, `AMD_LOG_LEVEL` (diagnostics), `GPU_MAX_HW_QUEUES`, `HIP_LAUNCH_BLOCKING=1` (debug only).

## 4. CMake

CUDA (CMake >= 3.27 recommended):
```cmake
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "80-real;86-real;89-real;90a-real;100a-real;120a-real")  # add -virtual/PTX entry if wanted
target_compile_options(my_ops PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo -Xptxas=-v --threads 0>)
```
- Family-specific (`100f`) and newer suffixes need a recent CMake (check release notes); if unsupported, set `CMAKE_CUDA_FLAGS` with explicit `-gencode` lines and `CMAKE_CUDA_ARCHITECTURES OFF`.
- Per-source arch: use `set_source_files_properties(... COMPILE_OPTIONS ...)` or separate object libraries so `sm_90a`-only kernels do not break other-arch builds.

HIP:
```cmake
find_package(hip REQUIRED CONFIG)
set(CMAKE_HIP_STANDARD 23)
set(CMAKE_HIP_ARCHITECTURES "gfx90a;gfx942;gfx950;gfx1100;gfx1201")
enable_language(HIP)
target_compile_options(my_ops PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-munsafe-fp-atomics>)
```
Use `CMAKE_HIP_COMPILER` = ROCm's clang (`/opt/rocm/llvm/bin/clang++`); require >= 21 via a configure-time check.

Shared project structure: one target per backend (`ops_cuda`, `ops_hip`), a common interface library for headers/specs/tests, and runtime dispatch in plain C++ (no device-language dependency).

## 5. Inspecting compiled kernels

Use `scripts/inspect_kernel.sh`:
```
scripts/inspect_kernel.sh cuda kernel.cu sm_90a      # -> resource usage (+ SASS excerpt)
scripts/inspect_kernel.sh hip  kernel.hip gfx942     # -> kernel-resource-usage remarks + ISA .s
```
Manual equivalents:
- NVIDIA: `cuobjdump -sass a.out | less`, `nvdisasm -c -g file.cubin`, `cuobjdump -ptx`, `cuobjdump -res-usage a.out` (registers, smem, local mem per kernel), `nvcc -ptx` to read PTX.
- AMD: `hipcc --offload-device-only -S` for GCN/RDNA asm; `roc-obj-ls`/`roc-obj-extract` to pull code objects from a fat binary then `llvm-objdump -d --mcpu=gfx942`; `llvm-readelf --notes` (AMDGPU metadata: `.vgpr_count`, `.agpr_count`, `.sgpr_count`, `.private_segment_fixed_size`, `.group_segment_fixed_size`, `.wavefront_size`); `-save-temps` retains `.s`, `.bc`, `.o` per arch.

What to look for is listed in `methodology.md` section 5.

## 6. Compile-time and binary-size control

- Template tile configs multiply compile time: instantiate only configs the heuristic can choose, and put each in its own TU.
- Restrict dev builds to the local arch (`CMAKE_CUDA_ARCHITECTURES=native`, single `--offload-arch`). CI builds the full matrix.
- Avoid including full CUTLASS/CK headers in every TU; use forward declarations and explicit instantiations.
- Large unrolled MFMA/wgmma loops: watch ptxas/LLVM scheduling time and I-cache pressure; consider `#pragma unroll N` caps.

## 7. Runtime dispatch checklist

1. Read device arch: CUDA `cudaDeviceGetAttribute(&major/minor)`, HIP `hipDeviceProp_t::gcnArchName` (strip target-ID features such as `:sramecc+:xnack-`).
2. Map to a kernel tier (tables in `SKILL.md`) and verify the binary was built for it (query function attributes: `cudaFuncGetAttributes` / `hipFuncGetAttributes` returns error if the kernel is absent for the arch).
3. Set opt-in attributes (dynamic smem/LDS size, cluster size, carveout) once at init.
4. Choose the tile config from a table keyed by (arch, dtype, shape bucket), fall back to the vendor library when no tuned entry exists.
5. Log the chosen variant under a debug flag so performance reports are attributable.
6. Support `NVIDIA/AMD` unavailability gracefully: return a clear error instead of launching a mismatched kernel (silent wrong-arch launches fail with "invalid device function" / "no kernel image").

## 8. NovaNN build wiring

In NovaNN, sections 2 to 4 are implemented by the CMake layer; never invoke
nvcc/hipcc manually for repo code and never add ad hoc `-gencode` or
optimization flags at source level.

- Configure, build and test through the preset scripts:
  `scripts/build-presets.sh <preset>` then `scripts/compile-presets.sh -j $(nproc) <preset>`
  then, for `-test-*` presets, `scripts/run-tests.sh <preset>`. Backend
  presets are `cuda-*` (Windows/Linux) and `hip-*` (Linux only); CUDA and
  HIP are mutually exclusive at configure time.
- Arch lists live in the CMake layer (`cmake/Modules/NovaNNCUDA.cmake`
  validates `CMAKE_CUDA_ARCHITECTURES`). To retarget or add an arch,
  change it there, not per source file.
- Per-source arch isolation follows the existing object-library targets
  (`ncore::obj::*` aggregated by `ncore::native`); new backends' kernels
  join the matching `ncore/native/<backend>/kernels/CMakeLists.txt`.
- A clean tree must rebuild warning free. Incremental builds hide warnings
  in skipped translation units: measure warnings only after
  `./clean.sh --target build/<preset>`.
- Confirm the build succeeded before interpreting test results: a failed
  Ninja compile deletes object files and ctest then runs a stale binary.
  Grep `build/logs/<preset>.log` for `error:`, and remember the logs
  append across runs.
- The vcpkg submodule is the registry baseline; after
  `scripts/update-vcpkg.sh` re-run the GPU gates (moving it re-resolves
  the ports).
- Sanitizer presets get their runtime environment through
  `NOVA_SANITIZER_ENV` in `ncore/tests/CMakeLists.txt` and `suppr.txt`;
  leave that wiring intact.
