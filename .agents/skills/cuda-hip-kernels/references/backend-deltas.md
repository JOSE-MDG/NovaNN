# CUDA ↔ HIP backend deltas

Port by invariant. Every item below has broken a real 1:1 port;
check each when touching the sibling backend.

## Execution width

- CUDA warp is always 32 lanes. HIP wavefront is 64 on CDNA
  (wave64-only) and 32 or 64 on RDNA (selected at compile time).
- Never hardcode 32. Derive spans from `warpSize` (device) or
  `__AMDGCN_WAVEFRONT_SIZE__` (compile time). Query wavefront size
  on host from device properties.
- Divergence costs scale with width: the same branch idles up to 64
  lanes on wave64. Tighter uniformity discipline on HIP.

## Masks and ballots

- `__ballot`/`__activemask` return 64-bit masks on wave64 AMD;
  32-bit code silently drops half the lanes. Use 64-bit mask types
  (`1ull << lane`) with a `lane_mask_t` alias selected per arch.
- Shuffle exists on AMD (DPP, `ds_permute`/`ds_bpermute` routes
  through LDS hardware without consuming it) but spellings and
  availability are arch-specific; guard with `__gfx942__` /
  `__AMDGCN_WAVEFRONT_SIZE__` checks. Prefer portable shuffle
  wrappers; reach for `__builtin_amdgcn_*` only with an arch guard.

## Launch bounds conversion

- CUDA: `__launch_bounds__(MAX_THREADS, MIN_BLOCKS_PER_SM)`.
- HIP: same spelling, second parameter is
  `MIN_WARPS_PER_EXECUTION_UNIT` — convert blocks→warps explicitly,
  do not copy the number across.
- Both force spilling to meet the bound; compiler defaults are
  usually within noise of optimal, so bound only with ≥5% measured
  cause (NVIDIA guidance).

## Matrix units

- CUDA: `mma` PTX / wmma, shaped per dtype per arch.
- HIP: MFMA builtins (`__builtin_amdgcn_mfma_*`), wavefront-wide,
  shape-specific register layouts. Prefer Composable Kernel over
  hand-rolled builtins; consult per-arch MFMA tables first.

## Compilation and language

- This project: `.cu` at C++20 (GCC host, nvcc device), HIP/host at
  C++23. Shared headers stay in the C++20 subset; C++23 library
  features are host-only.
- HIP needs `__HIP_PLATFORM_AMD__` defined when neither platform
  macro is set (tooling parses otherwise).
- HIP compiles with Clang: stricter warnings than nvcc/GCC paths.
  New code must be clean under `-Wconversion -Wsign-conversion
  -Wshadow -Wold-style-cast` on all three compilers. Explicit
  casts, `U`/`ULL` suffixes, no shadowing (watch locals vs struct
  members), no C-style casts.
- `__syncthreads` is a block barrier only; it does not order async
  copies — use event/waitcnt semantics for that on both backends.
- `__global__` must return `void`; results travel through pointers.
