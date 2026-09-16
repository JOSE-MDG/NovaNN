# Launch sizing (distilled from launch_heuristics_tiers.md)

Grid/block geometry is derived, never guessed. Two paths exist
because analysis has a cost: closed-form for small launches,
measured analysis only when the shape proves it worthwhile.

## The router (all integer ops on cached values)

1. **Sub-wave work → cheap path.** Work below one full resident wave
   (`smCount * maxThreadsPerSM`, or output tiles below `smCount`)
   has nothing to balance; occupancy refinement there is noise.
2. **Launch-dominated time → cheap path.** Roofline estimate below
   ~10x launch overhead (~5 us isolated, ~0 captured) cannot be
   beaten by any configuration. Use 0.7 bandwidth efficiency with
   proven aligned-contiguous access, 0.4 otherwise (scalar
   misaligned access really is that slow); err toward analysis.
3. **Outside the cheap-path bound → analysis.** Each pattern has a
   fixed bound (single-block reduction/scan, rank ≤ 2 layout,
   all-dims ≤ 128 GEMM, ≤16B gather segments, ≤4 keepable fused
   stages). Beyond it no fixed decision is valid.
4. **Cache the analysis** by (pattern, shape bucket, dtype, arch).
   A hit costs one hash probe; a miss recomputes inline, never in
   the background on the hot path.
5. **Captured sequences analyze once.** Replay is free, so config
   quality dominates; isolated tiny launches stay cheap.

## Cheap-path constants (fixed, not tuned per launch)

256 threads (8 warps / 4 wavefronts), 4 elements per thread, grid
capped at 8 blocks per SM, masked tails, single pass. Threads step
down whole warps at register-quantum crossings (the 32→33 cliff);
kernel authors hold ≤32 regs/thread to keep 256. Shared clamps
always apply: warp multiples, device/kernel max threads, per-axis
grid limits, all-ones floor.

## What gets asserted in tests

Pin the decisions, not just legality: exact thread/block counts for
reference shapes on stub caps, coverage-boundary true/false at the
exact limits, register fit-down steps, cache hit returning stored
geometry without recomputation, floor on empty input. Geometry
without assertions rots back into magic numbers.
