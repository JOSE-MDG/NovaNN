# Pattern playbook (distilled from launch_config_requirements.md)

Read the section matching the kernel's pattern. Each list is priority
order: do the first items before the later ones. "Why" is one line so
the reasoning survives without the source doc.

## ElementWise (streaming maps, casts, activations)

1. Prove 8/16B vector transactions from alignment + contiguity. Why:
   one wide transaction replaces up to 4 scalar ones at equal latency.
2. Coarsen 4 elements/thread when bandwidth-bound (the usual case).
   Why: fewer blocks, better instruction overlap, same traffic.
3. Pick ILP vs occupancy from the roofline knee, not by feel. Why:
   compute-bound kernels waste threads that could be registers.
4. Mask tails; separate tail kernels only when the remainder fills a
   wave. Why: extra launches cost microseconds each.
5. L2-persist only data the next kernel rereads immediately. Why:
   the persistent window is small and shared.

## Reduction (sum/max/argmax over an axis x batch)

1. Vectorize the load, coarsen per thread first. Why: global traffic
   dominates; combining is cheap next to it.
2. Shuffle within the warp before touching shared memory. Why:
   register exchange has no bank conflicts and no `__syncthreads`.
3. One shared sweep across warps, then one combine. Why: CUB's
   warp-reductions variant needs a single barrier and zero bank
   conflicts for most types.
4. Stateful ops (argmax) carry index+value: double partial storage
   everywhere in the budget. Why: silent truncation otherwise.
5. Multi-block needs an explicit combine: ordered single-pass atomics
   or classic two-pass. Why: arrival order is nondeterministic, so
   deterministic results forbid the aggressive path.
6. Accumulate wider than input (fp16 in, fp32 accum) and budget the
   extra live registers up front. Why: precision change is a register
   change.

## LayoutTransformation (transpose, permute, contiguous)

1. Tile through shared memory; never transpose in place globally. Why:
   strided global writes cannot coalesce.
2. Pad one column (or swizzle addresses) for bank conflicts. Why:
   a power-of-two stride maps every lane to one bank.
3. Keep both sides coalesced: tile edge a multiple of the transaction
   width. Why: 32/64/128B per warp transaction is the only efficient
   granularity.
4. Rows-per-thread from register headroom, warps before width. Why:
   registers are free until the quantum cliff.
5. Skip manual staging where hardware transpose-on-copy exists. Why:
   dead code on capable architectures.

## ParallelScan (cumsum, prefix sums, offsets)

1. Warp shuffle scan first, shared upsweep/downsweep across warps.
   Why: same register-exchange win as reduction.
2. Odd elements-per-thread while it fits. Why: breaks bank-stride
   alignment for free.
3. Double-buffer only when overlapping next-tile loads AND 2x fits.
   Why: otherwise it halves occupancy for nothing.
4. Multi-block: chained single-pass needs ordered atomics plus a
   reorderable op; else classic multi-pass. Why: prefixes must see
   predecessors in order.
5. Non-associative float ops forbid reordering entirely. Why:
   (a+b)+c != a+(b+c) is a correctness bug, not a tuning knob.

## SortAndHistogram (radix sort, counting)

1. Privatize by contention: registers, then warp-aggregated single
   atomic, then shared per block, then global. Why: atomics serialize;
   depth of privatization sets throughput.
2. Skewed input privatizes one level deeper at equal volume. Why:
   concentration turns shared atomics into a serial queue.
3. Fix radix width at 8 bits unless digit tables overflow shared.
   Why: fewer passes, predictable tables.
4. Budget key+value traffic together. Why: pairs double shared and
   bandwidth accounting.

## SpatialNeighborhood (stencil, conv, pooling)

1. Stage tile+halo in shared; score tiles by useful-payload over
   loaded bytes. Why: large radius relative to tile loads data used
   once at edges.
2. Borders clamp once at load, not per element in the loop. Why:
   per-element predicates diverge every warp touching an edge.
3. 1x1 filters stream like element-wise: no halo, no staging. Why:
   staging without reuse is pure overhead.
4. Small fixed filters belong in the read-only path, out of the
   shared budget. Why: constant cache broadcast is free bandwidth.
5. Dense small-filter many-channel conv is a GEMM in disguise when
   tensor/matrix units exist. Why: 10x+ throughput gap.

## GEMM (dense matmul)

1. Tile for the memory hierarchy (register fragment <- warp tile <-
   shared tile), sized from the MMA shapes of the dtype. Why: every
   level reuses data the level below cannot hold.
2. Check wave quantization before fancy schedules: tiles vs SMs. Why:
   an idle last wave dwarfs tile-shape gains; split-K/Stream-K exist
   for this case only.
3. Persistent CTAs when tiles dwarf SMs; swizzled traversal for L2
   reuse between neighbors. Why: fewer launches, hotter cache.
4. Fuse bias/activation/norm from registers/shared, never round-trip
   through global. Why: memory traffic is the cost, ALU is free.
5. Warp-specialize (load vs compute groups) at 3+ pipeline stages
   with async engines. Why: overlap needs role separation.
6. Closed form is insufficient here: shortlist analytically, measure
   empirically, cache by (shape, dtype, arch). Why: the space is too
   rough for formulas (CUTLASS/nvMatmulHeuristics pattern).

## GatherScatter (indexing, embedding lookup)

1. Assign by segment size: thread per element, warp per segment,
   block per segment with staging. Why: each level matches a
   coalescing granularity.
2. Dedup within the warp before emitting shared transactions. Why:
   one transaction replaces up to a warp of duplicates.
3. Presort indices only when sort cost < locality gain (scattered,
   low locality, large volume). Why: sorting is itself memory-bound.
4. Write-heavy scattered traffic stays coarser at equal sizes. Why:
   atomic contention scales worse than read duplication.

## FusedOperator (softmax, norm, attention-style)

1. Fuse along dataflow while intermediates fit registers/shared.
   Why: every global round-trip costs what the ALU does for free.
2. Budget the live peak, not the stage total. Why: equal op counts
   hide very different register cliffs depending on release order.
3. Recompute cheap values instead of keeping them live when the math
   favors it. Why: ALU is cheap, pressure is not.
4. Break the fusion explicitly at the first illegal boundary and
   launch the remainder separately. Why: silent global spill inside
   one "fused" launch is the worst of both worlds.
5. Warp-only barriers for single-warp-streamable fusions. Why:
   `__syncthreads` for one warp is pure overhead.
