# Methodology: measure, diagnose, validate

## Contents
1. Roofline and the theoretical floor
2. Benchmark protocol
3. NVIDIA profiling workflow (Nsight Systems / Nsight Compute)
4. AMD profiling workflow (rocprofv3 / rocprof-compute)
5. Reading resource usage and ISA
6. Diagnosis table (symptom -> cause -> fix)
7. Numerical validation
8. NovaNN protocol notes

---

## 1. Roofline and the theoretical floor

For each op compute:
- `bytes_min` = compulsory reads + writes (each unique input element once, each output once).
- `flops` = 2 * MACs for matrix ops, per-element op counts for pointwise ops (count transcendental ops separately: they run at 1/4 - 1/8 of FMA rate on SFU/MUFU).
- `AI = flops / bytes_min`.
- Ridge point = `peak_flops / peak_bw` for the dtype/pipeline in use (tensor vs CUDA-core/vector ALU differ by 10x+).
- Floor time = `max(bytes_min / achievable_bw, flops / achievable_flops)`.

Use **achievable** numbers measured on the actual box, not datasheet numbers:
- Achievable bandwidth: a vectorized device-to-device copy over a buffer much larger than the last-level cache (typically 85-93 % of datasheet on HBM parts, ~80-90 % on GDDR/LPDDR).
- Achievable compute: vendor GEMM at a large, well-shaped size for the same dtype, with clocks as they run under sustained load (tensor-heavy kernels usually run below boost clock; power throttling is real on consumer cards and on MI-series under full MFMA load).

Beyond DRAM: also consider L2 bandwidth, shared-memory/LDS bandwidth, issue slots, and special-function throughput. An op can be "compute-bound" on the MUFU unit (softmax exp in attention on Blackwell) or on LDS bandwidth (small-tile GEMM) rather than on the headline FLOPs.

## 2. Benchmark protocol

- Time on the device (CUDA/HIP events), never host wall-clock without full sync.
- Warm up (>= 10-20 launches) to pay JIT/lazy-loading, clock ramp, and cache cost.
- Run >= 100 iterations (more for sub-10 us kernels); report **median** and min; p90 reveals noise.
- Two regimes, report both when relevant:
  - *Hot*: back-to-back launches (inputs may live in L2). Representative of fused pipelines with small tensors.
  - *Cold*: flush L2 (and Infinity Cache / MALL on AMD) between iterations by writing a buffer larger than the cache. Representative of large-tensor inference/training steps. On MI300-class parts the Infinity Cache is 256 MB: flush >= 512 MB.
- For kernels shorter than ~10 us, event overhead dominates: capture N launches in a CUDA/HIP Graph and divide, or use kernel-internal clock counters.
- Pin clocks when comparing variants (NVIDIA: `nvidia-smi -lgc <min,max>`, may need root; AMD: `rocm-smi --setperfdeterminism <MHz>` or `amd-smi` equivalents). Always record the clocks used. Unpinned results swing 5-15 % with temperature.
- Randomize or at least use realistic data: all-zero inputs lower power draw and flatter tensor-core throughput (data-dependent power); do not benchmark with zeros.
- Compare against: (a) vendor library, (b) previous best of your own, (c) the roofline floor.
- Sweep shapes: powers of two hide tail/alignment bugs. Include prime-ish dims, M=1 / small-batch decode shapes, and the actual shapes from the model.
- Use `cudaFuncSetAttribute` / `hipFuncSetAttribute` for opt-in dynamic smem **outside** the timed region.

## 3. NVIDIA workflow

Timeline first (is the GPU even busy?):
```
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true -o out ./app
```
Look for: gaps between kernels (launch/sync overhead -> Graphs, PDL, fusion), H2D/D2H copies on the critical path, serialized streams, tiny kernels.

Kernel deep dive:
```
ncu --set full -k regex:my_kernel -s 5 -c 1 --import-source yes -o prof ./app
```
(`-s` skips warmup launches; `--import-source` plus compiling with `-lineinfo` gives per-source-line SASS correlation.)

Sections to read, in order:
1. **GPU Speed Of Light** (SM% vs DRAM%/L2%): which resource is the top limiter, and how far from 100.
2. **Memory Workload Analysis**: sectors/request (coalescing: ideal is 4 sectors/request for 32 x 4 B fully coalesced), L1/L2 hit rates, shared-memory bank conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st`), uncoalesced/ excessive global transactions.
3. **Warp State Statistics / Scheduler Statistics**: stall reasons. Interpretation:
   - `Stall Long Scoreboard`: waiting on global/local loads -> more bytes in flight, prefetch, cp.async/TMA, bigger tiles.
   - `Stall Short Scoreboard`: smem/MUFU/shuffle results -> reduce smem round-trips, vectorize.
   - `Stall MIO Throttle` / `LG Throttle`: memory instruction queue full -> wider vector instructions, fewer larger accesses.
   - `Stall Barrier`: imbalance between warps at `__syncthreads` -> restructure, use warp specialization/named barriers.
   - `Stall Math Pipe Throttle`: pipeline saturated (good for compute-bound).
   - `Stall Not Selected` high + `Eligible warps` > 1: issue-bound.
   - `Stall Wait` / `Stall Dispatch`: fixed-latency dependent chains -> ILP.
4. **Occupancy**: theoretical vs achieved, limiter (registers / smem / blocks / warps).
5. **Compute Workload Analysis**: pipe utilization (FMA, ALU, tensor, XU/MUFU); tensor pipe < ~80 % in a GEMM means feed starvation.
6. **Source counters**: per-line stall samples, spills (`local` loads/stores), branch divergence (`smsp__sass_branch_targets_threads_divergent`).

Correctness/safety tooling: `compute-sanitizer --tool memcheck|racecheck|synccheck|initcheck`. Run racecheck/synccheck whenever you touch barriers, mbarriers, async copies or warp specialization.

## 4. AMD workflow

ROCm 7 default profiler is **rocprofv3** (v1/v2 `rocprof` are legacy). `rocprof-compute` is the renamed Omniperf; `rocprof-sys` is the system tracer.

```
rocprofv3 --kernel-trace --stats --hip-trace -d out -- ./app        # timeline + per-kernel stats
rocprofv3 --pmc SQ_WAVES SQ_INSTS_VALU SQ_LDS_BANK_CONFLICT -d out -- ./app
rocprof-compute profile -n run1 -- ./app                            # full counter sweep (multi-pass)
rocprof-compute analyze -p workloads/run1/<arch>                    # roofline, LDS, L2, wavefront stats
```
Look for:
- VALU / MFMA utilization and **MFMA busy** vs issue stalls.
- LDS: bank conflict cycles (`SQ_LDS_BANK_CONFLICT`) relative to LDS access cycles; LDS allocation limiting occupancy.
- L2 hit rate and, on MI300/MI350, **per-XCD** L2 locality (workgroup->XCD round-robin hurts if neighbouring tiles share data).
- Scratch (spill) usage: any non-zero `scratch` in the kernel descriptor on a hot kernel is a bug to fix.
- Wave occupancy per SIMD vs VGPR/AGPR/LDS usage; instruction-cache misses on very large unrolled kernels (CDNA I-cache is shared between CUs: huge unrolled bodies can stall).
- Clock behaviour: MI-series can drop clocks under heavy MFMA; report `SCLK` alongside results.

Static view always available: `-Rpass-analysis=kernel-resource-usage` at compile time (VGPRs, AGPRs, SGPRs, scratch, LDS, occupancy in waves/SIMD).

## 5. Reading resource usage and ISA

NVIDIA:
```
nvcc -std=c++20 -O3 -arch=sm_90a -lineinfo -Xptxas -v -cubin k.cu -o k.cubin
cuobjdump -sass k.cubin | less        # or nvdisasm -g -c k.cubin
```
Check: registers/thread, `spill stores/loads` (want 0), `smem`, `cmem`, and in SASS: `LDG.E.128` (vectorized), `LDGSTS` (cp.async), `UTMALDG` (TMA), `HGMMA` (wgmma), `UTCHMMA/UTCQMMA` and friends (tcgen05), `LDSM` (ldmatrix), `BAR.SYNC` count in the loop, `STL/LDL` (local memory), long `MUFU` chains.

AMD:
```
hipcc -std=c++23 -O3 --offload-arch=gfx942 --offload-device-only -S k.hip -o k.s
hipcc -std=c++23 -O3 --offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage -c k.hip -o k.o
```
Check in the `.s`: `global_load_dwordx4` / `buffer_load_dwordx4` (vectorized), `ds_read_b128`/`ds_write_b128`, `v_mfma_*` density inside the main loop, `s_waitcnt` placement (an over-conservative `s_waitcnt vmcnt(0)` in the loop serializes the pipeline), `scratch_*` ops (spills), `.vgpr_count`/`.agpr_count`/`.sgpr_spill_count` metadata.

## 6. Diagnosis table

| Symptom | Likely cause | First fix |
|---|---|---|
| DRAM% low, SM% low, few warps | Latency-bound, too little parallelism | More work per SM (split, smaller tiles), more bytes in flight, vector loads |
| DRAM% < 80 % on streaming op | Narrow loads, misalignment, 64-bit index overhead, tail waves | 16 B vectors, align, 32-bit indexing, grid sizing, persistent grid-stride |
| Tensor pipe < 70 % in GEMM | Feed starvation / sync stalls | Deeper pipeline, async copies, warp specialization, bigger tile, fix bank conflicts |
| High spills | Register pressure | `__launch_bounds__`, smaller tile / accumulate tile, `maxnreg`, restructure live ranges |
| Perf cliff at some shapes | Wave quantization / tile mismatch | Stream-K/split-K, tile-size selection heuristic, persistent scheduler |
| Gaps in nsys timeline | Launch/sync overhead | Graphs, PDL, fusion, avoid host syncs |
| Good single-kernel, poor end-to-end | Memory-bound neighbours | Fuse, reuse in L2/smem, remove layout transposes |
| MI300: poor L2 hit despite tiling | Workgroups of neighbouring tiles landed on different XCDs | XCD-aware workgroup-id remap |
| Float atomics very slow on AMD | CAS loop instead of HW atomic | `-munsafe-fp-atomics` / `unsafeAtomicAdd`, warp aggregation, restructure as reduction |

## 7. Numerical validation

- Reference: fp64 CPU or fp64 GPU path of the same math. Compare relative error normalised by `sqrt(K)*eps(dtype)` for dot-product-like ops rather than a fixed `1e-3`.
- Typical reasonable bounds: FP32 accumulate of FP16/BF16 products -> relative error ~1e-3 (bf16 inputs) / ~1e-4 (fp16); TF32 ~1e-3; FP8 requires scale-aware comparison (compare against reference computed from the *dequantized* operands, not the original fp32 data).
- Always test: random data with large dynamic range, all-equal, all-negative (softmax/max), -inf/NaN propagation policy, denormals (`-ftz`/`--use_fast_math` changes behaviour), and sizes not divisible by any tile.
- Fast-math is opt-in per op: `__expf`, `__fdividef`, `rsqrtf`, `ex2.approx` have error bounds you must state. Do not enable global `--use_fast_math` / `-ffast-math` in a library without a per-op accuracy audit.
- Determinism: atomics-based reductions are order-nondeterministic; offer a deterministic path (fixed-order tree, two-pass) when training reproducibility matters.

## 8. NovaNN protocol notes

- Benchmarks of ncore kernels use the repo harnesses
  (`assets/bench_harness_cuda.cuh` / `assets/bench_harness_hip.hpp`) and
  report against the floor computed in section 1, on the actual target
  device, not a datasheet.
- Correctness pins live in `ncore/tests/` (GTest): a tuning change that
  alters numeric behavior needs an updated pin, and intentional
  GTEST_SKIPs carry reason strings and are part of the spec; never widen
  or remove them without cause.
- Never run the GPU suites with `HIP_LAUNCH_BLOCKING=1`: it serializes
  every launch, hides the null-stream drop, and makes the gates
  meaningless (see the engine-stream rules in `SKILL.md`).
- Before profiling a NovaNN kernel, confirm the build succeeded (grep
  `build/logs/<preset>.log` for `error:`); a stale binary profiles the
  wrong code.
- Keep benchmark and scenario evidence in the session folder under
  `agents/memory/` (context, plan, evidence, state) so an auto-compact
  resumes from the repository.
