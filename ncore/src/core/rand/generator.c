/**
 * @file generator.c
 * @brief Generator handle lifecycle, bulk fills, and process defaults.
 *
 * @details
 * Locking follows one rule: a single lock covers a whole fill or init,
 * never a single sample. User handles carry no lock and are not
 * thread-safe when shared, the two process defaults serialize through
 * their own static mutexes. Device fills run in device memory through
 * the native bridge. The host only derives the base block and advances
 * the offset on success.
 */

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/mtx.h>
#include <ncore/native/kernels/random/rng.h>
#include <ncore/rand/generator.h>
#include <ncore/rand/philox.h>
#include <ncore/rand/xoshiro256.h>
#include <ncore/threading/threads.h>

#ifdef NOVA_OPENMP
#include <ncore/threading/parallel.h>
#endif

#define RAND_PHILOX_LANES 4
#define RAND_U32_PER_BLOCK 4
#define RAND_U64_PER_BLOCK 2
#define RAND_LANE_BITS 32
#define RAND_XOSHIRO256_SNAPSHOT 5
#define RAND_PHILOX_SNAPSHOT 7

#ifdef __linux__
static once_flag defaults_once = {};
#elifdef _WIN64
static INIT_ONCE defaults_once = INIT_ONCE_STATIC_INIT;
#endif

static MTX_T cpu_default_mtx;
static MTX_T gpu_default_mtx;
static Generator_ cpu_default = {};
static Generator_ gpu_default = {};

/**
 * @brief Initialize both process defaults to the zero seed. Runs once
 * via the platform one-shot guard.
 */
#ifdef __linux__
static void init_defaults() {
  (void)mtx_init(&cpu_default_mtx, mtx_plain);
  (void)mtx_init(&gpu_default_mtx, mtx_plain);
  (void)generator_init(&cpu_default, Xoshiro256, DEVICE_CPU, 0, 0);
  (void)generator_init(&gpu_default, Philox4x32, DEVICE_GPU, 0, 0);
}
#elifdef _WIN64
static BOOL CALLBACK init_defaults(PINIT_ONCE once, PVOID param, PVOID *ctx) {
  (void)once;
  (void)param;
  (void)ctx;
  InitializeCriticalSection(&cpu_default_mtx);
  InitializeCriticalSection(&gpu_default_mtx);
  (void)generator_init(&cpu_default, Xoshiro256, DEVICE_CPU, 0, 0);
  (void)generator_init(&gpu_default, Philox4x32, DEVICE_GPU, 0, 0);
  return TRUE;
}
#endif

/** Runs init_defaults once per platform. */
static void ensure_defaults() {
#ifdef __linux__
  call_once(&defaults_once, init_defaults);
#elifdef _WIN64
  InitOnceExecuteOnce(&defaults_once, init_defaults, nullptr, nullptr);
#endif
}

/// Lanes a single fill may park in the handle cache. Philox leaves at
/// most three mask lanes of its last block; Xoshiro leaves at most one
/// sample half.

/**
 * @brief Parks @p count lanes starting at @p first of a fresh block.
 *
 * @pre The cache has room: parked lanes never exceed
 *      @c RAND_LANE_CACHE_CAP.
 */
static inline void lane_cache_put(Generator_ *gen, const uint32_t lanes[4],
                                  uint32_t first, uint32_t count) {
  for (uint32_t lane = first; lane < first + count; ++lane) {
    gen->lane_cache[gen->lane_count++] = lanes[lane];
  }
}

/// Drops @p count already emitted lanes from the front of the cache.
static inline void lane_cache_drop(Generator_ *gen, uint32_t count) {
  gen->lane_count -= count;
  for (uint32_t lane = 0; lane < gen->lane_count; ++lane) {
    gen->lane_cache[lane] = gen->lane_cache[count + lane];
  }
}

/**
 * @brief Reports success with the module standard message.
 */
static inline novaStatus_t ok() {
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}

/**
 * @brief Shared fill preflight: handle, destination, demand, placement.
 *
 * @details
 * Returns @c false with @p st set whenever the fill must stop without
 * running: a null handle, a null destination under a live demand (the
 * message text belongs to the caller, which knows its value width),
 * an empty demand (success travels in @p st so callers forward it and
 * skip state validation, matching the documented zero-demand
 * behavior), or a placement without a fill path. Passing checks leave
 * @p st untouched.
 *
 * @param[in]  gen      Handle; must not be nullptr for a live demand.
 * @param[in]  out      Destination; must not be nullptr when n > 0.
 * @param[in]  n        Demand values.
 * @param[in]  no_room  Message reported for a null destination.
 * @param[out] st       Status forwarded when returning @c false.
 */
static inline bool fill_demand_ok(Generator_ *gen, const void *out, size_t n,
                                  const char *no_room, novaStatus_t *st) {
  if (gen == nullptr) {
    *st = (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Fills need a live generator. Got null.\n",
    };
    return false;
  }
  if (out == nullptr && n > 0) {
    *st = (novaStatus_t){
        .err = novaInvalidPointer,
        .message = no_room,
    };
    return false;
  }
  if (n == 0) {
    *st = ok();
    return false;
  }
  if (gen->device != DEVICE_CPU && gen->device != DEVICE_GPU) {
    *st = (novaStatus_t){
        .err = novaInvalidDevice,
        .message = "Unknown placement for a fill. Fills use DEVICE_CPU or "
                   "device memory.\n",
    };
    return false;
  }
  return true;
}

novaStatus_t generator_init(Generator_ *gen, RngEngine engine, Device_ device,
                            uint64_t seed, uint64_t subsequence) {
  if (gen == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Initialization needs a live Generator_. Got null.\n",
    };
  }
  if (engine != Philox4x32 && engine != Xoshiro256) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Unknown engine. Use Philox4x32 or Xoshiro256.\n",
    };
  }
  if (engine == Xoshiro256 && device == DEVICE_GPU) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Xoshiro owns no device fill path. Philox4x32 backs "
                   "GPU streams.\n",
    };
  }
  if (device != DEVICE_CPU && device != DEVICE_GPU) {
    return (novaStatus_t){
        .err = novaInvalidDevice,
        .message = "Unknown placement for a generator handle. Use "
                   "DEVICE_CPU or DEVICE_GPU.\n",
    };
  }

  Generator_ tmp = {};
  tmp.seed = seed;
  tmp.subsequence = subsequence;
  tmp.device = device;
  tmp.engine = engine;
  if (engine == Xoshiro256) {
    auto init = xoshiro256_init(&tmp.state.xoshiro, seed);
    if (init.err != novaSuccess) {
      return init;
    }
    for (uint64_t w = 0; w < subsequence; ++w) {
      xoshiro256_jump(&tmp.state.xoshiro);
    }
  } else {
    auto init = philox_init(&tmp.state.philox, seed, subsequence);
    if (init.err != novaSuccess) {
      return init;
    }
  }
  *gen = tmp;
  return ok();
}

novaStatus_t generator_fill_u64(Generator_ *gen, uint64_t *out, size_t n) {
  novaStatus_t st = {};
  if (!fill_demand_ok(
          gen, out, n,
          "Nowhere to write 64-bit values. Provide storage for n values.\n",
          &st)) {
    return st;
  }
  if (gen->device == DEVICE_GPU) {
    if (gen->engine != Philox4x32) {
      return (novaStatus_t){
          .err = novaInvalidValue,
          .message = "Xoshiro cannot run on device. Use Philox4x32 for GPU "
                     "streams.\n",
      };
    }
    size_t fresh = n;
    bool pair_pending = false;
    uint64_t pair = 0;
    if (gen->lane_count >= 2U) {
      pair_pending = true;
      pair =
          (uint64_t)gen->lane_cache[0] | ((uint64_t)gen->lane_cache[1] << 32);
      fresh = n - 1U;
    }
    if (fresh > 0U) {
      const uint64_t blocks =
          (fresh + (RAND_U64_PER_BLOCK - 1U)) / RAND_U64_PER_BLOCK;
      uint32_t last_ctr[4] = {};
      philox_ctr_add(gen->state.philox.ctr, blocks - 1U, last_ctr);
      uint32_t last_block[4] = {};
      philox4x32_10(last_ctr, gen->state.philox.key, last_block);
      const uint32_t leftover = (uint32_t)((4U * blocks) - (2U * fresh));

      if (pair_pending) {
        auto staged =
            transfer_to(DEVICE_CPU, DEVICE_GPU, &pair, out, sizeof(uint64_t));
        if (staged.err != novaSuccess) {
          return staged;
        }
      }
      auto launched =
          launchRngFillU64(gen->state.philox.key, gen->state.philox.ctr,
                           out + (n - fresh), fresh);
      if (launched.err != novaSuccess) {
        return launched;
      }
      if (pair_pending) {
        lane_cache_drop(gen, 2U);
      }
      lane_cache_put(gen, last_block, 2U, leftover);
      uint32_t next[4] = {};
      philox_ctr_add(gen->state.philox.ctr, blocks, next);
      for (size_t c = 0; c < 4; ++c) {
        gen->state.philox.ctr[c] = next[c];
      }
    } else {
      auto staged =
          transfer_to(DEVICE_CPU, DEVICE_GPU, &pair, out, sizeof(uint64_t));
      if (staged.err != novaSuccess) {
        return staged;
      }
      lane_cache_drop(gen, 2U);
    }
    gen->offset += n;
    return ok();
  }
  if (gen->engine == Xoshiro256) {
    for (size_t i = 0; i < n; ++i) {
      out[i] = xoshiro256_next(&gen->state.xoshiro);
    }
    gen->offset += n;
    return ok();
  }
  {
    size_t i = 0;
    while (gen->lane_count >= 2U && i < n) {
      out[i++] =
          (uint64_t)gen->lane_cache[0] | ((uint64_t)gen->lane_cache[1] << 32);
      lane_cache_drop(gen, 2U);
    }
    const uint32_t key[2] = {gen->state.philox.key[0],
                             gen->state.philox.key[1]};
    const uint32_t ctr0[4] = {
        gen->state.philox.ctr[0], gen->state.philox.ctr[1],
        gen->state.philox.ctr[2], gen->state.philox.ctr[3]};
    const size_t fresh = n - i;
    const size_t full = fresh / RAND_U64_PER_BLOCK;
    const size_t tail = fresh % RAND_U64_PER_BLOCK;
#ifdef NOVA_OPENMP
    const uint32 threads = get_num_threads_from(ParallelComputeGroup, &st);
    if (st.err != novaSuccess) {
      return st;
    }
    const RngWork work = {
        .lanes = fresh,
        .cost_per_elem = 4U,
        .valid = true,
    };
    const ParallelDecision decision = decide_rng(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
    for (size_t b = 0; b < full; ++b) {
      uint32_t local[4] = {};
      philox_ctr_add(ctr0, b, local);
      uint32_t block[4] = {};
      philox4x32_10(local, key, block);
      uint64_t *const block_out = out + i + (b * RAND_U64_PER_BLOCK);
      block_out[0] = (uint64_t)block[0] | ((uint64_t)block[1] << 32);
      block_out[1] = (uint64_t)block[2] | ((uint64_t)block[3] << 32);
    }
    i += full * RAND_U64_PER_BLOCK;
    if (tail > 0U) {
      uint32_t local[4] = {};
      philox_ctr_add(ctr0, full, local);
      uint32_t block[4] = {};
      philox4x32_10(local, key, block);
      out[i++] = (uint64_t)block[0] | ((uint64_t)block[1] << 32);
      lane_cache_put(gen, block, 2U, 2U);
    }
    philox_ctr_add(ctr0, full + (tail > 0U ? 1U : 0U), gen->state.philox.ctr);
  }
  gen->offset += n;
  return ok();
}

novaStatus_t generator_fill_u32(Generator_ *gen, uint32_t *out, size_t n) {
  novaStatus_t st = {};
  if (!fill_demand_ok(
          gen, out, n,
          "Nowhere to write 32-bit values. Provide storage for n values.\n",
          &st)) {
    return st;
  }
  if (gen->device == DEVICE_GPU) {
    if (gen->engine != Philox4x32) {
      return (novaStatus_t){
          .err = novaInvalidValue,
          .message = "Xoshiro cannot run on device. Use Philox4x32 for GPU "
                     "streams.\n",
      };
    }
    const size_t take = n < gen->lane_count ? n : (size_t)gen->lane_count;
    const size_t fresh = n - take;
    if (fresh > 0U) {
      const uint64_t blocks =
          (fresh + (RAND_U32_PER_BLOCK - 1U)) / RAND_U32_PER_BLOCK;
      uint32_t last_ctr[4] = {};
      philox_ctr_add(gen->state.philox.ctr, blocks - 1U, last_ctr);
      uint32_t last_block[4] = {};
      philox4x32_10(last_ctr, gen->state.philox.key, last_block);
      const uint32_t leftover = (uint32_t)((4U * blocks) - fresh);

      if (take > 0U) {
        auto staged = transfer_to(DEVICE_CPU, DEVICE_GPU, gen->lane_cache, out,
                                  take * sizeof(uint32_t));
        if (staged.err != novaSuccess) {
          return staged;
        }
      }
      auto launched = launchRngFillU32(
          gen->state.philox.key, gen->state.philox.ctr, out + take, fresh);
      if (launched.err != novaSuccess) {
        return launched;
      }
      if (take > 0U) {
        lane_cache_drop(gen, (uint32_t)take);
      }
      if (leftover > 0U) {
        lane_cache_put(gen, last_block, (uint32_t)(fresh % RAND_U32_PER_BLOCK),
                       leftover);
      }
      uint32_t next[4] = {};
      philox_ctr_add(gen->state.philox.ctr, blocks, next);
      for (size_t c = 0; c < 4; ++c) {
        gen->state.philox.ctr[c] = next[c];
      }
    } else {
      auto staged = transfer_to(DEVICE_CPU, DEVICE_GPU, gen->lane_cache, out,
                                take * sizeof(uint32_t));
      if (staged.err != novaSuccess) {
        return staged;
      }
      lane_cache_drop(gen, (uint32_t)take);
    }
    gen->offset += n;
    return ok();
  }
  if (gen->engine == Xoshiro256) {
    size_t i = 0;
    if (gen->lane_count > 0U) {
      out[i++] = gen->lane_cache[0];
      lane_cache_drop(gen, 1U);
    }
    while (i < n) {
      const uint64_t sample = xoshiro256_next(&gen->state.xoshiro);
      out[i++] = (uint32_t)(sample >> RAND_LANE_BITS);
      if (i < n) {
        out[i++] = (uint32_t)sample;
      } else {
        gen->lane_cache[gen->lane_count++] = (uint32_t)sample;
      }
    }
    gen->offset += n;
    return ok();
  }
  {
    size_t i = 0;
    while (gen->lane_count > 0U && i < n) {
      out[i++] = gen->lane_cache[0];
      lane_cache_drop(gen, 1U);
    }
    const uint32_t key[2] = {gen->state.philox.key[0],
                             gen->state.philox.key[1]};
    const uint32_t ctr0[4] = {
        gen->state.philox.ctr[0], gen->state.philox.ctr[1],
        gen->state.philox.ctr[2], gen->state.philox.ctr[3]};
    const size_t fresh = n - i;
    const size_t full = fresh / RAND_U32_PER_BLOCK;
    const size_t tail = fresh % RAND_U32_PER_BLOCK;
#ifdef NOVA_OPENMP
    const uint32 threads = get_num_threads_from(ParallelComputeGroup, &st);
    if (st.err != novaSuccess) {
      return st;
    }
    const RngWork work = {
        .lanes = fresh,
        .cost_per_elem = 2U,
        .valid = true,
    };
    const ParallelDecision decision = decide_rng(&work, threads);
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
#endif
    for (size_t b = 0; b < full; ++b) {
      uint32_t local[4] = {};
      philox_ctr_add(ctr0, b, local);
      uint32_t block[4] = {};
      philox4x32_10(local, key, block);
      uint32_t *const block_out = out + i + (b * RAND_U32_PER_BLOCK);
      for (size_t l = 0; l < RAND_U32_PER_BLOCK; ++l) {
        block_out[l] = block[l];
      }
    }
    i += full * RAND_U32_PER_BLOCK;
    if (tail > 0U) {
      uint32_t local[4] = {};
      philox_ctr_add(ctr0, full, local);
      uint32_t block[4] = {};
      philox4x32_10(local, key, block);
      for (size_t l = 0; l < tail; ++l) {
        out[i++] = block[l];
      }
      lane_cache_put(gen, block, (uint32_t)tail,
                     (uint32_t)(RAND_PHILOX_LANES - tail));
    }
    philox_ctr_add(ctr0, full + (tail > 0U ? 1U : 0U), gen->state.philox.ctr);
  }
  gen->offset += n;
  return ok();
}

novaStatus_t generator_get_state(const Generator_ *gen, uint64_t *snapshot,
                                 size_t *n_snapshot) {
  if (gen == nullptr || snapshot == nullptr || n_snapshot == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Snapshotting needs a live handle, snapshot storage and a "
                   "count slot. Got null.\n",
    };
  }
  const size_t need = gen->engine == Xoshiro256 ? RAND_XOSHIRO256_SNAPSHOT
                                                : RAND_PHILOX_SNAPSHOT;
  if (*n_snapshot < need) {
    *n_snapshot = need;
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Snapshot storage too small for this engine. Pass a "
                   "buffer of the reported count.\n",
    };
  }
  if (gen->engine == Xoshiro256) {
    auto copy = xoshiro256_get_state(&gen->state.xoshiro, snapshot);
    if (copy.err != novaSuccess) {
      return copy;
    }
    snapshot[RAND_XOSHIRO256_SNAPSHOT - 1U] = gen->offset;
    *n_snapshot = RAND_XOSHIRO256_SNAPSHOT;
  } else {
    uint32_t ctr[4] = {};
    uint32_t key[2] = {};
    auto copy = philox_get_state(&gen->state.philox, ctr, key);
    if (copy.err != novaSuccess) {
      return copy;
    }
    snapshot[0] = ctr[0];
    snapshot[1] = ctr[1];
    snapshot[2] = ctr[2];
    snapshot[3] = ctr[3];
    snapshot[4] = key[0];
    snapshot[5] = key[1];
    snapshot[6] = gen->offset;
    *n_snapshot = RAND_PHILOX_SNAPSHOT;
  }
  return ok();
}

novaStatus_t generator_set_state(Generator_ *gen, const uint64_t *snapshot,
                                 size_t n_snapshot) {
  if (gen == nullptr || snapshot == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Restoring needs a live handle with a serialized state. "
                   "Got null.\n",
    };
  }
  if (gen->engine == Xoshiro256) {
    if (n_snapshot != RAND_XOSHIRO256_SNAPSHOT) {
      return (novaStatus_t){
          .err = novaInvalidValue,
          .message = "Wrong snapshot size for Xoshiro256. Pass the 5 entries "
                     "holding state then offset.\n",
      };
    }
    auto restored = xoshiro256_set_state(&gen->state.xoshiro, snapshot);
    if (restored.err != novaSuccess) {
      return restored;
    }
    gen->lane_count = 0U;
    gen->offset = snapshot[4];
    return ok();
  }
  if (n_snapshot != RAND_PHILOX_SNAPSHOT) {
    return (novaStatus_t){
        .err = novaInvalidValue,
        .message = "Wrong snapshot size for Philox4x32. Pass the 7 entries "
                   "holding counter, key, then offset.\n",
    };
  }
  for (size_t entry = 0; entry < 6U; ++entry) {
    if (snapshot[entry] > 0xFFFFFFFFULL) {
      return (novaStatus_t){
          .err = novaInvalidValue,
          .message = "A counter or key entry lies outside the 32-bit range "
                     "Philox carries. Use values up to 0xFFFFFFFF.\n",
      };
    }
  }
  const uint32_t ctr[4] = {
      (uint32_t)snapshot[0],
      (uint32_t)snapshot[1],
      (uint32_t)snapshot[2],
      (uint32_t)snapshot[3],
  };
  const uint32_t key[2] = {(uint32_t)snapshot[4], (uint32_t)snapshot[5]};
  auto restored = philox_set_state(&gen->state.philox, ctr, key);
  if (restored.err != novaSuccess) {
    return restored;
  }
  gen->lane_count = 0U;
  gen->offset = snapshot[6];
  return ok();
}

novaStatus_t generator_clone(const Generator_ *src, Generator_ *dst) {
  if (src == nullptr || dst == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Cloning needs two live handles; got null, pass valid "
                   "source and destination.\n",
    };
  }
  dst->seed = src->seed;
  dst->subsequence = src->subsequence;
  dst->offset = src->offset;
  dst->device = src->device;
  dst->engine = src->engine;
  if (src->engine == Xoshiro256) {
    dst->state.xoshiro = src->state.xoshiro;
  } else {
    dst->state.philox = src->state.philox;
  }
  dst->lane_count = src->lane_count;
  for (size_t lane = 0; lane < RAND_LANE_CACHE_CAP; ++lane) {
    dst->lane_cache[lane] = src->lane_cache[lane];
  }
  return ok();
}

novaStatus_t default_generator(Device_ device, Generator_ *gen) {
  if (gen == nullptr) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Copying the default needs a live handle. Got null.\n",
    };
  }
  if (device != DEVICE_CPU && device != DEVICE_GPU) {
    return (novaStatus_t){
        .err = novaInvalidDevice,
        .message = "No default generator exists for this placement. Use "
                   "DEVICE_CPU or DEVICE_GPU.\n",
    };
  }
  ensure_defaults();
  if (device == DEVICE_GPU) {
    MTX_LOCK(&gpu_default_mtx);
    *gen = gpu_default;
    MTX_UNLOCK(&gpu_default_mtx);
  } else {
    MTX_LOCK(&cpu_default_mtx);
    *gen = cpu_default;
    MTX_UNLOCK(&cpu_default_mtx);
  }
  return ok();
}

novaStatus_t manual_seed(uint64_t seed) {
  ensure_defaults();
  MTX_LOCK(&cpu_default_mtx);
  MTX_LOCK(&gpu_default_mtx);
  auto cpu = generator_init(&cpu_default, Xoshiro256, DEVICE_CPU, seed, 0);
  auto gpu = generator_init(&gpu_default, Philox4x32, DEVICE_GPU, seed, 0);
  MTX_UNLOCK(&gpu_default_mtx);
  MTX_UNLOCK(&cpu_default_mtx);
  if (cpu.err != novaSuccess) {
    return cpu;
  }
  return gpu;
}

/**
 * @brief Validates placement and demand shared by the default fills.
 *
 * @param[in] device Default to draw from.
 * @param[in] out    Destination values.
 * @param[in] n      Values to draw.
 *
 * @return @c novaSuccess when the call may proceed; otherwise the
 *         rejection status for a null destination with a live demand
 *         or for a placement without a default generator.
 */
static inline novaStatus_t default_fill_check(Device_ device, const void *out,
                                              size_t n) {
  if (out == nullptr && n > 0) {
    return (novaStatus_t){
        .err = novaInvalidPointer,
        .message = "Default draws need somewhere to land. Provide storage "
                   "for n values.\n",
    };
  }
  if (device != DEVICE_CPU && device != DEVICE_GPU) {
    return (novaStatus_t){
        .err = novaInvalidDevice,
        .message = "No default generator exists for this placement. Use "
                   "DEVICE_CPU or DEVICE_GPU.\n",
    };
  }
  return ok();
}

novaStatus_t default_fill_u32(Device_ device, uint32_t *out, size_t n) {
  auto checked = default_fill_check(device, out, n);
  if (checked.err != novaSuccess) {
    return checked;
  }
  ensure_defaults();
  if (device == DEVICE_GPU) {
    MTX_LOCK(&gpu_default_mtx);
    auto filled = generator_fill_u32(&gpu_default, out, n);
    MTX_UNLOCK(&gpu_default_mtx);
    return filled;
  }
  MTX_LOCK(&cpu_default_mtx);
  auto filled = generator_fill_u32(&cpu_default, out, n);
  MTX_UNLOCK(&cpu_default_mtx);
  return filled;
}

novaStatus_t default_fill_u64(Device_ device, uint64_t *out, size_t n) {
  auto checked = default_fill_check(device, out, n);
  if (checked.err != novaSuccess) {
    return checked;
  }
  ensure_defaults();
  if (device == DEVICE_GPU) {
    MTX_LOCK(&gpu_default_mtx);
    auto filled = generator_fill_u64(&gpu_default, out, n);
    MTX_UNLOCK(&gpu_default_mtx);
    return filled;
  }
  MTX_LOCK(&cpu_default_mtx);
  auto filled = generator_fill_u64(&cpu_default, out, n);
  MTX_UNLOCK(&cpu_default_mtx);
  return filled;
}
