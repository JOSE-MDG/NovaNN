/**
 * @file Contiguous.c
 * @brief CPU implementation of the contiguous layout operation.
 *
 * @details
 * Implements @ref contiguous_cpu_impl, which materializes a
 * row-major (C-contiguous) copy of an arbitrary tensor layout.
 * Scalars pass through unchanged, one-dimensional tensors are
 * handled directly, and multi-dimensional tensors are collapsed
 * via @ref collapse() before copying. Collapsed two-dimensional
 * views copy through cache-blocked tiles (plain row copies when
 * the inner stride is dense); deeper views parallelize the outer
 * dimension with an incremental odometer over the rest, so no
 * element pays full coordinate arithmetic.
 */

#include <string.h>

#ifdef NOVA_OPENMP
#include <omp.h>
#endif

#include <ncore/core/status.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/native/cpu/layout/contiguous.h>
#include <ncore/tensor.h>
#include <ncore/threading/parallel.h>
#include <ncore/threading/threads.h>

/**
 * @brief Copy one storage unit of a statically known width.
 *
 * @details
 * Constant sizes lower to a single load/store pair while remaining
 * strictly conforming on unaligned addresses. Falls back to @c memcpy
 * for widths outside 1/2/4/8.
 */
static inline void copy_storage_unit(unsigned char *restrict dst,
                                     const unsigned char *restrict src,
                                     size_t item_size) {
  switch (item_size) {
  case 1:
    memcpy(dst, src, 1);
    break;
  case 2:
    memcpy(dst, src, 2);
    break;
  case 4:
    memcpy(dst, src, 4);
    break;
  case 8:
    memcpy(dst, src, 8);
    break;
  default:
    memcpy(dst, src, item_size);
    break;
  }
}

/**
 * @def CONTIGUOUS_COPY_SPAN
 * @brief Defines a strided run copier for one storage-unit width.
 *
 * @details
 * Baking the width into the loop keeps the width dispatch out of the
 * inner loop. Left inside, GCC lowers it to a compare chain and Clang
 * to a jump table, both paid once per element. Streaming misses
 * dominate transposed layouts, where each unit sits on its own cache
 * line, so both streams prefetch ahead while the current units copy.
 */
#define CONTIGUOUS_COPY_SPAN(width)                                            \
  static inline void copy_span_##width(unsigned char *restrict dst,            \
                                       const unsigned char *restrict src,      \
                                       size_t count, size_t src_stride) {      \
    for (size_t i = 0; i < count; ++i) {                                       \
      if (i + 16 < count) {                                                    \
        __builtin_prefetch(src + (16 * src_stride), 0, 3);                     \
        __builtin_prefetch(dst + (16 * (width)), 1, 3);                        \
      }                                                                        \
      memcpy(dst, src, (width));                                               \
      dst += (width);                                                          \
      src += src_stride;                                                       \
    }                                                                          \
  }

CONTIGUOUS_COPY_SPAN(1)
CONTIGUOUS_COPY_SPAN(2)
CONTIGUOUS_COPY_SPAN(4)
CONTIGUOUS_COPY_SPAN(8)

/**
 * @brief Copy a strided run of storage units.
 *
 * @details
 * Resolves the width once per run instead of once per unit. Widths
 * outside 1/2/4/8 fall back to a @c memcpy per unit.
 *
 * @param[out] dst         Destination run, packed with no gaps.
 * @param[in]  src         First source unit.
 * @param[in]  count       Number of units to copy.
 * @param[in]  src_stride  Byte distance between consecutive source units.
 * @param[in]  item_size   Bytes per storage unit.
 */
static inline void copy_storage_span(unsigned char *restrict dst,
                                     const unsigned char *restrict src,
                                     size_t count, size_t src_stride,
                                     size_t item_size) {
  switch (item_size) {
  case 1:
    copy_span_1(dst, src, count, src_stride);
    break;
  case 2:
    copy_span_2(dst, src, count, src_stride);
    break;
  case 4:
    copy_span_4(dst, src, count, src_stride);
    break;
  case 8:
    copy_span_8(dst, src, count, src_stride);
    break;
  default:
    for (size_t i = 0; i < count; ++i) {
      memcpy(dst, src, item_size);
      dst += item_size;
      src += src_stride;
    }
    break;
  }
}

/**
 * @brief Tile edge for the blocked two-dimensional copy.
 *
 * @details
 * 64 by 64 storage units keep both tile working sets inside L1 on
 * reference hardware while dividing every tested shape evenly
 * enough for static scheduling.
 */
#define CONTIGUOUS_TILE_DIM 64

/**
 * @brief Copy a one-dimensional tensor into a contiguous buffer.
 *
 * @details
 * Gathers the storage units one by one from the offset-adjusted base
 * and recomputes the strides of @p dst for the contiguous layout.
 * Each unit is copied whole, so packed pairs (FP4) keep both nibbles
 * in order.
 *
 * @param[in]  src  Source tensor.  Must have exactly one dimension.
 * @param[out] dst  Destination tensor that receives the contiguous
 *                  data.
 *
 * @return @ref novaSuccess on success, or @ref novaInvalidTensor if
 *         @p src is not one-dimensional.
 */
static inline novaStatus_t contiguous_1d_tensor(const Tensor *restrict src,
                                                Tensor *restrict dst) {
  novaStatus_t st;
  const size_t stride = src->strides[0];
  const size_t item_size = src->item_size;
  const unsigned char *sbase = src->data.data + src->offset;
  unsigned char *dbase = dst->data.data;

  if (stride == item_size) {
    memcpy(dbase, sbase, src->size * item_size);
    compute_tensor_strides_(dst, src->ndims, src->shape, src->item_size);
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }

#ifdef NOVA_OPENMP

  auto threads = get_num_threads_from(ParallelComputeGroup, &st);

  if (st.err != novaSuccess) {
    return st;
  }

  const bool is_dense = (stride == src->item_size);
  const LayoutWork work = make_layout_work(src, is_dense);
  const ParallelDecision decision = decide_layout(&work, threads);

#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
  for (size_t item = 0; item < src->size; ++item) {
    copy_storage_unit(dbase + (item * item_size), sbase + (item * stride),
                      item_size);
  }

#else
  for (size_t item = 0; item < src->size; ++item) {
    copy_storage_unit(dbase + (item * item_size), sbase + (item * stride),
                      item_size);
  }
#endif
  compute_tensor_strides_(dst, src->ndims, src->shape, src->item_size);
  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);
  return st;
}

/**
 * @brief Copy a collapsed one-dimensional view into a contiguous
 *        buffer.
 *
 * @details
 * If the collapsed stride equals the source item size, the extent
 * (@c src->size units from the offset base) is dense and copied
 * with @c memcpy.  Otherwise the units are gathered one by one
 * using the collapsed stride from the offset base.
 *
 * @param[in]  cv   Collapsed view describing the source layout.
 *                  Must have exactly one dimension.
 * @param[in]  src  Source tensor backing @p cv.
 * @param[out] dst  Destination tensor that receives the contiguous
 *                  data.
 *
 * @return @ref novaSuccess on success, or @ref novaInvalidNdims if
 *         @p cv is not one-dimensional.
 */
static inline novaStatus_t contiguous_1d_cv(size_t stride,
                                            const Tensor *restrict src,
                                            Tensor *restrict dst) {
  novaStatus_t st;
  // Hoisted so the width and the base addresses stay loop-invariant.
  // Read through the tensor on every iteration, the byte stores keep
  // GCC from proving that, and the width switch never leaves the loop.
  const size_t item_size = src->item_size;
  const unsigned char *sbase = src->data.data + src->offset;
  unsigned char *dbase = dst->data.data;
  if (stride != src->item_size) {

#ifdef NOVA_OPENMP

    auto threads = get_num_threads_from(ParallelComputeGroup, &st);

    if (st.err != novaSuccess) {
      return st;
    }

    const LayoutWork work = make_layout_work(src, false);
    const ParallelDecision decision = decide_layout(&work, threads);

#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
    for (size_t item = 0; item < src->size; ++item) {
      copy_storage_unit(dbase + (item * item_size), sbase + (item * stride),
                        item_size);
    }

#else
    for (size_t item = 0; item < src->size; ++item) {
      copy_storage_unit(dbase + (item * item_size), sbase + (item * stride),
                        item_size);
    }
#endif
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }
  memcpy(dbase, sbase, src->size * item_size);
  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);
  return st;
}

/**
 * @brief Materialize a contiguous copy of @p src into @p dst.
 *
 * @details
 * Entry point of the CPU contiguous backend.  Scalars pass through
 * unchanged.  One-dimensional tensors are handled by
 * @ref contiguous_1d_tensor.  Other tensors are
 * collapsed with @ref collapse() and copied unit by unit,
 * following the collapsed strides (per-thread coordinates under
 * @c NOVA_OPENMP, an odometer walk otherwise).
 *
 * @param[in]  src  Source tensor with arbitrary layout.
 * @param[out] dst  Destination tensor that receives the contiguous
 *                  data.
 *
 * @return @ref novaSuccess on success, or the error status
 *         propagated from an internal helper.
 */
novaStatus_t contiguous_cpu_impl(const Tensor *restrict src,
                                 Tensor *restrict dst) {
  novaStatus_t st;

  if (src->ndims == 1) {
    return contiguous_1d_tensor(src, dst);
  }

  // Collapse contiguous dimensions if possible
  const CollapsedView cv = collapse(src);

  if (cv.ndims == 1) {
    return contiguous_1d_cv(cv.strides[0], src, dst);
  }

  const unsigned char *sbase = src->data.data + src->offset;
  unsigned char *dbase = dst->data.data;
  const size_t item_size = src->item_size;

#ifdef NOVA_OPENMP

  auto threads = get_num_threads_from(ParallelComputeGroup, &st);

  if (st.err != novaSuccess) {
    return st;
  }

  const LayoutWork work = make_layout_work(src, false);
  const ParallelDecision decision = decide_layout(&work, threads);

  if (cv.ndims == 2) {
    const size_t d0 = cv.shape[0];
    const size_t d1 = cv.shape[1];
    const size_t s0 = cv.strides[0];
    const size_t s1 = cv.strides[1];
    if (s1 == item_size) {
      // Inner-dense rows: one plain copy per row.
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
      for (size_t row = 0; row < d0; ++row) {
        memcpy(dbase + (row * d1 * item_size), sbase + (row * s0),
               d1 * item_size);
      }
    } else {
      // Cache-blocked tiles bound the strided working set; row
      // bases stay hoisted out of the inner copy loop. Parallelism
      // spans tile rows only, so each thread owns whole destination
      // rows and no cache line bounces between threads.
      const size_t n_tiles0 =
          (d0 + CONTIGUOUS_TILE_DIM - 1) / CONTIGUOUS_TILE_DIM;
      const size_t n_tiles1 =
          (d1 + CONTIGUOUS_TILE_DIM - 1) / CONTIGUOUS_TILE_DIM;
#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
      for (size_t tr = 0; tr < n_tiles0; ++tr) {
        for (size_t tc = 0; tc < n_tiles1; ++tc) {
          const size_t r0 = tr * CONTIGUOUS_TILE_DIM;
          const size_t c0 = tc * CONTIGUOUS_TILE_DIM;
          const size_t r_end =
              r0 + CONTIGUOUS_TILE_DIM < d0 ? r0 + CONTIGUOUS_TILE_DIM : d0;
          const size_t c_end =
              c0 + CONTIGUOUS_TILE_DIM < d1 ? c0 + CONTIGUOUS_TILE_DIM : d1;
          for (size_t row = r0; row < r_end; ++row) {
            const unsigned char *sp = sbase + (row * s0) + (c0 * s1);
            unsigned char *dp = dbase + (((row * d1) + c0) * item_size);
            copy_storage_span(dp, sp, c_end - c0, s1, item_size);
          }
        }
      }
    }
  } else {
    // Outer-parallel walk: each thread owns whole outer rows and
    // advances an incremental odometer over the rest, so no
    // element pays full coordinate arithmetic.
    const size_t nd = cv.ndims;
    const size_t outer = cv.shape[0];
    const size_t inner_total = src->size / outer;
    const size_t d_last = cv.shape[nd - 1];
    const size_t s_last = cv.strides[nd - 1];
    const bool inner_dense = (s_last == item_size);
    const size_t mid_count = inner_total / d_last;
    size_t dst_strides[NOVA_MAX_DIMS] = {0};
    dst_strides[nd - 1] = item_size;
    for (size_t dim = nd - 1; dim-- > 0;) {
      dst_strides[dim] = dst_strides[dim + 1] * cv.shape[dim + 1];
    }

#pragma omp parallel for num_threads(decision.num_threads)                     \
    schedule(static) if (decision.go)
    for (size_t row = 0; row < outer; ++row) {
      const unsigned char *srow = sbase + (row * cv.strides[0]);
      unsigned char *drow = dbase + (row * dst_strides[0]);
      coords_t mid = {0};
      size_t s_mid = 0;
      size_t d_mid = 0;
      for (size_t m = 0; m < mid_count; ++m) {
        if (inner_dense) {
          memcpy(drow + d_mid, srow + s_mid, d_last * item_size);
        } else {
          copy_storage_span(drow + d_mid, srow + s_mid, d_last, s_last,
                            item_size);
        }
        for (size_t dim = nd - 2; dim >= 1; --dim) {
          mid[dim]++;
          s_mid += cv.strides[dim];
          d_mid += dst_strides[dim];
          if (mid[dim] < cv.shape[dim]) {
            break;
          }
          mid[dim] = 0;
          s_mid -= cv.shape[dim] * cv.strides[dim];
          d_mid -= cv.shape[dim] * dst_strides[dim];
          if (dim == 1) {
            break;
          }
        }
      }
    }
  }

#else
  coords_t coords = {0};
  for (size_t item = 0; item < src->size; ++item) {

    size_t offset = src->offset;

    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    copy_storage_unit(dbase + (item * item_size), sbase, item_size);
    odometer(coords, cv.ndims, cv.shape);
  }
#endif

  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);

  return st;
}
