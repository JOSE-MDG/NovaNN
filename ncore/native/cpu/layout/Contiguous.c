/**
 * @file Contiguous.c
 * @brief CPU implementation of the contiguous layout operation.
 *
 * @details
 * Implements @ref contiguous_cpu_impl, which materializes a
 * row-major (C-contiguous) copy of an arbitrary tensor layout.
 * Scalars pass through unchanged, one-dimensional tensors are
 * handled directly, and multi-dimensional tensors are collapsed
 * via @ref collapse() before being copied element by element.
 */

#include <string.h>

#ifdef NOVA_OPENMP
#include <omp.h>
#endif

#include <ncore/core/status.h>
#include <ncore/headeronly/tensor_utils.h>
#include <ncore/native/cpu/layout/contiguous.h>
#include <ncore/tensor.h>
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
static inline novaStatus_t
contiguous_one_dimensional_tensor(const Tensor *restrict src,
                                  Tensor *restrict dst) {
  novaStatus_t st;

  if (src->ndims != 1) {
    st.err = novaInvalidTensor;
    st.message = "Cannot perform contiguous operation with a non "
                 "one-dimensional tensor";
    return st;
  }
  const size_t stride = src->strides[0];
  const size_t item_size = src->item_size;
  const unsigned char *sbase = src->data.data + src->offset;
  unsigned char *dbase = dst->data.data;

#ifdef NOVA_OPENMP

  auto threads = get_num_threads_from(ParallelComputeGroup, &st);

  if (st.err != novaSuccess) {
    return st;
  }

  const bool parallelize = is_parallelizable(
      src, threads, ParallelizableByElements | ParallelizableByBytes);

#pragma omp parallel for num_threads(threads) schedule(static) if (parallelize)
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
static inline novaStatus_t contiguous_one_dimensional_cv(
    const CollapsedView *cv, const Tensor *restrict src, Tensor *restrict dst) {

  novaStatus_t st;
  if (cv->ndims != 1) {
    st.err = novaInvalidNdims;
    st.message = "Cannot perform contiguous operation with a non "
                 "one-dimensional collapsed view";
    return st;
  }

  size_t stride = cv->strides[0];

  if (stride != src->item_size) {

#ifdef NOVA_OPENMP

    auto threads = get_num_threads_from(ParallelComputeGroup, &st);

    if (st.err != novaSuccess) {
      return st;
    }

    const bool parallelize = is_parallelizable(
        src, threads, ParallelizableByElements | ParallelizableByBytes);

#pragma omp parallel for num_threads(threads) schedule(static) if (parallelize)
    for (size_t item = 0; item < src->size; ++item) {
      copy_storage_unit(dst->data.data + (item * src->item_size),
                        src->data.data + src->offset + (item * stride),
                        src->item_size);
    }

#else
    for (size_t item = 0; item < src->size; ++item) {
      copy_storage_unit(dst->data.data + (item * src->item_size),
                        src->data.data + src->offset + (item * stride),
                        src->item_size);
    }
#endif
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }
  memcpy(dst->data.v, src->data.data + src->offset, src->size * src->item_size);
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
 * @ref contiguous_one_dimensional_tensor.  Other tensors are
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

  if (is_scalar(src)) {
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }

  if (src->ndims == 1) {
    return contiguous_one_dimensional_tensor(src, dst);
  }

  // Collapse contiguous dimensions if possible
  const CollapsedView cv = collapse(src);

  if (cv.ndims == 1) {
    return contiguous_one_dimensional_cv(&cv, src, dst);
  }

#ifdef NOVA_OPENMP

  auto threads = get_num_threads_from(ParallelComputeGroup, &st);

  if (st.err != novaSuccess) {
    return st;
  }

  const bool parallelize = is_parallelizable(
      src, threads, ParallelizableByElements | ParallelizableByBytes);

#pragma omp parallel for num_threads(threads) schedule(static) if (parallelize)
  for (size_t item = 0; item < src->size; ++item) {
    size_t offset = src->offset;
    coords_t coords = {0};
    compute_coords_from_linear_index_(item, cv.ndims, cv.shape, coords);

    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    copy_storage_unit(dst->data.data + (item * src->item_size),
                      src->data.data + offset, src->item_size);
  }

#else
  coords_t coords = {0};
  for (size_t item = 0; item < src->size; ++item) {

    size_t offset = src->offset;

    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    copy_storage_unit(dst->data.data + (item * src->item_size),
                      src->data.data + offset, src->item_size);
    odometer(coords, cv.ndims, cv.shape);
  }
#endif

  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);

  return st;
}
