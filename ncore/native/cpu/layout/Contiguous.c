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
 * @brief Copy a one-dimensional tensor into a contiguous buffer.
 *
 * @details
 * If @p src is already contiguous, @p dst becomes a view of the
 * same data via @ref create_view().  Otherwise the raw bytes are
 * copied with @c memcpy and the strides of @p dst are recomputed
 * for the contiguous layout.
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
  if (!is_contiguous(src)) {
    memcpy(dst->data.v, src->data.v, src->storage->size_bytes);
    compute_tensor_strides_(dst, src->ndims, src->shape, src->item_size);
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }

  *dst = create_view(src, src->shape, src->ndims, &st);
  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);
  return st;
}

/**
 * @brief Copy a collapsed one-dimensional view into a contiguous
 *        buffer.
 *
 * @details
 * If the collapsed stride equals the source item size, the data is
 * already contiguous and is copied with @c memcpy.  Otherwise the
 * elements are gathered one by one using the collapsed stride.
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
    omp_set_num_threads((int)threads);

    if (st.err != novaSuccess) {
      return st;
    }

#pragma omp parallel for schedule(                                             \
        static) if (is_parallelizable(src, threads,                            \
                                          ParallelizableByElements |           \
                                                  ParallelizableByTensor))
    for (size_t item = 0; item < src->size; ++item) {
      memcpy(dst->data.data + (item * src->item_size),
             src->data.data + (item * stride), src->item_size);
    }

#else
    for (size_t item = 0; item < src->size; ++item) {
      memcpy(dst->data.data + (item * src->item_size),
             src->data.data + (item * stride), src->item_size);
    }
#endif
    st.err = novaSuccess;
    st.message = nova_get_error_msg(st.err, nullptr);
    return st;
  }
  memcpy(dst->data.v, src->data.v, src->storage->size_bytes);
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
 * collapsed with @ref collapse() and copied element by element,
 * following the collapsed strides with an odometer walk.  When
 * @c NOVA_OPENMP is defined the multi-dimensional copy loop is
 * left unimplemented; the fallback loop is used otherwise.
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
  omp_set_num_threads((int)threads);

  if (st.err != novaSuccess) {
    return st;
  }

#pragma omp parallel for schedule(                                             \
        static) if (is_parallelizable(src, threads,                            \
                                          ParallelizableByElements |           \
                                                  ParallelizableByTensor))
  for (size_t item = 0; item < src->size; ++item) {
    size_t offset = src->offset;
    coords_t coords = {0};
    compute_coords_from_linear_index_(item, cv.ndims, cv.shape, coords);

    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    memcpy(dst->data.data + (item * src->item_size),
           src->data.data + offset, src->item_size);
  }

#else
  for (size_t item = 0; item < src->size; ++item) {

    size_t offset = src->offset;

    for (size_t dim = 0; dim < cv.ndims; ++dim) {
      offset += coords[dim] * cv.strides[dim];
    }

    memcpy(dst->data.data + (item * src->item_size),
           src->data.data + offset, src->item_size);
    odometer(coords, cv.ndims, cv.shape);
  }
#endif

  st.err = novaSuccess;
  st.message = nova_get_error_msg(st.err, nullptr);

  return st;
}
