/**
 * @file contiguous.h
 * @brief CPU backend entry point for making tensor data contiguous.
 *
 * @details
 * Declares the single CPU implementation used to produce a contiguous
 * (row-major) copy of a tensor.  The implementation lives in
 * @c ncore/native/cpu/layout/Contiguous.c and is dispatched to by
 * the layout layer.
 *
 * @see Contiguous.c  Implementation of the CPU contiguous copy.
 * @see collapse()    Collapsed-view helper used internally.
 */

#pragma once
#include <ncore/core/status.h>
#include <ncore/tensor.h>

/**
 * @brief Copy a tensor's data into a contiguous row-major layout on
 *        the CPU.
 *
 * @details
 * Writes the elements of @p src in standard row-major order into the
 * data buffer of @p dst.  The destination's backing storage must be
 * pre-allocated by the caller: it must be able to hold @c src->size
 * elements of @c src->item_size bytes each.
 *
 * The implementation collapses contiguous dimensions via
 * @ref collapse() and iterates over the collapsed view.  When @p src
 * is one-dimensional and already contiguous, no copy is performed
 * and @p dst is replaced by a view into @p src.  When @p src is a
 * scalar, the function succeeds without modifying @p dst.
 *
 * @param[in]  src  Source tensor.  Must not be @c nullptr.
 * @param[out] dst  Destination tensor with a caller-allocated data
 *                  buffer.  Must not be @c nullptr.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or an error status describing the failure.
 *
 * @pre  @p src and @p dst must not be @c nullptr.
 * @pre  @p dst's data buffer must be large enough for @c src->size
 *       elements.
 * @post On success, the elements of @p src are stored contiguously in
 *       @p dst's data buffer, or @p dst is a view of @p src when no
 *       copy was required.
 *
 * @see collapse()  Collapses contiguous dimensions before copying.
 * @see odometer()  Iterates over the collapsed view.
 */
novaStatus_t contiguous_cpu_impl(const Tensor *restrict src,
                                 Tensor *restrict dst);
