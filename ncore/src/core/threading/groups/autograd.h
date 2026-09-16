/**
 * @file autograd.h
 * @brief Interface for the autograd thread-pool configuration.
 */

#pragma once

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/threading/threads.h>

/**
 * @brief Return the number of threads assigned to the autograd group.
 *
 * @return Current autograd thread count. Always at least
 *         @ref MIN_THREADS_PER_GROUP; default is
 *         @ref MIN_THREADS_PER_GROUP before any call to
 *         @ref set_autograd_threads().
 *
 * @see set_autograd_threads()
 */
uint32 get_autograd_threads();

/**
 * @brief Assign a thread count to the autograd group.
 *
 * @param[in] threads  Number of threads for the autograd pool. Must be
 *                     at least @ref MIN_THREADS_PER_GROUP.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or @ref novaInvalidNumThreads when @p threads is
 *         less than @ref MIN_THREADS_PER_GROUP.
 *
 * @pre  @p threads must be at least @ref MIN_THREADS_PER_GROUP.
 *
 * @see get_autograd_threads()
 */
novaStatus_t set_autograd_threads(uint32 threads);
