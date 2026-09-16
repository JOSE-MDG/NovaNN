/**
 * @file compute.h
 * @brief Interface for the compute thread-pool configuration.
 */

#pragma once

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/threading/threads.h>

/**
 * @brief Return the number of threads assigned to the compute group.
 *
 * @return Current compute thread count. Always at least
 *         @ref MIN_THREADS_PER_GROUP; default is
 *         @ref MIN_THREADS_PER_GROUP before any call to
 *         @ref set_compute_threads().
 *
 * @see set_compute_threads()
 */
uint32 get_compute_threads();

/**
 * @brief Assign a thread count to the compute group.
 *
 * @param[in] threads  Number of threads for the compute pool. Must be
 *                     at least @ref MIN_THREADS_PER_GROUP.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or @ref novaInvalidNumThreads when @p threads is
 *         less than @ref MIN_THREADS_PER_GROUP.
 *
 * @pre  @p threads must be at least @ref MIN_THREADS_PER_GROUP.
 *
 * @see get_compute_threads()
 */
novaStatus_t set_compute_threads(uint32 threads);
