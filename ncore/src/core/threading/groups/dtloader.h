/**
 * @file dtloader.h
 * @brief Interface for the data-loader thread-pool configuration.
 */

#pragma once

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/threading/threads.h>

/**
 * @brief Return the number of threads assigned to the data-loader group.
 *
 * @return Current data-loader thread count. Always at least
 *         @ref MIN_THREADS_PER_GROUP; default is
 *         @ref MIN_THREADS_PER_GROUP before any call to
 *         @ref set_dtloader_threads().
 *
 * @see set_dtloader_threads()
 */
uint32 get_dtloader_threads();

/**
 * @brief Assign a thread count to the data-loader group.
 *
 * @param[in] threads  Number of threads for the data-loader pool. Must
 *                     be at least @ref MIN_THREADS_PER_GROUP.
 *
 * @return @ref novaStatus_t with @c err set to @ref novaSuccess on
 *         success, or @ref novaInvalidNumThreads when @p threads is
 *         less than @ref MIN_THREADS_PER_GROUP.
 *
 * @pre  @p threads must be at least @ref MIN_THREADS_PER_GROUP.
 *
 * @see get_dtloader_threads()
 */
novaStatus_t set_dtloader_threads(uint32 threads);
