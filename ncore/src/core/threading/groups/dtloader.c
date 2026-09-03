/**
 * @file dtloader.c
 * @brief Data-loader thread-pool counter.
 */

#include <stdatomic.h>

#include <ncore/core/status.h>

#include "dtloader.h"

/**
 * @var atomic_thread_counter
 * @brief Data-loader thread count.
 *
 * @details Determines the number of threads the data-loader pool is
 * allowed to use.
 */
static _Atomic uint32 atomic_thread_counter = MIN_THREADS_PER_GROUP;

uint32 get_dtloader_threads() {
  return atomic_load_explicit(&atomic_thread_counter, memory_order_relaxed);
}

novaStatus_t set_dtloader_threads(uint32 threads) {
  if (threads < MIN_THREADS_PER_GROUP) {
    return (novaStatus_t){
        .err = novaInvalidNumThreads,
        .message = nova_get_error_msg(novaInvalidNumThreads, nullptr),
    };
  }
  atomic_store_explicit(&atomic_thread_counter, threads, memory_order_relaxed);
  return (novaStatus_t){
      .err = novaSuccess,
      .message = nova_get_error_msg(novaSuccess, nullptr),
  };
}
