/**
 * @file compute.c
 * @brief Compute thread-pool counter.
 */

#include <stdatomic.h>

#include <ncore/core/status.h>

#include "compute.h"

/**
 * @var atomic_thread_counter
 * @brief Compute thread count.
 *
 * @details Determines the number of threads the compute pool is
 * allowed to use.
 */
static _Atomic uint32 atomic_thread_counter = MIN_THREADS_PER_GROUP;

uint32 get_compute_threads() {
  return atomic_load_explicit(&atomic_thread_counter, memory_order_relaxed);
}

novaStatus_t set_compute_threads(uint32 threads) {
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
