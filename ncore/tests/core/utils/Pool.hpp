/**
 * @file Pool.hpp
 * @brief Fixed-size worker thread pool for parallel test workloads.
 *
 * Provides a small thread pool built on `std::jthread`, a mutex, and a
 * condition variable. Submitted tasks run exactly once, in FIFO order,
 * on one of the pool's worker threads.
 */

#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <stop_token>
#include <thread>
#include <vector>

/**
 * @namespace tests::pool
 * @brief Utilities shared by the native core test suite.
 */
namespace tests::pool {

/**
 * @class ThreadPool
 * @brief Executes enqueued callables on a fixed set of worker threads.
 *
 * `ThreadPool` owns `workers` `std::jthread`s that block on a shared
 * FIFO queue. Work is submitted with enqueue() and runs exactly once on
 * the first idle worker.
 *
 * **Thread safety**: enqueue() is safe to call concurrently from any
 * number of producer threads.
 *
 * **Lifecycle**: destruction requests cooperative stop and joins all
 * workers. Tasks still queued at destruction time are drained before
 * the workers exit, so no work is abandoned.
 *
 * **Exception safety**: a task that throws terminates the process via
 * `std::terminate`. Only enqueue non-throwing tasks.
 */
class ThreadPool {
public:
  /// @brief Spawns `workers` worker threads and starts their loops.
  explicit ThreadPool(unsigned int workers);

  /// @brief Stops the workers and joins all worker threads.
  ~ThreadPool();

  /// @brief Queues `task` for execution by the next available worker.
  void enqueue(std::function<void()> task);

private:
  /// @brief Worker entry point: waits for and executes queued tasks.
  void workerLoop(std::stop_token st);

  std::queue<std::function<void()>> tasks_; ///< Pending tasks, guarded by `mtx_`.
  std::stop_source stop_source_;            ///< Cooperative stop for all workers.
  std::mutex mtx_;                          ///< Guards `tasks_` and the CV predicate.
  std::vector<std::jthread> workers_;       ///< Worker threads, joined on destruction.
  std::condition_variable cv_;              ///< Wakes workers on work or stop.
};

} // namespace tests::pool