/**
 * @file Pool.cpp
 * @brief Implementation of the ThreadPool worker pool.
 */

#include "Pool.hpp"

/**
 * @brief Spawns `workers` worker threads and starts their loops.
 *
 * Each worker captures a copy of the pool's stop token so that a stop
 * requested during destruction is visible to every thread.
 *
 * @param[in] workers  Number of worker threads to spawn.
 */
tests::pool::ThreadPool::ThreadPool(unsigned int workers) {

  workers_.reserve(workers);

  for (unsigned int worker = 0; worker < workers; ++worker) {
    workers_.emplace_back(
        [this, st = stop_source_.get_token()] -> void { workerLoop(st); });
  }
}

/**
 * @brief Stops the workers and joins all threads.
 *
 * Requests cooperative stop and wakes every blocked worker. Workers
 * drain remaining tasks before exiting and are then joined here.
 */
tests::pool::ThreadPool::~ThreadPool() {
  stop_source_.request_stop();
  cv_.notify_all();
}

/**
 * @brief Queues a task for execution by the next available worker.
 *
 * The task is appended to the shared FIFO under the mutex and a single
 * worker is woken. Safe to call concurrently from any thread.
 *
 * @param[in] task  Callable to execute. It must not throw.
 */
void tests::pool::ThreadPool::enqueue(std::function<void()> task) {
  {
    std::lock_guard<std::mutex> lock(mtx_);
    tasks_.push(std::move(task));
  }
  cv_.notify_one();
}

/**
 * @brief Worker entry point: waits for and executes queued tasks.
 *
 * Blocks until a task is available or a stop is requested. When the
 * stop token is set the worker still processes all remaining tasks
 * before returning, so queued work is never abandoned.
 *
 * @param[in] st  Stop token shared by all workers of the pool.
 */
void tests::pool::ThreadPool::workerLoop(std::stop_token st) {

  while (true) {
    std::function<void()> task;
    {
      std::unique_lock<std::mutex> lock(mtx_);
      cv_.wait(lock, [this, &st] -> bool {
        return st.stop_requested() || !tasks_.empty();
      });
      if (st.stop_requested() && tasks_.empty()) {
        return;
      }
      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}
