/**
 * @file MemoryAllocator_test.cpp
 * @brief Unit tests for the memory allocator (reserve/release/resize).
 *
 * The tests exercise the public C API in @ref storage.h, which routes
 * every allocation to the Rust storage layer (ncore/memory): CPU
 * allocations go through `RustStorage::allocate`, while device and
 * pinned-host allocations go through `RustStorage::allocate_device`
 * and the C++ bridge. Status codes follow the `StorageError` -> `NovaError`
 * mapping in `ncore/memory/src/error.rs`.
 *
 * @note Tests that interact with the GPU (device allocations and
 *       pinned-host memory, which is also allocated through the GPU
 *       backend) are guarded by `NOVA_HAS_CUDA`/`NOVA_HAS_HIP` so a
 *       CPU-only build still passes: without a backend the allocator
 *       would report novaBackendNotCompiled.
 */

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <thread>
#include <unordered_set>

#include <gtest/gtest.h>

#include <ncore/core/alloc.h>
#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>

#include "utils/Pool.hpp"

namespace {

/**
 * @class HandleTracker
 * @brief Tracks live handle IDs across threads.
 *
 * A thread-safe set that flags when two concurrent allocations receive
 * the same live handle ID. A failed insertion means the allocator
 * re-issued an ID that is still in use.
 */
class HandleTracker {
public:
  bool acquire(uint64_t id) {
    std::lock_guard<std::mutex> lock(mtx_);
    return live_ids_.insert(id).second;
  }

  void release(uint64_t id) {
    std::lock_guard<std::mutex> lock(mtx_);
    live_ids_.erase(id);
  }

private:
  std::mutex mtx_;
  std::unordered_set<uint64_t> live_ids_;
};

/**
 * @brief Returns a pseudo-random buffer size in the range [from, to].
 *
 * @param[in] from  Inclusive lower bound (default: 2).
 * @param[in] to    Inclusive upper bound (default: 2048).
 * @return A uniformly distributed size within the requested range.
 */
size_t getRandomBufferSize(size_t from = 2, size_t to = 2048) {
  static thread_local std::mt19937 gen(std::random_device{}());
  std::uniform_int_distribution<size_t> distrib(from, to);
  return distrib(gen);
}

} // namespace

/**
 * @brief Allocates and releases a host buffer.
 * @test reserve(512, "cpu", false, 64) routes to
 *       RustStorage::allocate() (ncore/memory/src/storage.rs), which
 *       builds a 64-byte-aligned layout and returns a handle with
 *       id != 0. The test fills the buffer, verifies the returned
 *       pointer is aligned to 64, then releases it: refcount 1 -> 0
 *       frees the buffer and zeroes the handle id, making
 *       is_valid_handle() return false.
 */
TEST(MemoryAllocator, HostAllocationAndDeallocation) {

  const size_t bytes = 512;
  const char *device = "cpu";
  const size_t align = 64;
  novaStatus_t st{};
  auto handle = reserve(bytes, device, false, align, &st);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";

  ASSERT_TRUE(is_valid_handle(&handle));

  float *ptr = static_cast<float *>(get_data_from(&handle));

  ASSERT_NE(ptr, nullptr)
      << "float *ptr = get_data_from(&handle) -> ptr == nullptr";
  EXPECT_TRUE((reinterpret_cast<std::uintptr_t>(ptr) % align) == 0)
      << "ptr alignment is diffent to " << align << "\n";

  for (size_t idx = 0; idx < (bytes / sizeof(float)); ++idx) {
    ptr[idx] = 1.0e-3f;
  }

  release(&handle, &st);

  EXPECT_TRUE(!is_valid_handle(&handle)) << st.message << "\n";
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
/**
 * @brief Allocates and releases a device buffer.
 * @test reserve(512, "device", false, 0) routes to
 *       RustStorage::allocate_device() (storage.rs), which asks the
 *       C++ bridge (deviceReserve) to reserve 512 bytes on the active
 *       CUDA/HIP backend. `align` is ignored on this path. The handle
 *       must be flagged as device memory and release must free the
 *       buffer.
 */
TEST(MemoryAllocator, DeviceAllocationAndDeallocation) {

  const size_t bytes = 512;
  const char *device = "device";
  const size_t align = 512; // is ignored
  novaStatus_t st{};

  auto handle = reserve(bytes, device, false, align, &st);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";

  ASSERT_TRUE(is_valid_handle(&handle));
  EXPECT_TRUE(is_device_memory_handle(&handle));

  float *ptr = static_cast<float *>(get_data_from(&handle));

  ASSERT_NE(ptr, nullptr);

  release(&handle, &st);

  EXPECT_TRUE(!is_valid_handle(&handle)) << st.message << "\n";
}

/**
 * @brief Allocates and releases pinned (page-locked) host memory.
 * @test reserve(512, "cpu", true, 0) routes to
 *       RustStorage::allocate_device() with pin_memory = true, which
 *       allocates page-locked host memory through the active GPU
 *       backend. The test writes the buffer and releases it.
 */
TEST(MemoryAllocator, PinnedHostAllocationAndDeallocation) {

  const size_t bytes = 512;
  const char *device = "cpu";
  const size_t align = 4096; // is ignored
  novaStatus_t st{};
  auto handle = reserve(bytes, device, true, align, &st);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";

  EXPECT_TRUE(is_valid_handle(&handle));

  float *ptr = static_cast<float *>(get_data_from(&handle));

  ASSERT_NE(ptr, nullptr);

  for (size_t idx = 0; idx < (bytes / sizeof(float)); ++idx) {
    ptr[idx] = 1.0e-3f;
  }

  release(&handle, &st);

  EXPECT_TRUE(!is_valid_handle(&handle)) << st.message << "\n";
}
#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

/**
 * @brief Rejects a zero-size reservation.
 * @test reserve(0, "cpu", false, 64) fails in RustStorage::allocate()
 *       with StorageError::InvalidSize, mapped to novaInvalidValue
 *       (error.rs). The returned handle has id == 0. The same applies
 *       to the device path, where allocate_device() performs the same
 *       size check.
 */
TEST(MemoryAllocator,
     InvalidReservedBytes) { // The same applies to `device = “device”`

  const size_t bytes = 0;
  novaStatus_t st{};

  auto handle = reserve(bytes, "cpu", false, 64, &st);

  EXPECT_EQ(st.err, novaInvalidValue) << st.message << "\n";
  EXPECT_TRUE(!is_valid_handle(&handle));
}

/**
 * @brief Rejects a zero-size resize.
 * @test resize(handle, 0) fails in RustStorage::resize() with
 *       StorageError::InvalidSize -> novaInvalidValue, leaving the
 *       handle registered and its cached size_bytes unchanged.
 */
TEST(MemoryAllocator,
     InvalidResizedBytes) { // The same applies to `device = “device”`

  const size_t bytes = 256;
  const size_t newBytes = 0;

  novaStatus_t st{};

  auto handle = reserve(bytes, "cpu", false, 64, &st);

  EXPECT_EQ(st.err, novaSuccess) << st.message << "\n";
  EXPECT_TRUE(is_valid_handle(&handle));

  st = resize(&handle, newBytes);
  EXPECT_EQ(st.err, novaInvalidValue);
  ASSERT_TRUE(is_valid_handle(&handle));
  EXPECT_EQ(handle.size_bytes, bytes);

  release(&handle, &st);
  EXPECT_EQ(st.err, novaSuccess) << st.message << "\n";
}

/**
 * @brief Resizes a host buffer.
 * @test resize() reallocs the 512-byte buffer to 1024 bytes on the CPU
 *       path (RustStorage::resize, std::realloc) and updates the
 *       handle's size_bytes cache to the new size on success.
 */
TEST(MemoryAllocator, HostBufferResizing) {

  const size_t bytes = 512;
  const size_t newBytes = 1024;
  const char *device = "cpu";
  const size_t align = 64;

  novaStatus_t st{};

  auto handle = reserve(bytes, device, false, align, &st);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  ASSERT_TRUE(is_valid_handle(&handle));
  EXPECT_TRUE(handle.size_bytes == bytes);

  st = resize(&handle, newBytes);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  EXPECT_TRUE(handle.size_bytes == newBytes);

  release(&handle, &st);
  EXPECT_EQ(st.err, novaSuccess) << st.message << "\n";
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
/**
 * @brief Resizes a device buffer.
 * @test resize() forwards to the C++ bridge (deviceResize), which
 *       allocates a new device buffer, copies the old contents, and
 *       frees the old one. The handle stays flagged as device memory
 *       and its size_bytes cache is updated.
 */
TEST(MemoryAllocator, DeviceBufferResizing) {

  const size_t bytes = 512;
  const size_t newBytes = 1024;
  const char *device = "device";
  const size_t align = 512; // is ignored

  novaStatus_t st{};

  auto handle = reserve(bytes, device, false, align, &st);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  ASSERT_TRUE(is_valid_handle(&handle));
  EXPECT_TRUE(is_device_memory_handle(&handle));
  EXPECT_EQ(handle.size_bytes, bytes);

  st = resize(&handle, newBytes);

  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  EXPECT_TRUE(handle.size_bytes == newBytes);

  release(&handle, &st);
  EXPECT_EQ(st.err, novaSuccess) << st.message << "\n";
}
#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

/**
 * @brief Rejects an oversized host allocation.
 * @test reserve(SIZE_MAX, "cpu", false, 64) fails when
 *       Layout::from_size_align(SIZE_MAX, 64) cannot be built (see the
 *       @note below), reporting novaInvalidMemoryLayout.
 */
TEST(MemoryAllocator, HostAllocationLimit) {

  const size_t bytes = SIZE_MAX; // -> 16EB
  const char *device = "cpu";
  const size_t align = 64;

  novaStatus_t st{};

  auto handle = reserve(bytes, device, false, align, &st);

  /**
   * @note Returned by the Rust storage layer: rounding SIZE_MAX up to the
   *       64-byte alignment overflows `isize::MAX`, so
   *       `Layout::from_size_align` fails and is mapped to
   *       `StorageError::InvalidMemoryLayout` → novaInvalidMemoryLayout.
   *       To see more details go to @c ncore/memory/src/storage.rs
   */
  ASSERT_EQ(st.err, novaInvalidMemoryLayout) << st.message << "\n";
  ASSERT_TRUE(!is_valid_handle(&handle));
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
/**
 * @brief Rejects an oversized device allocation.
 * @test reserve(SIZE_MAX, "device", false, 0) fails in the C++ GPU
 *       backend (cudaReserve/hipReserve), which reports
 *       novaOutOfMemory; the Rust layer propagates that code as-is
 *       (see the @note below).
 */
TEST(MemoryAllocator, DeviceAllocationLimit) {

  const size_t bytes = SIZE_MAX; // -> 16EB
  const char *device = "device";
  const size_t align = 512; // is ignored

  novaStatus_t st{};

  auto handle = reserve(bytes, device, false, align, &st);

  /**
   * @note Returned by the C++ device bridge backend: the allocator cannot provide
   *       SIZE_MAX bytes and reports novaOutOfMemory, which the Rust layer
   *       propagates as-is.
   */
  ASSERT_EQ(st.err, novaOutOfMemory) << st.message << "\n";
  ASSERT_TRUE(!is_valid_handle(&handle));
}
#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

/**
 * @brief Rejects releasing a handle that was already freed.
 * @test retain() raises the refcount from 1 to 2; the first release
 *       drops it to 1 (returns false, handle still valid); the second
 *       drops it to 0, frees the buffer, and zeroes the handle id
 *       (returns true); a third release finds no live allocation and
 *       fails with novaInvalidHandle.
 */
TEST(MemoryAllocator, DoubleFree) {

  const size_t bytes = 512;
  const char *device = "cpu";
  const size_t align = 64;

  novaStatus_t st{};

  auto handle = reserve(bytes, device, false, align, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  EXPECT_TRUE(is_valid_handle(&handle));

  // Increase the refcounter from 1 to 2
  st = retain(&handle);
  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";

  bool ffree = release(&handle, &st); // refcounter -= 1 -> 1
  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  EXPECT_FALSE(ffree);
  EXPECT_TRUE(is_valid_handle(&handle));
  bool sfree =
      release(&handle,
              &st); // refcounter -= 1 -> 0; freed and removed from the registry
  ASSERT_EQ(st.err, novaSuccess) << st.message << "\n";
  EXPECT_TRUE(sfree);
  EXPECT_TRUE(!is_valid_handle(&handle));
  bool tfree = release(
      &handle,
      &st); // The handle is no longer in the registry -> novaInvalidHandle
  ASSERT_EQ(st.err, novaInvalidHandle) << st.message << "\n";
  EXPECT_FALSE(tfree);
}

/**
 * @brief Stresses concurrent host allocations from a thread pool.
 * @test Spawns a ThreadPool with hardware_concurrency workers and
 *       submits exactly one task per worker; each task loops until a
 *       shared 30-second deadline, reserving a random-size
 *       (2..2048 bytes), 64-byte-aligned host buffer, asserting
 *       success and validity, tracking the handle id with
 *       HandleTracker (no duplicate live ids), and releasing it
 *       (refcount 1 -> 0, returns true). A worker stops at its first
 *       failure so a persistent error cannot flood the log; failures
 *       are counted in an atomic and asserted to be zero at the end.
 *
 *       The queue never holds more than one task per worker, so no
 *       unbounded backlog can accumulate: an enqueue-as-fast-as-you-can
 *       producer would queue millions of heap-allocated std::function
 *       tasks (ballooning RAM) and the ThreadPool destructor would
 *       still drain that entire backlog long after the deadline.
 */
TEST(MemoryAllocator, ConcurrentRandomHostAllocFree) {
  const unsigned int nthreads = std::thread::hardware_concurrency();
  const char *device = "cpu";
  const auto duration = std::chrono::seconds(30);
  const auto deadline = std::chrono::steady_clock::now() + duration;

  HandleTracker tracker;
  std::atomic<size_t> failures{0};

  {
    tests::pool::ThreadPool pool(nthreads);

    for (unsigned int worker = 0; worker < nthreads; ++worker) {
      pool.enqueue([&tracker, &failures, device, deadline] -> void {
        while (std::chrono::steady_clock::now() < deadline) {
          novaStatus_t st{};
          auto handle = reserve(getRandomBufferSize(), device, false, 64, &st);

          if (st.err != novaSuccess) {
            ADD_FAILURE() << "reserve() failed: " << st.message;
            failures.fetch_add(1);
            return;
          }
          if (!is_valid_handle(&handle)) {
            ADD_FAILURE() << "reserve() returned an invalid handle";
            failures.fetch_add(1);
            return;
          }
          if (!tracker.acquire(handle.id)) {
            ADD_FAILURE() << "Handle id " << handle.id
                          << " already live in another thread";
            failures.fetch_add(1);
            release(&handle, &st); // avoid leaking the live buffer
            return;
          }

          std::this_thread::yield();

          tracker.release(handle.id);

          if (!release(&handle, &st)) {
            ADD_FAILURE() << "release() failed: " << st.message;
            failures.fetch_add(1);
            return;
          }
          if (is_valid_handle(&handle)) {
            ADD_FAILURE() << "Handle remained valid after release()";
            failures.fetch_add(1);
          }
        }
      });
    }
  }

  EXPECT_EQ(failures.load(), 0u);
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
/**
 * @brief Stresses concurrent device allocations from a thread pool.
 * @test Like ConcurrentRandomHostAllocFree but on the device backend;
 *       each iteration additionally checks the handle is flagged as
 *       device memory.
 */
TEST(MemoryAllocator, ConcurrentRandomDeviceAllocFree) {
#if defined(NOVA_HAS_HIP) && defined(__clang__) && defined(__has_feature)
#if __has_feature(address_sanitizer)
  // The ASan runtime CHECK-fails ("unable to unmmap", sanitizer_common.cpp)
  // from a worker thread while the HSA runtime tears down after this
  // 30-second concurrent alloc/free storm; the test body itself passes.
  // Upstream ASan-vs-ROCr interaction, outside NovaNN control.
  GTEST_SKIP() << "HIP+ASan: sanitizer runtime crashes racing libhsa "
                  "teardown on this stress pattern";
#endif
#endif
  const unsigned int nthreads = std::thread::hardware_concurrency();
  const char *device = "device";
  const auto duration = std::chrono::seconds(30);
  const auto deadline = std::chrono::steady_clock::now() + duration;

  HandleTracker tracker;
  std::atomic<size_t> failures{0};

  {
    tests::pool::ThreadPool pool(nthreads);

    for (unsigned int worker = 0; worker < nthreads; ++worker) {
      pool.enqueue([&tracker, &failures, device, deadline] -> void {
        while (std::chrono::steady_clock::now() < deadline) {
          novaStatus_t st{};
          auto handle = reserve(getRandomBufferSize(), device, false, 512, &st);

          if (st.err != novaSuccess) {
            ADD_FAILURE() << "reserve() failed: " << st.message;
            failures.fetch_add(1);
            return;
          }
          if (!is_valid_handle(&handle)) {
            ADD_FAILURE() << "reserve() returned an invalid handle";
            failures.fetch_add(1);
            return;
          }
          if (!is_device_memory_handle(&handle)) {
            ADD_FAILURE() << "Handle is not marked as device memory";
            failures.fetch_add(1);
          }
          if (!tracker.acquire(handle.id)) {
            ADD_FAILURE() << "Handle id " << handle.id
                          << " already live in another thread";
            failures.fetch_add(1);
            release(&handle, &st); // avoid leaking the live buffer
            return;
          }

          std::this_thread::yield();

          tracker.release(handle.id);

          if (!release(&handle, &st)) {
            ADD_FAILURE() << "release() failed: " << st.message;
            failures.fetch_add(1);
            return;
          }
          if (is_valid_handle(&handle)) {
            ADD_FAILURE() << "Handle remained valid after release()";
            failures.fetch_add(1);
          }
        }
      });
    }
  }

  EXPECT_EQ(failures.load(), 0u);
}

/**
 * @brief Stresses concurrent pinned-host allocations from a thread pool.
 * @test Like ConcurrentRandomHostAllocFree but with pin_memory = true,
 *       allocating page-locked host memory through the active GPU
 *       backend.
 */
TEST(MemoryAllocator, ConcurrentRandomPinnedHostAllocFree) {
  const unsigned int nthreads = std::thread::hardware_concurrency();
  const char *device = "cpu";
  const auto duration = std::chrono::seconds(30);
  const auto deadline = std::chrono::steady_clock::now() + duration;

  HandleTracker tracker;
  std::atomic<size_t> failures{0};

  {
    tests::pool::ThreadPool pool(nthreads);

    for (unsigned int worker = 0; worker < nthreads; ++worker) {
      pool.enqueue([&tracker, &failures, device, deadline] -> void {
        while (std::chrono::steady_clock::now() < deadline) {
          novaStatus_t st{};
          auto handle = reserve(getRandomBufferSize(), device, true, 4096, &st);

          if (st.err != novaSuccess) {
            ADD_FAILURE() << "reserve() failed: " << st.message;
            failures.fetch_add(1);
            return;
          }
          if (!is_valid_handle(&handle)) {
            ADD_FAILURE() << "reserve() returned an invalid handle";
            failures.fetch_add(1);
            return;
          }
          if (!tracker.acquire(handle.id)) {
            ADD_FAILURE() << "Handle id " << handle.id
                          << " already live in another thread";
            failures.fetch_add(1);
            release(&handle, &st); // avoid leaking the live buffer
            return;
          }

          std::this_thread::yield();

          tracker.release(handle.id);

          if (!release(&handle, &st)) {
            ADD_FAILURE() << "release() failed: " << st.message;
            failures.fetch_add(1);
            return;
          }
          if (is_valid_handle(&handle)) {
            ADD_FAILURE() << "Handle remained valid after release()";
            failures.fetch_add(1);
          }
        }
      });
    }
  }

  EXPECT_EQ(failures.load(), 0u);
}
#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
