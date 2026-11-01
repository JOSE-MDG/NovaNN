/**
 * @file DeviceBuffer.hpp
 * @brief RAII owner of a device memory buffer descriptor.
 *
 * @details
 * The owner releases the buffer on destruction unless the test released
 * it explicitly, because the backend zeroes the descriptor of every free
 * it accepts.  A test that aborts on a failed assertion therefore leaves
 * no live allocation behind.
 *
 * The buffer is reserved on the backend compiled into the build, and the
 * device it lands on is the one the runtime has active.
 */

#pragma once

#include <cstddef>
#include <cstdint>

#include <ncore/memory/ffi.hpp>

#if defined(NOVA_HAS_CUDA) || defined(NOVA_HAS_HIP)

/**
 * @namespace tests::memory
 * @brief Device-memory helpers shared by the native core test suites.
 */
namespace tests::memory {

#ifdef NOVA_HAS_CUDA
/// @brief Backend compiled into this build (CUDA).
constexpr deviceKind_t kTestDeviceBackend = deviceKind_t::DeviceCUDA;
#else
/// @brief Backend compiled into this build (HIP).
constexpr deviceKind_t kTestDeviceBackend = deviceKind_t::DeviceHIP;
#endif

/**
 * @class ScopedDeviceBuffer
 * @brief Owns one device or pinned-host buffer until the test releases it.
 */
class ScopedDeviceBuffer final {
public:
  /// @brief Takes ownership of a buffer allocated on @p stream.
  explicit ScopedDeviceBuffer(void *stream = nullptr) : stream_(stream) {}

  ScopedDeviceBuffer(const ScopedDeviceBuffer &) = delete;
  ScopedDeviceBuffer &operator=(const ScopedDeviceBuffer &) = delete;

  ~ScopedDeviceBuffer() {
    if (buffer_.ptr != nullptr) {
      (void)deviceRelease(&buffer_, device_id_, stream_);
    }
  }

  /// @brief Reserves @p bytes on the backend of this build.
  novaStatus_t reserve(std::size_t bytes, bool pinned = false) {
    return deviceReserve(bytes, &buffer_, pinned, kTestDeviceBackend,
                         &device_id_, stream_);
  }

  /// @brief Releases the buffer and gives up ownership.
  novaStatus_t release() {
    return deviceRelease(&buffer_, device_id_, stream_);
  }

  /// @brief Resizes the buffer, which may relocate it.
  novaStatus_t resize(std::size_t bytes) {
    return deviceResize(&buffer_, bytes, device_id_, stream_);
  }

  /// @brief Descriptor of the owned buffer.
  deviceBuffer_t &get() { return buffer_; }

  /// @brief Descriptor of the owned buffer.
  [[nodiscard]] const deviceBuffer_t &get() const { return buffer_; }

  /// @brief Device the reservation landed on.
  [[nodiscard]] std::int32_t deviceId() const { return device_id_; }

private:
  deviceBuffer_t buffer_{};
  std::int32_t device_id_ = -1;
  void *stream_ = nullptr;
};

} // namespace tests::memory

#endif // NOVA_HAS_CUDA || NOVA_HAS_HIP
