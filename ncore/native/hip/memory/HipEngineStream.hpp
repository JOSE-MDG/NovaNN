/**
 * @file HipEngineStream.hpp
 * @brief Per-device non-blocking HIP engine stream.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include <ncore/core/status.h>
#include <ncore/headeronly/macros.h>

#include <ncore/headeronly/stream_types.h>

/**
 * @brief Owns the non-blocking HIP stream used by the engine for one device.
 *
 * @details
 * One instance per device slot in @ref slots_. The stream is created lazily
 * on the first call to @ref get, on the device the slot belongs to, and is
 * reclaimed by the HIP runtime at process exit. A slot whose creation
 * failed rejects later operations instead of retrying on a broken device.
 */
class HipEngineStream final {
public:
  /**
   * @brief Returns the engine stream of @p device_id, creating it on demand.
   *
   * @param[in]  device_id Runtime device ordinal owning the stream.
   * @param[out] out       Receives the engine stream. Must not be null.
   *
   * @return @ref novaStatus_t; @c novaSuccess on success.
   */
  static novaStatus_t get(std::int32_t device_id, hipStream_t *out);

  /**
   * @brief Returns the engine stream of the active device.
   *
   * @details
   * Resolves the device the calling thread has active, so launchers never
   * assume device zero.
   *
   * @param[out] out Receives the engine stream. Must not be null.
   *
   * @return @ref novaStatus_t; @c novaSuccess on success.
   */
  static novaStatus_t current(hipStream_t *out);

  /**
   * @brief Waits for the engine stream of @p device_id to drain.
   *
   * @param[in] device_id Runtime device ordinal owning the stream.
   *
   * @return @ref novaStatus_t; @c novaSuccess on success.
   */
  static novaStatus_t drain(std::int32_t device_id);

private:
  enum class State : std::uint8_t { Uninitialized, Ready, Failed };

  novaStatus_t initialize(std::int32_t device_id);

  std::mutex mutex_;
  State state_ = State::Uninitialized;
  hipStream_t stream_ = nullptr;

  static std::array<HipEngineStream, NCORE_COMPILE_TIME_MAX_GPUS> slots_;
};
