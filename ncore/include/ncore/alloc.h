/**
 * @file alloc.h
 * @brief Typed memory-allocation API for tensor data buffers on CPU and GPU.
 *
 * @details
 * Provides one allocation function per supported data type plus a
 * generic tensor-storage allocator.  Every entry point follows the
 * same pipeline:
 *
 * 1. Call @ref reserve() through the Rust FFI to obtain a
 *    @ref RustHandle.
 * 2. Validate the handle with @ref is_valid_handle().
 * 3. Resolve the pointer via @ref get_data_from() and cast to the
 *    requested element type.
 *
 * CPU and GPU backends are handled transparently — the caller
 * chooses the target device (and optionally pinned host memory)
 * through the `device` and `pin_memory` parameters.  META devices
 * always yield a `NULL` pointer because metatensors carry no
 * backing storage.
 *
 * ## Alignment
 *
 * The generic @ref allocate_tensor_buffer() selects alignment
 * automatically (512 B for GPU, 64 B for CPU).  The typed
 * `allocate_*_buffer()` functions accept an explicit `align`
 * parameter so callers can override the default when needed.
 *
 * ## Thread Safety
 *
 * All functions in this header are thread-safe.  The underlying
 * Rust allocator manages its own synchronisation.
 *
 * @see storage.h   RustHandle, reserve / retain / release lifecycle.
 * @see device.h    Device and DeviceKind enumerations.
 * @see dtype.h     DType_ enumeration and numeric type aliases.
 */

#pragma once

#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/storage.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Allocate a tensor-storage descriptor backed by an untyped
 *        data buffer.
 *
 * @details
 * Creates a @ref TensorStorage on the heap and wires it to a
 * Rust-managed allocation.  The buffer alignment is selected
 * automatically: 512 B for GPU, 64 B otherwise.
 *
 * For META devices the function returns `NULL` immediately without
 * calling into the Rust allocator.
 *
 * @param[in] bytes       Requested buffer size in bytes.  Must be > 0.
 * @param[in] device      Target device — determines alignment and
 *                        backend.
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only; rejected for `DEVICE_GPU`).
 *
 * @return Pointer to a valid @ref TensorStorage, or `NULL` on
 *         failure (allocation error or META device).
 *
 * @pre  @p bytes must be greater than zero.
 * @post On success, the returned @ref TensorStorage owns a valid
 *       @ref RustHandle and its `ptr.data` points to the allocated
 *       buffer.
 *
 * @see allocate_f32_buffer()  Typed variant for Float32.
 * @see TensorStorage          Storage descriptor struct.
 * @see reserve()              Rust FFI allocation function.
 */
TensorStorage *allocate_tensor_buffer(size_t bytes, Device device,
                                      bool pin_memory);

/**
 * @brief Allocate a typed 32-bit float (float32) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_f64_buffer()
 * @see allocate_tensor_buffer()
 */
float32 *allocate_f32_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed 64-bit float (float64 / double) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_f32_buffer()
 */
float64 *allocate_f64_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed half-precision 16-bit float (float16)
 *        buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_bf16_buffer()
 */
float16 *allocate_f16_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed brain 16-bit float (bfloat16) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_f16_buffer()
 */
bfloat16 *allocate_bf16_buffer(size_t bytes, size_t align, Device device,
                               bool pin_memory);

/**
 * @brief Allocate a typed signed 8-bit integer (int8) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_u8_buffer()
 */
int8 *allocate_s8_buffer(size_t bytes, size_t align, Device device,
                         bool pin_memory);

/**
 * @brief Allocate a typed unsigned 8-bit integer (uint8) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_s8_buffer()
 */
uint8 *allocate_u8_buffer(size_t bytes, size_t align, Device device,
                          bool pin_memory);

/**
 * @brief Allocate a typed signed 32-bit integer (int32) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_u32_buffer()
 */
int32 *allocate_s32_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed unsigned 32-bit integer (uint32) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_s32_buffer()
 */
uint32 *allocate_u32_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

/**
 * @brief Allocate a typed signed 64-bit integer (int64) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_u64_buffer()
 */
int64 *allocate_s64_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed unsigned 64-bit integer (uint64) buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_s64_buffer()
 */
uint64 *allocate_u64_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

/**
 * @brief Allocate a typed quantized signed 8-bit integer (qint8)
 *        buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_qu8_buffer()
 */
qint8 *allocate_qs8_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed quantized unsigned 8-bit integer (quint8)
 *        buffer.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] align       Alignment constraint (must be a power of two).
 * @param[in] device      Target device (`DEVICE_CPU`, `DEVICE_GPU`,
 *                        or `DEVICE_META`).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
 *
 * @return Pointer to the buffer, or `NULL` for META / on allocation
 *         failure.
 *
 * @see allocate_qs8_buffer()
 */
quint8 *allocate_qu8_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

#ifdef __cplusplus
}
#endif
