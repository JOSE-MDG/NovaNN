/**
 * @file alloc.h
 * @brief Typed memory-allocation API for tensor data buffers on CPU and GPU.
 *
 * Provides one allocation function per supported data type plus a generic
 * tensor-storage allocator.  Every entry point follows the same pipeline:
 *
 *   1. Call `reserve()` through the Rust FFI to obtain a handle.
 *   2. Validate the handle with `is_valid_handle()`.
 *   3. Resolve the pointer via `get_data_from()` and cast to the
 *      requested element type.
 *
 * CPU and GPU backends are handled transparently — the caller chooses
 * the target device (and optionally pinned host memory) through the
 * `device` and `pin_memory` parameters.  META devices always yield a
 * `NULL` pointer because metatensors carry no backing storage.
 *
 * @see storage.h  RustHandle, reserve / retain / release lifecycle.
 * @see device.h   Device and DeviceKind enumerations.
 */

#pragma once

#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/storage.h>
#include <stddef.h>

/**
 * @brief Allocate a tensor-storage descriptor backed by an untyped data buffer.
 *
 * Creates a `TensorStorage` on the heap and wires it to a Rust‑managed
 * allocation.  The buffer alignment is selected automatically:
 * 512 B for GPU, 64 B otherwise.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param device      Target device – determines alignment and backend.
 * @param pin_memory  If true, request page‑locked host memory (CPU only;
 *                    rejected for DEVICE_GPU).
 * @return Pointer to a valid `TensorStorage`, or `NULL` on failure
 *         (allocation error or META device).
 */
TensorStorage *allocate_tensor_buffer(size_t bytes, Device device,
                                      bool pin_memory);

/**
 * @brief Allocate a typed 32-bit float (float32) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
float32 *allocate_f32_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed 64-bit float (float64 / double) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
float64 *allocate_f64_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed half-precision 16-bit float (float16) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
float16 *allocate_f16_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed brain 16-bit float (bfloat16) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
bfloat16 *allocate_bf16_buffer(size_t bytes, size_t align, Device device,
                               bool pin_memory);

/**
 * @brief Allocate a typed signed 8-bit integer (int8) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
int8 *allocate_s8_buffer(size_t bytes, size_t align, Device device,
                         bool pin_memory);

/**
 * @brief Allocate a typed unsigned 8-bit integer (uint8) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
uint8 *allocate_u8_buffer(size_t bytes, size_t align, Device device,
                          bool pin_memory);

/**
 * @brief Allocate a typed signed 32-bit integer (int32) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
int32 *allocate_s32_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed unsigned 32-bit integer (uint32) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
uint32 *allocate_u32_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

/**
 * @brief Allocate a typed signed 64-bit integer (int64) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
int64 *allocate_s64_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed unsigned 64-bit integer (uint64) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
uint64 *allocate_u64_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

/**
 * @brief Allocate a typed quantized signed 8-bit integer (qint8) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
qint8 *allocate_qs8_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed quantized unsigned 8-bit integer (quint8) buffer.
 *
 * @param bytes       Requested buffer size in bytes.
 * @param align       Alignment constraint (must be a power of two).
 * @param device      Target device (CPU, GPU, or META).
 * @param pin_memory  If true, request page‑locked host memory (CPU only).
 * @return Pointer to the buffer, or `NULL` for META / on allocation failure.
 */
quint8 *allocate_qu8_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);
