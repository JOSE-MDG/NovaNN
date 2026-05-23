/**
 * @file alloc.h
 * @brief Typed memory allocation routines for tensor data buffers.
 *
 * Provides CPU/GPU allocation functions for every supported data type.
 * All functions reserve memory through the Rust FFI allocator, validate
 * the returned handle, and return a typed pointer to the buffer.
 *
 * GPU paths are currently stubs and will be implemented in a future
 * revision.  META device paths always return NULL since metatensors
 * do not occupy memory.
 */

#pragma once

#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/storage.h>
#include <stddef.h>

/**
 * @brief Allocate a tensor storage descriptor with an untyped data buffer.
 *
 * Creates a TensorStorage struct and backs it with a Rust-allocated buffer
 * aligned to 64 bytes. The storage keeps the RustHandle for reference
 * counting and eventual release.
 *
 * @param bytes Requested buffer size in bytes.
 * @param device Target device (CPU, GPU, or META).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to a newly allocated TensorStorage, or NULL on failure.
 */
TensorStorage *allocate_tensor_buffer(size_t bytes, Device device,
                                      bool pin_memory);

/**
 * @brief Allocate a typed 32-bit float buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
float32 *allocate_f32_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed 64-bit float (double) buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
float64 *allocate_f64_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed half-precision (16-bit float) buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
float16 *allocate_f16_buffer(size_t bytes, size_t align, Device device,
                             bool pin_memory);

/**
 * @brief Allocate a typed bfloat16 buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
bfloat16 *allocate_bf16_buffer(size_t bytes, size_t align, Device device,
                               bool pin_memory);

/**
 * @brief Allocate a typed signed 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
int8 *allocate_s8_buffer(size_t bytes, size_t align, Device device,
                         bool pin_memory);

/**
 * @brief Allocate a typed unsigned 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
uint8 *allocate_u8_buffer(size_t bytes, size_t align, Device device,
                          bool pin_memory);

/**
 * @brief Allocate a typed signed 32-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
int32 *allocate_s32_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed unsigned 32-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
uint32 *allocate_u32_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

/**
 * @brief Allocate a typed signed 64-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
int64 *allocate_s64_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed unsigned 64-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
uint64 *allocate_u64_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);

/**
 * @brief Allocate a typed quantized signed 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
qint8 *allocate_qs8_buffer(size_t bytes, size_t align, Device device,
                           bool pin_memory);

/**
 * @brief Allocate a typed quantized unsigned 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param device Target device.
 * @param align  Alignment constraint (must be a power of two).
 * @param pin_memory If true, request page-locked host memory when supported.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
quint8 *allocate_qu8_buffer(size_t bytes, size_t align, Device device,
                            bool pin_memory);
