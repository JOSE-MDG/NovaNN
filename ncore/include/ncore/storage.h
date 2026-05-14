/**
 * @file storage.h
 * @brief Memory storage primitives for tensor data.
 *
 * Provides the data_ptr union for type-safe access to tensor buffers,
 * the RustHandle abstraction for FFI-managed memory, and the
 * TensorStorage struct that binds them together.
 */

#pragma once

#include <ncore/dtype.h>
#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Typed pointer union for tensor data buffers.
 *
 * All members point to the same memory address; the active member is
 * determined by the tensor's DType_ at runtime.  Using the correct
 * member eliminates casts and enables natural pointer arithmetic.
 */
typedef union {
  void *v;             ///< Untyped pointer (generic access).
  unsigned char *data; ///< Raw byte pointer (serialisation / memcpy).
  float32 *f32;        ///< Pointer to 32-bit float elements.
  float64 *f64;        ///< Pointer to 64-bit float (double) elements.
  half *half;          ///< Pointer to IEEE 754 half-precision (16-bit) elements.
  bfloat16 *bf16;      ///< Pointer to Brain Float 16 elements.
  int8 *s8;            ///< Pointer to signed 8-bit integer elements.
  uint8 *u8;           ///< Pointer to unsigned 8-bit integer elements.
  int32 *s32;          ///< Pointer to signed 32-bit integer elements.
  uint32 *u32;         ///< Pointer to unsigned 32-bit integer elements.
  int64 *s64;          ///< Pointer to signed 64-bit integer elements.
  uint64 *u64;         ///< Pointer to unsigned 64-bit integer elements.
  qint8 *qs8;          ///< Pointer to quantised signed 8-bit elements.
  quint8 *qu8;         ///< Pointer to quantised unsigned 8-bit elements.
} data_ptr;

/**
 * @brief Opaque handle to a Rust-allocated memory region.
 *
 * Returned by reserve() and passed to retain() / release() for
 * reference-counted lifetime management.  The actual allocation lives
 * on the Rust side; this struct is a thin FFI bridge.
 */
typedef struct {
  int64_t id;          ///< Unique identifier for the allocation.
  size_t size_bytes;   ///< Usable size of the allocation in bytes.
  size_t align;        ///< Alignment constraint (e.g., 64 for cache-line alignment).
} RustHandle;

/**
 * @brief Allocate a new buffer via the Rust memory allocator.
 * @param size  Requested size in bytes.
 * @param align Required alignment (must be a power of two).
 * @return A valid RustHandle on success, or a handle with id == 0 on failure.
 */
RustHandle reserve(size_t size, size_t align);

/**
 * @brief Increment the reference count of a Rust allocation.
 * @param handle Pointer to a valid RustHandle.
 */
void retain(RustHandle *handle);

/**
 * @brief Decrement the reference count; free memory when it reaches zero.
 * @param handle Pointer to the RustHandle to release.
 * @return true if the underlying memory was freed, false if it is still alive.
 */
bool release(RustHandle *handle);

/**
 * @brief Resize an existing allocation (may move).
 * @param handle   Pointer to the RustHandle to resize.
 * @param new_size New size in bytes.
 * @return true on success, false on out-of-memory.
 */
bool resize(RustHandle *handle, size_t new_size);

/**
 * @brief Obtain the CPU-visible address of a Rust allocation.
 * @param handle Pointer to a valid RustHandle.
 * @return Pointer to the start of the data buffer, or NULL on error.
 */
void *get_data_from(RustHandle *handle);

/**
 * @brief Check whether a RustHandle refers to a live allocation.
 * @param handle Pointer to the RustHandle to validate.
 * @return true if the handle is valid, false otherwise.
 */
bool is_valid_handle(RustHandle *handle);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

/**
 * @brief Complete tensor storage descriptor.
 *
 * Bundles the typed pointer, byte count, alignment, and the Rust-side
 * handle so that the Tensor struct can manage memory transparently.
 */
typedef struct {
  data_ptr ptr;        ///< Typed pointer to the data buffer.
  size_t size_bytes;   ///< Total capacity in bytes.
  size_t align;        ///< Alignment of the buffer.
  RustHandle handle;   ///< Rust FFI handle for reference counting.
} TensorStorage;
