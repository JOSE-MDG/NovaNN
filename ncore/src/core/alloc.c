/**
 * @file alloc.c
 * @brief Implementation of typed memory allocation routines.
 *
 * Implements the allocation functions declared in alloc.h.  Every function
 * follows the same pattern:
 *   1. Call reserve() to obtain a Rust-managed allocation.
 *   2. Validate the returned handle with is_valid_handle().
 *   3. Dispatch on device type (CPU / GPU / META) and return a typed
 *      pointer into the buffer, or NULL for metatensors.
 *
 * GPU paths are stubs and will be filled in once a GPU memory backend
 * is integrated.  META devices always produce a NULL pointer because
 * metatensors carry no data.
 */

#include <ncore/alloc.h>
#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <ncore/storage.h>
#include <ncore/tensor.h>

/**
 * @brief Allocate a tensor storage descriptor with an untyped data buffer.
 *
 * Reserves a 64-byte-aligned Rust allocation, allocates a TensorStorage
 * struct on the heap (CPU only), and wires them together.
 *
 * @param bytes  Requested buffer size in bytes.
 * @param device Target device (CPU, GPU, or META).
 * @return Pointer to a newly allocated TensorStorage, or NULL on failure.
 */
TensorStorage *allocate_tensor_buffer(size_t bytes, Device device) {

  TensorStorage *storage = NULL;

  RustHandle handle = reserve(bytes, 64);

  NOVA_INTERNAL_ASSERT(is_valid_handle(&handle),
                       "[HANDLE] allocate_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation and TensorStorage creation
  }

  if (device == DEVICE_META) {
    return storage;
  }
  if (device == DEVICE_CPU) {
    storage = (TensorStorage *)malloc(sizeof(TensorStorage));

    if (storage == NULL) {
      return NULL;
    }

    storage->ptr.data = (unsigned char *)get_data_from(&handle);
    storage->align = handle.align;
    storage->handle = handle;
    storage->size_bytes = bytes;
  }

  return storage;
}

/**
 * @brief Allocate a typed 32-bit float buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
float32 *allocate_f32_buffer(size_t bytes, size_t align, Device device) {

  float32 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_f32_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for float32
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (float32 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed 64-bit float (double) buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
float64 *allocate_f64_buffer(size_t bytes, size_t align, Device device) {

  float64 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_f64_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for float64
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (float64 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed half-precision (16-bit float) buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
half *allocate_f16_buffer(size_t bytes, size_t align, Device device) {

  half *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_f16_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for float16
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (half *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed bfloat16 buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
bfloat16 *allocate_bf16_buffer(size_t bytes, size_t align, Device device) {

  bfloat16 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_bf16_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for bfloat16
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (bfloat16 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed signed 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
int8 *allocate_s8_buffer(size_t bytes, size_t align, Device device) {

  int8 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_s8_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for int8
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (int8 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed unsigned 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
uint8 *allocate_u8_buffer(size_t bytes, size_t align, Device device) {

  uint8 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_u8_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for uint8
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (uint8 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed signed 32-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
int32 *allocate_s32_buffer(size_t bytes, size_t align, Device device) {

  int32 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_s32_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for int32
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (int32 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed unsigned 32-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
uint32 *allocate_u32_buffer(size_t bytes, size_t align, Device device) {

  uint32 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_u32_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for uint32
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (uint32 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed signed 64-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
int64 *allocate_s64_buffer(size_t bytes, size_t align, Device device) {

  int64 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_s64_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for int64
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (int64 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed unsigned 64-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
uint64 *allocate_u64_buffer(size_t bytes, size_t align, Device device) {

  uint64 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_u64_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for uint64
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (uint64 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed quantized signed 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
qint8 *allocate_qs8_buffer(size_t bytes, size_t align, Device device) {

  qint8 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_qs8_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for qint8
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (qint8 *)get_data_from(&handle);
  }

  return ptr;
}

/**
 * @brief Allocate a typed quantized unsigned 8-bit integer buffer.
 * @param bytes  Requested size in bytes.
 * @param align  Alignment constraint (must be a power of two).
 * @param device Target device.
 * @return Pointer to the buffer, or NULL for META / on error.
 */
quint8 *allocate_qu8_buffer(size_t bytes, size_t align, Device device) {

  quint8 *ptr = NULL;
  RustHandle handle = reserve(bytes, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_qu8_buffer: Invalid rust memory handle")

  if (device == DEVICE_GPU) {
    // TODO: Implement GPU memory allocation for quint8
  }

  if (device == DEVICE_META) {
    return ptr;
  }
  if (device == DEVICE_CPU) {
    ptr = (quint8 *)get_data_from(&handle);
  }

  return ptr;
}
