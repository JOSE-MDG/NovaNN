/**
 * @file alloc.c
 * @brief Implementation of typed memory allocation routines.
 *
 * @details
 * Each public function follows the same pipeline:
 *
 * 1. Convert @ref Device to a C string via @ref map_device2string().
 * 2. Call @ref reserve() through the Rust FFI to obtain a
 *    @ref RustHandle.
 * 3. Assert the handle is valid with @ref is_valid_handle().
 * 4. For META devices, return `NULL` immediately (no backing
 *    storage).
 * 5. For CPU and GPU, resolve the data pointer via
 *    @ref get_data_from() and cast to the requested element type.
 *
 * The @ref allocate_tensor_buffer() variant additionally
 * heap-allocates a @ref TensorStorage struct that owns the
 * @ref RustHandle for lifetime tracking.
 *
 * ## Thread Safety
 *
 * All functions are thread-safe.  The underlying Rust allocator
 * manages its own synchronisation.
 *
 * @see alloc.h      Public API declarations.
 * @see storage.h    RustHandle and TensorStorage definitions.
 * @see device.h     Device enumeration.
 */

#include <ncore/alloc.h>
#include <ncore/device.h>
#include <ncore/dtype.h>
#include <ncore/macros.h>
#include <ncore/storage.h>
#include <ncore/tensor.h>

/**
 * @brief Translate a @ref Device enum to the C-string expected by
 *        the Rust FFI.
 *
 * @param[in] device  The device selection (CPU, GPU, or META).
 *
 * @return `"cpu"` for `DEVICE_CPU`, `"device"` for `DEVICE_GPU`,
 *         or `"none"` as a fallback for META / unknown values.
 */
static const char *map_device2string(Device device) {
  switch (device) {
  case DEVICE_CPU:
    return "cpu";
  case DEVICE_GPU:
    return "device";
  default:
    return "none";
  }
}

/**
 * @brief Allocate a tensor-storage descriptor backed by an untyped
 *        data buffer.
 *
 * @details
 * Unlike the typed `allocate_*_buffer()` helpers, this function
 * returns a heap-allocated @ref TensorStorage that holds the
 * @ref RustHandle and tracks size / alignment metadata.  Buffer
 * alignment is chosen automatically: 512 B for GPU devices, 64 B
 * otherwise.
 *
 * @param[in] bytes       Requested buffer size in bytes.
 * @param[in] device      Target device (CPU, GPU, or META).
 * @param[in] pin_memory  If `true`, request page-locked host memory
 *                        (CPU only).
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
                                      bool pin_memory) {

  TensorStorage *storage = NULL;

  if (device == DEVICE_META) {
    return storage;
  }

  size_t align = device == DEVICE_GPU ? 512 : 64;

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);

  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    storage = (TensorStorage *)malloc(sizeof(TensorStorage));

    if (storage == NULL) {
      return NULL;
    }

    storage->ptr.data = (unsigned char *)get_data_from(&handle);
    storage->align = handle.align;
    storage->handle = handle;
    storage->size_bytes = handle.size_bytes;
  }

  return storage;
}

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
                             bool pin_memory) {

  float32 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_f32_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (float32 *)get_data_from(&handle);
  }

  return ptr;
}

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
                             bool pin_memory) {

  float64 *ptr = NULL;
  if (device == DEVICE_META) {
    return ptr;
  }
  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_f64_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (float64 *)get_data_from(&handle);
  }

  return ptr;
}

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
                             bool pin_memory) {

  float16 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_f16_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (float16 *)get_data_from(&handle);
  }

  return ptr;
}

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
                               bool pin_memory) {

  bfloat16 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_bf16_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (bfloat16 *)get_data_from(&handle);
  }

  return ptr;
}

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
                         bool pin_memory) {

  int8 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_s8_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (int8 *)get_data_from(&handle);
  }

  return ptr;
}

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
                          bool pin_memory) {

  uint8 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_u8_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (uint8 *)get_data_from(&handle);
  }

  return ptr;
}

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
                           bool pin_memory) {

  int32 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_s32_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (int32 *)get_data_from(&handle);
  }

  return ptr;
}

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
                            bool pin_memory) {

  uint32 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_u32_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (uint32 *)get_data_from(&handle);
  }

  return ptr;
}

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
                           bool pin_memory) {

  int64 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_s64_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (int64 *)get_data_from(&handle);
  }

  return ptr;
}

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
                            bool pin_memory) {

  uint64 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_u64_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (uint64 *)get_data_from(&handle);
  }

  return ptr;
}

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
                           bool pin_memory) {

  qint8 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_qs8_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (qint8 *)get_data_from(&handle);
  }

  return ptr;
}

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
                            bool pin_memory) {

  quint8 *ptr = NULL;

  if (device == DEVICE_META) {
    return ptr;
  }

  RustHandle handle =
      reserve(bytes, map_device2string(device), pin_memory, align);
  NOVA_INTERNAL_ASSERT(
      is_valid_handle(&handle),
      "[HANDLE] allocate_qu8_buffer: Invalid rust memory handle: %s\n",
      get_last_reserve_error());

  if (device == DEVICE_CPU || device == DEVICE_GPU) {
    ptr = (quint8 *)get_data_from(&handle);
  }

  return ptr;
}
