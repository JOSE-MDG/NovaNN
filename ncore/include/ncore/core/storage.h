/**
 * @file storage.h
 * @brief Memory storage primitives for tensor data.
 *
 * Allocation, share, free and resize all go through the Rust FFI:
 *
 * @code
 * allocate:  reserve()  → RustHandle { id, requested, bytes, align } + novaStatus_t
 * share:     retain()   → increment refcount
 * free:      release()  → decrement refcount; free when zero + novaStatus_t
 * resize:    resize()   → may relocate the buffer + novaStatus_t
 * query:     get_data_from() → CPU-visible pointer
 * @endcode
 *
 * The Rust storage registry is thread-safe; the handle structs are
 * caller-owned values and need external synchronization for concurrent
 * mutation. Detailed status messages stay valid until the next storage
 * operation on the same thread.
 */

#pragma once

#include <stdalign.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <ncore/core/dtype.h>
#include <ncore/core/status.h>

#ifdef __cplusplus
extern "C" {
#endif

// clang-format off
/**
 * @union data_ptr
 * @brief Typed pointer union for tensor data buffers.
 *
 * All members alias the same address; the active member follows the
 * tensor's @ref DType_. @c v and @c data serve generic and raw-byte
 * access.
 */
typedef union {
  void *v;                        ///< Untyped pointer (generic access).
  unsigned char *data;            ///< Raw byte pointer (serialization / memcpy).
  float32 *f32;                   ///< Pointer to 32-bit float elements.
  float64 *f64;                   ///< Pointer to 64-bit float (double) elements.
  float16 *half;                  ///< Pointer to IEEE 754 half-precision (16-bit) elements.
  bfloat16 *bf16;                 ///< Pointer to Brain Float 16 elements.
  float8_e4m3fn *fp8e4m3fn;       ///< Pointer to 8-bit E4M3FN float elements.
  float8_e5m2 *fp8e5m2;           ///< Pointer to 8-bit E5M2 float elements.
  float4_e2m1fn_x2 *fp4e2m1fn_x2; ///< Pointer to 4-bit E2M1FN float elements.
  int8 *s8;                       ///< Pointer to signed 8-bit integer elements.
  uint8 *u8;                      ///< Pointer to unsigned 8-bit integer elements.
  int16 *s16;                     ///< Pointer to signed 16-bit integer elements.
  uint16 *u16;                    ///< Pointer to unsigned 16-bit integer elements.
  int32 *s32;                     ///< Pointer to signed 32-bit integer elements.
  uint32 *u32;                    ///< Pointer to unsigned 32-bit integer elements.
  int64 *s64;                     ///< Pointer to signed 64-bit integer elements.
  uint64 *u64;                    ///< Pointer to unsigned 64-bit integer elements.
  qint8 *qs8;                     ///< Pointer to quantized signed 8-bit elements.
  quint8 *qu8;                    ///< Pointer to quantized unsigned 8-bit elements.
  qint16 *qs16;                   ///< Pointer to quantized signed 16-bit elements.
  quint16 *qu16;                  ///< Pointer to quantized unsigned 16-bit elements.
  qint32 *qs32;                   ///< Pointer to quantized signed 32-bit elements.
  quint32 *qu32;                  ///< Pointer to quantized unsigned 32-bit elements.
} data_ptr;
// clang-format on

/**
 * @struct RustHandle
 * @brief Handle to a Rust-allocated memory region, FFI bridge.
 *
 * @li @c id — allocation ID; 0 marks an invalid or failed allocation.
 * @li @c requested_bytes — theoretical size asked by the caller.
 * @li @c bytes — real backend reservation, at least @c requested_bytes
 *   (size-class rounding may pad it).
 * @li @c align — alignment constraint in bytes.
 *
 * A handle is structurally valid when @c id != 0 and @c bytes != 0.
 * @c is_valid_handle() also checks that the ID is still registered.
 * Query functions tolerate null or invalid handles with sentinel
 * values, lifecycle and resize functions need a live handle.
 * @c gt_bytes() performs the structural check only: callers holding a
 * stale copy after the final release must not read it.
 */
typedef struct {
  uint64_t id;            ///< Unique identifier for the allocation.
  size_t requested_bytes; ///< Theoretical size asked by the caller, in bytes.
  size_t bytes;           ///< Real backend reservation in bytes.
  size_t align;           ///< Alignment constraint.
} RustHandle;

/**
 * @brief Allocate a buffer on the specified memory device.
 *
 * Routes to the Rust allocator: host RAM, page-locked host memory for
 * fast GPU transfers (@c pin_memory = true, CPU only), or GPU VRAM
 * through the active CUDA/HIP backend.
 *
 * On failure, @p status receives the error code and detailed message.
 *
 * @param[in]  size       Reservation size in bytes, already rounded to
 *                         its size class by the caller.  Must be > 0.
 * @param[in]  requested  Theoretical size asked by the caller.  Must be
 *                         > 0 and not greater than @p size.
 * @param[in]  device     Target device: @c cpu or @c device.
 *                         A null pointer selects @c cpu.
 * @param[in]  pin_memory If @c true and @p device is @c cpu,
 *                        allocate page-locked host memory.  Must be
 *                        @c false when @p device is @c device.
 * @param[in]  align      Required alignment in bytes (must be a
 *                        power of two). Ignored for device and
 *                        pinned host allocations.
 * @param[out] status     Receives the operation result. Must not be
 *                        @c nullptr.
 *
 * @return A valid @ref RustHandle on success (the caller owns one
 *         reference), or a handle with @c id == 0 on failure.
 */
RustHandle reserve(size_t size, size_t requested, const char *device,
                   bool pin_memory, size_t align, novaStatus_t *status);

/**
 * @brief Increment the reference count of a Rust allocation.
 *
 * @param[in,out] handle  Pointer to a valid @ref RustHandle.
 *                        Must not be @c nullptr.
 *
 * @pre  @p handle must point to a live handle (@c id != 0).
 *
 * @return @ref novaStatus_t describing the operation.
 *
 * Pair with a @c release() call or the allocation leaks.
 */
novaStatus_t retain(RustHandle *handle);

/**
 * @brief Decrement the reference count and free memory at zero.
 *
 * On the final successful release the underlying allocation is freed
 * and the handle is invalidated (@c id set to @c 0).
 *
 * @param[in,out] handle  Pointer to the @ref RustHandle to release.
 *                        Must not be @c nullptr.
 * @param[out] status     Receives the operation result. Must not be
 *                        @c nullptr.
 *
 * @return @c true if the underlying memory was freed (count reached
 *         zero), @c false if it is still alive or an error occurred.
 *         On failure the handle stays valid and the caller still owns
 *         the reference.
 */
bool release(RustHandle *handle, novaStatus_t *status);

/**
 * @brief Resize an existing allocation (may relocate).
 *
 * Device-backed storage rounds the request to its size class with the
 * same rules as the initial reservation. On success the handle's
 * cached @c requested_bytes and @c bytes are updated; @c align may
 * change when the backend relocates the buffer. On validation or
 * allocation failure the cache is unchanged. A backend-specific
 * failure must be treated according to its returned status and message.
 *
 * @param[in,out] handle    Pointer to the @ref RustHandle to resize.
 *                          Must not be @c nullptr.
 * @param[in]     requested New theoretical size in bytes.  Must be > 0.
 *
 * @return @ref novaStatus_t describing success or failure.
 */
novaStatus_t resize(RustHandle *handle, size_t requested);
/**
 * @brief Obtain the CPU-visible address of a Rust allocation.
 *
 * The pointer is host-accessible for CPU and pinned allocations. For
 * GPU device allocations it is only meaningful on the device side.
 *
 * @param[in] handle  Pointer to a valid @ref RustHandle.
 *                    Must not be @c nullptr.
 *
 * @return Pointer to the start of the data buffer, or @c nullptr on
 *         error.
 */
void *get_data_from(RustHandle *handle);

/**
 * @brief Check whether a RustHandle refers to a live allocation.
 *
 * True from a successful @c reserve() until the final @c release()
 * frees the memory. @p handle may be @c nullptr (returns @c false).
 *
 * @return @c true if the handle is valid (@c id != 0), @c false
 *         otherwise.
 */
bool is_valid_handle(RustHandle *handle);

/**
 * @brief Obtain the alignment recorded for a Rust allocation.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to query.
 *                    May be @c nullptr (returns @c 0).
 *
 * @return Alignment in bytes as requested or rounded up at
 *         reservation, or @c 0 if @p handle is @c nullptr or invalid.
 */
size_t get_align_from(RustHandle *handle);

/**
 * @brief Check whether a Rust allocation is backed by device-managed memory.
 *
 * True for GPU device VRAM and for pinned host allocations, which are
 * allocated through, and managed by, the active GPU backend.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to query.
 *                    May be @c nullptr (returns @c false).
 *
 * @return @c true for device-backed storage (including pinned
 *         host), @c false otherwise.
 */
bool is_device_memory_handle(RustHandle *handle);

/**
 * @brief Check whether a Rust allocation is pinned host memory.
 *
 * Pinned memory is host-accessible and readable by the GPU, which is
 * what makes asynchronous transfers possible.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to query.
 *                    May be @c nullptr (returns @c false).
 *
 * @return @c true for pinned host memory, @c false otherwise.
 */
bool is_pinned_handle(RustHandle *handle);

/**
 * @struct TensorStorage
 * @brief Typed pointer plus Rust handle, embedded in @c Tensor.
 *
 * The handle controls the allocation lifetime and carries both the
 * theoretical (@c requested_bytes) and the real reservation sizes.
 * @c safe_allocator() in @ref alloc.h creates it, @c retain()/@c release()
 * share and free it.
 */
typedef struct {
  data_ptr ptr;      ///< Typed pointer to the data buffer.
  RustHandle handle; ///< Rust FFI handle for reference counting.
} TensorStorage;

/**
 * @brief Round a byte count up to the small device step (512 bytes).
 *
 * @param[in]  n       Requested count.  Must be > 0.
 * @param[out] status  Receives the operation result.  Must not be
 *                     @c nullptr.
 *
 * @return Rounded count, or @c 0 with an error status on zero input
 *         or overflow.
 */
size_t round_up_device_small(size_t n, novaStatus_t *status);

/**
 * @brief Round a byte count up to the large device step (2 MiB).
 *
 * @param[in]  n       Requested count.  Must be > 0.
 * @param[out] status  Receives the operation result.  Must not be
 *                     @c nullptr.
 *
 * @return Rounded count, or @c 0 with an error status on zero input
 *         or overflow.
 */
size_t round_up_device_large(size_t n, novaStatus_t *status);

/**
 * @brief Round a byte count up to the small pinned step (4096 bytes).
 *
 * @param[in]  n       Requested count.  Must be > 0.
 * @param[out] status  Receives the operation result.  Must not be
 *                     @c nullptr.
 *
 * @return Rounded count, or @c 0 with an error status on zero input
 *         or overflow.
 */
size_t round_up_pinned_small(size_t n, novaStatus_t *status);

/**
 * @brief Round a byte count up to the next pinned geometric class.
 *
 * @details
 * Classes are 1x, 1.25x, 1.5x and 1.75x of each power of two, covering
 * the request with waste bounded near 25%.
 *
 * @param[in]  n       Requested count.  Must be > 0.
 * @param[out] status  Receives the operation result.  Must not be
 *                     @c nullptr.
 *
 * @return Rounded count, or @c 0 with an error status on zero input
 *         or overflow.
 */
size_t round_up_pinned_large(size_t n, novaStatus_t *status);

/**
 * @brief Resolve the real reservation size for a theoretical request.
 *
 * @details
 * Device requests use the 512-byte step up to 10 MiB and the 2 MiB step
 * above.  Pinned requests use the 4096-byte step up to 16 KiB and
 * geometric classes above.  Plain host reservations are identity.
 * Rounding applies only when a device backend is active. Otherwise the
 * request is returned unchanged.
 *
 * @param[in]  requested Requested count.  Must be > 0.
 * @param[in]  is_device @c true for GPU device reservations.
 * @param[in]  is_pinned @c true for page-locked host reservations.
 * @param[out] status    Receives the operation result.  Must not be
 *                       @c nullptr.
 *
 * @return Rounded count, or @c 0 with an error status on zero input,
 *         overflow, or a null status slot.
 */
size_t resolve_rounded_bytes(size_t requested, bool is_device, bool is_pinned,
                             novaStatus_t *status);

/**
 * @brief Return the real reservation size backing a tensor.
 *
 * @details
 * Validates the tensor, the status slot, and the tensor storage, then
 * returns the handle's real reservation size.
 *
 * @param[in]  ten    Tensor to query.  Must be allocated.
 * @param[out] status Receives the operation result.  Must not be
 *                    @c nullptr.
 *
 * @return Real reservation size, or @c 0 with an error status when
 *         @p ten or @p status is @c nullptr, the tensor storage is
 *         @c nullptr, or the handle is invalid.
 */
size_t gt_bytes(const Tensor *ten, novaStatus_t *status);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
