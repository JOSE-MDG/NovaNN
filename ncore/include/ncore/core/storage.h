/**
 * @file storage.h
 * @brief Memory storage primitives for tensor data.
 *
 * @details
 * Provides the low-level building blocks for tensor memory management:
 *
 * @li 1. data_ptr — A typed pointer union that enables zero-cost
 *    type-safe access to tensor element buffers.  The active member
 *    is selected at runtime based on the tensor's @ref DType_.
 *
 * @li 2. RustHandle — An opaque handle returned by the Rust FFI
 *    allocator (@c reserve()).  Carries the allocation ID, usable
 *    byte size, and alignment.  Reference-counted lifetime
 *    management is performed through @c retain() / @c release().
 *
 * @li 3. TensorStorage — A composite descriptor that bundles the
 *    typed pointer, byte count, alignment, and Rust handle into a
 *    single struct suitable for embedding in the @c Tensor struct.
 *
 * @section memory-lifecycle Memory Lifecycle
 *
 * @code
 * allocate:  reserve()  → RustHandle { id, size, align }
 *            safe_reserve() → same, but returns novaStatus_t
 * share:     retain()   → increment refcount
 * free:      release()  → decrement refcount; free when zero
 * resize:    resize()   → may relocate the buffer
 *            safe_resize() → same, but returns novaStatus_t
 * query:     get_data_from() → CPU-visible pointer
 * @endcode
 *
 * @section thread-safety Thread Safety
 *
 * All functions in this header are thread-safe.  The Rust allocator
 * handles concurrent access internally.  Error messages returned by
 * @c get_last_reserve_error() are thread-local and valid only until
 * the next @c reserve() call on the same thread.
 *
 * @see dtype.h       DType_ enumeration used by data_ptr members.
 * @see tensor.h      Tensor struct embedding a TensorStorage.
 * @see alloc.h       Higher-level safe_allocator() wrapper.
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
 * @details
 * All 14 members point to the same memory address; the active member
 * is determined by the tensor's @ref DType_ at runtime.  Using the
 * correct member eliminates casts and enables natural pointer
 * arithmetic (e.g., @c ptr.f32[i] for a Float32 tensor).
 *
 * The @c v member provides a type-agnostic @c void* for generic code,
 * while @c data provides a raw byte pointer for serialisation and
 * @c memcpy.
 *
 * @see DType_         Enum selecting the active union member.
 * @see dtype_size()   Returns the byte-width of a DType_.
 * @see TensorStorage  Embeds a data_ptr as its @c ptr field.
 */
typedef union {
  void *v;                        ///< Untyped pointer (generic access).
  unsigned char *data;            ///< Raw byte pointer (serialisation / memcpy).
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
  qint8 *qs8;                     ///< Pointer to quantised signed 8-bit elements.
  quint8 *qu8;                    ///< Pointer to quantised unsigned 8-bit elements.
  qint16 *qs16;                   ///< Pointer to quantised signed 16-bit elements.
  quint16 *qu16;                  ///< Pointer to quantised unsigned 16-bit elements.
  qint32 *qs32;                   ///< Pointer to quantised signed 32-bit elements.
  quint32 *qu32;                  ///< Pointer to quantised unsigned 32-bit elements.
} data_ptr;
// clang-format on

/**
 * @struct RustHandle
 * @brief Opaque handle to a Rust-allocated memory region.
 *
 * @details
 * Returned by @c reserve() and passed to @c retain() / @c release() for
 * reference-counted lifetime management.  The actual allocation lives
 * on the Rust side; this struct is a thin FFI bridge.
 *
 * @section fields Fields
 *
 * @li @c id — Unique identifier for the allocation.  A value of @c 0
 *   indicates an invalid / failed allocation.
 * @li @c size_bytes — Usable size of the allocation in bytes.  May be
 *   larger than the requested size due to alignment rounding.
 * @li @c align — Alignment constraint in bytes (e.g., 64 for
 *   cache-line alignment).
 *
 * @section validity Validity
 *
 * A handle is valid when @c id != 0.  Use @c is_valid_handle() to
 * check.  Invalid handles must not be passed to any function except
 * @c is_valid_handle() itself.
 *
 * @see reserve()        Creates a RustHandle.
 * @see retain()         Increments the reference count.
 * @see release()        Decrements the reference count.
 * @see is_valid_handle()  Validates a handle.
 * @see TensorStorage    Embeds a RustHandle as its @c handle field.
 */
typedef struct {
  int64_t id;        ///< Unique identifier for the allocation.
  size_t size_bytes; ///< Usable size of the allocation in bytes.
  size_t align; ///< Alignment constraint (e.g., 64 for cache-line alignment).
} RustHandle;

/**
 * @brief Allocate a buffer on the specified memory device.
 *
 * @details
 * Routes the allocation request to the Rust FFI allocator, which
 * supports:
 * @li CPU host RAM (@c cpu) — standard @c malloc-style allocation.
 * @li Pinned host memory (@c cpu + @c pin_memory=true) —
 *   page-locked memory for efficient GPU ↔ CPU transfers.
 * @li GPU device VRAM (@c device) — allocated through the active
 *   CUDA or HIP backend.
 *
 * On failure, the error message is stored in thread-local storage
 * and can be retrieved via @c get_last_reserve_error().
 *
 * @param[in]  size       Requested size in bytes.  Must be > 0.
 * @param[in]  device     Target device: @c cpu or @c device.
 * @param[in]  pin_memory If @c true and @p device is @c cpu,
 *                        allocate page-locked host memory.  Must be
 *                        @c false when @p device is @c device.
 * @param[in]  align      Required alignment in bytes (must be a
 *                        power of two).
 *
 * @return A valid @ref RustHandle on success, or a handle with
 *         @c id == 0 on failure.
 *
 * @pre  @c size must be > 0.
 * @pre  @c device must be @c cpu or @c device.
 * @pre  @c align must be a power of two.
 * @post On success, the returned handle has @c id != 0 and the
 *       caller owns one reference.
 * @post On failure, @c get_last_reserve_error() returns a
 *       non-nullptr error message.
 *
 * @see retain()           Increments the reference count.
 * @see release()          Decrements the reference count.
 * @see get_last_reserve_error()  Retrieves the failure reason.
 * @see TensorStorage      Embeds the returned handle.
 */
RustHandle reserve(size_t size, const char *device, bool pin_memory,
                   size_t align);

/**
 * @brief Allocate a buffer with structured error handling.
 *
 * @details
 * Wraps @ref reserve() and returns a @ref novaStatus_t instead of
 * requiring the caller to validate the handle.  On success, the
 * handle is written to @p handle with @c id != 0.  On failure, the
 * error code is set to @ref novaReserveError and the message is
 * retrieved from @ref get_last_reserve_error().
 *
 * @param[in]  bytes      Requested size in bytes.  Must be > 0.
 * @param[in]  device     Target device: @c cpu or @c device.
 * @param[in]  pin_memory If @c true and @p device is @c cpu,
 *                        allocate page-locked host memory.
 * @param[in]  align      Required alignment in bytes (power of two).
 * @param[out] handle     Pointer to receive the allocated handle.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success.
 *
 * @retval novaReserveError  The underlying @ref reserve() call failed.
 * @retval novaSuccess       Allocation succeeded.
 *
 * @see reserve()         Low-level allocation without status.
 * @see safe_resize()     Resize with structured error handling.
 */
novaStatus_t safe_reserve(size_t bytes, const char *device, bool pin_memory,
                          size_t align, RustHandle *handle);

/**
 * @brief Increment the reference count of a Rust allocation.
 *
 * @details
 * Marks the allocation as shared by an additional owner.  Must be
 * paired with a corresponding @c release() call to avoid leaks.
 *
 * @param[in,out] handle  Pointer to a valid @ref RustHandle.
 *                        Must not be @c nullptr.
 *
 * @pre  @p handle must point to a valid RustHandle (@c id != 0).
 * @post The reference count is incremented by one.
 *
 * @see release()   Decrements the reference count.
 * @see reserve()   Creates a handle with an initial count of one.
 */
void retain(RustHandle *handle);

/**
 * @brief Decrement the reference count; free memory when it reaches zero.
 *
 * @details
 * Releases one owner reference.  If the count reaches zero, the
 * underlying Rust allocation is freed and the handle is invalidated
 * (@c id set to @c 0).
 *
 * @param[in,out] handle  Pointer to the @ref RustHandle to release.
 *                        Must not be @c nullptr.
 *
 * @pre  @p handle must point to a valid RustHandle (@c id != 0).
 * @post The reference count is decremented by one.
 * @post If the count reaches zero, @c handle->id is set to @c 0 and
 *       the underlying memory is freed.
 *
 * @return @c true if the underlying memory was freed (count reached
 *         zero), @c false if it is still alive.
 *
 * @see retain()    Increments the reference count.
 * @see is_valid_handle()  Check if the handle is still valid.
 */
bool release(RustHandle *handle);

/**
 * @brief Resize an existing allocation (may relocate).
 *
 * @details
 * Attempts to grow or shrink the allocation in place.  If the
 * allocator cannot expand the buffer, a new region is allocated,
 * the data is copied, and the old region is freed.  The handle's
 * @c id, @c size_bytes, and @c align fields are updated accordingly.
 *
 * @param[in,out] handle   Pointer to the @ref RustHandle to resize.
 *                         Must not be @c nullptr.
 * @param[in]     new_size New size in bytes.  Must be > 0.
 *
 * @pre  @p handle must point to a valid RustHandle (@c id != 0).
 * @pre  @c new_size must be > 0.
 * @post On success, @c handle->size_bytes == new_size (or larger).
 * @post On failure, the allocation is unchanged.
 *
 * @return @c true on success, @c false on out-of-memory.
 *
 * @see reserve()   Creates a new allocation.
 * @see release()   Frees the allocation.
 */
bool resize(RustHandle *handle, size_t new_size);

/**
 * @brief Resize an allocation with structured error handling.
 *
 * @details
 * Wraps @ref resize() and returns a @ref novaStatus_t.  On success,
 * the handle is updated with the new size.  On failure, the error
 * code is set to @ref novaResizeError or @ref novaOutOfMemory and
 * the message is retrieved from @ref get_last_reserve_error().
 *
 * @param[in,out] handle  Pointer to the @ref RustHandle to resize.
 *                         Must not be @c nullptr.
 * @param[in]     new_size New size in bytes.  Must be > 0.
 *
 * @return @ref novaStatus_t with @c novaSuccess on success.
 *
 * @retval novaOutOfMemory   The allocator could not grow the buffer.
 * @retval novaResizeError   The resized handle failed validation.
 * @retval novaSuccess       Resize succeeded.
 *
 * @see resize()        Low-level resize without status.
 * @see safe_reserve()  Allocation with structured error handling.
 */
novaStatus_t safe_resize(RustHandle *handle, size_t new_size);
/**
 * @brief Obtain the CPU-visible address of a Rust allocation.
 *
 * @details
 * For CPU and pinned-host allocations, this returns the direct
 * pointer to the data buffer.  For GPU device allocations, this
 * returns a staging pointer that is valid for host-side reads
 * (the actual device pointer is accessed through the CUDA/HIP
 * backend).
 *
 * @param[in] handle  Pointer to a valid @ref RustHandle.
 *                    Must not be @c nullptr.
 *
 * @pre  @p handle must point to a valid RustHandle (@c id != 0).
 *
 * @return Pointer to the start of the data buffer, or @c nullptr on
 *         error.
 *
 * @see data_ptr     Typed pointer union for element access.
 * @see TensorStorage  Embeds the returned pointer as @c ptr.
 */
void *get_data_from(RustHandle *handle);

/**
 * @brief Check whether a RustHandle refers to a live allocation.
 *
 * @details
 * A handle is valid when its @c id field is non-zero.  This is the
 * case after a successful @c reserve() and before the final
 * @c release() that frees the memory.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to validate.
 *                    May be @c nullptr (returns @c false).
 *
 * @return @c true if the handle is valid (@c id != 0), @c false
 *         otherwise (including @c nullptr).
 *
 * @see reserve()   Creates a valid handle.
 * @see release()   Invalidates a handle when refcount reaches zero.
 */
bool is_valid_handle(RustHandle *handle);

/**
 * @brief Obtain the alignment recorded for a Rust allocation.
 *
 * @details
 * Returns the alignment value that was either requested during
 * @c reserve() or rounded up by the allocator.  Useful for asserts
 * and SIMD-friendly buffer checks.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to query.
 *                    May be @c nullptr (returns @c 0).
 *
 * @return Alignment in bytes, or @c 0 if @p handle is @c nullptr or
 *         invalid.
 *
 * @see reserve()   Specifies the alignment constraint.
 * @see TensorStorage  Stores the alignment in its @c align field.
 */
size_t get_align_from(RustHandle *handle);

/**
 * @brief Check whether a Rust allocation is backed by device-managed memory.
 *
 * @details
 * Returns @c true for:
 * @li GPU device VRAM allocations (@c device == device).
 * @li Pinned host allocations (@c pin_memory == true), because they
 *   are allocated through the active GPU backend and are
 *   managed by it.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to query.
 *                    May be @c nullptr (returns @c false).
 *
 * @return @c true for device-backed storage (including pinned
 *         host), @c false otherwise.
 *
 * @see is_pinned_handle()  Narrower check for pinned-only.
 * @see reserve()           Creates allocations with device tracking.
 */
bool is_device_memory_handle(RustHandle *handle);

/**
 * @brief Check whether a Rust allocation is pinned host memory.
 *
 * @details
 * Returns @c true only for page-locked host allocations created
 * with @c pin_memory == true.  These allocations are accessible
 * from both CPU and GPU and enable asynchronous DMA transfers.
 *
 * @param[in] handle  Pointer to the @ref RustHandle to query.
 *                    May be @c nullptr (returns @c false).
 *
 * @return @c true for pinned host memory, @c false otherwise.
 *
 * @see is_device_memory_handle()  Broader check including device.
 * @see reserve()                  Creates pinned allocations.
 */
bool is_pinned_handle(RustHandle *handle);

/**
 * @brief Retrieve the last error message from a failed reserve() call.
 *
 * @details
 * Returns a pointer to a thread-local string describing the most
 * recent @c reserve() error on the current thread.  The pointer is
 * valid until the next call to @c reserve() on the same thread.
 *
 * This function is intended for diagnostic output and assertion
 * messages.  Do not cache the returned pointer across calls.
 *
 * @return Null-terminated error string, or @c nullptr if the last
 *         @c reserve() succeeded (or was never called).
 *
 * @see reserve()              The function whose errors are reported.
 * @see get_last_reserve_error_len()  Length of the error string.
 */
const char *get_last_reserve_error(void);

/**
 * @brief Return the length of the last reserve() error message.
 *
 * @details
 * Returns the length (excluding the null terminator) of the string
 * returned by @c get_last_reserve_error().  Returns @c 0 if there is
 * no error message.
 *
 * @return Length in bytes, or @c 0 on success / no error.
 *
 * @see get_last_reserve_error()  The error string itself.
 * @see reserve()                 The function whose errors are reported.
 */
int get_last_reserve_error_len(void);

/**
 * @struct TensorStorage
 * @brief Complete tensor storage descriptor.
 *
 * @details
 * Bundles the typed pointer, byte count, alignment, and the
 * Rust-side handle so that the @c Tensor struct can manage memory
 * transparently.  A @c Tensor owns a @c TensorStorage that holds
 * the backing buffer; the @c TensorStorage in turn holds the
 * @c RustHandle that controls the allocation lifetime.
 *
 * @section fields Fields
 *
 * @li @c ptr — Typed pointer union for element access.  The active
 *   member matches the tensor's @ref DType_.
 * @li @c size_bytes — Total usable capacity in bytes.
 * @li @c align — Alignment of the buffer in bytes.
 * @li @c handle — Rust FFI handle for reference-counted lifetime
 *   management.
 *
 * @section lifecycle Lifecycle
 *
 * @li 1. Created by @c safe_allocator() (in @ref alloc.h)
 *    which calls @c reserve() via @c safe_reserve().
 * @li 2. Shared via @c retain() when a view references the same data.
 * @li 3. Freed via @c release() when the last reference is dropped.
 *
 * @see data_ptr        Typed pointer stored in @c ptr.
 * @see RustHandle      FFI handle stored in @c handle.
 * @see alloc.h         safe_allocator() creates TensorStorage.
 * @see tensor.h        Tensor struct embedding a TensorStorage pointer.
 */
typedef struct {
  data_ptr ptr;      ///< Typed pointer to the data buffer.
  size_t size_bytes; ///< Total capacity in bytes.
  size_t align;      ///< Alignment of the buffer.
  RustHandle handle; ///< Rust FFI handle for reference counting.
} TensorStorage;

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */
