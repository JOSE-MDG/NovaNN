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
 * allocate:  reserve()  → RustHandle { id, size, align } + novaStatus_t
 * share:     retain()   → increment refcount
 * free:      release()  → decrement refcount; free when zero + novaStatus_t
 * resize:    resize()   → may relocate the buffer + novaStatus_t
 * query:     get_data_from() → CPU-visible pointer
 * @endcode
 *
 * @section thread-safety Thread Safety
 *
 * All functions in this header are thread-safe with respect to the Rust
 * storage registry.  Handles themselves remain caller-owned values and must
 * not be concurrently mutated by multiple threads without external
 * synchronisation. Detailed status messages are valid until the next storage
 * operation on the same thread.
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
 * while @c data provides a raw byte pointer for serialization and
 * @c memcpy.
 *
 * @see DType_         Enum selecting the active union member.
 * @see dtype_size()   Returns the byte-width of a DType_.
 * @see TensorStorage  Embeds a data_ptr as its @c ptr field.
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
 * A handle is structurally valid when @c id != 0.  Use
 * @c is_valid_handle() to check whether that ID is also registered. Query
 * functions tolerate null or invalid handles by returning their documented
 * sentinel values; lifecycle and resize functions require a live handle.
 *
 * @see reserve()        Creates a RustHandle.
 * @see retain()         Increments the reference count.
 * @see release()        Decrements the reference count.
 * @see is_valid_handle()  Validates a handle.
 * @see TensorStorage    Embeds a RustHandle as its @c handle field.
 */
typedef struct {
  uint64_t id;       ///< Unique identifier for the allocation.
  size_t size_bytes; ///< Usable size of the allocation in bytes.
  size_t align; ///< Alignment constraint (e.g., 64 for cache-line alignment).
} RustHandle;

/**
 * @brief Allocate a buffer on the specified memory device.
 *
 * @details
 * Routes the allocation request to the Rust allocator, which supports:
 * @li CPU host RAM (@c cpu) — standard @c malloc-style allocation.
 * @li Pinned host memory (@c cpu + @c pin_memory=true) —
 *   page-locked memory for efficient GPU ↔ CPU transfers.
 * @li GPU device VRAM (@c device) — allocated through the active
 *   CUDA or HIP backend.
 *
 * On failure, @p status receives the error code and detailed message.
 *
 * @param[in]  size       Requested size in bytes.  Must be > 0.
 * @param[in]  device     Target device: @c cpu or @c device.
 *                         A null pointer selects @c cpu.
 * @param[in]  pin_memory If @c true and @p device is @c cpu,
 *                        allocate page-locked host memory.  Must be
 *                        @c false when @p device is @c device.
 * @param[in]  align      Required alignment in bytes (must be a
 *                        power of two).
 * @param[out] status     Receives the operation result. Must not be
 *                        @c nullptr.
 *
 * @return A valid @ref RustHandle on success, or a handle with
 *         @c id == 0 on failure.
 *
 * @pre  @c size must be > 0.
 * @pre  If non-null, @c device must be @c cpu or @c device.
 * @pre  @c align must be a power of two.
 * @post On success, the returned handle has @c id != 0 and the
 *       caller owns one reference.
 * @post @p status describes the result of the operation.
 *
 * @see retain()           Increments the reference count.
 * @see release()          Decrements the reference count.
 * @see TensorStorage      Embeds the returned handle.
 */
RustHandle reserve(size_t size, const char *device, bool pin_memory,
                   size_t align, novaStatus_t *status);

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
 * @post On success, the allocation reference count is incremented by one.
 * @return @ref novaStatus_t describing the operation.
 *
 * @see release()   Decrements the reference count.
 * @see reserve()   Creates a handle with an initial count of one.
 */
novaStatus_t retain(RustHandle *handle);

/**
 * @brief Decrement the reference count; free memory when it reaches zero.
 *
 * @details
 * Releases one owner reference.  If the count reaches zero, the
 * underlying Rust allocation is explicitly freed before the handle is
 * invalidated (@c id set to @c 0).
 *
 * @param[in,out] handle  Pointer to the @ref RustHandle to release.
 *                        Must not be @c nullptr.
 * @param[out] status     Receives the operation result. Must not be
 *                        @c nullptr.
 *
 * @pre  @p handle must point to a valid RustHandle (@c id != 0).
 * @post On success, the reference count is decremented by one.
 * @post On successful final release, @c handle->id is set to @c 0 and
 *       the underlying memory is freed.
 * @post On failure, the handle remains valid and the reference ownership is
 *       retained so the caller can report or retry the operation.
 *
 * @return @c true if the underlying memory was freed (count reached
 *         zero), @c false if it is still alive or an error occurred.
 *
 * @see retain()    Increments the reference count.
 * @see is_valid_handle()  Check if the handle is still valid.
 */
bool release(RustHandle *handle, novaStatus_t *status);

/**
 * @brief Resize an existing allocation (may relocate).
 *
 * @details
 * Attempts to grow or shrink the allocation while preserving existing data.
 * The active allocator may relocate the buffer. The handle's cached
 * @c size_bytes field is updated after a successful resize.
 *
 * @param[in,out] handle   Pointer to the @ref RustHandle to resize.
 *                         Must not be @c nullptr.
 * @param[in]     new_size New size in bytes.  Must be > 0.
 *
 * @pre  @p handle must point to a valid RustHandle (@c id != 0).
 * @pre  @c new_size must be > 0.
 * @post On success, @c handle->size_bytes == @p new_size.
 * @post On ordinary validation or allocation failure, the handle cache is
 *       unchanged. A backend-specific failure must be treated according to
 *       its returned status and message.
 *
 * @return @ref novaStatus_t describing success or failure.
 *
 * @see reserve()   Creates a new allocation.
 * @see release()   Frees the allocation.
 */
novaStatus_t resize(RustHandle *handle, size_t new_size);
/**
 * @brief Obtain the CPU-visible address of a Rust allocation.
 *
 * @details
 * For CPU and pinned-host allocations, this returns a pointer that is
 * directly accessible by the host. For GPU device allocations, the returned
 * pointer has the device accessibility guarantees provided by the active
 * CUDA or HIP backend and must not be dereferenced by host code as if it were
 * CPU memory.
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
 * A handle is valid when its @c id field is non-zero and the ID remains in
 * the Rust registry. This is the case after a successful @c reserve() and
 * before the final successful @c release() that frees the memory.
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
 *    which calls @c reserve() with status propagation.
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
