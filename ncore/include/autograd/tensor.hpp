/**
 * @file tensor.hpp
 * @brief C++ RAII wrapper for the C Tensor struct.
 *
 * @details
 * Provides a fully encapsulated, exception-safe C++ class that owns a
 * C `Tensor` instance.  Implements the Rule of Five to guarantee correct
 * lifecycle management of storage, gradient graphs, and view metadata.
 *
 * All C API functions are called internally; the underlying `::Tensor`
 * struct is kept private and exposed only via `c_tensor()` for FFI
 * interop when absolutely necessary.
 *
 * ## Design
 *
 * - **RAII**: construction allocates, destruction collects.
 * - **Rule of Five**: copy/move constructors and assignment operators
 *   are explicitly defined to handle the C tensor's reference-counted
 *   storage correctly.
 * - **Dual cache**: shape and strides are cached as `std::vector`
 *   alongside the C tensor's fixed arrays, providing a convenient
 *   C++ API while the C tensor is used for all low-level operations.
 *
 * @see ncore/tensor.h  Underlying C Tensor definition.
 * @see ncore/copy.h    deepcopy() and move_tensor().
 */

#pragma once

#include "ncore/repr/tensor_repr.h"
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <ncore/tensor.h>
#include <ostream>
#include <vector>

namespace autograd {

/** @brief Dense shape type — `std::vector<size_t>`. */
using shape_t = std::vector<size_t>;

/** @brief Dense strides type — same layout as @ref shape_t. */
using strides_t = std::vector<size_t>;

/**
 * @class Tensor
 * @brief RAII tensor class wrapping the C `::Tensor` struct.
 *
 * @details
 * Manages allocation, deep-copying, moving, and automatic collection of
 * the underlying C tensor.  Follows the Rule of Five.
 *
 * ## Lifecycle
 *
 * 1. **Construct** via the shape/dtype/device overload, an
 *    `initializer_list`, or the default constructor.
 * 2. **Use** — access shape/strides via `get_shape()` / `get_strides()`,
 *    or reach the raw C struct via `c_tensor()`.
 * 3. **Destroy** — the destructor calls `collect()` on the C tensor,
 *    recursively freeing storage and gradients.
 *
 * ## Thread safety
 *
 * Not thread-safe.  Concurrent modification of the same `Tensor`
 * instance from multiple threads requires external synchronisation.
 */
class Tensor {
public:
  /**
   * @brief Default constructor.  Creates an empty, unallocated tensor.
   *
   * @post `c_tensor_` is zero-initialised; `shape_` and `strides_`
   *       are empty.
   */
  Tensor() noexcept;

  /**
   * @brief Construct an allocated tensor.
   *
   * @details
   * Delegates to `create_tensor()` from the C API, which allocates
   * the backing buffer and initialises all C tensor metadata.  The
   * C++ shape/strides cache is then synchronised from the C tensor.
   *
   * @param[in]  shape         Dimension sizes.
   * @param[in]  dtype         Element data type (@ref DType_).
   * @param[in]  device        Target device (`DEVICE_CPU`,
   *                           `DEVICE_GPU`, or `DEVICE_META`).
   * @param[in]  requires_grad If `true`, an unallocated gradient
   *                           tensor is created.
   * @param[in]  pin_memory    If `true`, request page-locked host
   *                           memory (CPU only).
   *
   * @post `c_tensor_` is fully allocated (unless META).
   * @post `shape_` and `strides_` match the C tensor.
   */
  Tensor(const shape_t &shape, DType_ dtype, Device device,
         bool requires_grad = false, bool pin_memory = false);

  /**
   * @brief Construct an allocated tensor from an `initializer_list`.
   *
   * @details
   * Convenience overload.  Converts the `initializer_list` to a
   * `shape_t` and delegates to the shape_t constructor.
   *
   * @param[in]  shape         Dimension sizes.
   * @param[in]  dtype         Element data type (@ref DType_).
   * @param[in]  device        Target device.
   * @param[in]  requires_grad If `true`, an unallocated gradient
   *                           tensor is created.
   * @param[in]  pin_memory    If `true`, request page-locked host
   *                           memory.
   */
  Tensor(std::initializer_list<size_t> shape, DType_ dtype, Device device,
         bool requires_grad = false, bool pin_memory = false);

  /**
   * @brief Destructor.  Recursively releases storage and gradient
   *        sub-graph.
   *
   * @details
   * Calls `collect()` on the C tensor, which decrements the Rust
   * reference count and recursively frees gradients.  Safe to call
   * on an unallocated tensor (no-op).
   */
  ~Tensor();

  /**
   * @brief Copy constructor.  Performs a deep copy of data and
   *        metadata.
   *
   * @details
   * Zero-initialises the destination C tensor, calls `deepcopy()`
   * to allocate new storage and copy all elements (including the
   * gradient sub-graph), then synchronises the C++ cache.
   *
   * @param[in] other Source tensor to copy.
   *
   * @post `this->c_tensor_` owns its own independent storage.
   * @post `this->shape_` and `this->strides_` match `other`.
   */
  Tensor(const Tensor &other);

  /**
   * @brief Move constructor.  Transfers ownership; leaves source
   *        empty.
   *
   * @details
   * Calls `move_tensor()` to transfer the C tensor's storage and
   * metadata, then zeroes the source and synchronises both C++
   * caches.
   *
   * @param[in,out] other Source tensor to move from.  Reset to
   *                      empty afterwards.
   *
   * @post `other` is in a valid but unallocated state.
   */
  Tensor(Tensor &&other) noexcept;

  /**
   * @brief Copy assignment.  Releases current resources and
   *        deep-copies.
   *
   * @details
   * Self-assignment safe.  Calls `collect()` on the current C
   * tensor, zeroes it, deep-copies from @p other, and
   * synchronises the cache.
   *
   * @param[in] other Source tensor to copy.
   * @return Reference to `*this`.
   *
   * @post `this->c_tensor_` owns independent storage.
   */
  Tensor &operator=(const Tensor &other);

  /**
   * @brief Move assignment.  Releases current resources and
   *        transfers ownership.
   *
   * @details
   * Self-assignment safe.  Calls `collect()`, then
   * `move_tensor()` from @p other, zeroes the source, and
   * synchronises both caches.
   *
   * @param[in,out] other Source tensor to move from.  Reset to
   *                      empty afterwards.
   * @return Reference to `*this`.
   *
   * @post `other` is in a valid but unallocated state.
   */
  Tensor &operator=(Tensor &&other) noexcept;

  /**
   * @brief Get the tensor's shape (read-only).
   *
   * @return Const reference to the cached shape (dimension sizes).
   *
   * @see get_strides()
   */
  [[nodiscard]] const shape_t &get_shape() const noexcept;

  /**
   * @brief Get the tensor's shape (mutable).
   *
   * @return Mutable reference to the cached shape.
   *
   * @see get_strides()
   */
  [[nodiscard]] shape_t &get_shape() noexcept;

  /**
   * @brief Get the tensor's strides (read-only).
   *
   * @return Const reference to the cached strides (byte strides).
   *
   * @see get_shape()
   */
  [[nodiscard]] const strides_t &get_strides() const noexcept;

  /**
   * @brief Get the tensor's strides (mutable).
   *
   * @return Mutable reference to the cached strides.
   *
   * @see get_shape()
   */
  [[nodiscard]] strides_t &get_strides() noexcept;

  /**
   * @brief Access the underlying C Tensor (read-only).
   *
   * @return Const reference to the raw C `::Tensor` struct.
   *
   * @see c_tensor() (mutable)
   */
  [[nodiscard]] const ::Tensor &c_tensor() const noexcept;

  /**
   * @brief Access the underlying C Tensor (mutable, for FFI calls).
   *
   * @return Mutable reference to the raw C `::Tensor` struct.
   *
   * @see c_tensor() (const)
   */
  ::Tensor &c_tensor() noexcept;

  /**
   * @brief Stream insertion operator.
   *
   * @details
   * Calls `tensor_repr()` to produce a human-readable string
   * representation of the tensor (shape, dtype, device, data
   * preview).  The returned C string is freed after streaming.
   *
   * @param[in,out] ostrm Output stream.
   * @param[in]     ten   Tensor to print.
   * @return Reference to @p ostrm.
   */
  friend std::ostream &operator<<(std::ostream &ostrm, const Tensor &ten) {
    char *repr = tensor_repr(&ten.c_tensor_);
    if (repr != nullptr) {
      ostrm << repr;
    }
    free(repr);
    return ostrm;
  }

private:
  /**
   * @brief Synchronise the cached `shape_` and `strides_` from
   *        `c_tensor_`.
   *
   * @details
   * Clears and repopulates the C++ vectors from the C tensor's
   * fixed-size arrays.  Called after every constructor, assignment
   * operator, and move operation to keep the dual cache
   * consistent.
   */
  void sync_metadata_() noexcept;

  ::Tensor c_tensor_{}; ///< Underlying C tensor (owned).
  shape_t shape_;       ///< Cached dimension sizes.
  strides_t strides_;   ///< Cached byte strides.
};

} // namespace autograd
