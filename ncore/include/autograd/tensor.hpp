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
 * @see ncore/tensor.h  Underlying C Tensor definition.
 */

#pragma once

#include "ncore/repr/tensor_repr.h"
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <ncore/tensor.h>
#include <ostream>
#include <vector>

using Shape = std::vector<size_t>;
using Strides = Shape;

namespace autograd {

/**
 * @brief RAII tensor class wrapping the C Tensor struct.
 *
 * Manages allocation, deep-copying, moving, and automatic collection of
 * the underlying C tensor.  Follows the Rule of Five.
 */
class Tensor {
public:
  /**
   * @brief Default constructor. Creates an empty, unallocated tensor.
   */
  Tensor() noexcept;

  /**
   * @brief Construct an allocated tensor.
   *
   * @param shape         Dimension sizes.
   * @param dtype         Element data type.
   * @param device        Target device (CPU, GPU, or META).
   * @param requires_grad If true, an unallocated gradient tensor is created.
   */
  Tensor(const std::vector<size_t> &shape, DType_ dtype, Device device,
         bool requires_grad = false);

  /**
   * @brief Construct an allocated tensor from an initializer list.
   *
   * @param shape         Dimension sizes.
   * @param dtype         Element data type.
   * @param device        Target device (CPU, GPU, or META).
   * @param requires_grad If true, an unallocated gradient tensor is created.
   */
  Tensor(std::initializer_list<size_t> shape, DType_ dtype, Device device,
         bool requires_grad = false);

  /**
   * @brief Destructor. Recursively releases storage and gradient sub-graph.
   */
  ~Tensor();

  /**
   * @brief Copy constructor. Performs a deep copy of data and metadata.
   *
   * Allocates new storage via deepcopy(), copies element data, and
   * recursively deep-copies the gradient graph.
   *
   * @param other Source tensor to copy.
   */
  Tensor(const Tensor &other);

  /**
   * @brief Move constructor. Transfers ownership; leaves source empty.
   *
   * @param other Source tensor to move from (reset to empty afterwards).
   */
  Tensor(Tensor &&other) noexcept;

  /**
   * @brief Copy assignment. Releases current resources and deep-copies.
   *
   * @param other Source tensor to copy.
   * @return Reference to this tensor.
   */
  Tensor &operator=(const Tensor &other);

  /**
   * @brief Move assignment. Releases current resources and transfers
   * ownership.
   *
   * @param other Source tensor to move from (reset to empty afterwards).
   * @return Reference to this tensor.
   */
  Tensor &operator=(Tensor &&other) noexcept;

  /* ---- Query predicates ---- */

  /**
   * @brief Get the tensor's shape vector.
   *
   * @return Const reference to the cached shape (dimension sizes).
   */
  [[nodiscard]] const Shape &get_shape() const noexcept;

  /**
   * @brief Get the tensor's strides vector.
   *
   * @return Const reference to the cached strides (byte strides).
   */
  [[nodiscard]] const Strides &get_strides() const noexcept;

  /* ---- C API interop ---- */

  /**
   * @brief Access the underlying C Tensor (read-only).
   *
   * @return Const reference to the raw C `::Tensor` struct.
   */
  [[nodiscard]] const ::Tensor &c_tensor() const noexcept;

  /**
   * @brief Access the underlying C Tensor (mutable, for FFI calls).
   *
   * @return Mutable reference to the raw C `::Tensor` struct.
   */
  ::Tensor &c_tensor() noexcept;

  /* ---- Stream output ---- */

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
   * @brief Synchronise the cached shape_ and strides_ from c_tensor_.
   */
  void sync_metadata_() noexcept;

  ::Tensor c_tensor_{};
  Shape shape_;
  Strides strides_;
};

} // namespace autograd
