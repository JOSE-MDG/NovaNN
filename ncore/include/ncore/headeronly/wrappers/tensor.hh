/**
 * @file tensor.hh
 * @brief C++ wrapper for the NovaNN tensor, providing a high-level interface
 * over the C tensor core.
 *
 * @details
 * Defines the @ref ncore::wrappers::TensorCXX class that wraps the low-level
 * C @ref Tensor structure with RAII semantics, copy/move semantics, element
 * access, dtype/device transfer helpers, and stream output. This header is
 * the primary entry point for C++ code interacting with NovaNN tensors.
 *
 * @see tensor.h    C tensor definition.
 * @see device.h    Device enumeration (@ref Device_).
 * @see dtype.h     Data-type enumeration (@ref DType_).
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ncore/core/copy.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/macros.h>
#include <ncore/repr/tensor_repr.h>
#include <ncore/tensor.h>

/**
 * @namespace ncore::wrappers
 * @brief High-level C++ wrappers over the ncore C core.
 */
namespace ncore::wrappers {

/**
 * @class TensorCXX
 * @brief RAII wrapper around the C @ref Tensor, exposing a safe C++ interface.
 *
 * @details
 * Manages the full lifecycle of a NovaNN tensor: allocation, shape
 * metadata, reference counting (via the C core), copy/move semantics,
 * element access, and device/dtype transfers.  The class is aligned to
 * 64 bytes for cache-line friendliness.
 *
 * @par Ownership
 * Constructing a @ref TensorCXX allocates the underlying buffer through
 * the C core; destruction releases it.  Copying performs a deep copy;
 * moving transfers ownership and leaves the source in a valid but empty
 * state.
 *
 * @par Thread safety
 * A single @ref TensorCXX instance must not be mutated concurrently from
 * multiple threads without external synchronisation.
 */
class alignas(64) TensorCXX {

public:
  // ================================================================
  // Constructors
  // ================================================================

  /**
   * @brief Default constructor. Creates an empty (uninitialised) tensor.
   */
  TensorCXX();

  /**
   * @brief Constructs a tensor with the given shape, dtype, and device.
   *
   * @details
   * Allocates the backing buffer through the C core via
   * @ref create_tensor().  The tensor is fully independent — it owns its
   * own storage.
   *
   * @param[in]  shape         Number of elements along each dimension.
   * @param[in]  dtype         Data type of the tensor elements.
   * @param[in]  device        Target device (CPU or GPU).
   * @param[in]  requires_grad Whether to track gradients for this tensor.
   * @param[in]  pin_memory    Whether to use page-locked (pinned) host memory.
   * @param[in,out] st         Status output; @ref novaSuccess on success.
   *
   * @see create_tensor()  Underlying C allocation.
   */
  TensorCXX(const std::vector<size_t> &shape, DType_ dtype, Device_ device,
            bool requires_grad, bool pin_memory, novaStatus_t *st);

  /**
   * @brief Destructor. Releases the underlying tensor buffer.
   */
  ~TensorCXX();

  /**
   * @brief Constructs a tensor from an initializer-list shape.
   *
   * @details
   * Delegates to the vector-shaped constructor.
   *
   * @param[in]  shape         Number of elements along each dimension.
   * @param[in]  dtype         Data type of the tensor elements.
   * @param[in]  device        Target device (CPU or GPU).
   * @param[in]  requires_grad Whether to track gradients for this tensor.
   * @param[in]  pin_memory    Whether to use page-locked (pinned) host memory.
   * @param[in,out] st         Status output; @ref novaSuccess on success.
   */
  TensorCXX(std::initializer_list<size_t> shape, DType_ dtype, Device_ device,
            bool requires_grad, bool pin_memory, novaStatus_t *st);

  /**
   * @brief Constructs a scalar tensor (0-d).
   *
   * @details
   * Allocates a single-element buffer via @ref create_scalar_tensor().
   *
   * @param[in]  dtype         Data type of the tensor elements.
   * @param[in]  device        Target device (CPU or GPU).
   * @param[in]  requires_grad Whether to track gradients for this tensor.
   * @param[in]  pin_memory    Whether to use page-locked (pinned) host memory.
   * @param[in,out] st         Status output; @ref novaSuccess on success.
   *
   * @see create_scalar_tensor()  Underlying C allocation.
   */
  TensorCXX(DType_ dtype, Device_ device, bool requires_grad, bool pin_memory,
            novaStatus_t *st);

  // ================================================================
  // Copy / Move
  // ================================================================

  /**
   * @brief Copy constructor. Performs a deep copy of the source tensor.
   *
   * @details
   * Allocates a new buffer via @ref create_tensor_like() and copies
   * the element data with @ref deepcopy().  The copy is fully
   * independent of @p ten.
   *
   * @param[in] ten The tensor to copy.
   */
  TensorCXX(const TensorCXX &ten);

  /**
   * @brief Copy assignment operator. Performs a deep copy.
   * @param[in] ten The tensor to copy.
   * @return Reference to @c this.
   */
  TensorCXX &operator=(const TensorCXX &ten);

  /**
   * @brief Move constructor. Transfers ownership from @p ten.
   * @param[in,out] ten The tensor to move from. Left in a valid empty state.
   */
  TensorCXX(TensorCXX &&ten) noexcept;

  /**
   * @brief Move assignment operator. Transfers ownership from @p ten.
   * @param[in,out] ten The tensor to move from. Left in a valid empty state.
   * @return Reference to @c this.
   */
  TensorCXX &operator=(TensorCXX &&ten) noexcept;

  // ================================================================
  // Element access
  // ================================================================

  /**
   * @brief Proxy for a single tensor element (read and write).
   *
   * @details
   * Returned by @ref at(). It supports assignment to the element and
   * implicit conversion to the element type. Works for both CPU
   * tensors (direct memory access) and GPU tensors (single-element
   * device<->host transfer). A plain reference cannot be returned
   * here: GPU elements are not host-addressable, and the compiler
   * rejects non-const references to some reduced types.
   *
   * @tparam type The element type (e.g. @c float, @c Half).
   */
  template <typename type> class ScalarRef {
  public:
    /**
     * @brief Constructs a proxy referencing a specific element.
     * @param[in,out] tensor The owning tensor.
     * @param[in]     index  Linear index of the element.
     */
    ScalarRef(TensorCXX &tensor, size_t index) noexcept;

    /**
     * @brief Reads the element value.
     * @return The element promoted to @p type.
     */
    operator type() const;

    /**
     * @brief Writes the element value.
     * @param[in] value The value to store.
     * @return Reference to @c this.
     */
    ScalarRef &operator=(type value);

    /**
     * @brief Element-wise copy from another @ref ScalarRef.
     * @param[in] other The source element.
     * @return Reference to @c this.
     */
    ScalarRef &operator=(const ScalarRef &other);

  private:
    TensorCXX *tensor_; ///< Pointer to the owning tensor.
    size_t index_;      ///< Linear index of the element.
  };

  /**
   * @brief Returns a proxy to the element at the given linear index.
   * @tparam type The element type to interpret the element as.
   * @param[in] index Linear index (row-major) into the flattened tensor.
   * @return A @ref ScalarRef that can read or write the element.
   */
  template <typename type> ScalarRef<type> at(size_t index);

  // ================================================================
  // Stream output and print options
  // ================================================================

  /**
   * @brief Stream output operator (debug representation).
   * @param[in,out] os  The output stream.
   * @param[in,out] ten The tensor to print.
   * @return Reference to @p os.
   */
  friend std::ostream &operator<<(std::ostream &os, TensorCXX &ten) {

    const char *repr = tensor_repr_debug(&ten.c_tensor);
    if (repr == nullptr) {
      os << "\033[31mError\033[0m: No tensor representation avaiable; Result: "
            "NULL\n";
      return os;
    }

    os << repr << "\n";
    std::free(const_cast<char *>(repr));

    return os;
  }

  /**
   * @brief Prints the tensor representation to @c std::cout.
   *
   * @details
   * Renders the tensor via the repr subsystem and writes the result
   * to @c std::cout followed by a newline.  When @p debug is @c true,
   * the debug representation (metadata included) is used; otherwise
   * the plain tensor representation is printed.
   *
   * @param[in] debug  If @c true, print the debug representation.
   *
   * @see tensor_repr_debug()  Debug representation.
   * @see tensor_repr()        Plain representation.
   */
  void print(bool debug = true) {

    const char *repr =
        debug ? tensor_repr_debug(&c_tensor) : tensor_repr(&c_tensor);
    if (repr == nullptr) {
      std::cout
          << "\033[31mError\033[0m: No tensor representation avaiable; Result: "
             "NULL\n";
    }

    std::cout << repr << "\n";
    std::free(const_cast<char *>(repr));
  }

  // ================================================================
  // Getters
  // ================================================================

  /**
   * @brief Returns the shape of the tensor.
   * @return Vector of dimension sizes.
   */
  std::vector<size_t> getShape() noexcept;

  /**
   * @brief Returns the strides of the tensor.
   * @return Vector of stride values (in bytes, not elements).
   */
  [[nodiscard]] std::vector<size_t> getStrides() const noexcept;

  /**
   * @brief Returns the total number of elements (product of shape dims).
   * @return Element count.
   */
  [[nodiscard]] size_t getSize() const noexcept;

  /**
   * @brief Returns the number of dimensions (rank).
   * @return Dimension count.
   */
  [[nodiscard]] size_t dims() const noexcept;

  /**
   * @brief Returns the size in bytes of a single element.
   * @return Byte count per element.
   */
  [[nodiscard]] size_t itemSize() const noexcept;

  /**
   * @brief Returns the logical size in bytes (product of shape × item size).
   * @return Logical byte count.
   */
  [[nodiscard]] size_t logicalSize() const noexcept;

  /**
   * @brief Returns the device this tensor resides on.
   * @return @ref Device_ enumerator value.
   */
  [[nodiscard]] Device_ getDevice() const noexcept;

  /**
   * @brief Returns the data type of this tensor.
   * @return @ref DType_ enumerator value.
   */
  [[nodiscard]] DType_ getDType() const noexcept;

  /**
   * @brief Returns the underlying C tensor (by value copy).
   * @return A copy of the internal @ref Tensor structure.
   */
  Tensor getCTensor() noexcept;

  // ================================================================
  // Device / dtype transfer helpers
  // ================================================================

  /**
   * @brief Transfers the tensor to a different device.
   *
   * @details
   * Allocates a new tensor on the target device and copies the data
   * through the C transfer API.  Only CPU ↔ GPU transfers are
   * supported; any other combination fails with
   * @ref novaInvalidTransfDirection.
   *
   * @param[in]  device Target device (must be the opposite of the
   *                    current one).
   * @param[out] st     Status output; @ref novaSuccess on success.
   * @return A new tensor on the target device.
   *
   * @pre  The current device and @p device must be one of the pairs
   *       @c DEVICE_CPU → @c DEVICE_GPU or @c DEVICE_GPU → @c DEVICE_CPU.
   *
   * @see transf_tensor_from_host()    CPU → GPU transfer.
   * @see transf_tensor_from_device()  GPU → CPU transfer.
   */
  TensorCXX to(Device_ device, novaStatus_t &st) noexcept;

  /**
   * @brief Returns a new tensor with the given dtype (same device).
   *
   * @details
   * Allocates a new tensor on the same device and casts the elements
   * via @ref cast().  When @p dtype equals the current dtype, a copy
   * of the tensor is returned unchanged.
   *
   * @param[in]  dtype Target data type.
   * @param[out] st    Status output; @ref novaSuccess on success.
   * @return A new tensor with elements cast to @p dtype.
   *
   * @see cast()  Underlying C casting routine.
   */
  [[nodiscard]] TensorCXX to(DType_ dtype, novaStatus_t &st) const noexcept;

  /**
   * @brief Transfers the tensor from CPU to GPU.
   *
   * @details
   * Convenience wrapper over @ref to(Device_, novaStatus_t&) with the
   * target fixed to @c DEVICE_GPU.  The tensor must currently reside
   * on CPU.
   *
   * @param[out] st Status output; @ref novaSuccess on success.
   * @return A new tensor on the CUDA device.
   *
   * @pre  The tensor must reside on CPU (@c DEVICE_CPU).
   */
  TensorCXX cuda(novaStatus_t &st) noexcept;

private:
  Tensor c_tensor{};         ///< Underlying C tensor handle.
  std::vector<size_t> shape; ///< Shape (number of elements per dim).
  data_ptr data;             ///< Pointer to the raw data buffer.
  size_t size;               ///< Total element count.
  size_t ndims;              ///< Number of dimensions.
  size_t item_size;          ///< Bytes per element.
  size_t logical_size;       ///< Logical byte count (shape × item_size).
};

// ================================================================
// Constructors — inline definitions
// ================================================================

/// @brief Default constructor. Delegates to the C core default.
inline TensorCXX::TensorCXX() = default;

/// @brief Allocates the backing buffer and mirrors the C tensor metadata.
inline TensorCXX::TensorCXX(const std::vector<size_t> &shape, DType_ dtype,
                            Device_ device, bool requires_grad, bool pin_memory,
                            novaStatus_t *st) {
  shape_t local_shape = {0};
  for (size_t i = 0; i < shape.size(); ++i) {
    local_shape[i] = shape[i];
  }
  c_tensor = create_tensor(local_shape, dtype, device, requires_grad,
                           pin_memory, shape.size(), st);

  this->shape = shape;
  size = c_tensor.size;
  ndims = c_tensor.ndims;
  item_size = c_tensor.item_size;
  logical_size = c_tensor.logical_size;
  data = c_tensor.data;
}

/// @brief Releases the underlying tensor buffer via the C core.
inline TensorCXX::~TensorCXX() { collect(&c_tensor); }

inline TensorCXX::TensorCXX(std::initializer_list<size_t> shape, DType_ dtype,
                            Device_ device, bool requires_grad, bool pin_memory,
                            novaStatus_t *st)
    : TensorCXX(std::vector<size_t>(shape), dtype, device, requires_grad,
                pin_memory, st) {}

/// @brief Deep-copy constructor. Allocates a new buffer and copies data.
inline TensorCXX::TensorCXX(const TensorCXX &ten) {
  novaStatus_t st = {};
  Tensor tmp = create_tensor_like(&ten.c_tensor, &st);
  if (st.err == novaSuccess) {
    deepcopy(&ten.c_tensor, &tmp, &st);
    move_tensor(&c_tensor, &tmp);
  }
  shape = ten.shape;
  size = c_tensor.size;
  ndims = c_tensor.ndims;
  item_size = c_tensor.item_size;
  logical_size = c_tensor.logical_size;
  data = c_tensor.data;
}

/// @brief Constructs a scalar (0-d) tensor with the given dtype and device.
inline TensorCXX::TensorCXX(DType_ dtype, Device_ device, bool requires_grad,
                            bool pin_memory, novaStatus_t *st) {
  Tensor ten =
      create_scalar_tensor(dtype, device, requires_grad, pin_memory, st);

  move_tensor(&c_tensor, &ten);
  shape = {0};
  size = c_tensor.size;
  ndims = c_tensor.ndims;
  item_size = c_tensor.item_size;
  logical_size = c_tensor.logical_size;
  data = c_tensor.data;
}

/// @brief Copy assignment. Uses the copy-and-swap idiom for exception safety.
inline TensorCXX &TensorCXX::operator=(const TensorCXX &ten) {
  if (this == &ten) {
    return *this;
  }
  TensorCXX tmp(ten);
  std::swap(c_tensor, tmp.c_tensor);
  std::swap(shape, tmp.shape);
  std::swap(size, tmp.size);
  std::swap(ndims, tmp.ndims);
  std::swap(item_size, tmp.item_size);
  std::swap(logical_size, tmp.logical_size);
  std::swap(data, tmp.data);
  return *this;
}

/// @brief Move constructor. Transfers ownership; source is left empty.
inline TensorCXX::TensorCXX(TensorCXX &&ten) noexcept {
  move_tensor(&c_tensor, &ten.c_tensor);
  shape = std::move(ten.shape);
  size = ten.size;
  ndims = ten.ndims;
  item_size = ten.item_size;
  logical_size = ten.logical_size;
  data = ten.data;

  ten.size = 0;
  ten.ndims = 0;
  ten.item_size = 0;
  ten.logical_size = 0;
  ten.data.data = nullptr;
}

/// @brief Move assignment. Transfers ownership; source is left empty.
inline TensorCXX &TensorCXX::operator=(TensorCXX &&ten) noexcept {
  move_tensor(&c_tensor, &ten.c_tensor);
  shape = std::move(ten.shape);
  size = ten.size;
  ndims = ten.ndims;
  item_size = ten.item_size;
  logical_size = ten.logical_size;
  data = ten.data;

  ten.size = 0;
  ten.ndims = 0;
  ten.item_size = 0;
  ten.logical_size = 0;
  ten.data.data = nullptr;
  return *this;
}

// ================================================================
// Getter functions — inline definitions
// ================================================================

/// @copydoc TensorCXX::getShape()
inline std::vector<size_t> TensorCXX::getShape() noexcept { return shape; }

/// @copydoc TensorCXX::getStrides()
inline std::vector<size_t> TensorCXX::getStrides() const noexcept {
  std::vector<size_t> strides(ndims);
  for (size_t idx = 0; idx < ndims; ++idx) {
    strides[idx] = c_tensor.strides[idx];
  }
  return strides;
}

/// @copydoc TensorCXX::getSize()
inline size_t TensorCXX::getSize() const noexcept { return size; }

/// @copydoc TensorCXX::dims()
inline size_t TensorCXX::dims() const noexcept { return ndims; }

/// @copydoc TensorCXX::itemSize()
inline size_t TensorCXX::itemSize() const noexcept { return item_size; }

/// @copydoc TensorCXX::logicalSize()
inline size_t TensorCXX::logicalSize() const noexcept { return logical_size; }

/// @copydoc TensorCXX::getCTensor()
inline Tensor TensorCXX::getCTensor() noexcept { return c_tensor; }

/// @copydoc TensorCXX::getDevice()
inline Device_ TensorCXX::getDevice() const noexcept { return c_tensor.device; }

/// @copydoc TensorCXX::getDType()
inline DType_ TensorCXX::getDType() const noexcept { return c_tensor.dtype; }

// ================================================================
// ScalarRef — inline definitions
// ================================================================

/// @brief Constructs the proxy with a tensor reference and linear index.
template <typename type>
inline TensorCXX::ScalarRef<type>::ScalarRef(TensorCXX &tensor,
                                             size_t index) noexcept
    : tensor_(&tensor), index_(index) {}

/// @brief Implicit read: copies the element to the host and promotes to @p
/// type.
template <typename type>
inline TensorCXX::ScalarRef<type>::operator type() const {
  type value{};
  if (tensor_->getDevice() == DEVICE_CPU) {
    value = reinterpret_cast<type *>(tensor_->c_tensor.data.data)[index_];
  } else if (tensor_->getDevice() == DEVICE_GPU) {
    novaStatus_t st =
        transfer_to(DEVICE_GPU, DEVICE_CPU,
                    tensor_->c_tensor.data.data + (index_ * tensor_->item_size),
                    &value, sizeof(type));
    if (st.err != novaSuccess) {
      value = static_cast<type>(0);
    }
  }
  return value;
}

/// @brief Writes the element, transferring from host to the target device.
template <typename type>
inline TensorCXX::ScalarRef<type> &
TensorCXX::ScalarRef<type>::operator=(type value) {
  if (tensor_->getDevice() == DEVICE_CPU) {
    reinterpret_cast<type *>(tensor_->c_tensor.data.data)[index_] = value;
  } else if (tensor_->getDevice() == DEVICE_GPU) {
    novaStatus_t st =
        transfer_to(DEVICE_CPU, DEVICE_GPU, &value,
                    tensor_->c_tensor.data.data + (index_ * tensor_->item_size),
                    sizeof(type));
    // No return channel; a failed transfer leaves the element unchanged
    // and is observable on the next read.
    (void)st;
  }
  return *this;
}

/// @brief Element-wise copy from another @ref ScalarRef.
template <typename type>
inline TensorCXX::ScalarRef<type> &
TensorCXX::ScalarRef<type>::operator=(const ScalarRef &other) {
  return operator=(static_cast<type>(other));
}

/// @brief Returns a proxy to the element at the given linear index.
template <typename type>
inline TensorCXX::ScalarRef<type> TensorCXX::at(size_t index) {
  if (getDevice() == DEVICE_META) {
    throw std::runtime_error("A meta tensor cannot be indexed");
  }
  return ScalarRef<type>(*this, index);
}

// ================================================================
// Helpers — inline definitions
// ================================================================

/// @brief Transfers the tensor to the specified device (CPU ↔ GPU).
inline TensorCXX TensorCXX::to(Device_ device, novaStatus_t &st) noexcept {
  TensorCXX ten;
  const auto current_device = getDevice();

  if (current_device == DEVICE_GPU && device == DEVICE_CPU) {
    TensorCXX out(shape, getDType(), device, c_tensor.requires_grad_, false,
                  &st);
    if (st.err != novaSuccess) {
      return out;
    }
    st = transf_tensor_from_device(&c_tensor, &out.c_tensor);
    return out;
  } else if (current_device == DEVICE_CPU && device == DEVICE_GPU) {
    TensorCXX out(shape, getDType(), device, c_tensor.requires_grad_, false,
                  &st);
    if (st.err != novaSuccess) {
      return out;
    }
    st = transf_tensor_from_host(&c_tensor, &out.c_tensor);
    return out;
  } else {
    st.err = novaInvalidTransfDirection;
    st.message = nova_get_error_msg(st.err, nullptr);
  }

  return ten;
}

/// @brief Casts the tensor to the given dtype, returning a new tensor.
inline TensorCXX TensorCXX::to(DType_ dtype, novaStatus_t &st) const noexcept {
  if (dtype != getDType()) {
    TensorCXX out(this->shape, dtype, getDevice(), c_tensor.requires_grad_,
                  c_tensor.is_pinned_, &st);
    if (st.err == novaSuccess) {
      cast(&c_tensor, &out.c_tensor, dtype);
    }
    return out;
  }
  return *this;
}

/// @brief Convenience wrapper: transfers the tensor from CPU to CUDA.
inline TensorCXX TensorCXX::cuda(novaStatus_t &st) noexcept {
  const auto device = getDevice();
  if (device != DEVICE_CPU) {
    st.err = novaInvalidTransfDirection;
    st.message = nova_get_error_msg(st.err, nullptr);
    return {};
  }
  TensorCXX out(shape, getDType(), DEVICE_GPU, c_tensor.requires_grad_, false,
                &st);

  if (st.err != novaSuccess) {
    return out;
  }
  st = transf_tensor_from_host(&c_tensor, &out.c_tensor);
  return out;
}
} // namespace ncore::wrappers
