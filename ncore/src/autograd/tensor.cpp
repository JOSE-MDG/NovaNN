/**
 * @file tensor.cpp
 * @brief Implementation of the autograd::Tensor RAII wrapper.
 *
 * @details
 * Every method delegates to the C API defined in `tensor.h` and
 * `copy.h`.  The C++ class adds no new logic beyond lifecycle
 * management and the dual shape/strides cache.
 *
 * ## Cache synchronisation
 *
 * The `sync_metadata_()` helper clears and repopulates the C++
 * vectors from the C tensor's fixed-size arrays.  It is called
 * after every operation that may change the C tensor's shape or
 * strides.
 *
 * @see tensor.hpp     Class declaration.
 * @see ncore/tensor.h C Tensor API.
 * @see ncore/copy.h   deepcopy(), move_tensor().
 */

#include <autograd/tensor.hpp>
#include <cstring>
#include <ncore/copy.h>

/**
 * @brief Synchronise the cached `shape_` and `strides_` from
 *        `c_tensor_`.
 *
 * @details
 * Clears both vectors, reserves capacity for `c_tensor_.ndims`
 * elements, and copies each dimension's size and byte stride
 * from the C tensor's fixed arrays into the C++ vectors.
 *
 * Called after every constructor, assignment operator, and move
 * operation.
 *
 * @post `shape_.size() == c_tensor_.ndims`.
 * @post `strides_.size() == c_tensor_.ndims`.
 * @post Each element matches the corresponding C tensor entry.
 */
void autograd::Tensor::sync_metadata_() noexcept {
  shape_.clear();
  strides_.clear();

  const size_t ndims = c_tensor_.ndims;
  shape_.reserve(ndims);
  strides_.reserve(ndims);
  for (size_t i = 0; i < ndims; i++) {
    shape_.push_back(c_tensor_.shape[i]);
    strides_.push_back(c_tensor_.strides[i]);
  }
}

/**
 * @brief Default constructor.  Creates an empty, unallocated tensor.
 *
 * @details
 * Zero-initialises the C tensor via `memset`, then synchronises
 * the C++ cache (resulting in empty `shape_` and `strides_`
 * vectors).
 *
 * @post `c_tensor_` is all-zeros; `is_allocated_` is false.
 * @post `shape_` and `strides_` are empty.
 */
autograd::Tensor::Tensor() noexcept {
  std::memset(&c_tensor_, 0, sizeof(c_tensor_));
  sync_metadata_();
}

/**
 * @brief Construct an allocated tensor.
 *
 * @details
 * Delegates to `create_tensor()` from the C API, which:
 * 1. Zero-initialises the C tensor.
 * 2. Copies shape metadata and computes strides.
 * 3. Allocates the backing buffer via `allocate_tensor_buffer()`.
 * 4. Optionally creates an unallocated gradient tensor.
 *
 * The C++ shape/strides cache is then synchronised from the
 * resulting C tensor.
 *
 * @param[in]  shape         Dimension sizes.
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device (`DEVICE_CPU`,
 *                           `DEVICE_GPU`, or `DEVICE_META`).
 * @param[in]  requires_grad If `true`, an unallocated gradient
 *                           tensor is created in `c_tensor_.grad`.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory (CPU only).
 *
 * @post `c_tensor_` is fully allocated (unless META).
 * @post `shape_` and `strides_` match the C tensor.
 *
 * @see create_tensor()  Underlying C allocation.
 */
autograd::Tensor::Tensor(const shape_t &shape, DType_ dtype, Device device,
                         bool requires_grad, bool pin_memory)
    : c_tensor_(create_tensor(shape.data(), dtype, device, requires_grad,
                              pin_memory, shape.size())) {
  sync_metadata_();
}

/**
 * @brief Construct an allocated tensor from an `initializer_list`.
 *
 * @details
 * Converts the `initializer_list` to a `shape_t` and delegates
 * to the shape_t constructor.  This allows concise tensor
 * creation:
 *
 * @code{.cpp}
 * Tensor t({2, 3, 4}, Float32, DEVICE_CPU);
 * @endcode
 *
 * @param[in]  shape         Dimension sizes.
 * @param[in]  dtype         Element data type (@ref DType_).
 * @param[in]  device        Target device.
 * @param[in]  requires_grad If `true`, an unallocated gradient
 *                           tensor is created.
 * @param[in]  pin_memory    If `true`, request page-locked host
 *                           memory.
 */
autograd::Tensor::Tensor(std::initializer_list<size_t> shape, DType_ dtype,
                         Device device, bool requires_grad, bool pin_memory)
    : Tensor(shape_t(shape), dtype, device, requires_grad, pin_memory) {}

/**
 * @brief Destructor.  Recursively releases storage and gradient
 *        sub-graph.
 *
 * @details
 * Calls `collect()` on the C tensor, which:
 * 1. Decrements the Rust reference count via `release()`.
 * 2. If the count reaches zero, frees the `TensorStorage`.
 * 3. Recursively collects and frees the gradient sub-graph.
 *
 * Safe to call on an unallocated tensor (no-op).
 */
autograd::Tensor::~Tensor() { collect(&c_tensor_); }

/**
 * @brief Copy constructor.  Performs a deep copy of data and
 *        metadata.
 *
 * @details
 * 1. Zero-initialises the destination C tensor.
 * 2. Calls `deepcopy()` to allocate new storage and copy all
 *    elements, including a recursive deep copy of the gradient
 *    graph.
 * 3. Synchronises the C++ shape/strides cache.
 *
 * @param[in] other Source tensor to copy.
 *
 * @post `this->c_tensor_` owns independent storage (not shared).
 * @post `this->shape_` and `this->strides_` match `other`.
 *
 * @see deepcopy()  Underlying C deep copy.
 */
autograd::Tensor::Tensor(const Tensor &other) {
  std::memset(&c_tensor_, 0, sizeof(c_tensor_));
  deepcopy(&other.c_tensor_, &c_tensor_);
  sync_metadata_();
}

/**
 * @brief Move constructor.  Transfers ownership; leaves source
 *        empty.
 *
 * @details
 * 1. Calls `move_tensor()` to transfer the C tensor's storage and
 *    metadata.
 * 2. Zeroes the source C tensor (so its destructor is a no-op).
 * 3. Synchronises both C++ caches.
 *
 * @param[in,out] other Source tensor to move from.  Reset to
 *                      empty afterwards.
 *
 * @post `other` is in a valid but unallocated state.
 * @post `other.shape_` and `other.strides_` are empty.
 *
 * @see move_tensor()  Underlying C move.
 */
autograd::Tensor::Tensor(Tensor &&other) noexcept {
  move_tensor(&c_tensor_, &other.c_tensor_);
  std::memset(&other.c_tensor_, 0, sizeof(other.c_tensor_));
  sync_metadata_();
  other.sync_metadata_();
}

/**
 * @brief Copy assignment.  Releases current resources and
 *        deep-copies.
 *
 * @details
 * Self-assignment safe (checked via pointer comparison).  Steps:
 * 1. `collect()` existing resources.
 * 2. Zero the C tensor.
 * 3. `deepcopy()` from @p other.
 * 4. Synchronise the C++ cache.
 *
 * @param[in] other Source tensor to copy.
 * @return Reference to `*this`.
 *
 * @post `this->c_tensor_` owns independent storage.
 */
autograd::Tensor &autograd::Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    collect(&c_tensor_);
    std::memset(&c_tensor_, 0, sizeof(c_tensor_));
    deepcopy(&other.c_tensor_, &c_tensor_);
    sync_metadata_();
  }
  return *this;
}

/**
 * @brief Move assignment.  Releases current resources and
 *        transfers ownership.
 *
 * @details
 * Self-assignment safe.  Steps:
 * 1. `collect()` existing resources.
 * 2. `move_tensor()` from @p other.
 * 3. Zero the source C tensor.
 * 4. Synchronise both C++ caches.
 *
 * @param[in,out] other Source tensor to move from.  Reset to
 *                      empty afterwards.
 * @return Reference to `*this`.
 *
 * @post `other` is in a valid but unallocated state.
 */
autograd::Tensor &autograd::Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    collect(&c_tensor_);
    move_tensor(&c_tensor_, &other.c_tensor_);
    std::memset(&other.c_tensor_, 0, sizeof(other.c_tensor_));
    sync_metadata_();
    other.sync_metadata_();
  }
  return *this;
}

/**
 * @brief Get the tensor's shape (read-only).
 *
 * @return Const reference to the cached shape (dimension sizes).
 *
 * @see get_strides()
 */
const autograd::shape_t &autograd::Tensor::get_shape() const noexcept {
  return shape_;
}

/**
 * @brief Get the tensor's shape (mutable).
 *
 * @return Mutable reference to the cached shape.
 *
 * @see get_strides()
 */
autograd::shape_t &autograd::Tensor::get_shape() noexcept { return shape_; }

/**
 * @brief Get the tensor's strides (read-only).
 *
 * @return Const reference to the cached strides (byte strides).
 *
 * @see get_shape()
 */
const autograd::strides_t &autograd::Tensor::get_strides() const noexcept {
  return strides_;
}

/**
 * @brief Get the tensor's strides (mutable).
 *
 * @return Mutable reference to the cached strides.
 *
 * @see get_shape()
 */
autograd::strides_t &autograd::Tensor::get_strides() noexcept {
  return strides_;
}

/**
 * @brief Access the underlying C Tensor (read-only).
 *
 * @details
 * Provides read-only access to the raw C struct for interop with
 * C APIs that accept `const Tensor *`.
 *
 * @return Const reference to the raw C `::Tensor` struct.
 *
 * @see c_tensor() (mutable)
 */
[[nodiscard]] const ::Tensor &autograd::Tensor::c_tensor() const noexcept {
  return c_tensor_;
}

/**
 * @brief Access the underlying C Tensor (mutable, for FFI calls).
 *
 * @details
 * Provides mutable access to the raw C struct for interop with
 * C APIs that accept `Tensor *` (e.g., in-place operations).
 *
 * @return Mutable reference to the raw C `::Tensor` struct.
 *
 * @see c_tensor() (const)
 */
::Tensor &autograd::Tensor::c_tensor() noexcept { return c_tensor_; }
