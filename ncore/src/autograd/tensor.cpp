#include <autograd/tensor.hpp>
#include <cstring>
#include <ncore/copy.h>

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

/* ---- Constructors & Destructor ---- */

/**
 * @brief Default constructor. Creates an empty, unallocated tensor.
 */
autograd::Tensor::Tensor() noexcept {
  std::memset(&c_tensor_, 0, sizeof(c_tensor_));
  sync_metadata_();
}

/**
 * @brief Construct an allocated tensor.
 *
 * Initialises shape, strides, and size metadata from the given shape
 * vector, allocates a data buffer on the specified device, and
 * synchronises the C++ shape/strides cache.
 *
 * @param shape         Dimension sizes.
 * @param dtype         Element data type.
 * @param device        Target device (CPU, GPU, or META).
 * @param requires_grad If true, an unallocated gradient tensor is created.
 */
autograd::Tensor::Tensor(const std::vector<size_t> &shape, DType_ dtype,
                         Device device, bool requires_grad)
    : c_tensor_(create_tensor(shape.data(), dtype, device, requires_grad,
                              shape.size())) {
  sync_metadata_();
}

/**
 * @brief Construct an allocated tensor from an initializer list.
 *
 * Convenience overload that delegates to the vector constructor.
 *
 * @param shape         Dimension sizes.
 * @param dtype         Element data type.
 * @param device        Target device (CPU, GPU, or META).
 * @param requires_grad If true, an unallocated gradient tensor is created.
 */
autograd::Tensor::Tensor(std::initializer_list<size_t> shape, DType_ dtype,
                         Device device, bool requires_grad)
    : Tensor(std::vector<size_t>(shape), dtype, device, requires_grad) {}

/**
 * @brief Destructor. Recursively releases storage and gradient sub-graph.
 */
autograd::Tensor::~Tensor() { collect(&c_tensor_); }

/* ---- Rule of Five ---- */

/**
 * @brief Copy constructor. Performs a deep copy of data and metadata.
 *
 * Allocates new storage for the destination via deepcopy(), copies all
 * element data, and recursively deep-copies the gradient graph.  The
 * C++ shape/strides cache is synchronised from the copied C tensor.
 *
 * @param other Source tensor to copy.
 */
autograd::Tensor::Tensor(const Tensor &other) {
  std::memset(&c_tensor_, 0, sizeof(c_tensor_));
  deepcopy(&other.c_tensor_, &c_tensor_);
  sync_metadata_();
}

/**
 * @brief Move constructor. Transfers ownership; leaves source empty.
 *
 * Uses move_tensor() to transfer the underlying C storage and metadata,
 * then zeroes the source C tensor and re-synchronises both C++ caches.
 *
 * @param other Source tensor to move from (reset to empty afterwards).
 */
autograd::Tensor::Tensor(Tensor &&other) noexcept {
  move_tensor(&c_tensor_, &other.c_tensor_);
  std::memset(&other.c_tensor_, 0, sizeof(other.c_tensor_));
  sync_metadata_();
  other.sync_metadata_();
}

/**
 * @brief Copy assignment. Releases current resources and deep-copies.
 *
 * Frees existing storage via collect(), zeroes the struct, performs a
 * deep copy, and re-synchronises the shape/strides cache.
 *
 * @param other Source tensor to copy.
 * @return Reference to this tensor.
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
 * @brief Move assignment. Releases current resources and transfers ownership.
 *
 * Frees existing storage via collect(), transfers the source tensor's
 * C data with move_tensor(), zeroes source, and re-synchronises both
 * C++ caches.
 *
 * @param other Source tensor to move from (reset to empty afterwards).
 * @return Reference to this tensor.
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

/* ---- Query predicates ---- */

/**
 * @brief Get the tensor's shape vector.
 *
 * @return Const reference to the cached shape (vector of dimension sizes).
 */
const Shape &autograd::Tensor::get_shape() const noexcept { return shape_; }

/**
 * @brief Get the tensor's strides vector.
 *
 * @return Const reference to the cached strides (vector of byte strides).
 */
const Strides &autograd::Tensor::get_strides() const noexcept {
  return strides_;
}

/* ---- C API interop ---- */

/**
 * @brief Access the underlying C Tensor (read-only).
 *
 * @return Const reference to the raw C `::Tensor` struct.
 */
[[nodiscard]] const ::Tensor &autograd::Tensor::c_tensor() const noexcept {
  return c_tensor_;
}

/**
 * @brief Access the underlying C Tensor (mutable, for FFI calls).
 *
 * @return Mutable reference to the raw C `::Tensor` struct.
 */
::Tensor &autograd::Tensor::c_tensor() noexcept { return c_tensor_; }
