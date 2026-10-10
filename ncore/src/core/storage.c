/**
 * @file storage.c
 * @brief Size-class rounding and tensor byte-size queries.
 *
 * @details
 * Device reservations round the theoretical size to a 512-byte step up to
 * 10 MiB and to a shared 2 MiB block step above.  Pinned host reservations
 * round to a 4096-byte step up to 16 KiB and to geometric classes
 * (1x, 1.25x, 1.5x, 1.75x of each power of two) above, bounding waste near
 * 25%.  Plain host reservations keep the requested size.  Rounding applies
 * only while a device backend is active.
 *
 * @see storage.h  Public declarations and struct definitions.
 */

#include <stddef.h>
#include <stdint.h>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/tensor.h>

static constexpr size_t device_large_threshold = ((size_t)10 << 20);
static constexpr size_t pinned_linear_limit = ((size_t)16 << 10);
static constexpr size_t device_small_step = 512;
static constexpr size_t pinned_small_step = 4096;
static constexpr size_t large_step = ((size_t)2 << 20);

_Static_assert((device_small_step & (device_small_step - 1)) == 0,
               "device step must be a power of two");
_Static_assert((pinned_small_step & (pinned_small_step - 1)) == 0,
               "pinned step must be a power of two");
_Static_assert((large_step & (large_step - 1)) == 0,
               "large step must be a power of two");

/**
 * @brief Round @p n up to a multiple of @p step.
 *
 * @param[in]  n       Requested count.  Must be greater than zero.
 * @param[in]  step    Power-of-two step to round to.
 * @param[out] status  Receives the operation result.
 *
 * @return Rounded count, or @c 0 with an error status on zero input
 *         or overflow.
 */
static size_t align_up(size_t n, size_t step, novaStatus_t *status) {
  if (n == 0) {
    status->err = novaInvalidValue;
    status->message = nova_get_error_msg(status->err, nullptr);
    return 0;
  }
  const size_t add = step - 1;
  if (n > SIZE_MAX - add) {
    status->err = novaInvalidValue;
    status->message = nova_get_error_msg(status->err, nullptr);
    return 0;
  }
  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return (n + add) & ~add;
}

/**
 * @brief Round @p n up to the next geometric class.
 *
 * @details
 * Classes are 1x, 1.25x, 1.5x and 1.75x of each power of two, computed
 * with integer shifts only.  The result always covers @p n with waste
 * bounded near 25%.
 *
 * @param[in]  n       Requested count.  Must be greater than zero.
 * @param[out] status  Receives the operation result.
 *
 * @return Rounded count, or @c 0 with an error status on zero input
 *         or when no class fits in @c size_t.
 */
static size_t geometric_up(size_t n, novaStatus_t *status) {
  if (n == 0) {
    status->err = novaInvalidValue;
    status->message = nova_get_error_msg(status->err, nullptr);
    return 0;
  }
  size_t base = 1;
  for (;;) {
    const size_t cap = base + (base >> 1) + (base >> 2);
    if (cap >= n) {
      break;
    }
    if (base > SIZE_MAX / 2) {
      status->err = novaInvalidValue;
      status->message = nova_get_error_msg(status->err, nullptr);
      return 0;
    }
    base *= 2;
  }
  const size_t quarter = base >> 2;
  const size_t half = base >> 1;
  size_t out = base;
  if (n > base) {
    out = base + quarter;
  }
  if (n > out) {
    out = base + half;
  }
  if (n > out) {
    out = base + half + quarter;
  }
  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return out;
}

/**
 * @brief Report whether a device backend is usable in this process.
 *
 * @return @c true when CUDA or HIP reports an available device.
 */
static inline bool device_active() {
  return (is_cuda_available() || is_hip_available()) != 0;
}

size_t round_up_device_small(size_t n, novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  return align_up(n, device_small_step, status);
}

size_t round_up_device_large(size_t n, novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  return align_up(n, large_step, status);
}

size_t round_up_pinned_small(size_t n, novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  return align_up(n, pinned_small_step, status);
}

size_t round_up_pinned_large(size_t n, novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  return geometric_up(n, status);
}

size_t resolve_rounded_bytes(size_t requested, bool is_device, bool is_pinned,
                             novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  if (requested == 0) {
    status->err = novaInvalidValue;
    status->message = nova_get_error_msg(status->err, nullptr);
    return 0;
  }
  if ((!is_device && !is_pinned) || !device_active()) {
    status->err = novaSuccess;
    status->message = nova_get_error_msg(status->err, nullptr);
    return requested;
  }
  if (is_pinned) {
    if (requested <= pinned_linear_limit) {
      return round_up_pinned_small(requested, status);
    }
    return round_up_pinned_large(requested, status);
  }
  if (requested <= device_large_threshold) {
    return round_up_device_small(requested, status);
  }
  return round_up_device_large(requested, status);
}

size_t gt_bytes(const Tensor *ten, novaStatus_t *status) {
  if (status == nullptr) {
    return 0;
  }
  if (ten == nullptr) {
    status->err = novaInvalidPointer;
    status->message = nova_get_error_msg(status->err, nullptr);
    return 0;
  }
  if (ten->storage == nullptr || ten->storage->handle.id == 0 ||
      ten->storage->handle.bytes == 0) {
    status->err = novaInvalidTensor;
    status->message = nova_get_error_msg(status->err, nullptr);
    return 0;
  }
  status->err = novaSuccess;
  status->message = nova_get_error_msg(status->err, nullptr);
  return ten->storage->handle.bytes;
}
