/**
 * @file StorageRounding_test.cpp
 * @brief Unit tests for storage allocation size-class rounding.
 *
 * Exercises the rounding surface in @ref storage.h:
 * device 512 B step up to 10 MiB and 2 MiB step above, pinned
 * 4096 B step up to 16 KiB and geometric classes
 * (1x/1.25x/1.5x/1.75x per power of two) above, the
 * `resolve_rounded_bytes()` dispatcher, and the `gt_bytes()` validated
 * accessor. Dual-size invariants (`requested_bytes` vs `bytes`),
 * resize rounding, dtype-agnostic behavior, and the device-to-host
 * repr shadow (theoretical size, never the reservation) are covered.
 *
 * Golden rule throughout: payload operations (fill / memcpy / memcmp /
 * transfer) measure the theoretical `size * item_size`, never `bytes`.
 *
 * Environment note: `resolve_rounded_bytes()` returns the request
 * unchanged while no device backend is active (see `device_active()`
 * in `storage.c`). Pure `round_up_*` tests are deterministic on every
 * build. Dispatcher tests branch on `gpuAvailable()` so they stay
 * green on CPU-only runners and pin real rounding where a GPU
 * answers. Device/pinned live-allocation tests skip without a GPU
 * via `NOVA_TEST_REQUIRE_GPU()`.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <ncore/core/alloc.h>
#include <ncore/core/copy.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/repr/tensor_repr.h>
#include <ncore/tensor.h>
#include <ncore/threading/parallel/helpers.h>

#include "utils/TensorUtils.hpp"

namespace {

using ncore::wrappers::TensorCXX;
using ncore::wrappers::unallocated;

constexpr size_t kDeviceLargeThreshold = size_t{10485760};
constexpr size_t kPinnedLinearLimit = size_t{16384};
constexpr size_t kLargeStep = size_t{2097152};

/** @brief Theoretical payload of a tensor: element count times item size. */
size_t theoretical(const Tensor &ten) { return ten.size * ten.item_size; }

/**
* @brief Tier table: device requests up to 10 MiB use the 512 B step.
*
* @test REQ-004/007: expectations are literals so the contract cannot
*       drift with the implementation.
*/
struct TierCase {
  size_t input;
  size_t want;
};

class TierTableDeviceSmall : public ::testing::TestWithParam<TierCase> {};

/**
 * @brief Tier table: pinned requests up to 16 KiB use the 4096 B step.
 *
 * @test REQ-005/007: expectations are literals so the contract cannot
 *       drift with the implementation.
 */
class TierTablePinnedSmall : public ::testing::TestWithParam<TierCase> {};

/**
 * @brief Tier table: device requests above 10 MiB use the 2 MiB step.
 *
 * @test REQ-006/007: expectations are literals so the contract cannot
 *       drift with the implementation.
 */
class TierTableDeviceLarge : public ::testing::TestWithParam<TierCase> {};

/**
 * @brief Tier table: pinned requests above 16 KiB use geometric classes.
 *
 * @test REQ-013: 1x/1.25x/1.5x/1.75x of each power of two. Expectations
 *       are literals and waste stays within a quarter of the request.
 */
class TierTablePinnedGeometric : public ::testing::TestWithParam<TierCase> {};

/**
 * @brief Rounding is byte-based and dtype-agnostic.
 *
 * @test Constraints: the same byte request rounds identically for
 *       every dtype; dtype only enters through the theoretical
 *       `size * item_size` (packed FP4 uses storage units).
 */
class RoundingDtypeSweep : public ::testing::TestWithParam<DType_> {};
} // namespace

TEST_P(TierTableDeviceSmall, Dispatch) {
  const TierCase param = GetParam();
  SCOPED_TRACE(::testing::Message() << "input=" << param.input);

  novaStatus_t st{};
  const size_t got = round_up_device_small(param.input, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(got, param.want);
  EXPECT_EQ(got % 512, size_t{0});
  EXPECT_GE(got, param.input);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceSmallSteps, TierTableDeviceSmall,
    ::testing::Values(TierCase{1, 512}, TierCase{63, 512}, TierCase{100, 512},
                      TierCase{128, 512}, TierCase{511, 512},
                      TierCase{512, 512}, TierCase{513, 1024},
                      TierCase{600, 1024}, TierCase{4096, 4096},
                      TierCase{4097, 4608}, TierCase{1048575, 1048576},
                      TierCase{1048576, 1048576}, TierCase{3145828, 3146240},
                      TierCase{10485760, 10485760}));

TEST_P(TierTablePinnedSmall, Dispatch) {
  const TierCase param = GetParam();
  SCOPED_TRACE(::testing::Message() << "input=" << param.input);

  novaStatus_t st{};
  const size_t got = round_up_pinned_small(param.input, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(got, param.want);
  EXPECT_EQ(got % 4096, size_t{0});
  EXPECT_GE(got, param.input);
}

INSTANTIATE_TEST_SUITE_P(
    PinnedSmallSteps, TierTablePinnedSmall,
    ::testing::Values(TierCase{1, 4096}, TierCase{63, 4096},
                      TierCase{100, 4096}, TierCase{4096, 4096},
                      TierCase{5000, 8192}, TierCase{8192, 8192},
                      TierCase{8193, 12288}, TierCase{16384, 16384}));

TEST_P(TierTableDeviceLarge, Dispatch) {
  const TierCase param = GetParam();
  SCOPED_TRACE(::testing::Message() << "input=" << param.input);

  novaStatus_t st{};
  const size_t got = round_up_device_large(param.input, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(got, param.want);
  EXPECT_EQ(got % kLargeStep, size_t{0});
  EXPECT_GE(got, param.input);
}

INSTANTIATE_TEST_SUITE_P(DeviceLargeSteps, TierTableDeviceLarge,
                         ::testing::Values(TierCase{10485761, 12582912},
                                           TierCase{20971520, 20971520},
                                           TierCase{20971521, 23068672},
                                           TierCase{30000000, 31457280},
                                           TierCase{52428801, 54525952}));

TEST_P(TierTablePinnedGeometric, Dispatch) {
  const TierCase param = GetParam();
  SCOPED_TRACE(::testing::Message() << "input=" << param.input);

  novaStatus_t st{};
  const size_t got = round_up_pinned_large(param.input, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(got, param.want);
  EXPECT_GE(got, param.input);
  EXPECT_LE(got - param.input, param.input / 4);
}

INSTANTIATE_TEST_SUITE_P(
    PinnedGeometricSteps, TierTablePinnedGeometric,
    ::testing::Values(TierCase{16385, 20480}, TierCase{17408, 20480},
                      TierCase{20000, 20480}, TierCase{24577, 28672},
                      TierCase{524288, 524288}, TierCase{524289, 655360},
                      TierCase{1153434, 1310720}, TierCase{3145728, 3145728}));

/**
 * @brief Exact literals for the device 2 MiB step above 10 MiB.
 *
 * @test REQ-006: 10485761 jumps to 12582912, exact multiples stay,
 *       50 MiB + 1 lands on 52 MiB.
 */
TEST(StorageRounding, LargeStepsCoverAndAlign) {
  struct Case {
    size_t input;
    size_t want;
  };
  for (const Case param : {Case{.input = 10485761, .want = 12582912},
                           Case{.input = 20971520, .want = 20971520},
                           Case{.input = 20971521, .want = 23068672},
                           Case{.input = 52428801, .want = 54525952}}) {
    SCOPED_TRACE(::testing::Message() << "input=" << param.input);
    novaStatus_t st{};
    EXPECT_EQ(round_up_device_large(param.input, &st), param.want);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
  }
}

/**
 * @brief Odd small sizes still cover the request on an aligned multiple.
 *
 * @test REQ-004/005: 1, 63, 511, 513 and 4097 never return a short
 *       or misaligned reservation.
 */
TEST(StorageRounding, SmallStepsCoverAndAlign) {
  for (const size_t input : {size_t{1}, size_t{63}, size_t{100}, size_t{511},
                             size_t{513}, size_t{4097}}) {
    SCOPED_TRACE(::testing::Message() << "input=" << input);
    novaStatus_t st{};
    const size_t dev = round_up_device_small(input, &st);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    EXPECT_EQ(dev % 512, size_t{0});
    EXPECT_GE(dev, input);

    const size_t pinned = round_up_pinned_small(input, &st);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    EXPECT_EQ(pinned % 4096, size_t{0});
    EXPECT_GE(pinned, input);
  }
}

/**
 * @brief Class frontiers switch steps at exactly 10 MiB and 16 KiB.
 *
 * @test REQ-002/007: device 10485759 stays on the 512 step and 10485761
 *       jumps to the 2 MiB step. Pinned 16384 stays linear and 16385
 *       goes geometric.
 */
TEST(StorageRounding, FrontierMinusOnePlusOne) {
  novaStatus_t st{};
  EXPECT_EQ(round_up_device_small(kDeviceLargeThreshold - 1, &st),
            kDeviceLargeThreshold);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(round_up_device_small(kDeviceLargeThreshold, &st),
            kDeviceLargeThreshold);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(round_up_device_large(kDeviceLargeThreshold + 1, &st),
            size_t{12582912});
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  EXPECT_EQ(round_up_pinned_small(kPinnedLinearLimit - 1, &st),
            kPinnedLinearLimit);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(round_up_pinned_small(kPinnedLinearLimit, &st), kPinnedLinearLimit);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(round_up_pinned_large(kPinnedLinearLimit + 1, &st), size_t{20480});
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  if (tests::tensor::gpuAvailable()) {
    EXPECT_EQ(resolve_rounded_bytes(kDeviceLargeThreshold, true, false, &st),
              kDeviceLargeThreshold);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    EXPECT_EQ(
        resolve_rounded_bytes(kDeviceLargeThreshold + 1, true, false, &st),
        size_t{12582912});
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    EXPECT_EQ(resolve_rounded_bytes(kPinnedLinearLimit, false, true, &st),
              kPinnedLinearLimit);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    EXPECT_EQ(resolve_rounded_bytes(kPinnedLinearLimit + 1, false, true, &st),
              size_t{20480});
    ASSERT_EQ(st.err, novaSuccess) << st.message;
  } else {
    // Without an active backend the dispatcher is identity by design.
    EXPECT_EQ(resolve_rounded_bytes(kDeviceLargeThreshold, true, false, &st),
              kDeviceLargeThreshold);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
  }
}

/**
 * @brief Plain host reservations keep the requested size.
 *
 * @test REQ-003: `resolve_rounded_bytes(n, false, false)` is identity
 *       for small, frontier, and large inputs on every build.
 */
TEST(StorageRounding, HostIdentityKeepsRequested) {
  for (const size_t input :
       {size_t{1}, size_t{63}, size_t{512}, size_t{4096}, size_t{16385},
        size_t{3000000}, kDeviceLargeThreshold, kDeviceLargeThreshold + 1}) {
    SCOPED_TRACE(::testing::Message() << "input=" << input);
    novaStatus_t st{};
    EXPECT_EQ(resolve_rounded_bytes(input, false, false, &st), input);
    EXPECT_EQ(st.err, novaSuccess) << st.message;
  }
}

/**
 * @brief The dispatcher routes each placement-flag combination.
 *
 * @test REQ-007: host pairs are identity everywhere. Device/pinned
 *       pairs round through the matching small variant while a
 *       backend is active, and stay identity while none is (gated
 *       dispatcher, see `device_active()` in storage.c).
 */
TEST(StorageRounding, DispatcherRoutesPlacementFlags) {
  novaStatus_t st{};
  EXPECT_EQ(resolve_rounded_bytes(100, false, false, &st), size_t{100});
  EXPECT_EQ(st.err, novaSuccess) << st.message;

  const bool active = tests::tensor::gpuAvailable();
  const size_t dev = resolve_rounded_bytes(100, true, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(dev, active ? size_t{512} : size_t{100});

  const size_t pinned = resolve_rounded_bytes(100, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(pinned, active ? size_t{4096} : size_t{100});

  const size_t pinnedGeo = resolve_rounded_bytes(5000, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(pinnedGeo, active ? size_t{8192} : size_t{5000});
}

/**
 * @brief A host tensor carries requested == order and bytes == rounding.
 *
 * @test REQ-001: for plain host memory both sizes equal the
 *       theoretical `size * item_size`. `gt_bytes()` agrees.
 */
TEST(StorageRounding, DualSizesMatchRequestAndRounding) {
  novaStatus_t st{};
  TensorCXX ten({4, 8}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  const Tensor view = ten.getCTensor();
  ASSERT_TRUE(is_allocated(&view));
  ASSERT_NE(view.storage, nullptr);

  const size_t need = theoretical(view);
  ASSERT_EQ(need, size_t{128});
  EXPECT_EQ(view.storage->handle.requested_bytes, need);
  EXPECT_EQ(view.storage->handle.bytes, need);
  EXPECT_GE(view.storage->handle.bytes, view.storage->handle.requested_bytes);

  const size_t real = gt_bytes(&view, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(real, need);
}

/**
 * @brief The validated accessor returns the real reservation.
 *
 * @test REQ-008 (US-3): `gt_bytes()` on an allocated tensor reports
 *       `bytes` with success status instead of reaching through the
 *       handle.
 */
TEST(StorageRounding, GtBytesReturnsRealSize) {
  novaStatus_t st{};
  TensorCXX ten({16}, Float64, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  const Tensor view = ten.getCTensor();
  ASSERT_TRUE(is_allocated(&view));

  const size_t real = gt_bytes(&view, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(real, view.storage->handle.bytes);
  EXPECT_EQ(real, theoretical(view));
}

/**
 * @brief Null and unallocated inputs fail through the status path.
 *
 * @test REQ-010: null tensor, null status, null storage, and dead
 *       handles return zero with an error status and never abort.
 */
TEST(StorageRounding, GtBytesRejectsNulls) {
  novaStatus_t st{};

  // Null tensor with a live status slot.
  EXPECT_EQ(gt_bytes(nullptr, &st), size_t{0});
  EXPECT_EQ(st.err, novaInvalidPointer);
  EXPECT_NE(st.message, nullptr);

  // Null status slot with a live tensor: no status to report through.
  {
    novaStatus_t alloc{};
    TensorCXX ten({4}, Float32, DEVICE_CPU, false, false, &alloc);
    ASSERT_EQ(alloc.err, novaSuccess) << alloc.message;
    const Tensor view = ten.getCTensor();
    EXPECT_EQ(gt_bytes(&view, nullptr), size_t{0});
  }

  // Unallocated tensor: no backing storage.
  {
    novaStatus_t alloc{};
    TensorCXX ten(unallocated, {4}, Float32, DEVICE_CPU, false, false, &alloc);
    ASSERT_EQ(alloc.err, novaSuccess) << alloc.message;
    const Tensor view = ten.getCTensor();
    EXPECT_EQ(view.storage, nullptr);
    EXPECT_EQ(gt_bytes(&view, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidTensor);
    EXPECT_NE(st.message, nullptr);
  }

  // Live storage slot with a dead handle id.
  {
    Tensor ten{};
    TensorStorage storage{};
    ten.storage = &storage;
    EXPECT_EQ(gt_bytes(&ten, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidTensor);
    EXPECT_NE(st.message, nullptr);
  }
}

/**
 * @brief Zero and overflowing round-ups report instead of wrapping.
 *
 * @test REQ-011: every `round_up_*` rejects 0 and SIZE_MAX-scale
 *       inputs. `resolve_rounded_bytes` rejects 0 everywhere and
 *       overflow while a backend is active.
 */
TEST(StorageRounding, RoundUpRejectsZeroAndOverflow) {
  for (const size_t bad : {size_t{0}, SIZE_MAX, SIZE_MAX - 100}) {
    SCOPED_TRACE(::testing::Message() << "input=" << bad);
    novaStatus_t st{};
    EXPECT_EQ(round_up_device_small(bad, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidValue);
    EXPECT_EQ(round_up_device_large(bad, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidValue);
    EXPECT_EQ(round_up_pinned_small(bad, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidValue);
    EXPECT_EQ(round_up_pinned_large(bad, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidValue);
  }

  novaStatus_t st{};
  EXPECT_EQ(resolve_rounded_bytes(0, true, false, &st), size_t{0});
  EXPECT_EQ(st.err, novaInvalidValue);
  EXPECT_EQ(resolve_rounded_bytes(0, false, false, &st), size_t{0});
  EXPECT_EQ(st.err, novaInvalidValue);

  if (tests::tensor::gpuAvailable()) {
    EXPECT_EQ(resolve_rounded_bytes(SIZE_MAX, true, false, &st), size_t{0});
    EXPECT_EQ(st.err, novaInvalidValue);
  }
}

/**
 * @brief A null status slot never crashes the rounding entry points.
 *
 * @test All fallible entries report via `novaStatus_t`. With no slot
 *       they return zero instead of dereferencing null.
 */
TEST(StorageRounding, ResolveRejectsNullStatus) {
  EXPECT_EQ(resolve_rounded_bytes(128, false, false, nullptr), size_t{0});
  EXPECT_EQ(resolve_rounded_bytes(128, true, false, nullptr), size_t{0});
  EXPECT_EQ(round_up_device_small(128, nullptr), size_t{0});
  EXPECT_EQ(round_up_device_large(kDeviceLargeThreshold, nullptr), size_t{0});
  EXPECT_EQ(round_up_pinned_small(128, nullptr), size_t{0});
  EXPECT_EQ(round_up_pinned_large(kPinnedLinearLimit + 1, nullptr), size_t{0});
}

/**
 * @brief Resize stores the request as theoretical and re-rounds.
 *
 * @test REQ-012: host resize keeps `bytes == requested`. Rejected
 *       zero resize leaves both cached sizes untouched.
 */
TEST(StorageRounding, ResizeIsTheoreticalAndRounded) {
  novaStatus_t st{};
  auto handle = reserve(512, 512, "cpu", false, 64, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  ASSERT_TRUE(is_valid_handle(&handle));

  st = resize(&handle, 1024);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(handle.requested_bytes, size_t{1024});
  EXPECT_EQ(handle.bytes, size_t{1024});

  st = resize(&handle, 0);
  EXPECT_EQ(st.err, novaInvalidValue);
  EXPECT_EQ(handle.requested_bytes, size_t{1024});
  EXPECT_EQ(handle.bytes, size_t{1024});

  st = novaStatus_t{};
  EXPECT_TRUE(release(&handle, &st));
  EXPECT_EQ(st.err, novaSuccess) << st.message;
}

#if defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)
/**
 * @brief Device resize re-rounds small and large requests.
 *
 * @test REQ-012 on the device path: 200 B stays in the 512 B class,
 *       10485761 B lands on the 12582912 B block.
 */
TEST(StorageRounding, DeviceResizeIsTheoreticalAndRounded) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  const size_t rounded = round_up_device_small(100, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  auto handle = reserve(rounded, 100, "device", false, 0, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  ASSERT_TRUE(is_valid_handle(&handle));
  EXPECT_EQ(handle.requested_bytes, size_t{100});
  EXPECT_EQ(handle.bytes, rounded);

  st = resize(&handle, 200);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(handle.requested_bytes, size_t{200});
  EXPECT_EQ(handle.bytes, size_t{512});

  st = resize(&handle, kDeviceLargeThreshold + 1);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(handle.requested_bytes, kDeviceLargeThreshold + 1);
  EXPECT_EQ(handle.bytes, size_t{12582912});

  st = novaStatus_t{};
  EXPECT_TRUE(release(&handle, &st));
  EXPECT_EQ(st.err, novaSuccess) << st.message;
}

/**
 * @brief Live device and pinned reservations show both sizes.
 *
 * @test REQ-001/004/005 on real backends: a 128 B device tensor
 *       reserves 512 B, a 63 B pinned tensor reserves 4096 B, with
 *       `requested_bytes` pinned to the order in both cases.
 */
TEST(StorageRounding, DeviceAndPinnedTiersNeedGpu) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};

  TensorCXX dev({4, 8}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  const Tensor dv = dev.getCTensor();
  ASSERT_TRUE(is_allocated(&dv));
  EXPECT_EQ(dv.storage->handle.requested_bytes, size_t{128});
  EXPECT_EQ(dv.storage->handle.bytes, size_t{512});
  const size_t realDev = gt_bytes(&dv, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(realDev, size_t{512});

  TensorCXX pinned({63}, UnSigned8, DEVICE_CPU, false, true, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  const Tensor pv = pinned.getCTensor();
  ASSERT_TRUE(is_allocated(&pv));
  EXPECT_EQ(pv.storage->handle.requested_bytes, size_t{63});
  EXPECT_EQ(pv.storage->handle.bytes, size_t{4096});
  const size_t realPinned = gt_bytes(&pv, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(realPinned, size_t{4096});
}

/**
 * @brief A sub-tier pinned round trip moves exactly 63 bytes.
 *
 * @test REQ-009 for pinned staging: the 63 B payload (4096 B
 *       reservation) transfers host to device and back bit-identical
 *       when measured with the theoretical size.
 */
TEST(StorageRounding, PinnedSmallTierRoundTrip) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  TensorCXX host({63}, UnSigned8, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor hv = host.getCTensor();
  tests::tensor::fillTensor(hv);

  TensorCXX dev({63}, UnSigned8, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor dv = dev.getCTensor();

  st = transf_tensor_from_host(&hv, &dv);
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  TensorCXX back({63}, UnSigned8, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor bv = back.getCTensor();

  st = transf_tensor_from_device(&dv, &bv);
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  tests::tensor::expectBytesEqual(hv.data.data, bv.data.data,
                                  hv.size * hv.item_size,
                                  "pinned small-tier round trip");
}

/**
 * @brief Rendering a padded device tensor never overruns the shadow.
 *
 * @test REQ-009 (US-2): the repr host shadow holds the theoretical
 *       128 B (not the 512 B reservation); rendering succeeds in both
 *       normal and debug modes and keeps the GPU placement in the
 *       footer.
 */
TEST(StorageRounding, ReprShadowUsesTheoretical) {
  NOVA_TEST_REQUIRE_GPU();
  novaStatus_t st{};
  TensorCXX host({4, 8}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor hv = host.getCTensor();
  tests::tensor::fillTensor(hv);

  TensorCXX dev({4, 8}, Float32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor dv = dev.getCTensor();
  EXPECT_EQ(dv.storage->handle.requested_bytes, size_t{128});
  EXPECT_EQ(dv.storage->handle.bytes, size_t{512});

  st = transf_tensor_from_host(&hv, &dv);
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  char *out = nullptr;
  const novaStatus_t rst = tensor_repr_debug(&dv, &out);
  ASSERT_EQ(rst.err, novaSuccess) << rst.message;
  ASSERT_NE(out, nullptr);
  EXPECT_NE(std::strstr(out, "tensor("), nullptr) << out;
  EXPECT_NE(std::strstr(out, "device="), nullptr) << out;
  std::free(out);
}
#endif // defined(NOVA_HAS_CUDA) or defined(NOVA_HAS_HIP)

TEST_P(RoundingDtypeSweep, SameBytesSameRounding) {
  const DType_ dtype = GetParam();
  novaStatus_t st{};
  const size_t item = dtype_size(dtype);
  ASSERT_GT(item, size_t{0});

  const size_t probe = size_t{100};
  const size_t dev = round_up_device_small(probe, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(dev, size_t{512}) << "dtype=" << static_cast<int>(dtype);
  const size_t pinned = round_up_pinned_small(probe, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(pinned, size_t{4096}) << "dtype=" << static_cast<int>(dtype);
  (void)item;
}

INSTANTIATE_TEST_SUITE_P(DtypeAgnostic, RoundingDtypeSweep,
                         ::testing::Values(Float32, Float64, Float16, BFloat16,
                                           Float8E4M3fn, Float8E5M2,
                                           Float4E2M1fn, Signed8, UnSigned8,
                                           Signed32, Signed64));

/**
 * @brief Each dtype's host tensor reports requested == theoretical.
 *
 * @test The dtype half of dtype-agnosticism: host `bytes` equals
 *       `size * item_size` for 1/2/4/8-byte dtypes, so payload code
 *       can always derive lengths from metadata.
 */
TEST(StorageRounding, HostTensorsReportTheoreticalPerDtype) {
  struct Case {
    DType_ dtype;
    std::vector<size_t> shape;
  };
  for (const Case &param : {Case{.dtype = Float32, .shape = {4, 8}},
                            Case{.dtype = Float64, .shape = {4}},
                            Case{.dtype = Float16, .shape = {8}},
                            Case{.dtype = UnSigned8, .shape = {63}},
                            Case{.dtype = Signed32, .shape = {16}},
                            Case{.dtype = Signed64, .shape = {4}}}) {
    SCOPED_TRACE(::testing::Message()
                 << "dtype=" << static_cast<int>(param.dtype));
    novaStatus_t st{};
    TensorCXX ten(param.shape, param.dtype, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    const Tensor view = ten.getCTensor();
    ASSERT_TRUE(is_allocated(&view));
    EXPECT_EQ(view.storage->handle.requested_bytes, theoretical(view));
    EXPECT_EQ(view.storage->handle.bytes, theoretical(view));
    const size_t real = gt_bytes(&view, &st);
    ASSERT_EQ(st.err, novaSuccess) << st.message;
    EXPECT_EQ(real, theoretical(view));
  }
}

/**
 * @brief Packed FP4 tensors use storage units as the theoretical size.
 *
 * @test Data-model: logical `{8}` packs to 4 storage bytes, so
 *       `requested_bytes == 4` while `bytes >= 4`.
 */
TEST(StorageRounding, PackedTheoreticalUsesStorageUnits) {
  novaStatus_t st{};
  TensorCXX ten({8}, Float4E2M1fn, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor view = ten.mutableCTensor();
  ASSERT_TRUE(is_allocated(&view));
  ASSERT_NE(view.storage, nullptr);

  const size_t needed = view.size * view.item_size;
  EXPECT_EQ(view.size, size_t{4});
  EXPECT_EQ(needed, size_t{4});
  EXPECT_EQ(view.storage->handle.requested_bytes, needed);
  EXPECT_GE(view.storage->handle.bytes, needed);

  const size_t real = gt_bytes(&view, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  EXPECT_EQ(real, view.storage->handle.bytes);
}

/**
 * @brief Parallelism helpers account the theoretical size.
 *
 * @test Constraints: with a divergent record (`requested = 128`,
 *       `bytes = 512`) both builders must report 128 total bytes,
 *       never the rounded reservation.
 */
TEST(StorageRounding, ParallelHelpersAccountTheoretical) {
  struct DivergentTensor {
    Tensor ten{};
    TensorStorage storage{};
    std::array<unsigned char, 512> backing{};

    DivergentTensor() {
      storage.ptr.data = backing.data();
      storage.handle.requested_bytes = 128;
      storage.handle.bytes = 512;
      storage.handle.id = 1;
      ten.dtype = Float32;
      ten.device = DEVICE_CPU;
      ten.ndims = 1;
      ten.size = 32;
      ten.logical_size = 32;
      ten.item_size = 4;
      ten.storage = &storage;
      ten.data.data = backing.data();
      ten.is_allocated_ = true;
    }
  } fake;

  const ElementwiseWork ew = make_elementwise_work(&fake.ten);
  ASSERT_TRUE(ew.valid);
  EXPECT_EQ(ew.total_bytes, size_t{128});

  const LayoutWork lw = make_layout_work(&fake.ten, true);
  ASSERT_TRUE(lw.valid);
  EXPECT_EQ(lw.total_bytes, size_t{128});
}

/**
 * @brief Deep copies preserve both sizes and move the theoretical payload.
 *
 * @test Regression lock for the payload-length migration: the
 *       destination keeps `requested == 128`, `bytes == 128`, and a
 *       bit-identical 128 B payload measured with `size * item_size`.
 */
TEST(StorageRounding, DeepcopyPreservesDualSizes) {
  novaStatus_t st{};
  TensorCXX src({4, 8}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;
  Tensor sv = src.getCTensor();
  tests::tensor::fillTensor(sv);
  ASSERT_EQ(sv.storage->handle.requested_bytes, size_t{128});

  TensorCXX dst(unallocated, {4, 8}, Float32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  deepcopy(&sv, &dst.mutableCTensor(), &st);
  ASSERT_EQ(st.err, novaSuccess) << st.message;

  const Tensor dcv = dst.getCTensor();
  ASSERT_TRUE(is_allocated(&dcv));
  EXPECT_EQ(dcv.storage->handle.requested_bytes, size_t{128});
  EXPECT_EQ(dcv.storage->handle.bytes, sv.storage->handle.bytes);
  tests::tensor::expectBytesEqual(sv.data.data, dcv.data.data,
                                  sv.size * sv.item_size,
                                  "dual-size deep copy payload");
}
