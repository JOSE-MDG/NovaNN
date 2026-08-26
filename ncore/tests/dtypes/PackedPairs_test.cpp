/**
 * @file PackedPairs_test.cpp
 * @brief Pair-packed storage model tests for Float4_e2m1fn_x2.
 *
 * Covers the value level (two 4-bit lanes per byte, low nibble first) and
 * the tensor level: Float4E2M1fn is the only dtype with packing factor 2,
 * so creation takes logical shapes, divides the last dimension by two
 * (rejecting odd last dimensions with novaInvalidShape), and reports
 * size/logical_size in storage units versus logical elements.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/fp_utils.h>
#include <ncore/core/status.h>
#include <ncore/core/storage.h>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

namespace {

using ncore::dtypes::Float4_e2m1fn;
using ncore::dtypes::Float4_e2m1fn_x2;
using ncore::wrappers::TensorCXX;

} // namespace

/**
 * @brief Verifies the lane order at struct level: low nibble holds the
 *        first lane, high nibble the second.
 */
TEST(PackedPairs, StructPacksLowNibbleFirst) {
  // +0.5 encodes to nibble 0x1; -6.0 to nibble 0xF.
  const Float4_e2m1fn_x2 packed(0.5F, -6.0F);
  EXPECT_EQ(packed.val_, UINT8_C(0xF1));

  EXPECT_FLOAT_EQ(static_cast<float>(packed.low()), 0.5F);
  EXPECT_FLOAT_EQ(static_cast<float>(packed.high()), -6.0F);
}

/**
 * @brief Verifies that the scalar-struct constructor matches the float
 *        constructor for all 16x16 lane combinations.
 */
TEST(PackedPairs, StructCtorFromScalarStructsMatchesFloatCtor) {
  for (uint32_t lo = 0; lo < 16U; ++lo) {
    for (uint32_t hi = 0; hi < 16U; ++hi) {
      const auto loBits = static_cast<uint8_t>(lo);
      const auto hiBits = static_cast<uint8_t>(hi);
      const Float4_e2m1fn loLane{loBits, Float4_e2m1fn::from_bits()};
      const Float4_e2m1fn hiLane{hiBits, Float4_e2m1fn::from_bits()};

      const Float4_e2m1fn_x2 fromLanes(loLane, hiLane);
      const Float4_e2m1fn_x2 fromFloats(static_cast<float>(loLane),
                                        static_cast<float>(hiLane));
      EXPECT_EQ(fromLanes.val_, fromFloats.val_)
          << "lo=0x" << std::hex << static_cast<unsigned>(loBits) << " hi=0x"
          << std::hex << static_cast<unsigned>(hiBits);
      EXPECT_EQ(fromLanes.val_,
                static_cast<uint8_t>((hiBits << 4) | (loBits & 0x0FU)));
    }
  }
}

/**
 * @brief Verifies that the from_bits scalar constructor masks off any bits
 *        above the low nibble.
 */
TEST(PackedPairs, ScalarFromBitsMasksHighNibble) {
  const Float4_e2m1fn lane{UINT8_C(0xAB), Float4_e2m1fn::from_bits()};
  EXPECT_EQ(lane.x, UINT8_C(0x0B));
}

/**
 * @brief Verifies byte-level round trip through the C conversion API for
 *        all 256 packed bytes.
 */
TEST(PackedPairs, CTensorApiRoundTripsAllBytes) {
  novaStatus_t st{};
  TensorCXX ten({2}, DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor view = ten.mutableCTensor();
  ASSERT_TRUE(is_allocated(&view));
  ASSERT_EQ(view.size, size_t{1}); // One storage byte.

  for (uint32_t b = 0; b < 256U; ++b) {
    const auto byte = static_cast<uint8_t>(b);
    view.data.data[0] = byte;

    float lo = 0.0F;
    float hi = 0.0F;
    fp4e2m1x2_to_floats(view.data.fp4e2m1fn_x2[0], &lo, &hi);
    const auto repacked = fp4e2m1x2_from_floats(lo, hi);
    EXPECT_EQ(static_cast<uint8_t>(repacked), byte)
        << "byte=0x" << std::hex << b;
  }
}

/**
 * @brief Verifies creation divides the last logical dimension by the
 *        packing factor and derives size/logical_size accordingly.
 */
TEST(PackedPairs, TensorCreationDividesLastDimensionByTwo) {
  {
    novaStatus_t st{};
    TensorCXX ten({8}, DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    const Tensor view = ten.getCTensor();
    EXPECT_EQ(view.shape[0], size_t{4});
    EXPECT_EQ(view.size, size_t{4});
    EXPECT_EQ(view.logical_size, size_t{8});
    EXPECT_EQ(view.item_size, size_t{1});
  }
  {
    novaStatus_t st{};
    TensorCXX ten({2, 6}, DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    const Tensor view = ten.getCTensor();
    EXPECT_EQ(view.shape[0], size_t{2});
    EXPECT_EQ(view.shape[1], size_t{3});
    EXPECT_EQ(view.size, size_t{6});
    EXPECT_EQ(view.logical_size, size_t{12});
    // Row-major strides in bytes with item_size == 1.
    EXPECT_EQ(view.strides[0], size_t{3});
    EXPECT_EQ(view.strides[1], size_t{1});
  }
}

/**
 * @brief Verifies that an odd last logical dimension is rejected with
 *        novaInvalidShape and the documented message.
 *
 * This is an in-scope error path: the argument is well-formed but logically
 * inconsistent for a packed dtype, and ncore itself surfaces the failure.
 */
TEST(PackedPairs, OddLastDimensionRejectedWithInvalidShape) {
  {
    novaStatus_t st{};
    TensorCXX ten({3}, DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
    EXPECT_EQ(st.err, novaInvalidShape);
    ASSERT_NE(st.message, nullptr);
    EXPECT_NE(std::strstr(st.message, "multiple of the dtype packing"), nullptr)
        << st.message;
  }
  {
    novaStatus_t st{};
    TensorCXX ten({2, 5}, DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
    EXPECT_EQ(st.err, novaInvalidShape);
    ASSERT_NE(st.message, nullptr);
    EXPECT_NE(std::strstr(st.message, "multiple of the dtype packing"), nullptr)
        << st.message;
  }
}

/**
 * @brief Verifies the scalar-tensor special case: one storage unit holding
 *        two logical elements.
 */
TEST(PackedPairs, ScalarTensorHasLogicalSizeTwo) {
  novaStatus_t st{};
  TensorCXX ten(DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  const Tensor view = ten.getCTensor();
  EXPECT_TRUE(is_scalar(&view));
  EXPECT_EQ(view.size, size_t{1});
  EXPECT_EQ(view.logical_size, size_t{2});
  EXPECT_EQ(view.ndims, size_t{0});
}

/**
 * @brief Verifies dtype_packing_factor reports 2 exclusively for FP4.
 */
TEST(PackedPairs, PackingFactorIsReportedAsTwo) {
  EXPECT_EQ(dtype_packing_factor(DType_::Float4E2M1fn), size_t{2});
  for (size_t i = 0; i < NUM_DTYPES; ++i) {
    const auto dtype = static_cast<DType_>(i);
    if (dtype == DType_::Float4E2M1fn) {
      continue;
    }
    EXPECT_EQ(dtype_packing_factor(dtype), size_t{1}) << "dtype index=" << i;
  }
}

/**
 * @brief Verifies the Rust-backed allocation covers at least the storage
 *        extent and that every storage unit is writable through the typed
 *        pointer.
 */
TEST(PackedPairs, AllocationSizeMatchesStorageUnits) {
  for (const size_t logical : {size_t{2}, size_t{8}, size_t{64}}) {
    SCOPED_TRACE(::testing::Message() << "logical=" << logical);

    novaStatus_t st{};
    TensorCXX ten({logical}, DType_::Float4E2M1fn, DEVICE_CPU, false, false,
                  &st);
    ASSERT_EQ(st.err, novaSuccess);

    Tensor view = ten.mutableCTensor();
    ASSERT_TRUE(is_allocated(&view));
    ASSERT_NE(view.storage, nullptr);
    const size_t needed = view.size * view.item_size;
    EXPECT_GE(view.storage->handle.size_bytes, needed);

    for (size_t i = 0; i < view.size; ++i) {
      view.data.fp4e2m1fn_x2[i] = static_cast<uint8_t>(i + 1U);
    }
    for (size_t i = 0; i < view.size; ++i) {
      EXPECT_EQ(view.data.fp4e2m1fn_x2[i], static_cast<uint8_t>(i + 1U))
          << "storage unit=" << i;
    }
  }
}
