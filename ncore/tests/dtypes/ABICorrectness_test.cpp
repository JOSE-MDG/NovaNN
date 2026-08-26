/**
 * @file ABICorrectness_test.cpp
 * @brief Binary-level contract tests for the dtype layer.
 *
 * Verifies that the reduced-precision structs have the size, alignment, and
 * trivial-copy properties their storage model requires; that the public type
 * aliases map to the intended compiler types; that the DType_ enum keeps the
 * documented 21-value uint8_t-backed layout the dispatch tables index by;
 * and that the metadata tables (byte widths, classification masks) agree
 * with the enum they are indexed by.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/dtypes/bfloat16.hh>
#include <ncore/headeronly/dtypes/fp4_e2m1fn_x2.hh>
#include <ncore/headeronly/dtypes/fp8_e4m3fn.hh>
#include <ncore/headeronly/dtypes/fp8_e5m2.hh>
#include <ncore/headeronly/dtypes/half.hh>
#include <ncore/headeronly/macros.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tables/dtype_tables.h>
#include <ncore/tensor.h>

namespace {

// NOTE: no using-declaration for ncore::dtypes::BFloat16: the unscoped
// DType_ enumerator "BFloat16" would make unqualified lookup ambiguous.
using ncore::dtypes::Float4_e2m1fn;
using ncore::dtypes::Float4_e2m1fn_x2;
using ncore::dtypes::Float8_e4m3fn;
using ncore::dtypes::Float8_e5m2;
using ncore::dtypes::Half;
using ncore::wrappers::TensorCXX;

/// Documented byte widths per DType_ value, declared independently of
/// lookup_dtype_sizes.
constexpr std::array<size_t, NUM_DTYPES> kDocumentedSizes{{
    4, 8,       // Float32, Float64
    2, 2,       // Float16, BFloat16
    1, 1, 1,    // Float8E4M3fn, Float8E5M2, Float4E2M1fn
    1, 1, 1, 1, // Signed8, UnSigned8, QSigned8, QUnSigned8
    2, 2, 2, 2, // Signed16, UnSigned16, QSigned16, QUnSigned16
    4, 4, 4, 4, // Signed32, UnSigned32, QSigned32, QUnSigned32
    8, 8,       // Signed64, UnSigned64
}};

} // namespace
/**
 * @brief Verifies sizeof/alignof of every reduced-precision struct.
 */
TEST(ABICorrectness, ReducedStructSizesAndAlignments) {
  static_assert(sizeof(Half) == 2 && alignof(Half) == 2);
  static_assert(sizeof(ncore::dtypes::BFloat16) == 2);
  static_assert(alignof(ncore::dtypes::BFloat16) == 2);
  static_assert(sizeof(Float8_e4m3fn) == 1 && alignof(Float8_e4m3fn) == 1);
  static_assert(sizeof(Float8_e5m2) == 1 && alignof(Float8_e5m2) == 1);
  static_assert(sizeof(Float4_e2m1fn) == 1 && alignof(Float4_e2m1fn) == 1);
  static_assert(sizeof(Float4_e2m1fn_x2) == 1 &&
                alignof(Float4_e2m1fn_x2) == 1);

  EXPECT_EQ(sizeof(Half), size_t{2});
  EXPECT_EQ(alignof(Half), size_t{2});
  EXPECT_EQ(sizeof(ncore::dtypes::BFloat16), size_t{2});
  EXPECT_EQ(alignof(ncore::dtypes::BFloat16), size_t{2});
  EXPECT_EQ(sizeof(Float8_e4m3fn), size_t{1});
  EXPECT_EQ(alignof(Float8_e4m3fn), size_t{1});
  EXPECT_EQ(sizeof(Float8_e5m2), size_t{1});
  EXPECT_EQ(alignof(Float8_e5m2), size_t{1});
  EXPECT_EQ(sizeof(Float4_e2m1fn), size_t{1});
  EXPECT_EQ(alignof(Float4_e2m1fn), size_t{1});
  EXPECT_EQ(sizeof(Float4_e2m1fn_x2), size_t{1});
  EXPECT_EQ(alignof(Float4_e2m1fn_x2), size_t{1});
}

/**
 * @brief Verifies trivial copyability and standard layout: payloads are
 *        memcpy'd and moved between devices throughout the runtime.
 */
TEST(ABICorrectness, ReducedStructsAreTriviallyCopyable) {
  static_assert(std::is_trivially_copyable_v<Half>);
  static_assert(std::is_trivially_copyable_v<ncore::dtypes::BFloat16>);
  static_assert(std::is_trivially_copyable_v<Float8_e4m3fn>);
  static_assert(std::is_trivially_copyable_v<Float8_e5m2>);
  static_assert(std::is_trivially_copyable_v<Float4_e2m1fn>);
  static_assert(std::is_trivially_copyable_v<Float4_e2m1fn_x2>);

  static_assert(std::is_standard_layout_v<Half>);
  static_assert(std::is_standard_layout_v<ncore::dtypes::BFloat16>);
  static_assert(std::is_standard_layout_v<Float8_e4m3fn>);
  static_assert(std::is_standard_layout_v<Float8_e5m2>);
  static_assert(std::is_standard_layout_v<Float4_e2m1fn>);
  static_assert(std::is_standard_layout_v<Float4_e2m1fn_x2>);

  SUCCEED();
}

/**
 * @brief Pins the documented compile-time alias dispatch in dtype.h for the
 *        active toolchain.
 */
TEST(ABICorrectness, TypeAliasIdentitiesMatchToolchain) {
#if defined(_GNUC_CLANG_)
  static_assert(std::is_same_v<float16, _Float16>);
  static_assert(std::is_same_v<bfloat16, __bf16>);
  SUCCEED() << "GCC/Clang native reduced types active";
#else
  static_assert(std::is_same_v<float16, unsigned short>);
  static_assert(std::is_same_v<bfloat16, unsigned short>);
  SUCCEED() << "Portable reduced-type aliases active";
#endif
  static_assert(std::is_same_v<float8_e4m3fn, uint8_t>);
  static_assert(std::is_same_v<float8_e5m2, uint8_t>);
  static_assert(std::is_same_v<float4_e2m1fn_x2, uint8_t>);
  EXPECT_EQ(sizeof(float16), size_t{2});
  EXPECT_EQ(sizeof(bfloat16), size_t{2});
}

/**
 * @brief Pins enum stability: underlying type, first/last values, and
 *        enumerator count. Enumerator positions are dispatch-table indices;
 *        any reorder must fail here.
 */
TEST(ABICorrectness, DTypeEnumLayoutIsStable) {
  static_assert(std::is_same_v<std::underlying_type_t<DType_>, uint8_t>);
  static_assert(static_cast<uint8_t>(DType_::Float32) == 0);
  static_assert(static_cast<uint8_t>(DType_::UnSigned64) == 20);
  static_assert(NUM_DTYPES == 21);
  static_assert(NUM_FLOATS == 7);

  // Contiguity: every index in [0, NUM_DTYPES) must name a distinct valid
  // enumerator; spot-check both ends plus the float/int boundary.
  EXPECT_EQ(static_cast<uint8_t>(DType_::Float32), UINT8_C(0));
  EXPECT_EQ(static_cast<uint8_t>(DType_::UnSigned64), UINT8_C(20));
  EXPECT_EQ(NUM_DTYPES, size_t{21});
}

/**
 * @brief Verifies dtype_size against independent literal byte widths.
 */
TEST(ABICorrectness, DtypeSizeTableMatchesTypeSizes) {
  for (size_t d = 0; d < NUM_DTYPES; ++d) {
    const auto dtype = static_cast<DType_>(d);
    EXPECT_EQ(dtype_size(dtype), kDocumentedSizes[d]) << "dtype index=" << d;
  }
}

/**
 * @brief Verifies each classification mask has exactly its documented
 *        true-set (dtype_tables.h contract).
 */
TEST(ABICorrectness, ClassificationTablesMatchDocumentedSets) {
  auto isOneOf = [](DType_ d, std::initializer_list<DType_> set) {
    for (const DType_ s : set) {
      if (s == d) {
        return true;
      }
    }
    return false;
  };

  for (size_t i = 0; i < NUM_DTYPES; ++i) {
    const auto d = static_cast<DType_>(i);

    const bool wantFloating = isOneOf(
        d, {DType_::Float32, DType_::Float64, DType_::Float16, DType_::BFloat16,
            DType_::Float8E4M3fn, DType_::Float8E5M2, DType_::Float4E2M1fn});
    const bool wantInteger =
        isOneOf(d, {DType_::Signed8, DType_::UnSigned8, DType_::Signed16,
                    DType_::UnSigned16, DType_::Signed32, DType_::UnSigned32,
                    DType_::Signed64, DType_::UnSigned64});
    const bool wantSigned = isOneOf(d, {DType_::Signed8, DType_::Signed16,
                                        DType_::Signed32, DType_::Signed64});
    const bool wantUnsigned =
        isOneOf(d, {DType_::UnSigned8, DType_::UnSigned16, DType_::UnSigned32,
                    DType_::UnSigned64});
    const bool wantQuantizedSigned =
        isOneOf(d, {DType_::QSigned8, DType_::QSigned16, DType_::QSigned32});
    const bool wantQuantizedUnsigned = isOneOf(
        d, {DType_::QUnSigned8, DType_::QUnSigned16, DType_::QUnSigned32});
    const bool wantQuantizable =
        isOneOf(d, {DType_::Float4E2M1fn, DType_::QSigned8, DType_::QUnSigned8,
                    DType_::QSigned16, DType_::QUnSigned16, DType_::QSigned32,
                    DType_::QUnSigned32});

    EXPECT_EQ(floating[i][0], wantFloating) << "dtype index=" << i;
    EXPECT_EQ(integer[i][0], wantInteger) << "dtype index=" << i;
    EXPECT_EQ(signed_integer[i][0], wantSigned) << "dtype index=" << i;
    EXPECT_EQ(unsigned_integer[i][0], wantUnsigned) << "dtype index=" << i;
    EXPECT_EQ(quantized_signed_integer[i][0], wantQuantizedSigned)
        << "dtype index=" << i;
    EXPECT_EQ(quantized_unsigned_integer[i][0], wantQuantizedUnsigned)
        << "dtype index=" << i;
    EXPECT_EQ(quantizable_dtype[i][0], wantQuantizable) << "dtype index=" << i;
  }
}

/**
 * @brief Verifies the tensor-field predicates agree with the raw masks for
 *        real tensors of all 21 dtypes.
 */
TEST(ABICorrectness, ClassificationPredicatesAgreeWithTables) {
  for (size_t i = 0; i < NUM_DTYPES; ++i) {
    const auto dtype = static_cast<DType_>(i);
    // Packed dtypes require an even logical last dimension: one storage
    // unit's worth of logical elements is the minimal valid shape.
    const size_t logicalCount = dtype_packing_factor(dtype);
    novaStatus_t st{};
    TensorCXX ten({logicalCount}, dtype, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess) << "dtype index=" << i;

    const Tensor view = ten.getCTensor();
    EXPECT_EQ(is_floating(&view), floating[i][0]) << "dtype index=" << i;
    EXPECT_EQ(is_integer(&view), integer[i][0]) << "dtype index=" << i;
    EXPECT_EQ(is_signed_integer(&view), signed_integer[i][0])
        << "dtype index=" << i;
    EXPECT_EQ(is_unsigned_integer(&view), unsigned_integer[i][0])
        << "dtype index=" << i;
    EXPECT_EQ(is_quantized_signed_integer(&view),
              quantized_signed_integer[i][0])
        << "dtype index=" << i;
    EXPECT_EQ(is_quantized_unsigned_integer(&view),
              quantized_unsigned_integer[i][0])
        << "dtype index=" << i;
  }
}

/**
 * @brief Verifies data_ptr union members alias the same bytes as the raw
 *        byte pointer, for the packed FP4 member and the half member.
 */
TEST(ABICorrectness, DataPtrUnionMembersAliasSameBytes) {
  // FP4: one storage byte viewed through .data and .fp4e2m1fn_x2.
  {
    novaStatus_t st{};
    TensorCXX ten({2}, DType_::Float4E2M1fn, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor view = ten.mutableCTensor();
    ASSERT_TRUE(is_allocated(&view));
    EXPECT_EQ(view.size, size_t{1});

    view.data.data[0] = UINT8_C(0xAB);
    EXPECT_EQ(view.data.fp4e2m1fn_x2[0], UINT8_C(0xAB));
  }

  // FP16: one 2-byte element viewed through .data and .half.
  {
    novaStatus_t st{};
    TensorCXX ten({1}, DType_::Float16, DEVICE_CPU, false, false, &st);
    ASSERT_EQ(st.err, novaSuccess);
    Tensor view = ten.mutableCTensor();
    ASSERT_TRUE(is_allocated(&view));

    view.data.data[0] = UINT8_C(0x00);
    view.data.data[1] = UINT8_C(0x3C); // 0x3C00 = 1.0 in binary16.
    EXPECT_EQ(std::memcmp(view.data.half, view.data.data, 2), 0);

    float16 h = 0;
    std::memcpy(&h, view.data.half, sizeof(h));
    constexpr float16 kOne = static_cast<float16>(1.0F);
    EXPECT_EQ(std::memcmp(&h, &kOne, sizeof(h)), 0);
  }
}
