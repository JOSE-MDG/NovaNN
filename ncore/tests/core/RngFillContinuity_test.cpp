/**
 * @file RngFillContinuity_test.cpp
 * @brief Same-width fill composability over engine and placement paths.
 *
 * A split fill on one handle must match a whole fill of the concatenated
 * demand on an identically seeded handle: fill(a)+fill(b) packs the same
 * lanes as fill(a+b). Sizes that are not a multiple of the engine block
 * or sample width are the interesting cases, because a naive path would
 * drop the trailing lanes of the last block and desynchronize the two
 * schedules.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include <ncore/core/dtype.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/rand/generator.h>

#include "utils/TensorUtils.hpp"

namespace {

using ncore::wrappers::TensorCXX;

constexpr uint64_t kSeed = 42;

/// Asserts @p st holds the success code (message included on failure).
void expectOk(const novaStatus_t st, const char *ctx) {
  ASSERT_EQ(st.err, novaSuccess)
      << ctx << (st.message != nullptr ? st.message : "");
}

TEST(RngFillContinuity, PhiloxU32SplitFillsMatchWholeFill) {
  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 0),
           "whole init");
  uint32_t wholeOut[10] = {};
  expectOk(generator_fill_u32(&whole, wholeOut, 10), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_CPU, kSeed, 0),
           "split init");
  uint32_t splitOut[10] = {};
  expectOk(generator_fill_u32(&split, splitOut, 5), "split fill a");
  expectOk(generator_fill_u32(&split, splitOut + 5, 5), "split fill b");

  EXPECT_EQ(0, std::memcmp(wholeOut, splitOut, sizeof splitOut))
      << "split Philox u32 fills consumed a different lane schedule";
}

TEST(RngFillContinuity, PhiloxU64SplitFillsMatchWholeFill) {
  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 0),
           "whole init");
  uint64_t wholeOut[10] = {};
  expectOk(generator_fill_u64(&whole, wholeOut, 10), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_CPU, kSeed, 0),
           "split init");
  uint64_t splitOut[10] = {};
  expectOk(generator_fill_u64(&split, splitOut, 5), "split fill a");
  expectOk(generator_fill_u64(&split, splitOut + 5, 5), "split fill b");

  EXPECT_EQ(0, std::memcmp(wholeOut, splitOut, sizeof splitOut))
      << "split Philox u64 fills consumed a different lane schedule";
}

TEST(RngFillContinuity, XoshiroU32OddSplitFillsMatchWholeFill) {
  Generator_ whole = {};
  expectOk(generator_init(&whole, Xoshiro256, DEVICE_CPU, kSeed, 0),
           "whole init");
  uint32_t wholeOut[9] = {};
  expectOk(generator_fill_u32(&whole, wholeOut, 9), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Xoshiro256, DEVICE_CPU, kSeed, 0),
           "split init");
  uint32_t splitOut[9] = {};
  expectOk(generator_fill_u32(&split, splitOut, 5), "split fill a");
  expectOk(generator_fill_u32(&split, splitOut + 5, 4), "split fill b");

  EXPECT_EQ(0, std::memcmp(wholeOut, splitOut, sizeof splitOut))
      << "odd-split Xoshiro u32 fills lost the low half of a sample";
}

TEST(RngFillContinuity, XoshiroU32WholeThenOddSplitMatchWholeFill) {
  Generator_ whole = {};
  expectOk(generator_init(&whole, Xoshiro256, DEVICE_CPU, kSeed, 0),
           "whole init");
  uint32_t wholeOut[11] = {};
  expectOk(generator_fill_u32(&whole, wholeOut, 11), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Xoshiro256, DEVICE_CPU, kSeed, 0),
           "split init");
  uint32_t splitOut[11] = {};
  expectOk(generator_fill_u32(&split, splitOut, 5), "split odd fill");
  expectOk(generator_fill_u32(&split, splitOut + 5, 6), "split even fill");

  EXPECT_EQ(0, std::memcmp(wholeOut, splitOut, sizeof splitOut))
      << "an odd fill before an even one must not re-alter the schedule";
}

TEST(RngFillContinuity, PhiloxU64OddTailCarriesAcrossSplits) {
  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 0),
           "whole init");
  uint64_t wholeOut[13] = {};
  expectOk(generator_fill_u64(&whole, wholeOut, 13), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_CPU, kSeed, 0),
           "split init");
  uint64_t splitOut[13] = {};
  expectOk(generator_fill_u64(&split, splitOut, 5), "split fill a");
  expectOk(generator_fill_u64(&split, splitOut + 5, 8), "split fill b");

  EXPECT_EQ(0, std::memcmp(wholeOut, splitOut, sizeof splitOut))
      << "an odd u64 demand must not desync the later schedule";
}

TEST(RngFillContinuity, PhiloxU32LargeSplitFillsMatchWholeFill) {
  constexpr size_t kWhole = 163841;
  constexpr size_t kSplitA = 98305;

  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 0),
           "whole init");
  std::vector<uint32_t> wholeOut(kWhole);
  expectOk(generator_fill_u32(&whole, wholeOut.data(), kWhole), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_CPU, kSeed, 0),
           "split init");
  std::vector<uint32_t> splitOut(kWhole);
  expectOk(generator_fill_u32(&split, splitOut.data(), kSplitA),
           "split fill a");
  expectOk(
      generator_fill_u32(&split, splitOut.data() + kSplitA, kWhole - kSplitA),
      "split fill b");

  EXPECT_EQ(0, std::memcmp(wholeOut.data(), splitOut.data(),
                           kWhole * sizeof(uint32_t)))
      << "large split Philox u32 fills consumed a different lane schedule";
}

TEST(RngFillContinuity, PhiloxU64LargeSplitFillsMatchWholeFill) {
  constexpr size_t kWhole = 98305;
  constexpr size_t kSplitA = 65536;

  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 0),
           "whole init");
  std::vector<uint64_t> wholeOut(kWhole);
  expectOk(generator_fill_u64(&whole, wholeOut.data(), kWhole), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_CPU, kSeed, 0),
           "split init");
  std::vector<uint64_t> splitOut(kWhole);
  expectOk(generator_fill_u64(&split, splitOut.data(), kSplitA),
           "split fill a");
  expectOk(
      generator_fill_u64(&split, splitOut.data() + kSplitA, kWhole - kSplitA),
      "split fill b");

  EXPECT_EQ(0, std::memcmp(wholeOut.data(), splitOut.data(),
                           kWhole * sizeof(uint64_t)))
      << "large split Philox u64 fills consumed a different lane schedule";
}

TEST(RngFillContinuity, GpuPhiloxU32SplitFillsMatchWholeFill) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 11;

  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 4),
           "whole init");
  uint32_t wholeOut[kN] = {};
  expectOk(generator_fill_u32(&whole, wholeOut, kN), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_GPU, kSeed, 4),
           "split init");

  const std::vector<size_t> shape{kN};
  novaStatus_t st{};
  TensorCXX dev(shape, DType_::UnSigned32, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  TensorCXX home(shape, DType_::UnSigned32, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  uint32_t *devPtr = dev.mutableCTensor().data.u32;

  expectOk(generator_fill_u32(&split, devPtr, 6), "split first");
  expectOk(generator_fill_u32(&split, devPtr + 6, 5), "split second");

  Tensor devTen = dev.mutableCTensor();
  Tensor homeTen = home.mutableCTensor();
  expectOk(transf_tensor_from_device(&devTen, &homeTen), "fetch");
  const uint32_t *homePtr = homeTen.data.u32;
  EXPECT_EQ(0, std::memcmp(homePtr, wholeOut, kN * sizeof(uint32_t)))
      << "the GPU split fills must reproduce the whole fill";
}

TEST(RngFillContinuity, GpuPhiloxU64SplitFillsMatchWholeFill) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 5;

  Generator_ whole = {};
  expectOk(generator_init(&whole, Philox4x32, DEVICE_CPU, kSeed, 4),
           "whole init");
  uint64_t wholeOut[kN] = {};
  expectOk(generator_fill_u64(&whole, wholeOut, kN), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Philox4x32, DEVICE_GPU, kSeed, 4),
           "split init");

  const std::vector<size_t> shape{kN};
  novaStatus_t st{};
  TensorCXX dev(shape, DType_::UnSigned64, DEVICE_GPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  TensorCXX home(shape, DType_::UnSigned64, DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  uint64_t *devPtr = dev.mutableCTensor().data.u64;

  expectOk(generator_fill_u64(&split, devPtr, 3), "split first");
  expectOk(generator_fill_u64(&split, devPtr + 3, 2), "split second");

  Tensor devTen = dev.mutableCTensor();
  Tensor homeTen = home.mutableCTensor();
  expectOk(transf_tensor_from_device(&devTen, &homeTen), "fetch");
  const uint64_t *homePtr = homeTen.data.u64;
  EXPECT_EQ(0, std::memcmp(homePtr, wholeOut, kN * sizeof(uint64_t)))
      << "the GPU split fills must reproduce the whole fill";
}

TEST(RngFillContinuity, XoshiroU64SplitFillsMatchWholeFill) {
  Generator_ whole = {};
  expectOk(generator_init(&whole, Xoshiro256, DEVICE_CPU, kSeed, 0),
           "whole init");
  uint64_t wholeOut[9] = {};
  expectOk(generator_fill_u64(&whole, wholeOut, 9), "whole fill");

  Generator_ split = {};
  expectOk(generator_init(&split, Xoshiro256, DEVICE_CPU, kSeed, 0),
           "split init");
  uint64_t splitOut[9] = {};
  expectOk(generator_fill_u64(&split, splitOut, 5), "split odd fill");
  expectOk(generator_fill_u64(&split, splitOut + 5, 4), "split even fill");

  EXPECT_EQ(0, std::memcmp(wholeOut, splitOut, sizeof splitOut))
      << "whole u64 samples pack continuously across splits";
}

} // namespace
