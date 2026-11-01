/**
 * @file RngHandleValidation_test.cpp
 * @brief Rejection guards for engine, device, and engine/device pairs.
 *
 * A handle must never silently run an impossible configuration: an
 * unknown device value must not fall through to the CPU path, an
 * unknown engine value must not fall through to Philox, and Xoshiro
 * must be rejected at init time when the handle targets device memory.
 * Failed inits must also leave the caller's handle untouched.
 */

#include <gtest/gtest.h>

#include <cstdint>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/rand/generator.h>

namespace {

constexpr uint64_t kSeed = 11;

/** Sentinel fields proving a failed call left the handle untouched. */
void stampSentinel(Generator_ *gen) {
  gen->seed = 0xA5A5A5A5A5A5A5A5ULL;
  gen->subsequence = 7;
  gen->offset = 9;
  gen->device = DEVICE_CPU;
  gen->engine = Philox4x32;
}

void expectSentinel(const Generator_ &gen, const char *ctx) {
  EXPECT_EQ(gen.seed, 0xA5A5A5A5A5A5A5A5ULL) << ctx;
  EXPECT_EQ(gen.subsequence, 7U) << ctx;
  EXPECT_EQ(gen.offset, 9U) << ctx;
  EXPECT_EQ(gen.device, DEVICE_CPU) << ctx;
  EXPECT_EQ(gen.engine, Philox4x32) << ctx;
}

TEST(RngHandleValidation, InitRejectsXoshiroOnGpuAndKeepsSentinel) {
  Generator_ gen = {};
  stampSentinel(&gen);

  const novaStatus_t st =
      generator_init(&gen, Xoshiro256, DEVICE_GPU, kSeed, 0);

  EXPECT_EQ(st.err, novaInvalidValue)
      << "Xoshiro owns no device fill path; init must reject the pair"
      << (st.message != nullptr ? st.message : "");
  expectSentinel(gen, "fields written before validation");
}

TEST(RngHandleValidation, InitRejectsUnknownEngine) {
  Generator_ gen = {};
  stampSentinel(&gen);

  const novaStatus_t st =
      generator_init(&gen, static_cast<RngEngine>(77), DEVICE_CPU, kSeed, 0);

  EXPECT_EQ(st.err, novaInvalidValue)
      << "an unknown engine must not fall through to the Philox branch"
      << (st.message != nullptr ? st.message : "");
  expectSentinel(gen, "fields written before validation");
}

TEST(RngHandleValidation, InitRejectsUnknownDevice) {
  Generator_ gen = {};
  stampSentinel(&gen);

  const novaStatus_t st =
      generator_init(&gen, Philox4x32, static_cast<Device_>(77), kSeed, 0);

  EXPECT_EQ(st.err, novaInvalidDevice)
      << "an unknown device must not fall through to the CPU path"
      << (st.message != nullptr ? st.message : "");
  expectSentinel(gen, "fields written before validation");
}

TEST(RngHandleValidation, FillRejectsUnknownDeviceInsteadOfCpuFallback) {
  Generator_ gen = {};
  stampSentinel(&gen);
  gen.device = static_cast<Device_>(77);

  uint32_t out[4] = {};
  const novaStatus_t st = generator_fill_u32(&gen, out, 4);

  EXPECT_EQ(st.err, novaInvalidDevice)
      << "a fill on an unknown device must not run the CPU path"
      << (st.message != nullptr ? st.message : "");
  EXPECT_EQ(out[0], 0U) << "no value may land in the buffer";
}

TEST(RngHandleValidation, FillOnMetaIsRejected) {
  Generator_ gen = {};
  stampSentinel(&gen);
  gen.device = DEVICE_META;

  uint32_t out[4] = {};
  const novaStatus_t st = generator_fill_u32(&gen, out, 4);

  EXPECT_EQ(st.err, novaInvalidDevice)
      << (st.message != nullptr ? st.message : "");
  EXPECT_EQ(out[0], 0U) << "no value may land in the buffer";
}

TEST(RngHandleValidation, FillOnGpuWithXoshiroStaysRejected) {
  Generator_ gen = {};
  ASSERT_EQ(generator_init(&gen, Philox4x32, DEVICE_GPU, kSeed, 0).err,
            novaSuccess);
  gen.engine = Xoshiro256;

  uint32_t out[4] = {};
  const novaStatus_t st = generator_fill_u32(&gen, out, 4);

  EXPECT_EQ(st.err, novaInvalidValue)
      << (st.message != nullptr ? st.message : "");
}

} // namespace
