/**
 * @file RngGenerator_test.cpp
 * @brief Tests for the production generator wrapper
 *        (<ncore/headeronly/wrappers/generator.hh>).
 *
 * Determinism, subsequence independence, byte to lane mapping, zero
 * demand, snapshot and restore, clone, host device parity and the
 * status surfacing of a failed init.
 *
 * Tests needing a device call NOVA_TEST_REQUIRE_GPU() first.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/generator.hh>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include <tests/common/Gpu.hpp>
#include <tests/common/Status.hpp>
#include <tests/tensor/DeviceMirror.hpp>
#include <tests/tensor/TensorUtils.hpp>

namespace {

using ncore::wrappers::GeneratorCXX;
using ncore::wrappers::TensorCXX;
using tests::common::expectOk;
using tests::tensor::DeviceMirror;

constexpr uint64_t kSeed = 123;
constexpr uint64_t kSub = 7;
constexpr uint64_t kSubDev = 11;

/** One CPU Philox handle for the host side checks. */
GeneratorCXX hostGen(uint64_t seed, uint64_t subsequence) {
  return GeneratorCXX(Philox4x32, DEVICE_CPU, seed, subsequence);
}

/** Identical inputs reproduce bit-identical lanes. */
TEST(RngGenerator, SameInputsProduceSameBits) {
  constexpr size_t kN = 256;
  GeneratorCXX a = hostGen(kSeed, kSub);
  GeneratorCXX b = hostGen(kSeed, kSub);
  uint32_t x[kN] = {};
  uint32_t y[kN] = {};
  expectOk(a.fillU32(x, kN), "a fill");
  expectOk(b.fillU32(y, kN), "b fill");
  EXPECT_EQ(0, std::memcmp(x, y, sizeof x));
}

/** Distinct subsequences draw independently. */
TEST(RngGenerator, DistinctSubsequencesDrawIndependently) {
  constexpr size_t kN = 256;
  GeneratorCXX a = hostGen(kSeed, 3);
  GeneratorCXX b = hostGen(kSeed, 4);
  uint32_t x[kN] = {};
  uint32_t y[kN] = {};
  expectOk(a.fillU32(x, kN), "a fill");
  expectOk(b.fillU32(y, kN), "b fill");
  EXPECT_NE(0, std::memcmp(x, y, sizeof x));
}

/** The host byte fill writes lanes little end first and the tail peels
    the low bytes of the next lane. */
TEST(RngGenerator, ByteFillMapsLanesLittleEndFirst) {
  constexpr size_t kLanes = 2;
  constexpr size_t kTail = 3;
  constexpr size_t kBytes = (kLanes * 4) + kTail;

  uint32_t lanes[kLanes] = {};
  uint32_t tailLane = 0;
  GeneratorCXX laneStream = hostGen(kSeed, kSub);
  expectOk(laneStream.fillU32(lanes, kLanes), "lane fill");
  expectOk(laneStream.fillU32(&tailLane, 1), "tail lane fill");

  uint8_t bytes[kBytes] = {};
  GeneratorCXX byteStream = hostGen(kSeed, kSub);
  expectOk(byteStream.fillBytes(bytes, kBytes), "byte fill");

  EXPECT_EQ(0, std::memcmp(bytes, lanes, kLanes * 4));
  for (size_t i = 0; i < kTail; ++i) {
    EXPECT_EQ(bytes[(kLanes * 4) + i],
              static_cast<uint8_t>(tailLane >> (8 * i)))
        << "tail byte " << i;
  }
}

/** A zero demand succeeds without advancing the stream. */
TEST(RngGenerator, ZeroDemandDoesNotAdvanceTheStream) {
  uint32_t fresh[8] = {};
  expectOk(hostGen(kSeed, kSub).fillU32(fresh, 8), "fresh fill");

  GeneratorCXX advanced = hostGen(kSeed, kSub);
  EXPECT_EQ(advanced.fillU32(nullptr, 0).err, novaSuccess);

  uint32_t cont[8] = {};
  expectOk(advanced.fillU32(cont, 8), "continued fill");
  EXPECT_EQ(0, std::memcmp(fresh, cont, sizeof fresh));
}

/** A restore resumes the exact stream position of the snapshot. */
TEST(RngGenerator, SnapshotRestoreReproducesBoundaryFills) {
  GeneratorCXX a = hostGen(kSeed, kSub + 1);

  uint32_t first[4] = {};
  expectOk(a.fillU32(first, 4), "first fill");
  uint64_t snap[7] = {};
  expectOk(a.snapshot(snap), "snapshot");
  uint32_t second[4] = {};
  expectOk(a.fillU32(second, 4), "second fill");

  GeneratorCXX b = hostGen(kSeed, kSub + 1);
  uint32_t mirror[4] = {};
  expectOk(b.fillU32(mirror, 4), "mirror fill");
  EXPECT_EQ(0, std::memcmp(first, mirror, sizeof first));
  expectOk(b.restore(snap), "restore");
  uint32_t last[4] = {};
  expectOk(b.fillU32(last, 4), "restored fill");
  EXPECT_EQ(0, std::memcmp(second, last, sizeof second));
}

/** A clone continues the shared stream at its own pace. */
TEST(RngGenerator, CloneForksAndDiverges) {
  GeneratorCXX base = hostGen(kSeed, kSub + 2);
  uint32_t prefix[4] = {};
  expectOk(base.fillU32(prefix, 4), "prefix fill");

  GeneratorCXX forked = base.clone();
  uint32_t baseNext[4] = {};
  uint32_t baseThird[4] = {};
  uint32_t forkNext[4] = {};
  expectOk(base.fillU32(baseNext, 4), "base next");
  expectOk(base.fillU32(baseThird, 4), "base third");
  expectOk(forked.fillU32(forkNext, 4), "fork next");

  EXPECT_EQ(0, std::memcmp(baseNext, forkNext, sizeof baseNext));
  EXPECT_NE(0, std::memcmp(baseThird, forkNext, sizeof baseThird));
}

/** An implicit copy and a clone continue the source identically,
    parked lanes included. */
TEST(RngGenerator, CopyForkMatchesClone) {
  GeneratorCXX base = hostGen(kSeed, kSub + 3);
  uint32_t prefix[5] = {};
  expectOk(base.fillU32(prefix, 5), "prefix fill");

  GeneratorCXX copied = base;
  GeneratorCXX forked = base.clone();
  uint32_t copyNext[4] = {};
  uint32_t cloneNext[4] = {};
  expectOk(copied.fillU32(copyNext, 4), "copy fill");
  expectOk(forked.fillU32(cloneNext, 4), "clone fill");
  EXPECT_EQ(0, std::memcmp(copyNext, cloneNext, sizeof copyNext));

  uint32_t baseNext[4] = {};
  expectOk(base.fillU32(baseNext, 4), "base fill");
  EXPECT_EQ(0, std::memcmp(copyNext, baseNext, sizeof baseNext));
}

/** A clone of a failed handle reports the same failure. */
TEST(RngGenerator, FailedInitCloneSurfacesOnNextCall) {
  GeneratorCXX bad(Xoshiro256, DEVICE_GPU, kSeed, 0);
  GeneratorCXX forked = bad.clone();
  const novaStatus_t st = forked.fillU32(nullptr, 0);
  EXPECT_EQ(st.err, novaInvalidValue)
      << (st.message != nullptr ? st.message : "");
}

/** A failed init surfaces on the next call. */
TEST(RngGenerator, FailedInitSurfacesOnNextCall) {
  GeneratorCXX bad(Xoshiro256, DEVICE_GPU, kSeed, 0);
  const novaStatus_t st = bad.fillU32(nullptr, 0);
  EXPECT_EQ(st.err, novaInvalidValue)
      << (st.message != nullptr ? st.message : "");
}

/** A handle and tensor placement mix reports novaInvalidDevice. */
TEST(RngGenerator, TensorPlacementMixReportsInvalidDevice) {
  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, 0);
  novaStatus_t st{};
  TensorCXX homeTen(std::vector<size_t>{4}, DType_::UnSigned32, DEVICE_CPU,
                    false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor ten = homeTen.mutableCTensor();
  EXPECT_EQ(dev.fill(ten).err, novaInvalidDevice);
}

/** Host and device placements emit the same lanes. */
TEST(RngGenerator, DeviceFillMatchesHostBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kN = 64;

  GeneratorCXX host = hostGen(kSeed, kSubDev);
  uint32_t hostBits[kN] = {};
  expectOk(host.fillU32(hostBits, kN), "host fill");

  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, kSubDev);
  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(dev.fillU32(mirror.devTen().data.u32, kN), "device fill");
  mirror.fetch();

  EXPECT_EQ(
      0, std::memcmp(hostBits, mirror.homeTen().data.u32, kN * sizeof(uint32_t)));
}

/** Masked tail and parked lane continuation stay aligned on device. */
TEST(RngGenerator, DeviceTailAndContinuationMatchHost) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kHead = 5;
  constexpr size_t kNext = 8;
  constexpr size_t kN = kHead + kNext;

  GeneratorCXX host = hostGen(kSeed, kSubDev + 1);
  uint32_t hostBits[kN] = {};
  expectOk(host.fillU32(hostBits, kHead), "host head");
  expectOk(host.fillU32(hostBits + kHead, kNext), "host tail");

  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, kSubDev + 1);
  DeviceMirror mirror(kN, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(dev.fillU32(mirror.devTen().data.u32, kHead), "device head");
  expectOk(dev.fillU32(mirror.devTen().data.u32 + kHead, kNext), "device tail");
  mirror.fetch();

  EXPECT_EQ(
      0, std::memcmp(hostBits, mirror.homeTen().data.u32, kN * sizeof(uint32_t)));
}

/** A nonzero device tail is rejected before any lane is consumed. */
TEST(RngGenerator, DeviceByteTailRejected) {
  NOVA_TEST_REQUIRE_GPU();
  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, 0);
  DeviceMirror mirror(4, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  const novaStatus_t rejected = dev.fillBytes(mirror.devTen().data.data, 6);
  EXPECT_EQ(rejected.err, novaInvalidValue)
      << (rejected.message != nullptr ? rejected.message : "");
}

/** Host and device u64 placements emit the same pairs, odd fresh
    counts and parked pair continuation included. */
TEST(RngGenerator, DeviceU64FillMatchesHostBitwise) {
  NOVA_TEST_REQUIRE_GPU();
  constexpr size_t kHead = 5;
  constexpr size_t kNext = 8;
  constexpr size_t kN = kHead + kNext;

  GeneratorCXX host = hostGen(kSeed, kSubDev + 2);
  uint64_t hostBits[kN] = {};
  expectOk(host.fillU64(hostBits, kHead), "host head");
  expectOk(host.fillU64(hostBits + kHead, kNext), "host tail");

  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, kSubDev + 2);
  DeviceMirror mirror(kN, DType_::UnSigned64);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(dev.fillU64(mirror.devTen().data.u64, kHead), "device head");
  expectOk(dev.fillU64(mirror.devTen().data.u64 + kHead, kNext), "device tail");
  mirror.fetch();

  EXPECT_EQ(
      0, std::memcmp(hostBits, mirror.homeTen().data.u64, kN * sizeof(uint64_t)));
}

/** A device byte fill round-trips through the tensor layer. */
TEST(RngGenerator, DeviceByteFillRoundTrip) {
  NOVA_TEST_REQUIRE_GPU();
  GeneratorCXX host = hostGen(kSeed, kSubDev + 3);
  uint8_t hostBits[8] = {};
  expectOk(host.fillBytes(hostBits, 8), "host byte fill");

  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, kSubDev + 3);
  DeviceMirror mirror(2, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(dev.fillBytes(mirror.devTen().data.data, 8), "device byte fill");
  mirror.fetch();

  EXPECT_EQ(0, std::memcmp(hostBits, mirror.homeTen().data.data, 8));
}

/** A fill survives a host side delay before the read: the null-path
    drop shape documented on the block comments of the null paths,
    guarded by the launcher sync and the wrapper drain. */
TEST(RngGenerator, DeviceFillSurvivesHostDelayBeforeRead) {
  NOVA_TEST_REQUIRE_GPU();
  GeneratorCXX host = hostGen(kSeed, kSubDev + 4);
  uint32_t hostBits[8] = {};
  expectOk(host.fillU32(hostBits, 8), "host fill");

  GeneratorCXX dev(Philox4x32, DEVICE_GPU, kSeed, kSubDev + 4);
  DeviceMirror mirror(8, DType_::UnSigned32);
  ASSERT_EQ(mirror.st.err, novaSuccess)
      << (mirror.st.message != nullptr ? mirror.st.message : "");
  expectOk(dev.fillU32(mirror.devTen().data.u32, 8), "device fill");
  std::this_thread::sleep_for(std::chrono::milliseconds(2));
  mirror.fetch();

  EXPECT_EQ(
      0, std::memcmp(hostBits, mirror.homeTen().data.u32, 8 * sizeof(uint32_t)));
}

} // namespace
