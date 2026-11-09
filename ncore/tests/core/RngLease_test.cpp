/**
 * @file RngLease_test.cpp
 * @brief Exclusive leases over the process default generators.
 *
 * default_generator_lease() must hand out the live default of a
 * placement and hold its mutex until default_generator_return()
 * releases it, so a multi step fill draws from one unbroken shared
 * stream. Unknown placements and null slots must be rejected, and a
 * return without a hold is a no-op. Nested leases on one device
 * deadlock under the non recursive POSIX mutex and stay untested on
 * purpose.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/rand/generator.h>

#include <tests/common/Status.hpp>

namespace {

using tests::common::expectOk;

constexpr size_t kN = 8;
constexpr uint64_t kSeed = 77;

/** Returns a lease on scope exit so a failed assertion cannot strand
    the default mutex for the remaining tests of the process. */
struct ScopedLease {
  DefaultGeneratorLease lease{};

  ScopedLease() = default;
  ScopedLease(const ScopedLease &) = delete;
  ScopedLease &operator=(const ScopedLease &) = delete;

  ~ScopedLease() { (void)default_generator_return(&lease); }
};

TEST(RngLease, CpuLeaseHandsOutTheLiveDefault) {
  DefaultGeneratorLease lease{};
  expectOk(default_generator_lease(DEVICE_CPU, &lease), "lease");
  ASSERT_NE(lease.gen, nullptr);
  EXPECT_EQ(lease.device, DEVICE_CPU);

  expectOk(default_generator_return(&lease), "return");
  EXPECT_EQ(lease.gen, nullptr) << "a returned lease must drop the default";
}

/** Draws through lease.gen must advance the very stream the default
    fills consume, and a second lease cycle proves the return truly
    unlocked the mutex. */
TEST(RngLease, DrawsThroughTheLeaseAdvanceTheSharedDefault) {
  expectOk(manual_seed(kSeed), "seed");

  Generator_ ref = {};
  expectOk(generator_init(&ref, Xoshiro256, DEVICE_CPU, kSeed, 0), "ref init");
  std::vector<uint32_t> refWhole(kN * 3);
  expectOk(generator_fill_u32(&ref, refWhole.data(), refWhole.size()),
           "ref fill");

  std::vector<uint32_t> leased(kN);
  {
    ScopedLease hold;
    expectOk(default_generator_lease(DEVICE_CPU, &hold.lease), "lease");
    ASSERT_NE(hold.lease.gen, nullptr);
    expectOk(generator_fill_u32(hold.lease.gen, leased.data(), kN),
             "leased fill");
  }
  EXPECT_EQ(0,
            std::memcmp(leased.data(), refWhole.data(), kN * sizeof(uint32_t)))
      << "a leased draw must consume the shared default stream";

  std::vector<uint32_t> shared(kN);
  expectOk(default_fill_u32(DEVICE_CPU, shared.data(), kN), "default fill");
  EXPECT_EQ(0, std::memcmp(shared.data(), refWhole.data() + kN,
                           kN * sizeof(uint32_t)))
      << "the default must continue where the lease left off";
  EXPECT_NE(0, std::memcmp(leased.data(), shared.data(), kN * sizeof(uint32_t)))
      << "successive draws must hand out disjoint values";

  std::vector<uint32_t> again(kN);
  {
    ScopedLease hold;
    expectOk(default_generator_lease(DEVICE_CPU, &hold.lease), "second lease");
    ASSERT_NE(hold.lease.gen, nullptr);
    expectOk(generator_fill_u32(hold.lease.gen, again.data(), kN),
             "second leased fill");
  }
  EXPECT_EQ(0, std::memcmp(again.data(), refWhole.data() + 2 * kN,
                           kN * sizeof(uint32_t)))
      << "the second lease must resume the shared stream";
}

/** The lease use case: several draws of both widths inside one hold
    must pack like one uninterrupted run on the shared stream. */
TEST(RngLease, LeaseSpansMultiStepFillsOfBothWidths) {
  expectOk(manual_seed(kSeed), "seed");

  constexpr size_t kHead = 5;
  constexpr size_t kMid = 9;
  constexpr size_t kLast = 3;

  Generator_ ref = {};
  expectOk(generator_init(&ref, Xoshiro256, DEVICE_CPU, kSeed, 0), "ref init");
  uint32_t refHead[kHead] = {};
  uint64_t refMid[kMid] = {};
  uint32_t refLast[kLast] = {};
  uint32_t refTail[kN] = {};
  expectOk(generator_fill_u32(&ref, refHead, kHead), "ref u32");
  expectOk(generator_fill_u64(&ref, refMid, kMid), "ref u64");
  expectOk(generator_fill_u32(&ref, refLast, kLast), "ref u32 again");
  expectOk(generator_fill_u32(&ref, refTail, kN), "ref u32 tail");

  uint32_t leasedHead[kHead] = {};
  uint64_t leasedMid[kMid] = {};
  uint32_t leasedLast[kLast] = {};
  {
    ScopedLease hold;
    expectOk(default_generator_lease(DEVICE_CPU, &hold.lease), "lease");
    ASSERT_NE(hold.lease.gen, nullptr);
    expectOk(generator_fill_u32(hold.lease.gen, leasedHead, kHead),
             "leased u32");
    expectOk(generator_fill_u64(hold.lease.gen, leasedMid, kMid), "leased u64");
    expectOk(generator_fill_u32(hold.lease.gen, leasedLast, kLast),
             "leased u32 again");
  }

  uint32_t tail[kN] = {};
  expectOk(default_fill_u32(DEVICE_CPU, tail, kN), "default fill");

  EXPECT_EQ(0, std::memcmp(leasedHead, refHead, sizeof refHead));
  EXPECT_EQ(0, std::memcmp(leasedMid, refMid, sizeof refMid))
      << "the leased u64 draw must continue the u32 head on one stream";
  EXPECT_EQ(0, std::memcmp(leasedLast, refLast, sizeof refLast))
      << "the leased u32 draw must see the half lane the u64 kept";
  EXPECT_EQ(0, std::memcmp(tail, refTail, sizeof tail))
      << "the default must resume exactly where the lease stopped";
}

TEST(RngLease, LeaseRejectsNullSlotAndUnknownPlacements) {
  EXPECT_EQ(default_generator_lease(DEVICE_CPU, nullptr).err,
            novaInvalidPointer);

  Generator_ scratch = {};
  DefaultGeneratorLease lease{};
  lease.gen = &scratch;
  lease.device = DEVICE_META;

  const novaStatus_t meta = default_generator_lease(DEVICE_META, &lease);
  EXPECT_EQ(meta.err, novaInvalidDevice)
      << (meta.message != nullptr ? meta.message : "");
  const novaStatus_t unknown =
      default_generator_lease(static_cast<Device_>(77), &lease);
  EXPECT_EQ(unknown.err, novaInvalidDevice)
      << (unknown.message != nullptr ? unknown.message : "");
  EXPECT_EQ(lease.gen, &scratch) << "a rejected lease must stay untouched";
  EXPECT_EQ(lease.device, DEVICE_META)
      << "a rejected lease must stay untouched";
}

/** A return without a hold is a documented no-op; a null lease slot
    must be rejected. */
TEST(RngLease, ReturnWithoutAHoldIsInert) {
  DefaultGeneratorLease lease{};
  expectOk(default_generator_return(&lease), "no-op return");
  EXPECT_EQ(lease.gen, nullptr);

  EXPECT_EQ(default_generator_return(nullptr).err, novaInvalidPointer);
}

/** The GPU default leases the same way without probing any device:
    no fill runs here, so the cycle is safe without a GPU. */
TEST(RngLease, GpuLeaseCyclesWithoutTouchingTheDevice) {
  DefaultGeneratorLease lease{};
  expectOk(default_generator_lease(DEVICE_GPU, &lease), "lease");
  ASSERT_NE(lease.gen, nullptr);
  EXPECT_EQ(lease.device, DEVICE_GPU);
  expectOk(default_generator_return(&lease), "return");
  EXPECT_EQ(lease.gen, nullptr);
}

} // namespace
