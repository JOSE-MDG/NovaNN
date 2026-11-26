/**
 * @file CapsUnits_test.cpp
 * @brief Unit pins for the device-capability unit conversions.
 *
 * The runtime reports clocks in kilohertz, bus width in bits and every
 * capacity in bytes. These cases keep the derived roofs and the copied
 * fields in those units, with no KiB or MiB scaling along the way.
 */

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

#include <ncore/headeronly/heuristics/kernels/launch_config.hh>

namespace hk = ncore::heuristics::kernels;

namespace {

/// Minimal stand-in with the fields capsFromDetected reads.
struct FakeProps {
  uint32_t warpSize = 32U;
  uint32_t maxThreadsPerBlock = 1024U;
  uint32_t maxThreadsPerMultiProcessor = 2048U;
  uint32_t maxBlocksPerMultiProcessor = 32U;
  uint32_t multiProcessorCount = 132U;
  size_t sharedMemPerBlock = 49152U;
  size_t sharedMemPerMultiprocessor = 228224U;
  size_t sharedMemPerBlockOptin = 232448U;
  size_t reservedSharedMemPerBlock = 1024U;
  uint32_t regsPerMultiprocessor = 65536U;
  std::array<int, 3> maxGridSize{2147483647, 65535, 65535};
  int major = 9;
  int minor = 0;
  uint32_t memoryClockRate = 2619000U;
  uint32_t memoryBusWidth = 5120U;
  uint32_t l2CacheSize = 62914560U;
  uint32_t persistingL2CacheMaxSize = 16777216U;
  uint64_t peakFp32Flops = 66908160000000ULL;
};

} // namespace

/**
 * @brief Peak bandwidth follows the DDR formula in bytes per second.
 * @test The V100 reference from the CUDA best-practices guide, 877 MHz at
 * 4096 bits, rounds to 898 GB/s and is exactly 898048000000 B/s.
 */
TEST(CapsUnits, TheoreticalBandwidthIsBytesPerSecond) {
  EXPECT_EQ(hk::detail::theoreticalBandwidth(877000U, 4096U),
            898048000000ULL);
}

/**
 * @brief Peak FP32 follows cores per SM times clock times two.
 * @test H100 SXM: 128 cores, 132 SMs, 1.98 GHz gives 66.90816 TFLOPS.
 */
TEST(CapsUnits, CudaPeakFlopsIsPerSecond) {
  EXPECT_EQ(hk::detail::cudaPeakFp32Flops(900U, 132U, 1980000U),
            66908160000000ULL);
}

/**
 * @brief Copying detected facts keeps bytes as bytes and clocks as kHz.
 * @test A 48 KiB block budget stays 49152, a 60 MiB L2 stays 62914560,
 * and HBM3 at 2619 MHz on a 5120-bit bus gives about 3.35 TB/s.
 */
TEST(CapsUnits, CapsFromDetectedKeepsNativeUnits) {
  const FakeProps props{};
  const hk::DeviceCaps caps = hk::capsFromDetected(props);

  EXPECT_EQ(caps.smemPerBlock, 49152U);
  EXPECT_EQ(caps.smemPerBlockOptin, 232448U);
  EXPECT_EQ(caps.smemReservedPerBlock, 1024U);
  EXPECT_EQ(caps.l2Size, 62914560U);
  EXPECT_EQ(caps.l2MaxWindow, 16777216U);
  EXPECT_EQ(caps.maxThreadsPerSM, 2048U);
  EXPECT_EQ(caps.maxThreadsPerBlockDevice, 1024U);
  EXPECT_EQ(caps.regsPerSM, 65536U);
  EXPECT_EQ(caps.memBandwidthBytesPerSec, 3352320000000ULL);
  EXPECT_EQ(caps.peakFlops, 66908160000000ULL);
}
