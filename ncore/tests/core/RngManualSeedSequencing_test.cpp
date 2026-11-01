/**
 * @file RngManualSeedSequencing_test.cpp
 * @brief Cross-device coherence of the process defaults under reseeding.
 *
 * manual_seed() reseeds both process defaults. A CPU sample taken by
 * an observer at time t must never be newer than a GPU sample taken
 * later: the device defaults advance together through one seeding, so
 * a later GPU read can only see the same seed or a newer one. A mixed
 * window between the CPU and GPU seeding lets observers catch a GPU
 * default that lags a CPU sample taken after it, which this test
 * counts as a sequencing violation.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <thread>

#include <ncore/core/device.h>
#include <ncore/core/status.h>
#include <ncore/rand/generator.h>

namespace {

/// Seed steps the reseeding thread walks through per repetition.
constexpr uint64_t kSeedSteps = 40000;

/// Antenna thread pairs: read the CPU default, then the GPU default.
void observeMismatches(std::atomic<bool> *stop,
                       std::atomic<uint64_t> *violations) {
  while (!stop->load(std::memory_order_relaxed)) {
    Generator_ cpu = {};
    if (default_generator(DEVICE_CPU, &cpu).err != novaSuccess) {
      ADD_FAILURE() << "CPU default copy failed";
      return;
    }
    if (stop->load(std::memory_order_relaxed)) {
      return;
    }
    Generator_ gpu = {};
    if (default_generator(DEVICE_GPU, &gpu).err != novaSuccess) {
      ADD_FAILURE() << "GPU default copy failed";
      return;
    }
    if (gpu.seed < cpu.seed) {
      violations->fetch_add(1, std::memory_order_relaxed);
    }
  }
}

TEST(RngManualSeedSequencing, GpuDefaultNeverLagsAStaleCpuSample) {
  std::atomic<bool> stop{false};
  std::atomic<uint64_t> violations{0};

  std::thread seeder([&stop] {
    for (uint64_t step = 1; step <= kSeedSteps; ++step) {
      const novaStatus_t st = manual_seed(1000 + step);
      if (st.err != novaSuccess) {
        ADD_FAILURE() << "manual_seed failed: "
                      << (st.message != nullptr ? st.message : "");
        return;
      }
    }
    stop.store(true, std::memory_order_relaxed);
  });

  std::thread antennaA(observeMismatches, &stop, &violations);
  std::thread antennaB(observeMismatches, &stop, &violations);

  seeder.join();
  antennaA.join();
  antennaB.join();

  EXPECT_EQ(0, violations.load())
      << "a GPU sample lagged a CPU sample read after it";
}

} // namespace
