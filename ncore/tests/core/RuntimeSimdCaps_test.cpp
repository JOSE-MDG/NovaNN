/**
 * @file RuntimeSimdCaps_test.cpp
 * @brief Unit tests for runtime SIMD capability detection.
 *
 * Feeds synthetic CPUID snapshots through
 * get_simd_capabilities_from_cpuid() so features missing from the build
 * host (AMX, AVX10, AVX-512 FP16) are still covered.
 *
 * Bit positions are declared independently from the implementation, so a
 * wrong constant on either side fails instead of cancelling out.
 *
 * Synthetic models mirror the CPUID enumeration of their real-silicon
 * counterparts (e.g. Sapphire Rapids ships AMX-BF16/AMX-INT8 and
 * AVX-512 FP16 but neither AMX-FP16 nor AVX-VNNI-INT8; Granite Rapids
 * shipped AVX10 v1; Lunar Lake reports AVX10 v1 without enumerating
 * legacy AVX-512), so expectations stay calibrated against the data
 * actual hardware produces.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <ncore/simd/simd.h>

namespace {

/* ---- CPUID feature bits (Intel SDM Vol. 2A) ---- */

/* CPUID.(EAX=01H):ECX */
constexpr uint32_t kSse42Bit = 1U << 20;
constexpr uint32_t kFma3Bit = 1U << 12;
constexpr uint32_t kAvxBit = 1U << 28;
constexpr uint32_t kF16cBit = 1U << 29;

/* CPUID.(EAX=07H,ECX=00H):EBX */
constexpr uint32_t kAvx2Bit = 1U << 5;
constexpr uint32_t kAvx512FBit = 1U << 16;
constexpr uint32_t kAvx512DqBit = 1U << 17;
constexpr uint32_t kAvx512BwBit = 1U << 30;
constexpr uint32_t kAvx512VlBit = 1U << 31;

/* CPUID.(EAX=07H,ECX=00H):ECX */
constexpr uint32_t kAvx512VnniBit = 1U << 11;

/* CPUID.(EAX=07H,ECX=00H):EDX */
constexpr uint32_t kAmxBf16Bit = 1U << 22;
constexpr uint32_t kAvx512Fp16Bit = 1U << 23;
constexpr uint32_t kAmxInt8Bit = 1U << 25;

/* CPUID.(EAX=07H,ECX=01H):EAX */
constexpr uint32_t kAvxVnniBit = 1U << 4;
constexpr uint32_t kAvx512Bf16Bit = 1U << 5;
constexpr uint32_t kAmxFp16Bit = 1U << 21;

/* CPUID.(EAX=07H,ECX=01H):EDX */
constexpr uint32_t kAvxVnniInt8Bit = 1U << 4;
constexpr uint32_t kAvx10PresenceBit = 1U << 19;

/**
 * @enum Reg
 * @brief Index of one output register within a CPUID leaf array.
 */
enum class Reg : std::size_t { kEax = 0, kEbx = 1, kEcx = 2, kEdx = 3 };

/// Member pointer to one boolean flag of SIMDCapabilities.
using FlagField = const bool SIMDCapabilities::*;

/// One flag expectation: which field must hold which value.
using FlagExpectation = std::pair<FlagField, bool>;

/**
 * @struct CpuModel
 * @brief One synthetic processor: raw CPUID values plus expected flags.
 */
struct CpuModel {
  std::string name;
  SIMDCpuidSnapshot raw{};
  std::vector<FlagExpectation> expected;
};

/**
 * @brief Sets bits in one register of a CPUID snapshot.
 *
 * @param[in,out] regs Registers of one leaf (index 0..3 = EAX..EDX).
 * @param[in] reg Target register.
 * @param[in] mask Bit mask to OR in.
 */
void setBits(std::span<uint32_t, 4> regs, Reg reg, uint32_t mask) {
  regs[static_cast<std::size_t>(reg)] |= mask;
}

/**
 * @brief Builds expectations with every flag false.
 */
std::vector<FlagExpectation> allFlagsFalse() {
  /* Extent deduced; the assert below guards flag-count drift. */
  constexpr std::array fields = {
      &SIMDCapabilities::sse4_2_,      &SIMDCapabilities::avx_,
      &SIMDCapabilities::avx2_,        &SIMDCapabilities::avx2_int8_,
      &SIMDCapabilities::avx2_vnni_,   &SIMDCapabilities::f16c_,
      &SIMDCapabilities::vnni_,        &SIMDCapabilities::fma3_,
      &SIMDCapabilities::avx512f_,     &SIMDCapabilities::avx512_bw_,
      &SIMDCapabilities::avx512_dq_,   &SIMDCapabilities::avx512_vl_,
      &SIMDCapabilities::avx512_vnni_, &SIMDCapabilities::avx512_fp16_,
      &SIMDCapabilities::avx512_bf16_, &SIMDCapabilities::avx10_1_,
      &SIMDCapabilities::avx10_2_,     &SIMDCapabilities::amx_,
      &SIMDCapabilities::amx_fp16_,    &SIMDCapabilities::amx_bf16_,
      &SIMDCapabilities::amx_int8_,
  };
  static_assert(fields.size() == 21,
                "SIMDCapabilities has 21 flags; keep this list in sync");

  std::vector<FlagExpectation> out;
  out.reserve(fields.size());
  for (const auto field : fields) {
    out.emplace_back(field, false);
  }
  return out;
}

/**
 * @brief Marks one flag as expected true in an expectation list.
 */
void expectTrue(std::vector<FlagExpectation> &flags, FlagField field) {
  for (auto &[member, value] : flags) {
    if (member == field) {
      value = true;
      return;
    }
  }
}

/**
 * @brief Marks several flags as expected true in an expectation list.
 */
void expectTrue(std::vector<FlagExpectation> &flags,
                std::initializer_list<FlagField> fields) {
  for (const auto field : fields) {
    expectTrue(flags, field);
  }
}

/**
 * @brief Builds the catalog of synthetic processors under test.
 */
std::vector<CpuModel> buildCpuCatalog() {
  std::vector<CpuModel> cpus;

  cpus.push_back(
      CpuModel{.name = "AllZero", .raw = {}, .expected = allFlagsFalse()});

  {
    CpuModel cpu{.name = "Haswell", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx,
            kSse42Bit | kFma3Bit | kAvxBit | kF16cBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx, kAvx2Bit);
    expectTrue(cpu.expected,
               {&SIMDCapabilities::sse4_2_, &SIMDCapabilities::fma3_,
                &SIMDCapabilities::avx_, &SIMDCapabilities::f16c_,
                &SIMDCapabilities::avx2_});
    cpus.push_back(std::move(cpu));
  }

  {
    CpuModel cpu{.name = "Zen4", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx,
            kSse42Bit | kFma3Bit | kAvxBit | kF16cBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx,
            kAvx2Bit | kAvx512FBit | kAvx512DqBit | kAvx512BwBit |
                kAvx512VlBit);
    setBits(cpu.raw.leaf7_0, Reg::kEcx, kAvx512VnniBit);
    setBits(cpu.raw.leaf7_1, Reg::kEax, kAvx512Bf16Bit);
    expectTrue(cpu.expected,
               {&SIMDCapabilities::sse4_2_, &SIMDCapabilities::fma3_,
                &SIMDCapabilities::avx_, &SIMDCapabilities::f16c_,
                &SIMDCapabilities::avx2_, &SIMDCapabilities::avx512f_,
                &SIMDCapabilities::avx512_dq_, &SIMDCapabilities::avx512_bw_,
                &SIMDCapabilities::avx512_vl_, &SIMDCapabilities::avx512_vnni_,
                &SIMDCapabilities::avx512_bf16_, &SIMDCapabilities::vnni_});
    cpus.push_back(std::move(cpu));
  }

  {
    /* Sapphire Rapids (Xeon 4th gen): AMX-BF16/AMX-INT8 and AVX-512
       FP16, but neither AMX-FP16 nor AVX-VNNI-INT8 — both arrived with
       Granite Rapids. */
    CpuModel cpu{
        .name = "SapphireRapids", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx,
            kSse42Bit | kFma3Bit | kAvxBit | kF16cBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx,
            kAvx2Bit | kAvx512FBit | kAvx512DqBit | kAvx512BwBit |
                kAvx512VlBit);
    setBits(cpu.raw.leaf7_0, Reg::kEcx, kAvx512VnniBit);
    setBits(cpu.raw.leaf7_0, Reg::kEdx,
            kAmxBf16Bit | kAvx512Fp16Bit | kAmxInt8Bit);
    setBits(cpu.raw.leaf7_1, Reg::kEax, kAvxVnniBit | kAvx512Bf16Bit);
    expectTrue(cpu.expected,
               {&SIMDCapabilities::sse4_2_, &SIMDCapabilities::fma3_,
                &SIMDCapabilities::avx_, &SIMDCapabilities::f16c_,
                &SIMDCapabilities::avx2_, &SIMDCapabilities::avx512f_,
                &SIMDCapabilities::avx512_dq_, &SIMDCapabilities::avx512_bw_,
                &SIMDCapabilities::avx512_vl_, &SIMDCapabilities::avx512_vnni_,
                &SIMDCapabilities::amx_bf16_, &SIMDCapabilities::avx512_fp16_,
                &SIMDCapabilities::amx_int8_, &SIMDCapabilities::avx2_vnni_,
                &SIMDCapabilities::avx512_bf16_, &SIMDCapabilities::amx_,
                &SIMDCapabilities::vnni_});
    cpus.push_back(std::move(cpu));
  }

  {
    /* Granite Rapids (Xeon 6 gen-1): the Sapphire Rapids feature set
       plus AMX-FP16 and AVX-VNNI-INT8, shipping AVX10 version 1. */
    CpuModel cpu{
        .name = "GraniteRapids", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx,
            kSse42Bit | kFma3Bit | kAvxBit | kF16cBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx,
            kAvx2Bit | kAvx512FBit | kAvx512DqBit | kAvx512BwBit |
                kAvx512VlBit);
    setBits(cpu.raw.leaf7_0, Reg::kEcx, kAvx512VnniBit);
    setBits(cpu.raw.leaf7_0, Reg::kEdx,
            kAmxBf16Bit | kAvx512Fp16Bit | kAmxInt8Bit);
    setBits(cpu.raw.leaf7_1, Reg::kEax,
            kAvxVnniBit | kAvx512Bf16Bit | kAmxFp16Bit);
    setBits(cpu.raw.leaf7_1, Reg::kEdx, kAvxVnniInt8Bit | kAvx10PresenceBit);
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEbx)] = 1u;
    expectTrue(
        cpu.expected,
        {&SIMDCapabilities::sse4_2_,      &SIMDCapabilities::fma3_,
         &SIMDCapabilities::avx_,         &SIMDCapabilities::f16c_,
         &SIMDCapabilities::avx2_,        &SIMDCapabilities::avx512f_,
         &SIMDCapabilities::avx512_dq_,   &SIMDCapabilities::avx512_bw_,
         &SIMDCapabilities::avx512_vl_,   &SIMDCapabilities::avx512_vnni_,
         &SIMDCapabilities::amx_bf16_,    &SIMDCapabilities::avx512_fp16_,
         &SIMDCapabilities::amx_int8_,    &SIMDCapabilities::avx2_vnni_,
         &SIMDCapabilities::avx512_bf16_, &SIMDCapabilities::amx_fp16_,
         &SIMDCapabilities::avx2_int8_,   &SIMDCapabilities::amx_,
         &SIMDCapabilities::vnni_,        &SIMDCapabilities::avx10_1_});
    cpus.push_back(std::move(cpu));
  }

  {
    /* Clearwater Forest: first part with AVX10 version 2; enumerates
       the same vector/AMX feature set as Granite Rapids. Keeps the
       version >= 2 decode path covered with realistic data. */
    CpuModel cpu{
        .name = "ClearwaterForest", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx,
            kSse42Bit | kFma3Bit | kAvxBit | kF16cBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx,
            kAvx2Bit | kAvx512FBit | kAvx512DqBit | kAvx512BwBit |
                kAvx512VlBit);
    setBits(cpu.raw.leaf7_0, Reg::kEcx, kAvx512VnniBit);
    setBits(cpu.raw.leaf7_0, Reg::kEdx,
            kAmxBf16Bit | kAvx512Fp16Bit | kAmxInt8Bit);
    setBits(cpu.raw.leaf7_1, Reg::kEax,
            kAvxVnniBit | kAvx512Bf16Bit | kAmxFp16Bit);
    setBits(cpu.raw.leaf7_1, Reg::kEdx, kAvxVnniInt8Bit | kAvx10PresenceBit);
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEbx)] = 2u;
    expectTrue(
        cpu.expected,
        {&SIMDCapabilities::sse4_2_,      &SIMDCapabilities::fma3_,
         &SIMDCapabilities::avx_,         &SIMDCapabilities::f16c_,
         &SIMDCapabilities::avx2_,        &SIMDCapabilities::avx512f_,
         &SIMDCapabilities::avx512_dq_,   &SIMDCapabilities::avx512_bw_,
         &SIMDCapabilities::avx512_vl_,   &SIMDCapabilities::avx512_vnni_,
         &SIMDCapabilities::amx_bf16_,    &SIMDCapabilities::avx512_fp16_,
         &SIMDCapabilities::amx_int8_,    &SIMDCapabilities::avx2_vnni_,
         &SIMDCapabilities::avx512_bf16_, &SIMDCapabilities::amx_fp16_,
         &SIMDCapabilities::avx2_int8_,   &SIMDCapabilities::amx_,
         &SIMDCapabilities::vnni_,        &SIMDCapabilities::avx10_1_,
         &SIMDCapabilities::avx10_2_});
    cpus.push_back(std::move(cpu));
  }

  {
    /* Lunar Lake: AVX10 version 1 without legacy AVX-512 CPUID
       enumeration; every avx512_* flag must stay off despite the
       AVX10 presence bit. */
    CpuModel cpu{.name = "LunarLake", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx,
            kSse42Bit | kFma3Bit | kAvxBit | kF16cBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx, kAvx2Bit);
    setBits(cpu.raw.leaf7_1, Reg::kEdx, kAvx10PresenceBit);
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEbx)] = 1u;
    expectTrue(cpu.expected,
               {&SIMDCapabilities::sse4_2_, &SIMDCapabilities::fma3_,
                &SIMDCapabilities::avx_, &SIMDCapabilities::f16c_,
                &SIMDCapabilities::avx2_, &SIMDCapabilities::avx10_1_});
    cpus.push_back(std::move(cpu));
  }

  {
    CpuModel cpu{.name = "Avx10V0", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf7_1, Reg::kEdx, kAvx10PresenceBit);
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEbx)] = 0u;
    cpus.push_back(std::move(cpu));
  }

  {
    CpuModel cpu{
        .name = "Avx10GarbageLeaf", .raw = {}, .expected = allFlagsFalse()};
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEax)] = 0xFFFFFFFFu;
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEbx)] = 0xFFFFFFFFu;
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEcx)] = 0xFFFFFFFFu;
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEdx)] = 0xFFFFFFFFu;
    cpus.push_back(std::move(cpu));
  }

  /* Presence bit in the wrong subleaf (7.0) must not enable AVX10. */
  {
    CpuModel cpu{
        .name = "Avx10WrongLeaf", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf7_0, Reg::kEdx, kAvx10PresenceBit);
    cpu.raw.leaf24_0[static_cast<std::size_t>(Reg::kEbx)] = 2u;
    cpus.push_back(std::move(cpu));
  }

  {
    CpuModel cpu{.name = "NoSubleaf1", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf1, Reg::kEcx, kSse42Bit | kAvxBit);
    setBits(cpu.raw.leaf7_0, Reg::kEbx, kAvx2Bit | kAvx512FBit);
    expectTrue(cpu.expected,
               {&SIMDCapabilities::sse4_2_, &SIMDCapabilities::avx_,
                &SIMDCapabilities::avx2_, &SIMDCapabilities::avx512f_});
    cpus.push_back(std::move(cpu));
  }

  {
    CpuModel cpu{.name = "AmxFp16Only", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf7_1, Reg::kEax, kAmxFp16Bit);
    expectTrue(cpu.expected,
               {&SIMDCapabilities::amx_fp16_, &SIMDCapabilities::amx_});
    cpus.push_back(std::move(cpu));
  }

  /* fp16 alone: no gating on avx512f. */
  {
    CpuModel cpu{.name = "Fp16Only", .raw = {}, .expected = allFlagsFalse()};
    setBits(cpu.raw.leaf7_0, Reg::kEdx, kAvx512Fp16Bit);
    expectTrue(cpu.expected, {&SIMDCapabilities::avx512_fp16_});
    cpus.push_back(std::move(cpu));
  }

  return cpus;
}

/**
 * @class SimdCapsMapping
 * @brief Fixture running one synthetic CPU model per test case.
 */
class SimdCapsMapping : public ::testing::TestWithParam<CpuModel> {};

/**
 * @test Decodes each catalog CPU into its exact expected flag vector.
 */
TEST_P(SimdCapsMapping, MapsCpuidBitsToExpectedFlags) {
  const CpuModel &cpu = GetParam();

  SIMDCapabilities caps{};
  get_simd_capabilities_from_cpuid(&cpu.raw, &caps);

  SCOPED_TRACE(cpu.name);
  for (const auto &[field, want] : cpu.expected) {
    EXPECT_EQ(caps.*field, want);
  }
}

INSTANTIATE_TEST_SUITE_P(SyntheticCpus, SimdCapsMapping,
                         ::testing::ValuesIn(buildCpuCatalog()),
                         [](const ::testing::TestParamInfo<CpuModel> &info)
                             -> std::string { return info.param.name; });

} // namespace
