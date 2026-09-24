/**
 * @file GpuCasting.hpp
 * @brief Device tensor fixtures for the GPU dtype casting suites.
 *
 * Builds on the CPU oracle fixtures (utils/Oracle.hpp): a host pair
 * provides staging buffers, while matching DEVICE_GPU tensors carry
 * the conversion under test. Results come home through a
 * device-to-host transfer and compare byte-exact against the scalar
 * reference kernel, which is the pinned portable semantic both
 * backends mirror (strict bitwise contract, NaN payloads included).
 *
 * Tests needing a device call NOVA_TEST_REQUIRE_GPU() first (it must
 * run in the test body so GTEST_SKIP returns correctly); pure status
 * edges that need no device launch run unconditionally.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include "utils/Oracle.hpp"
#include "utils/TensorUtils.hpp"

#ifdef NOVA_HAS_CUDA
extern "C" novaStatus_t launchCudaDtypeCastingKernel(const Tensor *restrict src,
                                                     Tensor *restrict dst);
#endif
#ifdef NOVA_HAS_HIP
extern "C" novaStatus_t launchHipDtypeCastingKernel(const Tensor *restrict src,
                                                    Tensor *restrict dst);
#endif

namespace tests::gpu {

using ncore::wrappers::TensorCXX;

/**
 * @struct GpuCastPair
 * @brief A host reference pair plus matching device tensors.
 */
struct GpuCastPair {
  tests::casting::CastPair host; ///< CPU staging buffers and reference.
  TensorCXX devSrc;              ///< Source tensor on the active GPU backend.
  TensorCXX devDst;              ///< Destination tensor on the GPU backend.
  TensorCXX home;                ///< CPU buffer receiving the device result.
  bool ok = false;               ///< True when every step below succeeded.
};

/**
 * @brief Calls the compiled-in backend launcher directly.
 *
 * Used only for edges that need no device launch (shape mismatch and
 * empty tensors return before any device query). Exactly one backend
 * is compiled in per preset, so exactly one branch exists.
 */
inline novaStatus_t callBackendLauncher(const Tensor *src, Tensor *dst) {
#ifdef NOVA_HAS_CUDA
  return launchCudaDtypeCastingKernel(src, dst);
#elif defined(NOVA_HAS_HIP)
  return launchHipDtypeCastingKernel(src, dst);
#else
  (void)src;
  (void)dst;
  return {.err = novaBackendNotCompiled,
          .message = nova_get_error_msg(novaBackendNotCompiled, nullptr)};
#endif
}

/**
 * @brief Allocates a host reference pair plus matching device tensors.
 *
 * Mirrors makePair sizing (the count rounds up to even when either
 * side is packed) so storage-unit counts agree on both sides. The
 * host source keeps its deterministic fill; stageToDevice() copies it
 * to the device. Call NOVA_TEST_REQUIRE_GPU() before this helper so
 * missing hardware skips instead of failing allocation or transfer.
 *
 * @param[in] from         Source dtype.
 * @param[in] to           Destination dtype.
 * @param[in] logicalCount Logical element count (must be positive).
 * @param[in] seed         Deterministic fill seed.
 * @return Pair descriptor; check ok before use.
 */
[[nodiscard]] inline GpuCastPair
makeDevicePair(DType_ from, DType_ to, size_t logicalCount, uint32_t seed) {
  GpuCastPair pair;
  pair.host = tests::casting::makePair(from, to, logicalCount, seed);
  if (!pair.host.ok) {
    return pair;
  }
  novaStatus_t st{};
  size_t n = logicalCount;
  if (dtype_packing_factor(from) == 2 || dtype_packing_factor(to) == 2) {
    n += n % 2;
  }
  const std::vector<size_t> shape{n};
  constexpr bool kNoGrad = false;
  constexpr bool kNoPin = false;
  pair.devSrc = TensorCXX(shape, from, DEVICE_GPU, kNoGrad, kNoPin, &st);
  if (st.err != novaSuccess) {
    ADD_FAILURE() << "makeDevicePair: device src allocation failed: "
                  << (st.message != nullptr ? st.message : "");
    return pair;
  }
  pair.devDst = TensorCXX(shape, to, DEVICE_GPU, kNoGrad, kNoPin, &st);
  if (st.err != novaSuccess) {
    ADD_FAILURE() << "makeDevicePair: device dst allocation failed: "
                  << (st.message != nullptr ? st.message : "");
    return pair;
  }
  pair.home = TensorCXX(shape, to, DEVICE_CPU, kNoGrad, kNoPin, &st);
  if (st.err != novaSuccess) {
    ADD_FAILURE() << "makeDevicePair: home allocation failed: "
                  << (st.message != nullptr ? st.message : "");
    return pair;
  }
  if (pair.devSrc.getSize() != pair.host.src.getSize() ||
      pair.devDst.getSize() != pair.host.dst.getSize()) {
    ADD_FAILURE() << "makeDevicePair: device/host size mismatch";
    return pair;
  }
  Tensor staged = pair.host.src.mutableCTensor();
  Tensor device = pair.devSrc.mutableCTensor();
  st = transf_tensor_from_host(&staged, &device);
  if (st.err != novaSuccess) {
    ADD_FAILURE() << "makeDevicePair: stage to device failed: "
                  << (st.message != nullptr ? st.message : "");
    return pair;
  }
  pair.ok = true;
  return pair;
}

/**
 * @brief Re-stages the host source buffer to the device source.
 *
 * Call after writeRaw/writeFloatValue on the host source when a test
 * needs exact patterns instead of the deterministic fill.
 */
inline novaStatus_t stageToDevice(GpuCastPair &pair) {
  Tensor staged = pair.host.src.mutableCTensor();
  Tensor device = pair.devSrc.mutableCTensor();
  return transf_tensor_from_host(&staged, &device);
}

/**
 * @brief Runs cast() on the device pair (public entry point).
 */
inline novaStatus_t castOnDevice(GpuCastPair &pair, DType_ target) {
  Tensor src = pair.devSrc.mutableCTensor();
  Tensor dst = pair.devDst.mutableCTensor();
  return cast(&src, &dst, target);
}

/**
 * @brief Runs the scalar reference kernel on the host pair.
 *
 * The scalar kernel is the pinned portable semantic both backends
 * mirror; the SIMD dispatch wrapper is CPU-backend business covered
 * by the CPU ISA suites, so bitwise verdicts compare against scalar.
 */
inline void scalarRef(GpuCastPair &pair, const tests::casting::PairInfo *info) {
  Tensor src = pair.host.src.mutableCTensor();
  Tensor dst = pair.host.dst.mutableCTensor();
  info->scalar(&src, &dst);
}

/**
 * @brief Transfers the device result into the home buffer.
 */
inline novaStatus_t fetchHome(GpuCastPair &pair) {
  Tensor device = pair.devDst.mutableCTensor();
  Tensor home = pair.home.mutableCTensor();
  return transf_tensor_from_device(&device, &home);
}

/**
 * @brief Expects the fetched device result to equal the host reference
 *        byte for byte.
 */
inline void expectHomeEqualRef(GpuCastPair &pair, const char *label) {
  const Tensor &home = pair.home.getCTensor();
  const Tensor &ref = pair.host.dst.getCTensor();
  const size_t homeBytes = pair.home.getSize() * pair.home.itemSize();
  const size_t refBytes = pair.host.dst.getSize() * pair.host.dst.itemSize();
  ASSERT_EQ(homeBytes, refBytes) << label;
  EXPECT_EQ(std::memcmp(home.data.data, ref.data.data, homeBytes), 0) << label;
}

/**
 * @brief Sanitizes a pair label into a valid GoogleTest case name.
 */
inline std::string sanitizeLabel(const std::string &label) {
  std::string name;
  for (const char c : label) {
    name.push_back((c == '-' || c == '>') ? '_' : c);
  }
  return name;
}

} // namespace tests::gpu
