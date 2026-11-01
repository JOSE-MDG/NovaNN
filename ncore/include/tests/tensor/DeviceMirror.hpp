/**
 * @file DeviceMirror.hpp
 * @brief Device buffer with its host mirror for the RNG parity suites.
 */

#pragma once

#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

namespace tests::tensor {

/**
 * @class DeviceMirror
 * @brief GPU buffer plus its CPU mirror, allocated through the tensor layer.
 *
 * The mirrors are reached through accessors instead of cached copies:
 * a cached Tensor would freeze the metadata the tensor layer may update
 * in place after a transfer.
 */
class DeviceMirror final {
public:
  DeviceMirror(size_t n, DType_ dtype)
      : st{}, dev(std::vector<size_t>{n}, dtype, DEVICE_GPU, false, false, &st),
        home(std::vector<size_t>{n}, dtype, DEVICE_CPU, false, false, &st) {}

  /// Status of the allocation that backs both mirrors.
  novaStatus_t st{};

  /// Device tensor carrying the data under test.
  Tensor &devTen() { return dev.mutableCTensor(); }

  /// Host tensor receiving the device-to-host copy.
  Tensor &homeTen() { return home.mutableCTensor(); }

  /// D2H copy after a device fill.
  void fetch() {
    st = transf_tensor_from_device(&devTen(), &homeTen());
    EXPECT_EQ(st.err, novaSuccess) << (st.message != nullptr ? st.message : "");
  }

private:
  ncore::wrappers::TensorCXX dev;
  ncore::wrappers::TensorCXX home;
};

} // namespace tests::tensor
