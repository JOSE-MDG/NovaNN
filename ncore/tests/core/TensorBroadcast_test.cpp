/**
 * @file TensorBroadcast_test.cpp
 * @brief Unit tests for tensor broadcasting.
 *
 * Exercises `broadcast()` @c ncore/src/core/tensor.c per-input views with
 * the broadcast shape and stretched strides, scalar and META groups, the
 * packed-dtype last-dim rule, rejection of collected sources and of
 * destinations that already hold resources, refcount restoration through
 * `collect()`, and the `is_broadcast_view()` detection of repeated
 * elements.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>

#include <ncore/core/broadcast.h>
#include <ncore/core/device.h>
#include <ncore/core/dtype.h>
#include <ncore/core/status.h>
#include <ncore/headeronly/wrappers/tensor.hh>
#include <ncore/tensor.h>

#include <tests/rng/Fill.hpp>

namespace {

using ncore::wrappers::TensorCXX;
using tests::rng::fillTensor;

/**
 * @brief Collect every view of a broadcast group.
 *
 * @details
 * Broadcast outputs are raw C views: the C API writes one retained
 * reference per output into a plain array, so no wrapper can own them.
 * Collecting every view before leaving the scope restores the source
 * refcount. META views hold no reference, so the call is a safe no-op
 * for them.
 */
void collectViews(Tensor *outputs[], size_t count) {
  for (size_t i = 0; i < count; ++i) {
    SCOPED_TRACE(::testing::Message() << "output=" << i);
    EXPECT_EQ(collect(outputs[i]).err, novaSuccess);
  }
}

/// Views of a {2, 1} and a {1, 3} source over the common shape {2, 3}.
TEST(TensorBroadcast, CommonShapeProducesViews) {
  novaStatus_t st{};
  TensorCXX column({2, 1}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX row({1, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&column.mutableCTensor(), &row.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  st = broadcast(inputs, outputs, 2);
  ASSERT_EQ(st.err, novaSuccess);

  const shape_t expectedShape = {2, 3};
  const strides_t expectedStrides[2] = {{4, 0}, {0, 4}};
  for (size_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(::testing::Message() << "input=" << i);
    EXPECT_EQ(outputs[i]->dtype, DType_::Float32);
    EXPECT_EQ(outputs[i]->device, Device_::DEVICE_CPU);
    EXPECT_EQ(outputs[i]->item_size, size_t{4});
    EXPECT_EQ(outputs[i]->ndims, size_t{2});
    EXPECT_EQ(outputs[i]->size, size_t{6});
    EXPECT_EQ(outputs[i]->logical_size, size_t{6});
    for (size_t dim = 0; dim < 2; ++dim) {
      EXPECT_EQ(outputs[i]->shape[dim], expectedShape[dim])
          << "shape[" << dim << "]";
      EXPECT_EQ(outputs[i]->strides[dim], expectedStrides[i][dim])
          << "strides[" << dim << "]";
    }
    EXPECT_TRUE(outputs[i]->is_view_);
    EXPECT_FALSE(outputs[i]->is_leaf_);
    EXPECT_EQ(outputs[i]->storage, inputs[i]->storage);
    EXPECT_EQ(outputs[i]->data.data, inputs[i]->data.data);
    EXPECT_EQ(outputs[i]->offset, inputs[i]->offset);
  }

  collectViews(outputs, 2);
}

/// Each view holds one retained reference; collecting them leaves the
/// source with its own reference and a usable buffer.
TEST(TensorBroadcast, CollectingViewsLeavesSourceAlive) {
  novaStatus_t st{};
  TensorCXX src({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor sv = src.getCTensor();
  EXPECT_EQ(fillTensor(sv).err, novaSuccess);

  Tensor *inputs[2] = {&sv, &sv};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  ASSERT_EQ(broadcast(inputs, outputs, 2).err, novaSuccess);
  collectViews(outputs, 2);

  EXPECT_TRUE(is_allocated(&sv));
  EXPECT_NE(sv.storage, nullptr);
  sv.data.f32[0] = 1.5F;
  EXPECT_FLOAT_EQ(sv.data.f32[0], 1.5F);
}

/// A scalar contributes no dimension: its view stretches to the whole
/// broadcast shape and reads one element.
TEST(TensorBroadcast, ScalarWithTensorProducesTensorShape) {
  novaStatus_t st{};
  TensorCXX ssrc(DType_::Float32, Device_::DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor ssv = ssrc.getCTensor();
  ASSERT_TRUE(is_scalar(&ssv));

  TensorCXX dense({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                  &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&ssv, &dense.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  ASSERT_EQ(broadcast(inputs, outputs, 2).err, novaSuccess);

  const Tensor sov = *outputs[0];
  EXPECT_EQ(sov.ndims, size_t{2});
  EXPECT_EQ(sov.shape[0], size_t{2});
  EXPECT_EQ(sov.shape[1], size_t{3});
  EXPECT_EQ(sov.strides[0], size_t{0});
  EXPECT_EQ(sov.strides[1], size_t{0});
  EXPECT_EQ(sov.size, size_t{6});
  EXPECT_TRUE(sov.is_view_);
  EXPECT_EQ(sov.storage, inputs[0]->storage);
  EXPECT_EQ(sov.data.data, inputs[0]->data.data);

  const Tensor dov = *outputs[1];
  EXPECT_EQ(dov.ndims, size_t{2});
  EXPECT_EQ(dov.shape[0], size_t{2});
  EXPECT_EQ(dov.shape[1], size_t{3});
  EXPECT_EQ(dov.strides[0], size_t{12});
  EXPECT_EQ(dov.strides[1], size_t{4});
  EXPECT_EQ(dov.storage, inputs[1]->storage);

  collectViews(outputs, 2);
}

/// An all-scalar group has no dimensions to merge: the outputs stay
/// scalar and only the view flags flip. */
TEST(TensorBroadcast, AllScalarsGroupKeepsScalarRank) {
  novaStatus_t st{};
  TensorCXX first(DType_::Float32, Device_::DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX second(DType_::Float32, Device_::DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&first.mutableCTensor(), &second.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  ASSERT_EQ(broadcast(inputs, outputs, 2).err, novaSuccess);

  for (size_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(::testing::Message() << "output=" << i);
    EXPECT_EQ(outputs[i]->ndims, size_t{0});
    EXPECT_TRUE(is_scalar(outputs[i]));
    EXPECT_EQ(outputs[i]->size, size_t{1});
    EXPECT_EQ(outputs[i]->logical_size, size_t{1});
    EXPECT_TRUE(outputs[i]->is_view_);
    EXPECT_FALSE(outputs[i]->is_leaf_);
    EXPECT_EQ(outputs[i]->storage, inputs[i]->storage);
  }

  collectViews(outputs, 2);
}

/// A requires grad scalar keeps a scalar grad shell through the group:
/// the shell is created with zero dims, so no stride is written. */
TEST(TensorBroadcast, ScalarGradShellStaysScalarInGroup) {
  novaStatus_t st{};
  TensorCXX ssrc(DType_::Float32, Device_::DEVICE_CPU, true, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor ssv = ssrc.getCTensor();
  ASSERT_TRUE(is_scalar(&ssv));
  ASSERT_TRUE(ssv.grad != nullptr);
  ASSERT_TRUE(is_scalar_grad(ssv.grad));

  TensorCXX dense({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                  &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&ssv, &dense.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  ASSERT_EQ(broadcast(inputs, outputs, 2).err, novaSuccess);

  const Tensor sov = *outputs[0];
  EXPECT_EQ(sov.ndims, size_t{2});
  EXPECT_EQ(sov.shape[0], size_t{2});
  EXPECT_EQ(sov.shape[1], size_t{3});
  EXPECT_EQ(sov.strides[0], size_t{0});
  EXPECT_EQ(sov.strides[1], size_t{0});
  EXPECT_EQ(sov.size, size_t{6});
  EXPECT_TRUE(sov.is_view_);

  ASSERT_TRUE(sov.grad != nullptr);
  EXPECT_NE(sov.grad, inputs[0]->grad);
  EXPECT_TRUE(is_scalar_grad(sov.grad));
  EXPECT_EQ(sov.grad->ndims, size_t{0});
  EXPECT_EQ(sov.grad->shape[0], size_t{0});
  EXPECT_EQ(sov.grad->size, size_t{1});
  EXPECT_EQ(sov.grad->storage, nullptr);
  EXPECT_FALSE(sov.grad->is_allocated_);
  EXPECT_EQ(outputs[1]->grad, nullptr);

  collectViews(outputs, 2);
  EXPECT_TRUE(is_grad_collected(outputs[0]->grad));
}

/// META tensors carry no buffer: the group broadcasts as pure layout and
/// every output stays META with no storage to release. */
TEST(TensorBroadcast, MetaGroupProducesMetaViews) {
  novaStatus_t st{};
  TensorCXX column({2, 1}, DType_::Float32, Device_::DEVICE_META, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX row({1, 3}, DType_::Float32, Device_::DEVICE_META, false, false,
                &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&column.mutableCTensor(), &row.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  ASSERT_EQ(broadcast(inputs, outputs, 2).err, novaSuccess);

  const shape_t expectedShape = {2, 3};
  const strides_t expectedStrides[2] = {{4, 0}, {0, 4}};
  for (size_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(::testing::Message() << "input=" << i);
    EXPECT_EQ(outputs[i]->device, Device_::DEVICE_META);
    EXPECT_EQ(outputs[i]->storage, nullptr);
    EXPECT_EQ(outputs[i]->data.data, nullptr);
    EXPECT_FALSE(is_allocated(outputs[i]));
    EXPECT_TRUE(outputs[i]->is_view_);
    EXPECT_FALSE(outputs[i]->is_leaf_);
    EXPECT_EQ(outputs[i]->ndims, size_t{2});
    EXPECT_EQ(outputs[i]->size, size_t{6});
    for (size_t dim = 0; dim < 2; ++dim) {
      EXPECT_EQ(outputs[i]->shape[dim], expectedShape[dim])
          << "shape[" << dim << "]";
      EXPECT_EQ(outputs[i]->strides[dim], expectedStrides[i][dim])
          << "strides[" << dim << "]";
    }
  }

  collectViews(outputs, 2);
}

/* Creation takes logical shapes for the packed dtype, so the last dim
   lives in units of two. Both members agree there: nothing stretches
   and the last byte stride stays. */
TEST(TensorBroadcast, PackedGroupBroadcastsSameLastDim) {
  novaStatus_t st{};
  TensorCXX unit({1, 6}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, false,
                 false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX dense({4, 6}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, false,
                  false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&unit.mutableCTensor(), &dense.mutableCTensor()};
  ASSERT_EQ(inputs[0]->shape[0], size_t{1});
  ASSERT_EQ(inputs[0]->shape[1], size_t{3});
  ASSERT_EQ(inputs[1]->shape[0], size_t{4});
  ASSERT_EQ(inputs[1]->shape[1], size_t{3});

  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};
  st = broadcast(inputs, outputs, 2);
  ASSERT_EQ(st.err, novaSuccess);

  const shape_t expectedShape = {4, 3};
  const strides_t expectedStrides[2] = {{0, 1}, {3, 1}};
  for (size_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(::testing::Message() << "input=" << i);
    EXPECT_EQ(outputs[i]->item_size, size_t{1});
    EXPECT_EQ(outputs[i]->ndims, size_t{2});
    for (size_t dim = 0; dim < 2; ++dim) {
      EXPECT_EQ(outputs[i]->shape[dim], expectedShape[dim])
          << "shape[" << dim << "]";
      EXPECT_EQ(outputs[i]->strides[dim], expectedStrides[i][dim])
          << "strides[" << dim << "]";
    }
    EXPECT_EQ(outputs[i]->size, size_t{12});
    EXPECT_EQ(outputs[i]->logical_size, size_t{24});
    EXPECT_EQ(outputs[i]->storage, inputs[i]->storage);
    EXPECT_TRUE(outputs[i]->is_view_);
  }

  collectViews(outputs, 2);
}

/// The storage sizes of the last dim disagree (3 vs 4), so the packed
/// rule rejects the group before any view exists. */
TEST(TensorBroadcast, PackedGroupRejectsMismatchedLastDim) {
  novaStatus_t st{};
  TensorCXX first({2, 6}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, false,
                  false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX second({2, 8}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, false,
                   false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&first.mutableCTensor(), &second.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  st = broadcast(inputs, outputs, 2);
  EXPECT_EQ(st.err, novaInvalidShape);
  ASSERT_NE(st.message, nullptr);
  EXPECT_NE(std::strstr(st.message, "cannot stretch the last dim"), nullptr)
      << st.message;
  EXPECT_EQ(outputs[0]->storage, nullptr);
  EXPECT_EQ(outputs[1]->storage, nullptr);
}

/// A packed scalar contributes no dimension, so the packed rule skips it
/// and the scalar's view stretches to the dense member's shape. */
TEST(TensorBroadcast, PackedScalarGroupTakesTensorShape) {
  novaStatus_t st{};
  TensorCXX ssrc(DType_::Float4E2M1fn, Device_::DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  Tensor ssv = ssrc.getCTensor();
  ASSERT_TRUE(is_scalar(&ssv));
  ASSERT_EQ(ssv.logical_size, size_t{2});

  TensorCXX dense({2, 6}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, false,
                  false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&ssv, &dense.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  ASSERT_EQ(broadcast(inputs, outputs, 2).err, novaSuccess);

  const Tensor sov = *outputs[0];
  EXPECT_EQ(sov.ndims, size_t{2});
  EXPECT_EQ(sov.shape[0], size_t{2});
  EXPECT_EQ(sov.shape[1], size_t{3});
  EXPECT_EQ(sov.strides[0], size_t{0});
  EXPECT_EQ(sov.strides[1], size_t{0});
  EXPECT_EQ(sov.size, size_t{6});
  EXPECT_EQ(sov.logical_size, size_t{12});
  EXPECT_TRUE(sov.is_view_);
  EXPECT_EQ(sov.storage, inputs[0]->storage);
  EXPECT_EQ(sov.data.data, inputs[0]->data.data);

  const Tensor dov = *outputs[1];
  EXPECT_EQ(dov.strides[0], size_t{3});
  EXPECT_EQ(dov.strides[1], size_t{1});
  EXPECT_EQ(dov.size, size_t{6});
  EXPECT_EQ(dov.logical_size, size_t{12});

  collectViews(outputs, 2);
}

/* Requires grad sources give every view a fresh grad shell whose shape
   is the source's storage shape: the shell creation takes logical
   units, so the packed last dim survives shell creation. */
TEST(TensorBroadcast, PackedGradShellsFollowSourceStorageShape) {
  novaStatus_t st{};
  TensorCXX dense({2, 6}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, true,
                  false, &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX unit({1, 6}, DType_::Float4E2M1fn, Device_::DEVICE_CPU, true, false,
                 &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&dense.mutableCTensor(), &unit.mutableCTensor()};
  ASSERT_TRUE(inputs[0]->grad != nullptr);
  ASSERT_EQ(inputs[0]->grad->shape[0], size_t{2});
  ASSERT_EQ(inputs[0]->grad->shape[1], size_t{3});
  ASSERT_TRUE(inputs[1]->grad != nullptr);
  ASSERT_EQ(inputs[1]->grad->shape[0], size_t{1});
  ASSERT_EQ(inputs[1]->grad->shape[1], size_t{3});

  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};
  st = broadcast(inputs, outputs, 2);
  ASSERT_EQ(st.err, novaSuccess);

  const shape_t expectedShape = {2, 3};
  const strides_t expectedStrides[2] = {{3, 1}, {0, 1}};
  for (size_t i = 0; i < 2; ++i) {
    SCOPED_TRACE(::testing::Message() << "output=" << i);
    EXPECT_EQ(outputs[i]->shape[0], expectedShape[0]);
    EXPECT_EQ(outputs[i]->shape[1], expectedShape[1]);
    EXPECT_EQ(outputs[i]->strides[0], expectedStrides[i][0]);
    EXPECT_EQ(outputs[i]->strides[1], expectedStrides[i][1]);
    EXPECT_EQ(outputs[i]->size, size_t{6});
    EXPECT_EQ(outputs[i]->logical_size, size_t{12});
    EXPECT_TRUE(outputs[i]->is_view_);
    ASSERT_TRUE(outputs[i]->grad != nullptr);
    EXPECT_NE(outputs[i]->grad, inputs[i]->grad);
    EXPECT_EQ(outputs[i]->grad->dtype, DType_::Float4E2M1fn);
    EXPECT_EQ(outputs[i]->grad->shape[0], inputs[i]->grad->shape[0]);
    EXPECT_EQ(outputs[i]->grad->shape[1], inputs[i]->grad->shape[1]);
    EXPECT_EQ(outputs[i]->grad->storage, nullptr);
    EXPECT_FALSE(outputs[i]->grad->is_allocated_);
  }

  collectViews(outputs, 2);
  EXPECT_TRUE(is_grad_collected(outputs[0]->grad));
  EXPECT_TRUE(is_grad_collected(outputs[1]->grad));
}

TEST(TensorBroadcast, IncompatibleShapesRejected) {
  novaStatus_t st{};
  TensorCXX first({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                  &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX second({2, 4}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&first.mutableCTensor(), &second.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  st = broadcast(inputs, outputs, 2);
  EXPECT_EQ(st.err, novaInvalidShape);
  ASSERT_NE(st.message, nullptr);
  EXPECT_EQ(outputs[0]->storage, nullptr);
  EXPECT_EQ(outputs[1]->storage, nullptr);
}

TEST(TensorBroadcast, MixedDtypeGroupRejected) {
  novaStatus_t st{};
  TensorCXX first({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                  &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX second({2, 3}, DType_::BFloat16, Device_::DEVICE_CPU, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&first.mutableCTensor(), &second.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  st = broadcast(inputs, outputs, 2);
  EXPECT_EQ(st.err, novaInvalidDtype);
  ASSERT_NE(st.message, nullptr);
  EXPECT_EQ(outputs[0]->storage, nullptr);
}

TEST(TensorBroadcast, MixedDeviceGroupRejected) {
  novaStatus_t st{};
  TensorCXX cpu({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX meta({2, 3}, DType_::Float32, Device_::DEVICE_META, false, false,
                 &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&cpu.mutableCTensor(), &meta.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};

  st = broadcast(inputs, outputs, 2);
  EXPECT_EQ(st.err, novaInvalidDevice);
  ASSERT_NE(st.message, nullptr);
  EXPECT_EQ(outputs[0]->storage, nullptr);
}

/* The pre-filled destination is a retained view: the guard fires at
   index 0 before the fail path runs, so the reference is released
   here. */
TEST(TensorBroadcast, OccupiedDestinationRejected) {
  novaStatus_t st{};
  TensorCXX first({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                  &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX second({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor src = first.getCTensor();
  Tensor preset = create_view(&src, src.shape, src.ndims, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[2] = {&src, &second.mutableCTensor()};
  Tensor dst[2] = {};
  Tensor *outputs[2] = {&dst[0], &dst[1]};
  *outputs[0] = preset;

  st = broadcast(inputs, outputs, 2);
  EXPECT_EQ(st.err, novaInvalidTensor);
  ASSERT_NE(st.message, nullptr);
  EXPECT_NE(std::strstr(st.message, "empty destination"), nullptr)
      << st.message;
  EXPECT_EQ(outputs[1]->storage, nullptr);

  EXPECT_EQ(collect(outputs[0]).err, novaSuccess);
}

/* An empty group has nothing to broadcast: success with untouched
   destinations. */
TEST(TensorBroadcast, EmptyGroupSucceeds) {
  novaStatus_t st{};
  TensorCXX src({1}, DType_::Float32, Device_::DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[1] = {&src.mutableCTensor()};
  Tensor dst[1] = {};
  Tensor *outputs[1] = {&dst[0]};

  st = broadcast(inputs, outputs, 0);
  EXPECT_EQ(st.err, novaSuccess);
  EXPECT_EQ(outputs[0]->storage, nullptr);
}

TEST(TensorBroadcast, NullArraysRejected) {
  novaStatus_t st{};
  TensorCXX src({1}, DType_::Float32, Device_::DEVICE_CPU, false, false, &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *inputs[1] = {&src.mutableCTensor()};
  Tensor dst[1] = {};
  Tensor *outputs[1] = {&dst[0]};

  st = broadcast(nullptr, outputs, 1);
  EXPECT_EQ(st.err, novaInvalidPointer);
  EXPECT_NE(st.message, nullptr);

  st = broadcast(inputs, nullptr, 1);
  EXPECT_EQ(st.err, novaInvalidPointer);
  EXPECT_NE(st.message, nullptr);
}

/* Structural detection: a dim above one with stride zero is a broadcast
   view, a plain layout is not. */
TEST(TensorBroadcast, StretchedDimensionDetectsBroadcastView) {
  novaStatus_t st{};
  TensorCXX column({2, 1}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX row({1, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *stretchedIn[2] = {&column.mutableCTensor(), &row.mutableCTensor()};
  Tensor stretchedDst[2] = {};
  Tensor *stretchedOut[2] = {&stretchedDst[0], &stretchedDst[1]};
  ASSERT_EQ(broadcast(stretchedIn, stretchedOut, 2).err, novaSuccess);
  EXPECT_TRUE(is_broadcast_view(stretchedOut[0]));
  EXPECT_TRUE(is_broadcast_view(stretchedOut[1]));
  collectViews(stretchedOut, 2);

  TensorCXX first({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                  &st);
  ASSERT_EQ(st.err, novaSuccess);
  TensorCXX second({2, 3}, DType_::Float32, Device_::DEVICE_CPU, false, false,
                   &st);
  ASSERT_EQ(st.err, novaSuccess);

  Tensor *plainIn[2] = {&first.mutableCTensor(), &second.mutableCTensor()};
  Tensor plainDst[2] = {};
  Tensor *plainOut[2] = {&plainDst[0], &plainDst[1]};
  ASSERT_EQ(broadcast(plainIn, plainOut, 2).err, novaSuccess);
  EXPECT_FALSE(is_broadcast_view(plainOut[0]));
  EXPECT_FALSE(is_broadcast_view(plainOut[1]));
  collectViews(plainOut, 2);
}

} // namespace
