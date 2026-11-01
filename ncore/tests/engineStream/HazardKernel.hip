#include <hip/hip_runtime.h>

#include "HazardKernel.hpp"

extern "C" __global__ void novaTestNullWriteKernel(int *dst, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *dst = value;
  }
}

extern "C" void novaTestNullWrite(int *dst, int value, void *stream) {
  novaTestNullWriteKernel<<<1, 32, 0, static_cast<hipStream_t>(stream)>>>(
      dst, value);
}

extern "C" __global__ void novaTestWriteFloatKernel(float *dst, int n,
                                                    float value) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = value;
  }
}

extern "C" void novaTestWriteFloat(float *dst, int n, float value,
                                   void *stream) {
  novaTestWriteFloatKernel<<<64, 64, 0, static_cast<hipStream_t>(stream)>>>(
      dst, n, value);
}

extern "C" __global__ void novaTestScaleIntoKernel(const float *src, float *dst,
                                                   int n, float scale) {
  int i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i < n) {
    dst[i] = src[i] * scale;
  }
}

extern "C" void novaTestScaleInto(const float *src, float *dst, int n,
                                  float scale, void *stream) {
  novaTestScaleIntoKernel<<<64, 64, 0, static_cast<hipStream_t>(stream)>>>(
      src, dst, n, scale);
}
