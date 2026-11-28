// bench_harness_hip.hpp - drop-in timing harness (ROCm 7, clang 21, C++23)
//
// Usage:
//   #include "bench_harness_hip.hpp"
//   auto r = bench::run([&](hipStream_t s) { my_kernel<<<grid, block, lds, s>>>(...); });
//   std::printf("median %.3f us, %.1f GB/s\n", r.median_ms * 1e3, bench::gbps(bytes, r.median_ms));
//
// NOTE on cold-cache runs: MI300-class parts have a 256 MB Infinity Cache (MALL) that is NOT
// reflected by the L2 attribute. The default flush size is therefore >= 512 MiB. Override via
// Options::flush_bytes. Shorter kernels (< ~10 us): use bench::run_graph().
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#define BENCH_HIP_CHECK(expr)                                                          \
  do {                                                                                 \
    hipError_t err_ = (expr);                                                          \
    if (err_ != hipSuccess) {                                                          \
      std::fprintf(stderr, "HIP error %s:%d: %s (%s)\n", __FILE__, __LINE__,           \
                   hipGetErrorString(err_), #expr);                                    \
      std::abort();                                                                    \
    }                                                                                  \
  } while (0)

namespace bench {

struct Result {
  float median_ms = 0.f;
  float min_ms = 0.f;
  float p90_ms = 0.f;
};

struct Options {
  int warmup = 20;
  int iters = 100;
  bool flush_l2 = true;          // cold-cache measurement (L2 + Infinity Cache)
  std::size_t flush_bytes = 0;   // 0 -> max(2x L2, 512 MiB)
};

class CacheFlusher {
 public:
  explicit CacheFlusher(std::size_t bytes_override = 0) {
    int dev = 0;
    BENCH_HIP_CHECK(hipGetDevice(&dev));
    int l2 = 0;
    BENCH_HIP_CHECK(hipDeviceGetAttribute(&l2, hipDeviceAttributeL2CacheSize, dev));
    bytes_ = bytes_override ? bytes_override
                            : std::max<std::size_t>(2 * static_cast<std::size_t>(l2),
                                                    512ull << 20);
    BENCH_HIP_CHECK(hipMalloc(&buf_, bytes_));
  }
  ~CacheFlusher() { (void)hipFree(buf_); }
  CacheFlusher(const CacheFlusher&) = delete;
  CacheFlusher& operator=(const CacheFlusher&) = delete;

  void flush(hipStream_t s) { BENCH_HIP_CHECK(hipMemsetAsync(buf_, 0xA5, bytes_, s)); }

 private:
  void* buf_ = nullptr;
  std::size_t bytes_ = 0;
};

template <class Launch>
Result run(Launch&& launch, Options opt = {}, hipStream_t stream = nullptr) {
  std::optional<CacheFlusher> flusher;
  if (opt.flush_l2) flusher.emplace(opt.flush_bytes);
  for (int i = 0; i < opt.warmup; ++i) launch(stream);
  BENCH_HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<hipEvent_t> start(opt.iters), stop(opt.iters);
  for (int i = 0; i < opt.iters; ++i) {
    BENCH_HIP_CHECK(hipEventCreate(&start[i]));
    BENCH_HIP_CHECK(hipEventCreate(&stop[i]));
  }

  for (int i = 0; i < opt.iters; ++i) {
    if (flusher) flusher->flush(stream);
    BENCH_HIP_CHECK(hipEventRecord(start[i], stream));
    launch(stream);
    BENCH_HIP_CHECK(hipEventRecord(stop[i], stream));
  }
  BENCH_HIP_CHECK(hipStreamSynchronize(stream));
  BENCH_HIP_CHECK(hipGetLastError());

  std::vector<float> ms(opt.iters);
  for (int i = 0; i < opt.iters; ++i) {
    BENCH_HIP_CHECK(hipEventElapsedTime(&ms[i], start[i], stop[i]));
    (void)hipEventDestroy(start[i]);
    (void)hipEventDestroy(stop[i]);
  }
  std::sort(ms.begin(), ms.end());
  Result r;
  r.min_ms = ms.front();
  r.median_ms = ms[ms.size() / 2];
  r.p90_ms = ms[static_cast<std::size_t>(0.9 * (ms.size() - 1))];
  return r;
}

template <class Launch>
Result run_graph(Launch&& launch, int batch = 50, int reps = 20, int warmup = 3) {
  hipStream_t s;
  BENCH_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  launch(s);  // code object load / lazy init outside capture
  BENCH_HIP_CHECK(hipStreamSynchronize(s));

  hipGraph_t graph;
  hipGraphExec_t exec;
  BENCH_HIP_CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeThreadLocal));
  for (int i = 0; i < batch; ++i) launch(s);
  BENCH_HIP_CHECK(hipStreamEndCapture(s, &graph));
  BENCH_HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  for (int i = 0; i < warmup; ++i) BENCH_HIP_CHECK(hipGraphLaunch(exec, s));
  BENCH_HIP_CHECK(hipStreamSynchronize(s));

  hipEvent_t a, b;
  BENCH_HIP_CHECK(hipEventCreate(&a));
  BENCH_HIP_CHECK(hipEventCreate(&b));
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    BENCH_HIP_CHECK(hipEventRecord(a, s));
    BENCH_HIP_CHECK(hipGraphLaunch(exec, s));
    BENCH_HIP_CHECK(hipEventRecord(b, s));
    BENCH_HIP_CHECK(hipEventSynchronize(b));
    BENCH_HIP_CHECK(hipEventElapsedTime(&ms[i], a, b));
    ms[i] /= static_cast<float>(batch);
  }
  std::sort(ms.begin(), ms.end());
  (void)hipEventDestroy(a);
  (void)hipEventDestroy(b);
  (void)hipGraphExecDestroy(exec);
  (void)hipGraphDestroy(graph);
  (void)hipStreamDestroy(s);

  Result r;
  r.min_ms = ms.front();
  r.median_ms = ms[ms.size() / 2];
  r.p90_ms = ms[static_cast<std::size_t>(0.9 * (ms.size() - 1))];
  return r;
}

inline double gbps(double bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double tflops(double flops, double ms) { return flops / (ms * 1e-3) / 1e12; }

// Measured achievable bandwidth: vectorized device-to-device copy (use as the roofline).
__global__ void copy_u4_kernel(const uint4* __restrict__ in, uint4* __restrict__ out,
                               std::size_t n) {
  std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i];
}

inline double measure_copy_gbps(std::size_t bytes = 1ull << 30) {
  void *a = nullptr, *b = nullptr;
  BENCH_HIP_CHECK(hipMalloc(&a, bytes));
  BENCH_HIP_CHECK(hipMalloc(&b, bytes));
  BENCH_HIP_CHECK(hipMemset(a, 1, bytes));
  int dev = 0, cus = 0;
  BENCH_HIP_CHECK(hipGetDevice(&dev));
  BENCH_HIP_CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, dev));
  const std::size_t n = bytes / sizeof(uint4);
  Options opt;
  opt.flush_l2 = false;  // buffer is far larger than every cache level
  auto r = run(
      [&](hipStream_t s) {
        copy_u4_kernel<<<cus * 8, 256, 0, s>>>(static_cast<const uint4*>(a),
                                               static_cast<uint4*>(b), n);
      },
      opt);
  (void)hipFree(a);
  (void)hipFree(b);
  return gbps(2.0 * static_cast<double>(bytes), r.median_ms);  // read + write
}

}  // namespace bench
