// bench_harness_cuda.cuh - drop-in timing harness (CUDA 13, C++20)
//
// Usage:
//   #include "bench_harness_cuda.cuh"
//   auto r = bench::run([&](cudaStream_t s) { my_kernel<<<grid, block, smem, s>>>(...); });
//   std::printf("median %.3f us, %.1f GB/s\n", r.median_ms * 1e3, bench::gbps(bytes, r.median_ms));
//
// For kernels shorter than ~10 us, per-iteration event overhead matters: use
// bench::run_graph(), which captures `batch` launches in a CUDA Graph and divides.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <vector>

#define BENCH_CUDA_CHECK(expr)                                                         \
  do {                                                                                 \
    cudaError_t err_ = (expr);                                                         \
    if (err_ != cudaSuccess) {                                                         \
      std::fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,          \
                   cudaGetErrorString(err_), #expr);                                   \
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
  bool flush_l2 = true;          // cold-cache measurement
  std::size_t flush_bytes = 0;   // 0 -> 2x L2 size (min 64 MiB)
};

class L2Flusher {
 public:
  explicit L2Flusher(std::size_t bytes_override = 0) {
    int dev = 0;
    BENCH_CUDA_CHECK(cudaGetDevice(&dev));
    int l2 = 0;
    BENCH_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    bytes_ = bytes_override ? bytes_override
                            : std::max<std::size_t>(2 * static_cast<std::size_t>(l2),
                                                    64ull << 20);
    BENCH_CUDA_CHECK(cudaMalloc(&buf_, bytes_));
  }
  ~L2Flusher() { cudaFree(buf_); }
  L2Flusher(const L2Flusher&) = delete;
  L2Flusher& operator=(const L2Flusher&) = delete;

  void flush(cudaStream_t s) { BENCH_CUDA_CHECK(cudaMemsetAsync(buf_, 0xA5, bytes_, s)); }

 private:
  void* buf_ = nullptr;
  std::size_t bytes_ = 0;
};

// Times `launch(stream)` per iteration with CUDA events. The flush (if enabled) runs
// before the start event, so it is not included in the measurement.
template <class Launch>
Result run(Launch&& launch, Options opt = {}, cudaStream_t stream = nullptr) {
  std::optional<L2Flusher> flusher;
  if (opt.flush_l2) flusher.emplace(opt.flush_bytes);
  for (int i = 0; i < opt.warmup; ++i) launch(stream);
  BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<cudaEvent_t> start(opt.iters), stop(opt.iters);
  for (int i = 0; i < opt.iters; ++i) {
    BENCH_CUDA_CHECK(cudaEventCreate(&start[i]));
    BENCH_CUDA_CHECK(cudaEventCreate(&stop[i]));
  }

  for (int i = 0; i < opt.iters; ++i) {
    if (flusher) flusher->flush(stream);
    BENCH_CUDA_CHECK(cudaEventRecord(start[i], stream));
    launch(stream);
    BENCH_CUDA_CHECK(cudaEventRecord(stop[i], stream));
  }
  BENCH_CUDA_CHECK(cudaStreamSynchronize(stream));
  BENCH_CUDA_CHECK(cudaGetLastError());

  std::vector<float> ms(opt.iters);
  for (int i = 0; i < opt.iters; ++i) {
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms[i], start[i], stop[i]));
    cudaEventDestroy(start[i]);
    cudaEventDestroy(stop[i]);
  }
  std::sort(ms.begin(), ms.end());
  Result r;
  r.min_ms = ms.front();
  r.median_ms = ms[ms.size() / 2];
  r.p90_ms = ms[static_cast<std::size_t>(0.9 * (ms.size() - 1))];
  return r;
}

// Hot-cache timing of very short kernels: capture `batch` launches in a graph and
// report the per-launch average (median over `reps` graph replays).
template <class Launch>
Result run_graph(Launch&& launch, int batch = 50, int reps = 20, int warmup = 3) {
  cudaStream_t s;
  BENCH_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  launch(s);  // module load / lazy init outside capture
  BENCH_CUDA_CHECK(cudaStreamSynchronize(s));

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  BENCH_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < batch; ++i) launch(s);
  BENCH_CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  BENCH_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));

  for (int i = 0; i < warmup; ++i) BENCH_CUDA_CHECK(cudaGraphLaunch(exec, s));
  BENCH_CUDA_CHECK(cudaStreamSynchronize(s));

  cudaEvent_t a, b;
  BENCH_CUDA_CHECK(cudaEventCreate(&a));
  BENCH_CUDA_CHECK(cudaEventCreate(&b));
  std::vector<float> ms(reps);
  for (int i = 0; i < reps; ++i) {
    BENCH_CUDA_CHECK(cudaEventRecord(a, s));
    BENCH_CUDA_CHECK(cudaGraphLaunch(exec, s));
    BENCH_CUDA_CHECK(cudaEventRecord(b, s));
    BENCH_CUDA_CHECK(cudaEventSynchronize(b));
    BENCH_CUDA_CHECK(cudaEventElapsedTime(&ms[i], a, b));
    ms[i] /= static_cast<float>(batch);
  }
  std::sort(ms.begin(), ms.end());
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(s);

  Result r;
  r.min_ms = ms.front();
  r.median_ms = ms[ms.size() / 2];
  r.p90_ms = ms[static_cast<std::size_t>(0.9 * (ms.size() - 1))];
  return r;
}

inline double gbps(double bytes, double ms) { return bytes / (ms * 1e-3) / 1e9; }
inline double tflops(double flops, double ms) { return flops / (ms * 1e-3) / 1e12; }

// Measured achievable bandwidth: vectorized device-to-device copy over `bytes`.
// Use this (not the datasheet) as the roofline for memory-bound ops.
__global__ void copy_u4_kernel(const uint4* __restrict__ in, uint4* __restrict__ out,
                               std::size_t n) {
  std::size_t i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i];
}

inline double measure_copy_gbps(std::size_t bytes = 1ull << 30) {
  void *a = nullptr, *b = nullptr;
  BENCH_CUDA_CHECK(cudaMalloc(&a, bytes));
  BENCH_CUDA_CHECK(cudaMalloc(&b, bytes));
  BENCH_CUDA_CHECK(cudaMemset(a, 1, bytes));
  int dev = 0, sms = 0;
  BENCH_CUDA_CHECK(cudaGetDevice(&dev));
  BENCH_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const std::size_t n = bytes / sizeof(uint4);
  Options opt;
  opt.flush_l2 = false;  // buffer is far larger than L2 anyway
  auto r = run(
      [&](cudaStream_t s) {
        copy_u4_kernel<<<sms * 8, 256, 0, s>>>(static_cast<const uint4*>(a),
                                              static_cast<uint4*>(b), n);
      },
      opt);
  cudaFree(a);
  cudaFree(b);
  return gbps(2.0 * static_cast<double>(bytes), r.median_ms);  // read + write
}

}  // namespace bench
