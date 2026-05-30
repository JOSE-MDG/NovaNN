#include <ncore/tensor.h>
#include <ncore/device.h>
#include <ncore/copy.h>
#include <ncore/dtype.h>
#include <ncore/backend.h>
#include <ncore/repr/tensor_repr.h>
#include <ncore/cpp_ffi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) do { printf("  TEST: %s ... ", name); fflush(stdout); } while(0)
#define PASS() do { puts("PASS"); tests_passed++; } while(0)
#define FAIL(msg) do { puts("FAIL"); printf("    -> %s\n", msg); tests_failed++; } while(0)

int main(void) {
  puts("========================================");
  puts("  NovaNN Colab Smoke Test");
  puts("========================================");
  puts("");

  /* ── 1. CPU baseline ───────────────────── */
  puts("[CPU Baseline]");

  TEST("create 2x6 Float32 CPU");
  const shape_t shape_2x6 = {2, 6};
  Tensor cpu = create_tensor(shape_2x6, Float32, DEVICE_CPU, false, false, 2);
  if (cpu.data.f32 != NULL && cpu.size == 12) PASS();
  else FAIL("bad alloc or size");

  TEST("fill and print");
  for (size_t i = 0; i < cpu.size; ++i)
    cpu.data.f32[i] = (float)(i + 1) * 0.25f;
  tensor_print(&cpu);
  PASS();

  TEST("cast Float32 -> Float64");
  Tensor casted = create_tensor(shape_2x6, Float64, DEVICE_CPU, false, false, 2);
  cast(&cpu, Float64, &casted);
  if (casted.dtype == Float64) PASS();
  else FAIL("dtype not Float64");
  collect(&casted);

  TEST("scalar tensor");
  Tensor s = create_scalar_tensor(Float32, DEVICE_CPU, false, false);
  if (s.ndims == 0 && s.size == 1 && s.data.f32 != NULL) PASS();
  else FAIL("bad scalar");
  collect(&s);

  TEST("create_view 3x4 -> flat 12");
  const shape_t shape_3x4 = {3, 4};
  Tensor base = create_tensor(shape_3x4, Float32, DEVICE_CPU, false, false, 2);
  for (size_t i = 0; i < base.size; ++i) base.data.f32[i] = (float)i;
  const shape_t flat = {12};
  Tensor v = create_view(&base, flat, 1);
  if (v.ndims == 1 && v.size == 12 && v.is_view_) PASS();
  else FAIL("bad view");
  collect(&base);
  collect(&v);

  /* ── 2. GPU detection ──────────────────── */
  puts("");
  puts("[GPU Detection]");

  TEST("is_cuda_available()");
  int have_cuda = is_cuda_available();
  printf("%s", have_cuda ? "yes" : "no");
  PASS();

  TEST("is_device_available(CUDA_DEVICE, true)");
  int dev_avail = is_device_available(CUDA_DEVICE, true);
  printf("%s", dev_avail ? "yes" : "no");
  PASS();

  TEST("get_device_id()");
  int dev_id = get_device_id();
  printf("%d", dev_id);
  if (have_cuda && dev_id >= 0) PASS();
  else if (!have_cuda && dev_id == -1) PASS();
  else FAIL("unexpected device id");

  /* ── 3. GPU tensor ops (only if CUDA) ──── */
  puts("");
  puts("[GPU Operations]");

  if (have_cuda) {
    TEST("create 4x4 Float32 GPU");
    const shape_t gpu_shape = {4, 4};
    Tensor gpu = create_tensor(gpu_shape, Float32, DEVICE_GPU, false, false, 2);
    if (gpu.data.f32 && gpu.size == 16 && gpu.device == DEVICE_GPU) PASS();
    else FAIL("bad GPU tensor");

    TEST("fill with pattern (host-side)");
    for (size_t i = 0; i < gpu.size; ++i) gpu.data.f32[i] = (float)i * 1.0f;

    TEST("transfer_to CPU -> GPU");
    DeviceStatus st = transfer_to(
      DEVICE_GPU, DEVICE_CPU,
      cpu.data.v, gpu.data.v,
      false, cpu.size * cpu.item_size
    );
    if (st.code == 0) PASS();
    else FAIL(st.message);

    TEST("transfer_to GPU -> CPU verify");
    Tensor cpu_back = create_tensor(shape_2x6, Float32, DEVICE_CPU, false, false, 2);
    st = transfer_to(
      DEVICE_CPU, DEVICE_GPU,
      gpu.data.v, cpu_back.data.v,
      false, cpu_back.size * cpu_back.item_size
    );
    if (st.code != 0) { FAIL(st.message); goto gpu_cleanup; }
    int ok = 1;
    for (size_t i = 0; i < cpu.size; ++i)
      if (cpu.data.f32[i] != cpu_back.data.f32[i]) { ok = 0; break; }
    if (ok) PASS();
    else FAIL("data mismatch after transfer");
    collect(&cpu_back);

  gpu_cleanup:
    collect(&gpu);
  } else {
    puts("  (skipped — CUDA not available)\n");
    puts("  Make sure the library was built with CUDA:\n");
    puts("    cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=native -DUSE_HIP=OFF");
    puts("    cmake --build build");
    puts("  And check Colab has CUDA 12.6+:\n");
    puts("    !nvcc --version");
    puts("    !nvidia-smi");
  }

  /* ── summary ────────────────────────────── */
  puts("");
  puts("========================================");
  printf("  Results: %d passed, %d failed, %d total\n",
         tests_passed, tests_failed, tests_passed + tests_failed);
  puts("========================================");

  return tests_failed > 0 ? 1 : 0;
}
