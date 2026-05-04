#pragma once

#include <stdbool.h>

typedef struct {
  bool sse4_2_;

  bool avx_;
  bool avx2_;
  bool avx2_int8_;
  bool avx2_vnni_;
  bool f16c_;

  bool vnni_;

  bool fma3_;

  bool avx512f_;
  bool avx512_vnni_;
  bool avx512_fp16_;
  bool avx512_bf16_;

  bool amx_;
  bool amx_fp16_;
  bool amx_bf16_;
  bool amx_int8_;
} Capabilities_;

const Capabilities_ *get_cpu_capabilites();
