#pragma once

#include <ncore/macros.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ATTR(packed) {
  CUDA = 0,
  Rocm = 1,
  oneDNN = 2,
  Generic = 3,
  Meta = 4,
} Backend;

const Backend *get_current_running_backend();
bool set_next_execution_backend_to_(Backend backend);
bool is_backend_available(Backend backend);

#ifdef __cplusplus
}
#endif
