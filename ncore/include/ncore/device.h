#pragma once

#include <ncore/macros.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum ATTR(packed) {
  DEVICE_CPU = 0,
  DEVICE_GPU = 1,
  DEVICE_META = 2
} Device;

struct Tensor;
typedef struct Tensor Tensor;
typedef Tensor *TensorGrad;

const Device *get_current_global_device();
void set_current_gloval_device_to_(Device device);
Device get_current_device_from(const Tensor *ten);
int move_tensor_to_(Device device, Tensor *ten);
int move_grad_to_(Device device, TensorGrad grad);

#ifdef __cplusplus
}
#endif
