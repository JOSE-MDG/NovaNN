/**
 * @file stream_types.h
 * @brief Single definition point for the CUDA and HIP stream handles.
 *
 * @details
 * Both runtimes expose a stream as a pointer to an opaque struct owned
 * by the runtime.  A module that only names a stream in a signature
 * includes this file instead of redeclaring the opaque handle, so the
 * declaration exists once and the alias always denotes the runtime
 * type.
 *
 * When the runtime header is reachable the runtime definition is used.
 * The fallback exists for translation units compiled without the
 * runtime include path, where the backend headers still have to name a
 * stream.  Both spellings denote the same pointer type, so a
 * declaration and its definition agree even when they resolve the
 * handle differently.
 *
 * The fallback is never emitted next to the runtime definition: under
 * nvcc the runtime marks @c CUstream_st as device builtin, and a second
 * declaration of the same struct without that attribute is a different
 * type for the compiler.
 *
 * The fallback spells the alias with @c using under C++ and with
 * @c typedef under C.  Both introduce the same type, and the C++ form
 * matches the spelling the runtimes use.
 */

#pragma once

#if defined(NOVA_HAS_CUDA) && __has_include(<cuda_runtime_api.h>)
#include <cuda_runtime_api.h>
#elif defined(__cplusplus)
using cudaStream_t = struct CUstream_st *;
#else
typedef struct CUstream_st *cudaStream_t;
#endif

#if defined(NOVA_HAS_HIP) && __has_include(<hip/hip_runtime_api.h>)
/* The HIP runtime refuses to compile unless exactly one platform macro
   is set, and only the HIP compiler sets it on its own. */
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hip_runtime_api.h>
#elif defined(__cplusplus)
using hipStream_t = struct ihipStream_t *;
#else
typedef struct ihipStream_t *hipStream_t;
#endif
