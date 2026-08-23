# NovaNN — Deep Learning Ecosystem

## What is NovaNN?

NovaNN is a deep learning ecosystem built from scratch. Its goal is to be a unified tool that integrates all the capabilities needed for the model lifecycle: architecture definition, pretraining, fine-tuning (SFT, DPO), quantization, PEFT (LoRA/QLoRA), inference and deployment.

API design inspired by PyTorch. The ecosystem NovaNN aims to become is something like this: *torch + torchvision + transformers + datasets + PEFT + bitsandbytes + TRL*, NovaNN aims to provide everything in a single tool with a modern design, leveraging C23, C++23 and Rust for the high-performance core, with Cython as a bridge to a clean Python API.

```
Pretraining → SFT → DPO → Quantization (NF4, FP8) → PEFT (LoRA/QLoRA) → Forward → Deployment
                                  ── all in NovaNN ──
```

---

## Tech Stack

| Language | Role | Purpose |
|----------|------|---------|
| **C23** | Numerical core | Simplicity, low overhead, ABI compatible with other languages. All fundamental logic lives here: tensor, storage, device, dtype, dispatch, tensor repr and CPU backends. |
| **C++23** | Components that require it | Autograd engine (automatic differentiation) and CUDA/HIP kernels. Only where C is not sufficient. |
| **Rust** | Memory management | Absolute owner of memory: allocation, deallocation, ref-counting. The entire buffer lifecycle goes through Rust. |
| **Cython** | C → Python binding | Bridge between the native core and Python. C functions always export a `novaStatus_t`; Cython validates arguments and translates errors to Python exceptions with near-zero overhead. |
| **Python** | High-level API | Final interface for the user. |

## Call Flow (v5.0.0) — target design, not current state

> The three flows below describe the **target** pipeline of the API once complete. **They do not reflect the current state of the code**: as of today there are no Cython bindings to the native core (`nova/` is still v4.0.4 Python+NumPy), there are no matmul arithmetic ops in any backend, and the autograd engine in C++23 is paused without implementation. See "Project Status" below for the actual state.

### 1. Tensor Creation

```
    X = nova.rand(M, N)
    Y = nova.rand(N, K)

    ┌── Python calls nova.rand()
    │
    ├── Cython validates arguments (shape, dtype, device)
    │     │
    │     └── C creates the tensor:
    │           ├── calculates size in bytes
    │           ├── allocates memory
    │           │     ├── CPU  → Rust allocates via the system
    │           │     └── GPU  → Rust → C++ bridge → cudaMalloc()/hipMalloc() (or their Async variants if supported)
    │           ├── creates structure (shape, dtype, strides)
    │           ├── writes random values into the buffer
    │           └── returns status
    │
    └── Cython translates the status:
          ├── success → returns Tensor to Python
          └── error → raises Python exception
```

### 2. Tensor Operation (planned — matmul not yet implemented in any backend)

```
    Z = nova.matmul(X, Y)

    ┌── Python calls nova.matmul()
    │
    ├── Cython validates arguments (types, compatible shapes)
    │     │
    │     └── C executes the operation:
    │           ├── calculates result shape
    │           ├── allocates memory for the result
    │           ├── executes kernel according to device:
    │           │     ├── CPU  → C code with SIMD
    │           │     ├── CUDA → C++/CUDA kernel
    │           │     └── HIP  → C++/HIP kernel
    │           ├── if gradient required:
    │           │     autograd registers the operation in the graph
    │           └── returns status
    │
    └── Cython translates the status:
          ├── success → returns Tensor Z
          └── error → raises Python exception
```

### 3. Backward (planned — autograd engine paused, not implemented)

```
    loss = nova.mean(Z)
    loss.backward()

    ┌── Python calls backward()
    │
    ├── Cython initiates the backward pass
    │     │
    │     └── C++ (autograd engine):
    │           ├── traverses the graph in reverse order
    │           ├── for each operation:
    │           │     ├── computes gradients
    │           │     ├── allocates memory for new gradients
    │           │     ├── executes backward kernel (CPU/CUDA/HIP)
    │           │     └── accumulates into tensor's .grad
    │           ├── frees intermediate tensors (Rust releases memory)
    │           └── returns status
    │
    └── Cython translates the status:
          ├── success → None
          └── error → raises Python exception
```

---

## Core Subsystems (ncore)

### Core Runtime
Manages the complete tensor lifecycle:
- **Tensor**: N-dimensional structure with shape, strides, dtype and device.
- **Storage**: memory buffers with reference counting (delegated to Rust).
- **Device**: device detection and selection (CPU, CUDA, HIP, Meta).
- **DType**: system of 21 types (32/64-bit floats, low precision FP4/FP8/FP16/BF16, signed/unsigned/quantized integers).
- **Dispatch**: selects the correct implementation based on dtype and backend.
- **Copy**: copy between devices (CPU ↔ GPU).

### Representation (Repr)
Pipeline for serializing tensors to readable strings: scans data to determine optimal format, formats each element according to its dtype, renders the N-dimensional structure, and adds metadata (dtype, shape, device).

#### **Example:**
```
tensor([[2.3474e-01, 4.6948e-01, 7.0422e-01, ..., 7.0422e+00, 7.2769e+00, 7.5117e+00],
        [7.7464e+00, 7.9812e+00, 8.2159e+00, ..., 1.4554e+01, 1.4789e+01, 1.5023e+01],
        [1.5258e+01, 1.5493e+01, 1.5728e+01, ..., 2.2066e+01, 2.2300e+01, 2.2535e+01],
        ...,
        [2.1807e+02, 2.1831e+02, 2.1854e+02, ..., 2.2488e+02, 2.2512e+02, 2.2535e+02],
        [2.2559e+02, 2.2582e+02, 2.2605e+02, ..., 2.3239e+02, 2.3263e+02, 2.3286e+02],
        [2.3310e+02, 2.3333e+02, 2.3357e+02, ..., 2.3990e+02, 2.4014e+02, 2.4037e+02]],
       dtype=float32, shape=(32, 32), device=cuda, requires_grad=False)
```

### Dtypes (Reduced Types)
Soft-float implementation for low-precision types (FP4 E2M1, FP8 E4M3/E5M2, FP16, BF16) with conversions to float32 and native compiler support when available (`_Float16`, `__bf16`).

### Autograd (C++23, paused)
Reverse-mode automatic differentiation engine. Destined to be implemented in C++23.

---

## Hardware Backends

| Backend | Status | Current Capabilities |
|---------|--------|---------------------|
| **CPU** | Active | SIMD for dtype casting (SSE4.2 to AVX10.2). Layouts: contiguous implemented, others in progress. Arithmetic ops pending. |
| **CUDA** | Active | Device detection, GPU allocator, host↔device transfers, dtype casting kernel f32→fp16. |
| **HIP** | Active (Linux only) | Parallel to CUDA: detection, allocator, transfers, dtype casting kernel f32→fp16. Not available on Windows: RDNA 2/3 and CDNA consumer GPUs lack official Windows driver support. |
| **cuDNN** | Placeholder | No implementation. |
| **MIOpen** | Placeholder | No implementation. |
| **oneDNN** | Placeholder | No implementation. |
| **Quantized** | Placeholder | No implementation. |
| **Transformers** | Placeholder | No implementation. |

The three main backends (CPU, CUDA, HIP) are developed in parallel. CUDA and HIP are mutually exclusive at compile time (`-DUSE_CUDA=ON` or `-DUSE_HIP=ON`, not both). HIP is Linux-only; attempting `-DUSE_HIP=ON` on Windows produces a configure-time error.

---

## Memory Management (Rust)

`ncore_memory` is the Rust crate acting as the official memory allocator. All buffer allocation and deallocation goes through Rust.

```
C/C++ calls:  reserve() / retain() / release() / resize()
                      │
                      ▼
              ┌────────────────┐
              │  ncore_memory  │  (Rust staticlib, 0 external dependencies)
              │  ┌──────────┐  │
              │  │ manager  │  │  HashMap<ID, RustStorage> with Mutex
              │  │ storage  │  │  allocate/deallocate CPU (std::alloc) or GPU (via bridge)
              │  │ handle   │  │  RustHandle [id, size, align] in repr(C)
              │  │ ffi/     │  │  externally exported "C" functions
              │  └──────────┘  │
              └───────┬────────┘
                      │
              ┌───────▼──────────────────────────────────────────────────────────┐
              │   memorycsrc      (C++ staticlib bridge)                         │
              │   deviceReserve / deviceRelease / deviceResize / deviceTransfer  │
              │   → CUDA API or HIP API depending on detected backend            │
              └──────────────────────────────────────────────────────────────────┘
```

---

## Build System (CMake)

| Aspect | Detail |
|---------|--------|
| **Standards** | C23 and C++23 mandatory |
| **Compilers** | GCC ≥ 15.0 (Linux), Clang ≥ 20.1 (Linux and Windows, including clang-cl). **No MSVC/cl.exe support.** |
| **GPU Backends** | `-DUSE_CUDA=ON` (Windows/Linux) or `-DUSE_HIP=ON` (Linux only, mutually exclusive with CUDA) |
| **SIMD** | Automatic detection: SSE4.2, AVX/AVX2, AVX-512, AVX10, AMX |
| **Optimizations** | LTO enabled by default, hardening linker flags |
| **Sanitizers** | ASan and UBSan (`-DUSE_ASAN=ON` / `-DUSE_UBSAN=ON`) |
| **Tests** | GoogleTest + CTest (C/C++), pytest (Python) |

### Main Targets

| Target (alias) | Type | Content |
|--------|------|---------|
| `nova` | SHARED | Main native core library |
| `ncore_obj` (`ncore::obj::core`) | OBJECT | C core |
| `autograd_obj` (`ncore::obj::autograd`) | OBJECT | C++ autograd |
| `dtypes_obj` (`ncore::obj::dtypes`) | OBJECT | C++ reduced types |
| `native` (`ncore::native`) | STATIC | Aggregates all backends (cpu, cuda, hip, kernels) |
| `ncore_memory` (`ncore::memory`) | IMPORTED STATIC | Rust crate |
| `memorycsrc` (`ncore::memory::csrc`) | STATIC | C++ bridge for GPU memory |

### Building the Project

Prerequisites: CMake ≥ 3.27, Ninja, and vcpkg with the `VCPKG_ROOT` environment variable pointing to the vcpkg installation (all presets load its toolchain). Every preset builds out-of-source into `build/<preset-name>/`; list them all with `cmake --list-presets`.

#### Preset Naming Convention

Visible presets follow `<backend>-<config>[-<sanitizer>][-test][-os]`:

| Component | Values | Notes |
|-----------|--------|-------|
| `<backend>` | `cpu`, `cuda`, `hip` | `hip` presets are Linux-only |
| `<config>` | `release`, `debug` | Sets `CMAKE_BUILD_TYPE` |
| `<sanitizer>` | `asan`, `ubsan` | Optional. UBSan presets are Linux-only |
| `-test` | — | Optional. Enables GoogleTest + CTest (`BUILD_TESTING=ON`) |
| `-os` | `linux`, `windows` | Compilers: Linux → gcc/g++, Windows → clang-cl + lld-link. HIP forces clang and disables LTO |

Examples: `cpu-release-linux`, `cpu-asan-test-debug-linux`, `cuda-test-release-windows`, `hip-ubsan-test-debug-linux`.

#### Option 1 — Workflow Presets

Chain configure → build (→ test for `-test-*` presets) in one command:

```bash
cmake --workflow --preset cpu-release-linux      # configure + build
cmake --workflow --preset cpu-test-debug-linux   # configure + build + ctest
```

#### Option 2 — Step by Step

```bash
cmake --preset cpu-test-debug-linux           # 1. configure → build/cpu-test-debug-linux/
cmake --build --preset cpu-test-debug-linux   # 2. compile
ctest --preset cpu-test-debug-linux           # 3. run tests (-test-* presets only)
```

Test presets print output on failure and error out when no tests are registered.

#### Option 3 — Helper Scripts (bulk operations)

Configure or build many presets at once. Both scripts print one summary line per preset (with a live progress spinner on a TTY) and write full output to `build/logs/<preset>.log`.

| Script (bash / pwsh) | Purpose | Options |
|----------------------|---------|---------|
| `scripts/build-presets.sh` / `.ps1` | Configure matching presets (`cmake --preset`) | `-c/--continue`, `-l/--list` |
| `scripts/compile-presets.sh` / `.ps1` | Build already-configured presets (`cmake --build`) | `-C/--config Release\|Debug`, `-j/--jobs N`, `-c/--continue`, `-l/--list` |

```bash
scripts/build-presets.sh                  # configure every preset
scripts/build-presets.sh cpu              # configure cpu-* presets
scripts/compile-presets.sh cpu            # build configured cpu-* presets
scripts/compile-presets.sh -c -j $(nproc) cuda  # build cuda-* presets, 16 jobs, keep going on failure
```

- Filters match a backend prefix (`cpu`, `cuda`, `hip`) or an exact preset name; `compile-presets` accepts multiple filters.
- `--config` defaults to the value derived from the preset name (`*-debug*` → Debug, otherwise Release).
- Both scripts abort on the first failure unless `--continue` is given. Exit status: `0` success · `1` failure · `2` usage error.
- On Windows, configuring `cuda-*` presets requires the `CUDA_HOST_COMPILER` environment variable (e.g., pointing to MSVC's `cl.exe`); otherwise those presets are skipped.
- Typical verification loop: `scripts/build-presets.sh <preset>` → `scripts/compile-presets.sh <preset>` → `ctest --preset <preset>`.

---

## Project Status

- **v5.0.0** in active development: complete core rewrite with C23/C++23/Rust.
- **v4.0.4**: stable legacy version (Python + NumPy) published on PyPI. The current Python code in `nova/` belongs to this version and will be replaced by Cython bindings to the native core.
- Core runtime (tensor, storage, device, dtype, repr) is advanced. CPU backends with complete SIMD for dtype casting. GPU backends with base infrastructure. Autograd and arithmetic ops are the main pending items.
- No CI/CD yet.

### Roadmap (overview)

```
v4.0.4             v5.0.0-alpha              v5.0.0-beta               v5.0.0
  │                     │                         │                       │
  ▼                     ▼                         ▼                       ▼
──┴─────────────────────┴─────────────────────────┴───────────────────────▶

Python+NumPy      Native C core              Autograd C++              Ecosystem
(legacy)          Backends CPU/CUDA/HIP      Cython Bindings           Hub (models)
                  Dtypes, repr,              Arithmetic ops            PEFT, NF4
                  Memory (Rust)              Native Python API         SFT/DPO trainers
```

---

## Summary

NovaNN combines **C23** (portable numerical core), **C++23** (autograd and GPU kernels), **Rust** (safe memory), **Cython** (efficient bindings) and **Python** (API). It is in active development of its version 5.0.0, where the native core completely replaces NumPy. It supports CPU, CUDA and HIP, with a modular architecture aimed at being a complete "PyTorch + transformers + PEFT + datasets" ecosystem in a single tool.
