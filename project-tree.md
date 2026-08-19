# File Tree: NovaNN

**Generated:** 8/18/2026, 12:42:51 AM

```
NovaNN
.
├── benchmarks (old 4.0.4)
├── .agents
│   └── skills
│       ├── c23-features
│       │   └── SKILL.md
│       ├── cmake-rst-documentation
│       │   └── SKILL.md
│       ├── conventional-commit
│       │   └── SKILL.md
│       ├── cpp23-features
│       │   └── SKILL.md
│       ├── doxygen-c-cxx-documentation
│       │   └── SKILL.md
│       ├── gtest-cpp23
│       │   ├── references
│       │   │   ├── advanced.md
│       │   │   ├── basics.md
│       │   │   └── intermediate.md
│       │   └── SKILL.md
│       ├── python-docstrings
│       │   └── SKILL.md
│       └── skill-creator
│           └── SKILL.md
├── cmake
│   ├── config.h.in
│   ├── Detect
│   │   ├── cpu
│   │   │   ├── DetectAMX.cmake
│   │   │   ├── DetectAVX10.1.cmake
│   │   │   ├── DetectAVX10.2.cmake
│   │   │   ├── DetectAVX2.cmake
│   │   │   ├── DetectAVX512.cmake
│   │   │   ├── DetectAVX.cmake
│   │   │   └── DetectSSE.cmake
│   │   ├── lto
│   │   │   └── DetectLTO.cmake
│   │   ├── sanitizers
│   │   │   └── DetectSanitizers.cmake
│   │   ├── simd
│   │   │   └── DetectSIMD.cmake
│   │   ├── testing
│   │   │   └── DetectGTest.cmake
│   │   └── threading
│   │       ├── DetectOpenMP.cmake
│   │       └── DetectPThreads.cmake
│   ├── GenExportDef.cmake
│   ├── Modules
│   │   ├── NovaNNBuildFlags.cmake
│   │   ├── NovaNNCPU.cmake
│   │   ├── NovaNNCUDA.cmake
│   │   ├── NovaNNHandleCompilerRT.cmake
│   │   ├── NovaNNHIP.cmake
│   │   └── NovaNNRuntime.cmake
│   └── Utils
│       ├── CheckCompilerVersion.cmake
│       └── CheckInstructionSupport.cmake
├── examples (old 4.0.4)
├── ncore
│   ├── CMakeLists.txt
│   ├── include
│   │   └── ncore
│   │       ├── core
│   │       │   ├── alloc.h
│   │       │   ├── backend.h
│   │       │   ├── copy.h
│   │       │   ├── device.h
│   │       │   ├── dtype.h
│   │       │   ├── fp_utils.h
│   │       │   ├── status.h
│   │       │   └── storage.h
│   │       ├── headeronly
│   │       │   ├── cast.h
│   │       │   ├── dtypes
│   │       │   │   ├── bfloat16.hh
│   │       │   │   ├── fp4_e2m1fn_x2.hh
│   │       │   │   ├── fp8_e4m3fn.hh
│   │       │   │   ├── fp8_e5m2.hh
│   │       │   │   └── half.hh
│   │       │   ├── macros.h
│   │       │   ├── tensor_utils.h
│   │       │   └── wrappers
│   │       │       └── tensor.hh
│   │       ├── native
│   │       │   ├── cpu
│   │       │   │   ├── dtype
│   │       │   │   │   └── casting.h
│   │       │   │   └── layout
│   │       │   │       └── contiguous.h
│   │       │   └── kernels
│   │       │       └── casting.h
│   │       ├── repr
│   │       │   ├── repr_context.h
│   │       │   ├── repr_options.h
│   │       │   └── tensor_repr.h
│   │       ├── simd
│   │       │   └── simd.h
│   │       ├── tables
│   │       │   ├── cast_tables.h
│   │       │   └── dtype_tables.h
│   │       ├── tensor.h
│   │       └── threading
│   │           └── threads.h
│   ├── memory
│   │   ├── build.rs
│   │   ├── Cargo.toml
│   │   ├── CMakeLists.txt
│   │   ├── csrc
│   │   │   ├── admin.cpp
│   │   │   ├── admin.hpp
│   │   │   ├── CMakeLists.txt
│   │   │   ├── ffi.cpp
│   │   │   └── ffi.hpp
│   │   └── src
│   │       ├── counter.rs
│   │       ├── error.rs
│   │       ├── ffi
│   │       │   ├── cpp
│   │       │   │   └── bindings.rs
│   │       │   ├── cpp.rs
│   │       │   ├── lifecycle.rs
│   │       │   ├── query.rs
│   │       │   ├── reserve.rs
│   │       │   └── resize.rs
│   │       ├── ffi.rs
│   │       ├── handle.rs
│   │       ├── id.rs
│   │       ├── lib.rs
│   │       ├── manager.rs
│   │       ├── ops
│   │       │   ├── lifecycle.rs
│   │       │   ├── query.rs
│   │       │   ├── reserve.rs
│   │       │   └── resize.rs
│   │       ├── ops.rs
│   │       ├── status.rs
│   │       └── storage.rs
│   ├── native
│   │   ├── CMakeLists.txt
│   │   ├── cpu
│   │   │   ├── CMakeLists.txt
│   │   │   ├── dtype
│   │   │   │   └── DTypeCasting.c
│   │   │   └── layout
│   │   │       ├── Contiguous.c
│   │   │       ├── Indexing.c
│   │   │       ├── Permute.c
│   │   │       ├── Reshape.c
│   │   │       ├── Slicing.c
│   │   │       └── Transpose.c
│   │   ├── cuda
│   │   │   ├── CMakeLists.txt
│   │   │   ├── DetectCudaDevice.cpp
│   │   │   ├── DetectCudaDevice.hpp
│   │   │   ├── DetectCudaDeviceInfo.cpp
│   │   │   ├── DetectCudaDeviceInfo.hpp
│   │   │   ├── kernels
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── ContiguousLayoutKernel.cu
│   │   │   │   ├── ContiguousLayoutKernel.h
│   │   │   │   ├── DtypeCastingKernel.cu
│   │   │   │   └── DtypeCastingKernel.h
│   │   │   └── memory
│   │   │       ├── CudaAllocator.cpp
│   │   │       ├── CudaAllocator.hpp
│   │   │       ├── CudaIO.cpp
│   │   │       └── CudaIO.hpp
│   │   ├── hip
│   │   │   ├── CMakeLists.txt
│   │   │   ├── DetectHipDevice.cpp
│   │   │   ├── DetectHipDevice.hpp
│   │   │   ├── DetectHipDeviceInfo.cpp
│   │   │   ├── DetectHipDeviceInfo.hpp
│   │   │   ├── kernels
│   │   │   │   ├── CMakeLists.txt
│   │   │   │   ├── ContiguousLayoutKernel.h
│   │   │   │   ├── ContiguousLayoutKernel.hip
│   │   │   │   ├── DtypeCastingKernel.h
│   │   │   │   └── DtypeCastingKernel.hip
│   │   │   └── memory
│   │   │       ├── HipAllocator.cpp
│   │   │       ├── HipAllocator.hpp
│   │   │       ├── HipIO.cpp
│   │   │       └── HipIO.hpp
│   │   ├── kernels
│   │   │   ├── CastingDispatchImpl.cpp
│   │   │   └── CMakeLists.txt
│   │   ├── native_functions.yaml
│   │   └── native_stub.c
│   ├── src
│   │   ├── core
│   │   │   ├── alloc.c
│   │   │   ├── backend.c
│   │   │   ├── copy.c
│   │   │   ├── device.c
│   │   │   ├── dtype.c
│   │   │   ├── simd.c
│   │   │   ├── status.c
│   │   │   ├── tables
│   │   │   │   ├── cast_dispatch_tables.c
│   │   │   │   ├── cast_tables.c
│   │   │   │   ├── dtype_tables.c
│   │   │   │   └── status_dispatch_tables.c
│   │   │   ├── tensor.c
│   │   │   └── threading
│   │   │       ├── concurrency.c
│   │   │       ├── concurrency.h
│   │   │       ├── groups
│   │   │       │   ├── autograd.c
│   │   │       │   ├── autograd.h
│   │   │       │   ├── compute.c
│   │   │       │   ├── compute.h
│   │   │       │   ├── dtloader.c
│   │   │       │   └── dtloader.h
│   │   │       ├── manager.c
│   │   │       ├── manager.h
│   │   │       └── threads.c
│   │   ├── dtypes
│   │   │   ├── BFloat16.hpp
│   │   │   ├── DTypes.cpp
│   │   │   ├── DTypes.hpp
│   │   │   ├── Float4_e2m1fn_x2.hpp
│   │   │   ├── Float8_e4m3fn.hpp
│   │   │   ├── Float8_e5m2.hpp
│   │   │   └── Half.hpp
│   │   └── repr
│   │       ├── api
│   │       │   └── tensor_repr.c
│   │       ├── context
│   │       │   └── repr_context.c
│   │       ├── formatters
│   │       │   ├── element_fmt.c
│   │       │   ├── element_fmt.h
│   │       │   ├── float_formatter.c
│   │       │   ├── float_formatter.h
│   │       │   ├── int_formatter.c
│   │       │   ├── int_formatter.h
│   │       │   ├── qint_formatter.c
│   │       │   └── qint_formatter.h
│   │       ├── layouts
│   │       │   ├── dense_layout.c
│   │       │   ├── layouts.h
│   │       │   ├── strided_layout.c
│   │       │   └── summarized_layout.c
│   │       ├── metadata
│   │       │   ├── metadata_fmt.c
│   │       │   └── metadata_fmt.h
│   │       ├── options
│   │       │   └── repr_options.c
│   │       ├── string_builder
│   │       │   ├── string_builder.c
│   │       │   └── string_builder.h
│   │       └── traversal
│   │           ├── tensor_iterator.c
│   │           └── tensor_iterator.h
│   └── tests
│       ├── CMakeLists.txt
│       ├── core
│       │   ├── CMakeLists.txt
│       │   ├── DeviceDetection_test.cpp
│       │   ├── MemoryAllocator_test.cpp
│       │   ├── RuntimeSimdCaps_test.cpp
│       │   ├── StatusPropagation_test.cpp
│       │   ├── TensorCopies_test.cpp
│       │   └── TensorTransfer_test.cpp
│       ├── dtypeCasting
│       │   ├── CMakeLists.txt
│       │   ├── Dispatch_test.cpp
│       │   ├── ISAEquivalence_test.cpp
│       │   ├── SaturationInvariant_test.cpp
│       │   ├── ScalarOracle_test.cpp
│       │   ├── SpecialValueInvariant_test.cpp
│       │   └── utils
│       │       └── Oracle.h
│       └── dtypes
│           ├── ABICorrectness_test.cpp
│           ├── bitPatternIdentity
│           │   ├── BFloat16_test.cpp
│           │   ├── Float4_e2m1fn_x2_test.cpp
│           │   ├── Float8_e4m3fn_test.cpp
│           │   ├── Float8_e5m2_test.cpp
│           │   └── Half_test.cpp
│           ├── CMakeLists.txt
│           ├── IEEESemantic_test.cpp
│           ├── PackedPairs_test.cpp
│           ├── RangeAndDensity_test.cpp
│           ├── SpecialValuesClassification_test.cpp
│           └── utils
│               └── FloatingPointClassification.h
├── nova (old 4.0.4)
├── ports
│   ├── cuda
│   │   ├── portfile.cmake
│   │   ├── vcpkg_find_cuda.cmake
│   │   ├── vcpkg.json
│   │   └── vcpkg-port-config.cmake
│   ├── onednn
│   │   ├── portfile.cmake
│   │   └── vcpkg.json
│   ├── openblas
│   │   ├── android-exclude-sme.diff
│   │   ├── cmake-project-include.cmake
│   │   ├── disable-testing.diff
│   │   ├── getarch.diff
│   │   ├── portfile.cmake
│   │   ├── system-check-msvc.diff
│   │   ├── vcpkg.json
│   │   └── win32-uwp.diff
│   └── rocm
│       ├── portfile.cmake
│       ├── vcpkg_find_rocm.cmake
│       ├── vcpkg.json
│       └── vcpkg-port-config.cmake
├── scripts
│   ├── build-presets.ps1
│   ├── build-presets.sh
│   ├── compile-presets.ps1
│   └── compile-presets.sh
├── tests (old 4.0.4)
├── thirdParty
│   └── sleef
├── tools
│   └── codegen
│       ├── engine.py
│       ├── generate.py
│       ├── rules
│       │   └── dtype_casting
│       │       ├── cast_funcs_rules.json
│       │       ├── cast_tables_rules.json
│       │       └── dtype_casting_rules.json
│       ├── scripts
│       │   └── dtype_casting
│       │       ├── _build_cast_funcs_rules.py
│       │       ├── _build_cast_tables_rules.py
│       │       ├── gen_cast_funcs.py
│       │       ├── gen_cast_tables.py
│       │       └── gen_dtype_casting.py
│       └── templates
│           ├── dtype_casting
│           │   ├── CastFuncs.h.jinja
│           │   ├── CastTables.h.jinja
│           │   ├── CastTables.jinja
│           │   ├── DTypeCasting.h.jinja
│           │   └── DTypeCasting.jinja
│           └── utils
│               ├── DTypes.jinja
│               └── FileHeaderDocBlock.jinja
├── .clang-format
├── .clang-tidy
├── .clangd
├── .gitattributes
├── .gitignore
├── .python-version
├── AGENTS.md
├── CHANGELOG.es.md
├── CHANGELOG.md
├── CLAUDE.md -> AGENTS.md
├── CMakeLists.txt
├── CMakePresets.json
├── CONTRIBUTING.es.md
├── CONTRIBUTING.md
├── LICENCE
├── project-tree.md
├── pyproject.toml
├── README.es.md
├── README.md
├── ruff.toml
├── uv.lock
├── vcpkg-configuration.json
└── vcpkg.json
```

---
*Generated by FileTree Pro Extension*
