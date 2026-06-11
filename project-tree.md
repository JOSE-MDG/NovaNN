# File Tree: NovaNN

**Generated:** 6/11/2026, 7:34:04 PM

```
├── .agents
├── benchmarks
│   ├── autograd
│   │   ├── backward_overhead.py
│   │   ├── grad_accumulation.py
│   │   └── memory_footprint.py
│   ├── operations
│   │   ├── elementwise_cpu.py
│   │   └── reduction_ops.py
│   ├── training
│   │   └── end_to_end_cpu.py
│   ├── utils
│   │   ├── memory.py
│   │   └── timing.py
│   ├── README.es.md
│   └── README.md
├── bindings
│   └── cython
│       └── lib
├── cmake
│   ├── Detect
│   │   ├── cpu
│   │   │   ├── DetectAMX.cmake
│   │   │   ├── DetectAVX.cmake
│   │   │   ├── DetectAVX2.cmake
│   │   │   ├── DetectAVX512.cmake
│   │   │   └── DetectSSE.cmake
│   │   ├── lto
│   │   │   └── DetectLTO.cmake
│   │   ├── sanitizers
│   │   │   └── DetectSanitizers.cmake
│   │   ├── simd
│   │   │   └── DetectSIMD.cmake
│   │   └── threading
│   │       ├── DetectOpenMP.cmake
│   │       └── DetectPThreads.cmake
│   ├── Modules
│   │   ├── NovaNNBuildFlags.cmake
│   │   ├── NovaNNCPU.cmake
│   │   ├── NovaNNCUDA.cmake
│   │   ├── NovaNNHIP.cmake
│   │   └── NovaNNRuntime.cmake
│   ├── Utils
│   │   └── CheckInstructionSupport.cmake
│   └── config.h.in
├── examples
│   ├── binary_classification.py
│   ├── conv_example.py
│   ├── multiclass_classification.py
│   └── regression.py
├── images
│   ├── benchmarks
│   │   ├── autograd
│   │   │   ├── accumulation_framework_comparison.png
│   │   │   ├── accumulation_overhead.png
│   │   │   ├── accumulation_vs_microbatch.png
│   │   │   ├── memory_overhead.png
│   │   │   ├── memory_vs_batch.png
│   │   │   ├── memory_vs_depth.png
│   │   │   ├── nova_accumulation_comparison.png
│   │   │   ├── nova_fwd_vs_bwd.png
│   │   │   ├── nova_memory_fwd_vs_bwd.png
│   │   │   ├── overhead_vs_batch.png
│   │   │   ├── overhead_vs_depth.png
│   │   │   └── relative_overhead.png
│   │   ├── operations
│   │   │   ├── activation_comparison.png
│   │   │   ├── addition_performance.png
│   │   │   ├── arithmetic_speedup.png
│   │   │   ├── basic_reductions_comparison.png
│   │   │   ├── mean_performance.png
│   │   │   ├── minmax_performance.png
│   │   │   ├── multiplication_performance.png
│   │   │   ├── relu_performance.png
│   │   │   ├── sigmoid_performance.png
│   │   │   ├── statistical_reductions_comparison.png
│   │   │   ├── std_performance.png
│   │   │   ├── sum_performance.png
│   │   │   └── var_performance.png
│   │   └── training
│   │       ├── convnet_training_performance.png
│   │       ├── mlp_training_performance.png
│   │       ├── mlp_training_speedup.png
│   │       └── optimizer_comparison.png
│   ├── NovaNN Banners.png
│   └── graph.png
├── ncore
│   ├── backends
│   ├── doxygen
│   ├── generated
│   ├── include
│   │   ├── autograd
│   │   │   ├── node.hpp
│   │   │   └── tensor.hpp
│   │   └── ncore
│   │       ├── headeronly
│   │       │   ├── cast.h
│   │       │   └── tensor_utils.h
│   │       ├── repr
│   │       │   ├── repr_context.h
│   │       │   ├── repr_options.h
│   │       │   └── tensor_repr.h
│   │       ├── tables
│   │       │   ├── cast_tables.h
│   │       │   └── dtype_tables.h
│   │       ├── alloc.h
│   │       ├── backend.h
│   │       ├── copy.h
│   │       ├── cpp_ffi.h
│   │       ├── device.h
│   │       ├── dtype.h
│   │       ├── macros.h
│   │       ├── simd.h
│   │       ├── storage.h
│   │       └── tensor.h
│   ├── rust
│   │   ├── .cargo
│   │   │   └── config.toml
│   │   ├── csrc
│   │   │   ├── device
│   │   │   │   ├── cuda
│   │   │   │   │   ├── cuda_allocator.cpp
│   │   │   │   │   ├── cuda_allocator.hpp
│   │   │   │   │   ├── cuda_io.cpp
│   │   │   │   │   └── cuda_io.hpp
│   │   │   │   ├── hip
│   │   │   │   │   ├── hip_allocator.cpp
│   │   │   │   │   ├── hip_allocator.hpp
│   │   │   │   │   ├── hip_io.cpp
│   │   │   │   │   └── hip_io.hpp
│   │   │   │   ├── admin.cpp
│   │   │   │   └── admin.hpp
│   │   │   ├── CMakeLists.txt
│   │   │   ├── ffi.cpp
│   │   │   └── ffi.hpp
│   │   ├── src
│   │   │   ├── ffi
│   │   │   │   ├── cpp
│   │   │   │   │   └── bindings.rs
│   │   │   │   ├── cpp.rs
│   │   │   │   ├── lifecycle.rs
│   │   │   │   ├── query.rs
│   │   │   │   ├── reserve.rs
│   │   │   │   └── resize.rs
│   │   │   ├── ops
│   │   │   │   ├── lifecycle.rs
│   │   │   │   ├── query.rs
│   │   │   │   ├── reserve.rs
│   │   │   │   └── resize.rs
│   │   │   ├── pool
│   │   │   │   └── caching.rs
│   │   │   ├── error.rs
│   │   │   ├── ffi.rs
│   │   │   ├── handle.rs
│   │   │   ├── id.rs
│   │   │   ├── lib.rs
│   │   │   ├── manager.rs
│   │   │   ├── ops.rs
│   │   │   ├── pool.rs
│   │   │   └── storage.rs
│   │   ├── CMakeLists.txt
│   │   ├── Cargo.toml
│   │   └── build.rs
│   ├── src
│   │   ├── autograd
│   │   │   ├── threadPool
│   │   │   ├── engine.cpp
│   │   │   ├── node.cpp
│   │   │   └── tensor.cpp
│   │   ├── core
│   │   │   ├── detect
│   │   │   │   ├── cuda_device.c
│   │   │   │   └── hip_device.c
│   │   │   ├── tables
│   │   │   │   ├── cast_dispatch_tables.c
│   │   │   │   ├── cast_tables.c
│   │   │   │   └── dtype_tables.c
│   │   │   ├── alloc.c
│   │   │   ├── copy.c
│   │   │   ├── device.c
│   │   │   ├── dtype.c
│   │   │   ├── simd.c
│   │   │   └── tensor.c
│   │   ├── dtypes
│   │   ├── ops
│   │   │   ├── asm
│   │   │   │   └── kernels
│   │   │   ├── fused
│   │   │   └── transformers
│   │   └── repr
│   │       ├── api
│   │       │   └── tensor_repr.c
│   │       ├── context
│   │       │   └── repr_context.c
│   │       ├── formatters
│   │       │   ├── element_fmt.c
│   │       │   ├── element_fmt.h
│   │       │   ├── float_formatter.c
│   │       │   ├── float_formatter.h
│   │       │   ├── int_formatter.c
│   │       │   ├── int_formatter.h
│   │       │   ├── qint_formatter.c
│   │       │   └── qint_formatter.h
│   │       ├── layouts
│   │       │   ├── dense_layout.c
│   │       │   ├── dense_layout.h
│   │       │   ├── strided_layout.c
│   │       │   └── summarized_layout.c
│   │       ├── metadata
│   │       │   ├── metadata_fmt.c
│   │       │   └── metadata_fmt.h
│   │       ├── options
│   │       │   └── repr_options.c
│   │       ├── string_builder
│   │       │   ├── string_builder.c
│   │       │   └── string_builder.h
│   │       └── traversal
│   │           ├── tensor_iterator.c
│   │           └── tensor_iterator.h
│   ├── tests
│   └── CMakeLists.txt
├── nova
│   ├── _interfaces
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── _base_tensor.py
│   │   ├── _lr_scheduler.py
│   │   ├── _lr_scheduler.pyi
│   │   ├── _optimizer.py
│   │   └── _optimizer.pyi
│   ├── _internal
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── _binding.py
│   │   └── _generators.py
│   ├── _typing
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── _typing.py
│   ├── autograd
│   │   ├── _ops
│   │   │   ├── native
│   │   │   │   └── native_functions.yaml
│   │   │   ├── __init__.py
│   │   │   ├── _activation.py
│   │   │   ├── _arithmetic.py
│   │   │   ├── _comparison.py
│   │   │   ├── _convolution.py
│   │   │   ├── _creation.py
│   │   │   ├── _indexing.py
│   │   │   ├── _linalg.py
│   │   │   ├── _linear.py
│   │   │   ├── _loss.py
│   │   │   ├── _manipulation.py
│   │   │   ├── _normalization.py
│   │   │   ├── _random.py
│   │   │   ├── _reduction.py
│   │   │   ├── _trigonometric.py
│   │   │   ├── _view.py
│   │   │   └── utils.py
│   │   ├── engine
│   │   │   ├── __init__.py
│   │   │   ├── context.py
│   │   │   ├── context.pyi
│   │   │   ├── engine.py
│   │   │   └── engine.pyi
│   │   ├── tests
│   │   │   ├── op_signatures.py
│   │   │   ├── test_backward.py
│   │   │   ├── test_function.py
│   │   │   ├── test_gradients.py
│   │   │   └── test_operations.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   └── processing.py
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── function.py
│   │   ├── function.pyi
│   │   ├── grad.py
│   │   ├── grad.pyi
│   │   └── grad_mode.py
│   ├── core
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── constants.py
│   ├── metrics
│   │   ├── classification
│   │   │   ├── __init__.py
│   │   │   ├── _confusion.py
│   │   │   ├── _roc_auc.py
│   │   │   └── _stat.py
│   │   ├── regression
│   │   │   ├── __init__.py
│   │   │   ├── _error.py
│   │   │   └── _r2.py
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   └── metric.py
│   ├── nn
│   │   ├── modules
│   │   │   ├── __init__.py
│   │   │   ├── activation.py
│   │   │   ├── batchnorm.py
│   │   │   ├── container.py
│   │   │   ├── conv.py
│   │   │   ├── dropout.py
│   │   │   ├── flatten.py
│   │   │   ├── layernorm.py
│   │   │   ├── lazy.py
│   │   │   ├── linear.py
│   │   │   ├── loss.py
│   │   │   ├── module.py
│   │   │   ├── module.pyi
│   │   │   └── pooling.py
│   │   ├── utils
│   │   │   ├── __init__.py
│   │   │   ├── clip_grad.py
│   │   │   └── tensor_utils.py
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── functional.py
│   │   ├── functional.pyi
│   │   ├── init.py
│   │   ├── init.pyi
│   │   ├── parameter.py
│   │   └── parameter.pyi
│   ├── optim
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── adam.py
│   │   ├── adam.pyi
│   │   ├── adamw.py
│   │   ├── adamw.pyi
│   │   ├── lr_scheduler.py
│   │   ├── lr_scheduler.pyi
│   │   ├── rmsprop.py
│   │   ├── rmsprop.pyi
│   │   ├── sgd.py
│   │   └── sgd.pyi
│   ├── serialization
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── _safe_load.py
│   │   ├── load.py
│   │   ├── load.pyi
│   │   ├── save.py
│   │   └── save.pyi
│   ├── utils
│   │   ├── data
│   │   │   ├── __init__.py
│   │   │   ├── dataloader.py
│   │   │   ├── dataset.py
│   │   │   └── preprocessing.py
│   │   ├── datasets
│   │   │   ├── __init__.py
│   │   │   ├── fashion.py
│   │   │   └── mnist.py
│   │   ├── decorators
│   │   │   ├── __init__.py
│   │   │   ├── memory_usage.py
│   │   │   ├── registry.py
│   │   │   └── timing.py
│   │   ├── README.es.md
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── grad_checking.py
│   │   ├── hooks.py
│   │   ├── logger.py
│   │   ├── memory.py
│   │   └── to_tensor.py
│   ├── README.es.md
│   ├── README.md
│   ├── __init__.py
│   ├── _tensor.py
│   ├── _tensor.pyi
│   ├── dtypes.py
│   └── exceptions.py
├── skills
│   ├── conventional-commit
│   │   └── SKILL.md
│   └── doxygen-c-cxx-documentation
│       └── SKILL.md
├── tests
│   ├── functional
│   │   ├── test_activation.py
│   │   └── test_loss.py
│   ├── nn
│   │   ├── test_batchnorm.py
│   │   ├── test_clip_grad.py
│   │   ├── test_container.py
│   │   ├── test_conv.py
│   │   ├── test_dropout.py
│   │   ├── test_flatten.py
│   │   ├── test_layernorm.py
│   │   ├── test_lazy_variants.py
│   │   ├── test_linear.py
│   │   └── test_pooling.py
│   ├── optim
│   │   ├── test_adam.py
│   │   ├── test_adamw.py
│   │   ├── test_common.py
│   │   ├── test_rmsprop.py
│   │   ├── test_schedulers.py
│   │   └── test_sgd.py
│   ├── serialization
│   │   ├── test_load.py
│   │   └── test_save.py
│   ├── README.es.md
│   ├── README.md
│   ├── test_api.py
│   ├── test_binding_system.py
│   ├── test_conversion.py
│   ├── test_creation_and_casting.py
│   ├── test_dataset.py
│   ├── test_fashion_loader.py
│   ├── test_hooks.py
│   ├── test_loader.py
│   ├── test_memory_utils.py
│   ├── test_metrics.py
│   ├── test_mnist_loader.py
│   ├── test_normal_use.py
│   ├── test_preprocessing.py
│   ├── test_registry.py
│   └── test_timing_decorators.py
├── thirdParty
│   └── sleef
├── tools
│   └── codegen
│       ├── rules
│       ├── templates
│       └── engine.py
├── .clang-format
├── .clang-tidy
├── .clangd
├── .gitattributes
├── .gitignore
├── CHANGELOG.es.md
├── CHANGELOG.md
├── CMakeLists.txt
├── CONTRIBUTING.es.md
├── CONTRIBUTING.md
├── LICENCE
├── README.es.md
├── README.md
├── main
├── main.cpp
├── merge-commit.txt
├── project-tree.md
├── pyproject.toml
└── uv.lock
```
