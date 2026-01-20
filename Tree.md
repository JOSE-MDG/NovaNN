# File Tree: NovaNN

**Generated:** 1/20/2026, 8:55:03 PM
**Root Path:** `/home/juancho_col/Documents/NovaNN`

```
├── 📁 benchmarks
│   ├── 📁 autograd
│   │   ├── 🐍 backward_overhead.py
│   │   ├── 🐍 grad_accumulation.py
│   │   ├── 🐍 grad_checking.py
│   │   ├── 🐍 memory_footprint.py
│   │   └── 🐍 small_graphs.py
│   ├── 📁 ops
│   │   ├── 🐍 broadcasting_cost.py
│   │   ├── 🐍 elementwise_cpu.py
│   │   ├── 🐍 reduction_ops.py
│   │   └── 🐍 small_tensor_ops.py
│   ├── 📁 training
│   │   ├── 🐍 end_to_end_cpu.py
│   │   ├── 🐍 lr_scheduler_step.py
│   │   ├── 🐍 optimizer_step.py
│   │   └── 🐍 tiny_mlp_cpu.py
│   ├── 📁 utils
│   │   ├── 🐍 compare_with_torch.py
│   │   ├── 🐍 memory.py
│   │   ├── 🐍 report.py
│   │   └── 🐍 timing.py
│   ├── 📝 README.en.md
│   └── 📝 README.es.md
├── 📁 examples
│   ├── 🐍 binary_classification.py
│   ├── 🐍 conv_example.py
│   ├── 🐍 multiclass_classification.py
│   └── 🐍 regression.py
├── 📁 images
│   ├── 🖼️ NovaNN Banners.png
│   └── 🖼️ metrics.png
├── 📁 notebooks
│   └── 📄 exploration.ipynb
├── 📁 nova
│   ├── 📁 _interfaces
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 _base_tensor.py
│   │   ├── 🐍 _lr_scheduler.py
│   │   ├── 📄 _lr_scheduler.pyi
│   │   ├── 🐍 _optimizer.py
│   │   └── 📄 _optimizer.pyi
│   ├── 📁 _internal
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 _binding.py
│   │   └── 🐍 _generators.py
│   ├── 📁 _typing
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 _typing.py
│   ├── 📁 autograd
│   │   ├── 📁 _ops
│   │   │   ├── 📁 _activation
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 gelu.py
│   │   │   │   ├── 🐍 leaky_relu.py
│   │   │   │   ├── 🐍 prelu.py
│   │   │   │   ├── 🐍 relu.py
│   │   │   │   └── 🐍 sigmoid.py
│   │   │   ├── 📁 _basic
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   └── 🐍 arithmetic.py
│   │   │   ├── 📁 _comparison
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   └── 🐍 comparison.py
│   │   │   ├── 📁 _creation
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 factory.py
│   │   │   │   └── 🐍 random.py
│   │   │   ├── 📁 _indexing
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 getitem.py
│   │   │   │   └── 🐍 setitem.py
│   │   │   ├── 📁 _linalg
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 det.py
│   │   │   │   ├── 🐍 diag.py
│   │   │   │   ├── 🐍 dot.py
│   │   │   │   ├── 🐍 inv.py
│   │   │   │   ├── 🐍 matmul.py
│   │   │   │   ├── 🐍 norm.py
│   │   │   │   └── 🐍 trace.py
│   │   │   ├── 📁 _manipulation
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 clone.py
│   │   │   │   └── 🐍 manipulation.py
│   │   │   ├── 📁 _reduction
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   └── 🐍 reduce.py
│   │   │   ├── 📁 _trigonometric
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   └── 🐍 trigonometric.py
│   │   │   ├── 📁 native
│   │   │   │   └── ⚙️ native_functions.yaml
│   │   │   ├── 📁 utils
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   └── 🐍 unbroadcasting.py
│   │   │   ├── 📁 views
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 as_strided.py
│   │   │   │   ├── 🐍 extend.py
│   │   │   │   └── 🐍 view.py
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 engine
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 context.py
│   │   │   ├── 📄 context.pyi
│   │   │   ├── 🐍 engine.py
│   │   │   └── 📄 engine.pyi
│   │   ├── 📁 tests
│   │   │   ├── 🐍 op_signatures.py
│   │   │   ├── 🐍 test_backward.py
│   │   │   ├── 🐍 test_function.py
│   │   │   ├── 🐍 test_gradients.py
│   │   │   └── 🐍 test_operations.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 processing.py
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 function.py
│   │   ├── 📄 function.pyi
│   │   ├── 🐍 grad.py
│   │   ├── 📄 grad.pyi
│   │   └── 🐍 grad_mode.py
│   ├── 📁 core
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   └── 🐍 constants.py
│   ├── 📁 metrics
│   │   ├── 📁 classification
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 _confusion.py
│   │   │   ├── 🐍 _roc_auc.py
│   │   │   └── 🐍 _stat.py
│   │   ├── 📁 regression
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 _error.py
│   │   │   └── 🐍 _r2.py
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 metric.py
│   ├── 📁 nn
│   │   ├── 📁 modules
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 activation.py
│   │   │   ├── 🐍 batchnorm.py
│   │   │   ├── 🐍 container.py
│   │   │   ├── 🐍 conv.py
│   │   │   ├── 🐍 dropout.py
│   │   │   ├── 🐍 flatten.py
│   │   │   ├── 🐍 layernorm.py
│   │   │   ├── 🐍 lazy.py
│   │   │   ├── 🐍 linear.py
│   │   │   ├── 🐍 loss.py
│   │   │   ├── 🐍 module.py
│   │   │   ├── 📄 module.pyi
│   │   │   └── 🐍 pooling.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 clip_grad.py
│   │   │   └── 🐍 standardization.py
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 functional.py
│   │   ├── 📄 functional.pyi
│   │   ├── 🐍 init.py
│   │   ├── 📄 init.pyi
│   │   ├── 🐍 parameter.py
│   │   └── 📄 parameter.pyi
│   ├── 📁 optim
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 adam.py
│   │   ├── 📄 adam.pyi
│   │   ├── 🐍 adamw.py
│   │   ├── 📄 adamw.pyi
│   │   ├── 🐍 lr_scheduler.py
│   │   ├── 📄 lr_scheduler.pyi
│   │   ├── 🐍 rmsprop.py
│   │   ├── 📄 rmsprop.pyi
│   │   ├── 🐍 sgd.py
│   │   └── 📄 sgd.pyi
│   ├── 📁 serialization
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 _safe_load.py
│   │   ├── 🐍 load.py
│   │   ├── 📄 load.pyi
│   │   ├── 🐍 save.py
│   │   └── 📄 save.pyi
│   ├── 📁 utils
│   │   ├── 📁 decorators
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 registry.py
│   │   │   └── 🐍 timing.py
│   │   ├── 📝 README.en.md
│   │   ├── 📝 README.es.md
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 grad_checking.py
│   │   ├── 🐍 hooks.py
│   │   ├── 🐍 logger.py
│   │   ├── 🐍 to_tensor.py
│   │   └── 🐍 visualization.py
│   ├── 📝 README.en.md
│   ├── 📝 README.es.md
│   ├── 🐍 __init__.py
│   ├── 🐍 _tensor.py
│   ├── 📄 _tensor.pyi
│   ├── 🐍 dtypes.py
│   └── 🐍 exceptions.py
├── 📁 tests
│   ├── 📁 functional
│   │   ├── 🐍 test_activation.py
│   │   └── 🐍 test_loss.py
│   ├── 📁 nn
│   │   ├── 🐍 test_batchnorm.py
│   │   ├── 🐍 test_container.py
│   │   ├── 🐍 test_conv.py
│   │   ├── 🐍 test_dropout.py
│   │   ├── 🐍 test_flatten.py
│   │   ├── 🐍 test_layernorm.py
│   │   ├── 🐍 test_lazy_variants.py
│   │   ├── 🐍 test_linear.py
│   │   └── 🐍 test_pooling.py
│   ├── 📁 optim
│   │   ├── 🐍 test_adam.py
│   │   ├── 🐍 test_adamw.py
│   │   ├── 🐍 test_common.py
│   │   ├── 🐍 test_rmsprop.py
│   │   ├── 🐍 test_schedulers.py
│   │   └── 🐍 test_sgd.py
│   ├── 📁 serialization
│   │   ├── 🐍 test_load.py
│   │   └── 🐍 test_save.py
│   ├── 📝 README.en.md
│   ├── 📝 README.es.md
│   ├── 🐍 test_api.py
│   ├── 🐍 test_binding_system.py
│   ├── 🐍 test_conversion.py
│   ├── 🐍 test_creation_and_casting.py
│   ├── 🐍 test_hooks.py
│   ├── 🐍 test_loader.py
│   ├── 🐍 test_metrics.py
│   └── 🐍 test_registry.py
├── 📁 tutorials
│   ├── 📁 00_philosophy
│   │   ├── 📝 design_goals.md
│   │   ├── 🐍 gradients_are_graphs.py
│   │   ├── 🐍 tensors_are_values.py
│   │   └── 📝 why_nova.md
│   ├── 📁 01_basics
│   │   ├── 🐍 broadcasting.py
│   │   ├── 🐍 common_pitfalls.py
│   │   ├── 🐍 creation_and_dtypes.py
│   │   ├── 🐍 indexing_and_views.py
│   │   └── 🐍 tensors.py
│   ├── 📁 02_autograd
│   │   ├── 🐍 backward_basics.py
│   │   ├── 🐍 computational_graph.py
│   │   ├── 🐍 grad_accumulation.py
│   │   ├── 🐍 grad_debugging.py
│   │   ├── 🐍 no_grad_mode.py
│   │   └── 🐍 requires_grad.py
│   ├── 📁 03_nn
│   │   ├── 🐍 batchnorm_and_dropout.py
│   │   ├── 🐍 conv2d_step_by_step.py
│   │   ├── 🐍 linear_and_activation.py
│   │   ├── 🐍 modules_and_parameters.py
│   │   ├── 🐍 train_vs_eval.py
│   │   └── 🐍 weight_initialization.py
│   ├── 📁 04_training
│   │   ├── 🐍 gradient_clipping.py
│   │   ├── 🐍 lr_schedulers.py
│   │   ├── 🐍 optimizers_explained.py
│   │   ├── 🐍 saving_and_loading.py
│   │   └── 🐍 training_loop_from_scratch.py
│   ├── 📁 05_advanced
│   │   ├── 🐍 custom_autograd_function.py
│   │   ├── 🐍 grad_checking.py
│   │   ├── 🐍 hooks_and_profiling.py
│   │   ├── 🐍 numerical_stability.py
│   │   └── 📝 performance_notes.md
│   └── 📁 06_comparison
│       ├── 🐍 api_comparison_with_torch.py
│       └── 📝 design_tradeoffs.md
├── ⚙️ .gitignore
├── 📝 CHANGELOG.en.md
├── 📝 CHANGELOG.md
├── 📝 CONTRIBUTING.en.md
├── 📝 CONTRIBUTING.md
├── 📄 LICENCE
├── 📝 README.en.md
├── 📝 README.md
├── 📝 Tree.md
├── 🐍 main.py
├── 📄 poetry.lock
└── ⚙️ pyproject.toml
```

---

_Generated by FileTree Pro Extension_
