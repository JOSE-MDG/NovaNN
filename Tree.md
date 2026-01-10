# File Tree: NovaNN

**Generated:** 1/10/2026, 6:57:05 PM
**Root Path:** `/home/juancho_col/Documents/NovaNN`

```
├── 📁 benchmarks
│   ├── 📁 autograd
│   ├── 📁 ops
│   └── 📁 training
├── 📁 examples
│   ├── 🐍 binary_classification.py
│   ├── 🐍 conv_example.py
│   ├── 🐍 multiclass_classification.py
│   └── 🐍 regresion.py
├── 📁 images
│   ├── 🖼️ NovaNN Banners.png
│   └── 🖼️ metrics.png
├── 📁 notebooks
│   └── 📄 exploration.ipynb
├── 📁 nova
│   ├── 📁 _interfaces
│   │   ├── 🐍 _base_tensor.py
│   │   ├── 🐍 _lr_scheduler.py
│   │   └── 🐍 _optimizer.py
│   ├── 📁 _internal
│   │   ├── 🐍 _binding.py
│   │   └── 🐍 _generators.py
│   ├── 📁 _typing
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
│   │   │   └── 🐍 engine.py
│   │   ├── 📁 tests
│   │   │   ├── 🐍 op_signatures.py
│   │   │   ├── 🐍 test_backward.py
│   │   │   ├── 🐍 test_function.py
│   │   │   ├── 🐍 test_gradients.py
│   │   │   └── 🐍 test_operations.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 processing.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 function.py
│   │   ├── 🐍 grad.py
│   │   └── 🐍 grad_mode.py
│   ├── 📁 core
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
│   │   │   └── 🐍 pooling.py
│   │   ├── 📁 utils
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 clip_grad.py
│   │   │   └── 🐍 standardization.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 functional.py
│   │   ├── 🐍 init.py
│   │   ├── 🐍 parameter.py
│   │   └── 📄 parameter.pyi
│   ├── 📁 optim
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 adam.py
│   │   ├── 🐍 adamw.py
│   │   ├── 🐍 lr_scheduler.py
│   │   ├── 🐍 rmsprop.py
│   │   └── 🐍 sgd.py
│   ├── 📁 serialization
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 _safe_load.py
│   │   ├── 🐍 load.py
│   │   └── 🐍 save.py
│   ├── 📁 utils
│   │   ├── 📁 decorators
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 registry.py
│   │   │   └── 🐍 timing.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 grad_checking.py
│   │   ├── 🐍 hooks.py
│   │   ├── 🐍 logger.py
│   │   ├── 🐍 to_tensor.py
│   │   └── 🐍 visualization.py
│   ├── 🐍 __init__.py
│   ├── 🐍 _tensor.py
│   ├── 📄 _tensor.pyi
│   └── 🐍 dtypes.py
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
│   │   ├── 🐍 test_rmsprop.py
│   │   ├── 🐍 test_schedulers.py
│   │   └── 🐍 test_sgd.py
│   ├── 📁 serialization
│   │   ├── 🐍 test_load.py
│   │   └── 🐍 test_save.py
│   ├── 📝 README.en.md
│   ├── 📝 README.md
│   ├── 🐍 test_binding.py
│   ├── 🐍 test_conversion.py
│   ├── 🐍 test_creation_and_casting.py
│   ├── 🐍 test_hooks.py
│   ├── 🐍 test_loader.py
│   ├── 🐍 test_metrics.py
│   ├── 🐍 test_registry.py
│   └── 🐍 test_tensor_api.py
├── 📁 tutorials
│   ├── 📁 advanced
│   │   ├── 🐍 01_schedulers.py
│   │   ├── 🐍 02_metrics.py
│   │   ├── 🐍 03_no_grad.py
│   │   ├── 🐍 04_convolutionals.py
│   │   ├── 🐍 05_checkpoints.py
│   │   ├── 🐍 06_clipping.py
│   │   ├── 🐍 07_hooks.py
│   │   ├── 🐍 08_batchnorm.py
│   │   └── 🐍 09_cnns.py
│   ├── 📁 basic
│   │   ├── 🐍 01_nova_intro.py
│   │   ├── 🐍 02_tensor.py
│   │   ├── 🐍 03_basic_operations.py
│   │   ├── 🐍 04_sgd_optimizer.py
│   │   ├── 🐍 05_backward_and_gradients.py
│   │   └── 🐍 06_simple_mlp.py
│   ├── 📁 intermediate
│   │   ├── 🐍 01_optimizers.py
│   │   ├── 🐍 02_functinal.py
│   │   ├── 🐍 03_module.py
│   │   ├── 🐍 04_no_grad.py
│   │   ├── 🐍 05_training_loop.py
│   │   ├── 🐍 06_standarization.py
│   │   └── 🐍 07_classifier.py
│   └── 📝 philosophy.md
├── ⚙️ .gitignore
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
