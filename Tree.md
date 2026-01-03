# File Tree: NovaNN

**Generated:** 1/3/2026, 10:27:24 PM
**Root Path:** `/home/juancho_col/Documents/NovaNN`

```
├── 📁 benchmarks
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
│   │   │   │   ├── 📁 views
│   │   │   │   │   ├── 🐍 __init__.py
│   │   │   │   │   ├── 🐍 permute.py
│   │   │   │   │   ├── 🐍 reshape.py
│   │   │   │   │   └── 🐍 transpose.py
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 det.py
│   │   │   │   ├── 🐍 dot.py
│   │   │   │   ├── 🐍 inv.py
│   │   │   │   ├── 🐍 matmul.py
│   │   │   │   ├── 🐍 norm.py
│   │   │   │   └── 🐍 trace.py
│   │   │   ├── 📁 _manipulation
│   │   │   │   ├── 🐍 __init__.py
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
│   │   │   ├── 📁 ops
│   │   │   ├── 🐍 test_function.py
│   │   │   └── 🐍 test_grad_checking.py
│   │   ├── 📁 utils
│   │   │   ├── 📁 gradients
│   │   │   │   ├── 📁 clipping
│   │   │   │   │   ├── 🐍 __init__.py
│   │   │   │   │   ├── 🐍 clip_grad_norm.py
│   │   │   │   │   └── 🐍 clip_value.py
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 grad.py
│   │   │   │   └── 🐍 grad_checking.py
│   │   │   ├── 📁 hooks
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   └── 🐍 handle.py
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 processig.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 function.py
│   │   ├── 🐍 grad.py
│   │   └── 🐍 grad_mode.py
│   ├── 📁 core
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   └── 🐍 constants.py
│   ├── 📁 nn
│   │   ├── 📁 eval
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 metrics.py
│   │   ├── 📁 modules
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 activation.py
│   │   │   ├── 🐍 batchnorm.py
│   │   │   ├── 🐍 container.py
│   │   │   ├── 🐍 conv.py
│   │   │   ├── 🐍 dropout.py
│   │   │   ├── 🐍 flatten.py
│   │   │   ├── 🐍 layernorm.py
│   │   │   ├── 🐍 linear.py
│   │   │   ├── 🐍 loss.py
│   │   │   ├── 🐍 module.py
│   │   │   └── 🐍 pooling.py
│   │   ├── 📁 utils
│   │   │   └── 🐍 __init__.py
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 functional.py
│   │   ├── 🐍 init.py
│   │   └── 🐍 parameter.py
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
│   │   ├── 📁 log_config
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 logger.py
│   │   ├── 📁 train
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 train.py
│   │   ├── 📁 visualizations
│   │   │   └── 🐍 visualization.py
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 to_tensor.py
│   ├── 🐍 __init__.py
│   ├── 🐍 _tensor.py
│   ├── 📄 _tensor.pyi
│   └── 🐍 dtypes.py
├── 📁 tests
│   ├── 📝 README.en.md
│   ├── 📝 README.md
│   └── 🐍 test_dtype_casting.py
├── 📁 tutorials
├── ⚙️ .gitignore
├── 📝 CONTRIBUTING.en.md
├── 📝 CONTRIBUTING.md
├── 📄 LICENCE
├── 📝 README.en.md
├── 📝 README.md
├── 📝 Tree.md
├── 📄 checking.ipynb
├── 🐍 main.py
├── 📄 poetry.lock
└── ⚙️ pyproject.toml
```

---

_Generated by FileTree Pro Extension_
