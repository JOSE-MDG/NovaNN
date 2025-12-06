# File Tree: NovaNN

**Generated:** 12/6/2025, 8:20:44 PM
**Root Path:** `/home/juancho_col/Documents/NovaNN`

```
├── 📁 examples
│   ├── 🐍 binary_classification.py
│   ├── 🐍 conv_example.py
│   ├── 🐍 multiclass_classification.py
│   └── 🐍 regresion.py
├── 📁 images
│   └── 🖼️ comparison.png
├── 📁 notebooks
│   └── 📄 exploration.ipynb
├── 📁 novann
│   ├── 📁 _typing
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 _typing.py
│   ├── 📁 core
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 constants.py
│   │   └── 🐍 init.py
│   ├── 📁 layers
│   │   ├── 📁 activations
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 activations.py
│   │   │   ├── 🐍 relu.py
│   │   │   ├── 🐍 sigmoid.py
│   │   │   ├── 🐍 softmax.py
│   │   │   └── 🐍 tanh.py
│   │   ├── 📁 bn
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 batchnorm1d.py
│   │   │   └── 🐍 batchnorm2d.py
│   │   ├── 📁 convolutional
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 conv1d.py
│   │   │   └── 🐍 conv2d.py
│   │   ├── 📁 flatten
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 flatten.py
│   │   ├── 📁 linear
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 linear.py
│   │   ├── 📁 pooling
│   │   │   ├── 📁 gap
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 global_avg_pool1d.py
│   │   │   │   └── 🐍 global_avg_pool2d.py
│   │   │   ├── 📁 maxpool
│   │   │   │   ├── 🐍 __init__.py
│   │   │   │   ├── 🐍 maxpool1d.py
│   │   │   │   └── 🐍 maxpool2d.py
│   │   │   └── 🐍 __init__.py
│   │   ├── 📁 regularization
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 dropout.py
│   │   └── 🐍 __init__.py
│   ├── 📁 losses
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 functional.py
│   ├── 📁 metrics
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 metrics.py
│   ├── 📁 model
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 nn.py
│   ├── 📁 module
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 layer.py
│   │   └── 🐍 module.py
│   ├── 📁 optim
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 adam.py
│   │   ├── 🐍 rmsprop.py
│   │   └── 🐍 sgd.py
│   ├── 📁 utils
│   │   ├── 📁 decorators
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 timing.py
│   │   ├── 📁 gradient_checking
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 numerical.py
│   │   ├── 📁 log_config
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 logger.py
│   │   ├── 📁 train
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 train.py
│   │   ├── 📁 visualizations
│   │   │   └── 🐍 visualization.py
│   │   └── 🐍 __init__.py
│   └── 🐍 __init__.py
├── 📁 tests
│   ├── 📁 initializers
│   │   └── 🐍 test_init.py
│   ├── 📁 layers
│   │   ├── 📁 activations
│   │   │   ├── 🐍 test_leaky_relu.py
│   │   │   ├── 🐍 test_relu.py
│   │   │   ├── 🐍 test_sigmoid.py
│   │   │   ├── 🐍 test_softmax.py
│   │   │   └── 🐍 test_tanh.py
│   │   ├── 📁 batch_norm
│   │   │   ├── 🐍 test_batchnorm1d.py
│   │   │   └── 🐍 test_batchnorm2d.py
│   │   ├── 📁 conv
│   │   │   ├── 🐍 test_conv1d.py
│   │   │   └── 🐍 test_conv2d.py
│   │   ├── 📁 linear
│   │   │   └── 🐍 test_linear.py
│   │   ├── 📁 pooling
│   │   │   ├── 📁 gap
│   │   │   │   ├── 🐍 test_gap1d.py
│   │   │   │   └── 🐍 test_gap2d.py
│   │   │   └── 📁 maxpool
│   │   │       ├── 🐍 test_maxpooling1d.py
│   │   │       └── 🐍 test_maxpooling2d.py
│   │   └── 📁 regularization
│   │       └── 🐍 test_dropout.py
│   ├── 📁 optimizers
│   │   ├── 🐍 test_adam.py
│   │   ├── 🐍 test_rmsprop.py
│   │   └── 🐍 test_sgd.py
│   └── 📁 sequential
│       └── 🐍 test_sequential.py
├── ⚙️ .gitignore
├── 📄 LICENCE
├── 📝 NovaNNFiletree.md
├── 📝 README.en.md
├── 📝 README.md
├── 🐍 main.py
├── 📄 poetry.lock
└── ⚙️ pyproject.toml
```

---
*Generated by FileTree Pro Extension*