# Testing en NovaNN v2.0.0

## 🧪 Visión General

NovaNN incluye una suite completa de tests unitarios que verifica la correcta implementación de todos los componentes del framework. Con una cobertura >95%, los tests aseguran que cada capa, optimizador, función de pérdida y utilidad funcione correctamente tanto en forward como en backward pass.

## 📁 Estructura de Tests

```
📁 tests
├── 📁 initializers
│   └── 🐍 test_init.py
├── 📁 layers
│   ├── 📁 activations
│   │   ├── 🐍 test_leaky_relu.py
│   │   ├── 🐍 test_relu.py
│   │   ├── 🐍 test_sigmoid.py
│   │   ├── 🐍 test_softmax.py
│   │   └── 🐍 test_tanh.py
│   ├── 📁 batch_norm
│   │   ├── 🐍 test_batchnorm1d.py
│   │   └── 🐍 test_batchnorm2d.py
│   ├── 📁 conv
│   │   ├── 🐍 test_conv1d.py
│   │   └── 🐍 test_conv2d.py
│   ├── 📁 linear
│   │   └── 🐍 test_linear.py
│   ├── 📁 pooling
│   │   ├── 📁 gap
│   │   │   ├── 🐍 test_gap1d.py
│   │   │   └── 🐍 test_gap2d.py
│   │   └── 📁 maxpool
│   │       ├── 🐍 test_maxpooling1d.py
│   │       └── 🐍 test_maxpooling2d.py
│   └── 📁 regularization
│       └── 🐍 test_dropout.py
├── 📁 optimizers
│   ├── 🐍 test_adam.py
│   ├── 🐍 test_rmsprop.py
│   └── 🐍 test_sgd.py
└── 📁 sequential
    └── 🐍 test_sequential.py
```