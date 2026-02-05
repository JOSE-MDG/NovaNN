# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.3] - 2026-02-04

### 🎉 Lanzamiento Mayor - Refactorización Completa del Framework

La versión 4.0.3 representa una **reescritura completa** de NovaNN, transformándolo de un proyecto educativo básico a un framework de deep learning modular, extensible y profesional. Esta versión introduce cambios fundamentales en la arquitectura, API y filosofía del proyecto.

### ✨ Añadido

#### Sistema de Autograd Dinámico

- **Motor de diferenciación automática completo** inspirado en PyTorch
  - Construcción dinámica de grafos computacionales
  - Clase base `Function` para operaciones diferenciables
  - Sistema `Context` para cache de valores intermedios
  - Backpropagation automática con `tensor.backward()`
  - Gestión de gradientes con `grad_fn` y tracking automático
  - Modos `no_grad()` y `enable_grad()` para control de gradientes
- **80+ operaciones diferenciables** organizadas por categorías:
  - Operaciones básicas (aritméticas, exponenciación, logaritmos)
  - Activaciones (ReLU, LeakyReLU, PReLU, GELU, Sigmoide)
  - Pérdidas (mse, bce, bcewithlogits)
  - Álgebra lineal (matmul, dot, det, inv, norm, trace)
  - Capa lineal (densa)
  - Manipulación de tensores (reshape, permute, stack, concat, split)
  - Matmul convolucional (ConvMatmul1d, ConvMatmul2d, ConvMatmul3d)
  - Reducción (suma, media, var, mín, max)
  - Trigonométricas (sin, cos, tan, arcsin, arccos, arctan)
  - Normalización (batchnorm, layernorm)
  - Comparación (maximum, minimum, where)
  - Indexación avanzada (getitem, setitem con indexación sofisticada)
  - Vistas y pasos (as_strided, view, extend)

#### Abstracción de Tensores

- **Clase `Tensor` completa** como wrapper sobre NumPy arrays
  - Sistema de tipos con `dtype` support (float16-float256, uint8-uint64, int8-int64, complex64-256, bool)
  - Tracking automático de gradientes con `requires_grad`
  - Métodos in-place con sufijo `_` (add*, mul*, zero\_, etc.)
  - Operadores sobrecargados (`+`, `-`, `*`, `/`, `@`, `**`, etc.)
  - Sistema de binding dinámico desde YAML para generación de API
  - Propiedades avanzadas (shape, strides, ndim, device, is_leaf)

#### Módulo `nn` Completo

- **Capas fundamentales**:
  - `Linear` y `LazyLinear` (fully connected con lazy initialization)
  - `Conv1d`, `Conv2d`, `Conv3d` (convoluciones 1D/2D/3D)
  - `LazyConv1d`, `LazyConv2d`, `LazyConv3d` (versiones lazy)
  - `Flatten` para transición conv → fc
- **Normalización**:
  - `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`
  - `LazyBatchNorm1d`, `LazyBatchNorm2d`, `LazyBatchNorm3d`
  - `LayerNorm` para arquitecturas Transformer futuras
- **Activaciones como módulos**:
  - `ReLU`, `LeakyReLU`, `PReLU`, `GELU`, `Sigmoid`, `Tanh`, `Softmax`
- **Pooling**:
  - `MaxPool1d`, `MaxPool2d`, `MaxPool3d`
  - `AvgPool1d`, `AvgPool2d`, `AvgPool3d`
  - `GlobalAvgPool1d`, `GlobalAvgPool2d`, `GlobalAvgPool3d`
- **Regularización**:
  - `Dropout`, `Dropout2d`, `Dropout3d`
- **Contenedor `Sequential`** mejorado con auto-registro de submódulos
- **Sistema `Module`** completo:
  - Auto-registro de parámetros, buffers y submódulos
  - Modos `train()` / `eval()` propagados recursivamente
  - `state_dict()` / `load_state_dict()` para serialización
  - Iteradores `parameters()`, `named_parameters()`, `modules()`, `named_modules()`
  - Representación legible con `__repr__()` y `extra_repr()`

#### Módulo `nn.functional`

- **API funcional completa** sin estado para todas las operaciones
- Funciones de activación: `relu()`, `leaky_relu()`, `gelu()`, `sigmoid()`, `tanh()`, `softmax()`, `log_softmax()`
- Funciones de pérdida: `mse_loss()`, `l1_loss()`, `smooth_l1_loss()`, `binary_cross_entropy()`, `binary_cross_entropy_with_logits()`, `nll_loss()`, `cross_entropy()`, `kl_div()`
- Operaciones lineales/conv: `linear()`, `conv1d()`, `conv2d()`, `conv3d()`
- Pooling funcional: `max_pool1d/2d/3d()`, `avg_pool1d/2d/3d()`, `global_avg_pool1d/2d/3d()`
- Normalización: `batch_norm()`, `layer_norm()`, `normalize()`
- Dropout funcional: `dropout()`, `dropout2d()`, `dropout3d()`

#### Módulo `optim`

- **Optimizadores modernos**:
  - `SGD` con momentum y gradient clipping
  - `Adam` con coupled weight decay
  - `AdamW` con decoupled weight decay (recomendado)
  - `RMSprop` con centered variant
- **Learning rate schedulers**:
  - `StepLR` (decaimiento escalonado)
  - `CosineAnnealingLR` (decaimiento coseno)
  - `OneCycleLR` (super-convergence con momentum cycling)
- **Características comunes**:
  - Sistema de `param_groups` para learning rates diferenciados
  - Exclusión automática de parámetros BatchNorm del weight decay
  - Hooks pre/post step para logging y debugging
  - `state_dict()` / `load_state_dict()` para checkpointing

#### Módulo `metrics`

- **Sistema de métricas acumulativas** con patrón `reset()` → `update()` → `compute()`
- **Clasificación**:
  - `Accuracy`, `Precision`, `Recall`, `F1Score` con averaging (micro/macro/weighted)
  - `ConfusionMatrix` multi-clase eficiente con bincount
  - `ROCAUC` para clasificación binaria
- **Regresión**:
  - `MSE` (MSE/RMSE)
  - `MAE` (MAE)
  - `R2Score` (coeficiente de determinación)

#### Inicialización de Pesos

- **Módulo `nn.init`** completo con funciones profesionales:
  - Xavier/Glorot: `xavier_normal_()`, `xavier_uniform_()`
  - Kaiming/He: `kaiming_normal_()`, `kaiming_uniform_()`
  - Básicas: `uniform_()`, `normal_()`, `constant_()`, `zeros_()`, `ones_()`
  - `calculate_gain()` para ganancias por activación
  - `get_fans()` para cálculo de fan-in/fan-out

#### Serialización

- **Módulo `serialization`** para guardar/cargar modelos
  - `save()` y `load()` con soporte pickle seguro
  - Modo `weights_only=True` para seguridad
  - Sistema de registro para deserialización segura de clases personalizadas
  - Decorador `@registry_class` para auto-registro

#### Sistema de Tipos y Utilidades

- **Módulo `_typing`** con definiciones de tipos completas
  - Type hints para todas las APIs (Size, Dtype, Dim, etc.)
  - Stubs `.pyi` para mejor IDE support
- **Módulo `utils`**:
  - `hooks.py`: Sistema de hooks para módulos y optimizadores
  - `logger.py`: Logger profesional con niveles y formateo
  - `grad_checking.py`: Validación numérica de gradientes
  - `to_tensor.py`: Conversión flexible de datos a tensores
  - `visualization.py`: Utilidades para visualización de grafos y métricas
  - `clip_grad.py`: Gradient clipping (norm y value)

#### Benchmarks

- **Directorio `benchmarks/`** completo para análisis de rendimiento
  - Comparaciones con PyTorch en operaciones elementwise, reducción, autograd
  - Benchmarks de entrenamiento end-to-end
  - Análisis de memoria y overhead computacional
  - Scripts de reporte y visualización

#### Documentación

- **READMEs modulares** para cada submódulo con:
  - Descripción detallada de funcionalidades
  - Ejemplos de uso prácticos
  - Integración con otros módulos
  - Detalles técnicos de implementación
- **CONTRIBUTING.md** completo con guías de estilo y proceso de PR
- **CHANGELOG.md** estructurado (este archivo)

### 🔄 Cambiado

#### Arquitectura del Proyecto

- **Renombrado `novann/` → `nova/`** para API más limpia (`import nova` vs `import novann`)
- **Reorganización completa de módulos**:
  - `layers/` → distribuido en `nn/modules/` y `autograd/_ops/`
  - `model/nn.py` → `nn/modules/container.py` (Sequential)
  - `module/` → integrado en `nn/` como `Module` base
  - `losses/` → `nn/modules/loss.py` y `nn/functional.py`
  - `metrics/` → refactorizado con nueva API acumulativa
  - `utils/` → reorganizado por funcionalidad específica
- **Separación clara** entre:
  - API pública (`nova.nn`, `nova.optim`, etc.)
  - Implementaciones internas (`nova._internal`, `nova._interfaces`)
  - Sistema de tipos (`nova._typing`)

#### Sistema de Capas y Módulos

- **Nueva clase base `Module`** con metaclase para auto-registro
- **`Parameter` y `Buffer`** como clases independientes (no solo wrappers)
  - `Parameter` con `requires_grad=True` por defecto
  - `Buffer` para estadísticas no entrenables (BatchNorm)
  - Variantes `Uninitialized*` para lazy initialization
- **Lazy modules** con inicialización automática en primer forward
- **Sistema de hooks** para forward y backward passes

#### Optimizadores

- **API unificada** con clase base `Optimizer`
- **Weight decay desacoplado** en AdamW y RMSprop (mejor que v3.0.0)
- **Exclusión automática** de parámetros BatchNorm del weight decay (antes manual)
- **Soporte para `param_groups`** para learning rates diferenciados por capa

#### Métricas

- **Nueva API acumulativa** `reset()` → `update()` → `compute()` (antes cálculo directo)
- **Soporte para averaging** (micro/macro/weighted) en métricas de clasificación
- **Métricas por clase** además de globales

#### Serialización

- **Sistema de registro** para clases personalizadas (antes solo pickle básico)
- **Modo `weights_only=True`** para seguridad (previene ejecución de código arbitrario)

#### Testing

- **Cobertura reducida de 95% → 87%** debido a:
  - Expansión masiva del código (3.0.0: ~2000 líneas docs, 4.0.0: código modular)
  - Nuevos módulos aún sin tests completos (schedulers, algunas ops autograd)
  - Enfoque en arquitectura y funcionalidad sobre cobertura exhaustiva
- **Tests reorganizados** para reflejar nueva estructura de módulos

### ⚠️ Breaking Changes

#### Imports

```python
# v3.0.0
from novann.layers import Linear, ReLU
from novann.model import Sequential
from novann.losses import CrossEntropyLoss
from novann.optim import Adam

# v4.0.3
import nova.nn as nn
from nova.optim import Adam

model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU()
)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)
```

#### API de Módulos

```python
# v3.0.0 - capas sin auto-registro
class MyModel:
    def __init__(self):
        self.linear = Linear(10, 5)

    def parameters(self):
        return self.linear.parameters()  # manual

# v4.0.3 - auto-registro con Module
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)  # auto-registrado

    def forward(self, x):
        return self.linear(x)

    # parameters() heredado de Module
```

#### Inicialización de Pesos

```python
# v3.0.0 - inicialización manual en cada capa
layer = Linear(10, 5)
layer.reset_parameters(init_fn)

# v4.0.3 - inicialización en nn.init
from nova.nn import init
weight = nn.Parameter(nova.empty((10, 5)))
init.kaiming_normal_(weight, nonlinearity='relu')
```

#### Métricas

```python
# v3.0.0 - cálculo directo
acc = accuracy(model, dataloader)

# v4.0.3 - API acumulativa
from nova.metrics import Accuracy
metric = Accuracy(num_classes=10)
for input, target in dataloader:
    preds = model(input)
    metric.update(preds, target)
final_acc = metric.compute()
```

#### Training Loop

```python
# v3.0.0 - forward retorna output simple
output = model(x)
loss, grad = loss_fn(output, y)
model.backward(grad)

# v4.0.3 - con autograd automático
output = model(x)
loss = criterion(output, y)
loss.backward()  # calcula gradientes automáticamente
optimizer.step()
```

#### Parámetros y Gradientes

```python
# v3.0.0 - Parameters simple wrapper
class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

# v4.0.3 - Parameter con tracking completo
class Parameter(Tensor):
    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)
        # grad_fn, _inputs, etc. manejados por Tensor
```

### 🗑️ Eliminado

#### Módulos Deprecados

- ❌ `novann.functional` (API funcional básica) → reemplazado por `nova.nn.functional` completo
- ❌ `novann.utils.train.train()` (función de entrenamiento monolítica) → usuarios implementan loops personalizados
- ❌ `novann.utils.datasets` (loaders específicos) → usuarios cargan datos manualmente o usan `DataLoader` genérico
- ❌ Sistema de inicialización automática en `Sequential` basado en activaciones → reemplazado por `nn.init` explícito
- ❌ `novann.core.config` (mapas de inicialización hardcodeados)
- ❌ Clase `Layer` como abstracción separada → integrada en `Module`

#### Utilidades Removidas

- ❌ `utils/gradient_checking/numerical.py` → movido a `utils/grad_checking.py` con mejor API
- ❌ `utils/visualizations/visualization.py` → Eliminado. imagenes generadas por `benchmarks/`
- ❌ `utils/log_config/logger.py` → reemplazado por `utils/logger.py` con mejor configuración

#### Dependencias

- ❌ Dependencia implícita en estructura de carpetas específica
- ❌ Loaders hardcodeados para MNIST/Fashion-MNIST

### 🐛 Corregido

#### Autograd

- ✅ Propagación incorrecta de gradientes en operaciones con broadcasting
- ✅ Memory leaks en grafos computacionales largos
- ✅ Gradientes incorrectos en `MaxPool` con múltiples máximos iguales
- ✅ Unstable backward en `BatchNorm` con batch size = 1

#### Optimizadores

- ✅ Weight decay aplicado incorrectamente a parámetros BatchNorm en v3.0.0
- ✅ Momentum no inicializado correctamente en SGD
- ✅ Bias correction en Adam solo aplicado en primer step

#### Capas

- ✅ Padding incorrecto en `Conv2d` con stride > 1
- ✅ BatchNorm inestable en modo eval con running stats no inicializadas
- ✅ Dropout no desactivado correctamente en modo eval

#### Serialización

- ✅ Carga de modelos con arquitecturas personalizadas fallaba
- ✅ State dict no guardaba buffers persistentes de BatchNorm

#### Eficiencia

- ✅ Eficientes implementaciones de operaciones de manera nativa (ConvMatmul, Dense etc.)
- ✅ Reducción del tamaño del los grafos computacionales.

### 🔒 Seguridad

- 🔐 Modo `weights_only=True` en serialización previene ejecución de código arbitrario
- 🔐 Sistema de registro para clases seguras en deserialización
- 🔐 Validación de tipos en operaciones críticas

### 📊 Rendimiento

#### Mejoras

- ⚡ Operaciones autograd 15-25% más rápidas que v3.0.0 (optimización de loops)
- ⚡ Conv2d con im2col 30% más eficiente en CPU
- ⚡ Reducción de 40% en overhead de memoria del grafo computacional
- ⚡ Reducción del la duración del backward en un 20-30% en operaciones comunes nativas (Linear, BatchNorm, LazyerNorm)
- ⚡ Ahorro de momoria en un ~50% gracias a operaciones con pre-alocaciones y el core numpy
- ⚡ Metodo con inplementaciones in-place reales, no simuladas

#### Regresiones Conocidas

- 🐢 Backward en grafos muy profundos (>100 capas) puede ser lento vs PyTorch
- 🐢 Operaciones de indexing fancy ~2x más lentas que PyTorch (Python puro vs C++)

### 📝 Notas de Migración

#### Para usuarios de v3.0.0

1. **Actualizar imports**:

```python
   # Antes
   from novann.layers import Linear
   from novann.model import Sequential

   # Ahora
   import nova.nn as nn
```

2. **Adaptar training loops**:

```python
   # Antes
   loss, grad = criterion(output, target)
   model.backward(grad)

   # Ahora
   loss = criterion(output, target)
   loss.backward()
```

3. **Actualizar métricas**:

```python
   # Antes
   acc = accuracy(model, loader)

   # Ahora
   metric = Accuracy(num_classes=10)
   for batch in loader:
       metric.update(model(batch['x']), batch['y'])
   acc = metric.compute()
```

4. **Cambiar inicialización**:

```python
   # Antes
   # Sequential lo hacía automáticamente

   # Ahora
   from nova.nn import init
   for m in model.modules():
       if isinstance(m, nn.Linear):
           init.kaiming_normal_(m.weight)
           if m.bias is not None:
               init.zeros_(m.bias)
```

5. **Actualizar serialización**:

```python
   # Antes
   import pickle
   pickle.dump(model, f)

   # Ahora
   import nova
   nova.save(model.state_dict(), 'model.pth')
   # ...
   model.load_state_dict(nova.load('model.pth'))
```

## [3.0.0] - 2025-12-06

### Añadido

- Framework básico de deep learning con capas fully connected y convolucionales
- Optimizadores: SGD, Adam, AdamW, RMSprop
- Funciones de pérdida: MSE, MAE, CrossEntropy, BinaryCrossEntropy
- Métricas: accuracy, binary_accuracy, r2_score
- Capas: Linear, Conv1d, Conv2d, BatchNorm1d, BatchNorm2d, Dropout
- Activaciones: ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
- Pooling: MaxPool1d, MaxPool2d, GlobalAvgPool1d, GlobalAvgPool2d
- Contenedor Sequential con inicialización automática
- Sistema de logging
- Función `train()` para entrenamiento simplificado
- Loaders para MNIST y Fashion-MNIST
- Ejemplos de clasificación y regresión
- Tests unitarios (cobertura 95%)

### Notas

- Versión inicial funcional del framework
- README de 2000+ líneas explicando cada archivo (mala práctica)
- Sin sistema de autograd (backward manual)
- Sin tipado estático completo
- malcasting de tipos

## Comparación de Versiones

| Característica      | v3.0.0          | v4.0.3                 |
| ------------------- | --------------- | ---------------------- |
| **Autograd**        | ❌ Manual       | ✅ Automático dinámico |
| **API**             | Custom          | PyTorch-style          |
| **Tensors**         | Simple wrapper  | Clase completa con ops |
| **Operaciones**     | ~20             | 80+                    |
| **Módulos**         | Básicos         | Completos + Lazy       |
| **Schedulers**      | ❌              | ✅ 3 tipos             |
| **Métricas**        | 3 básicas       | 8 + averaging          |
| **Serialización**   | Pickle          | Safe + registro        |
| **Documentación**   | 1 README enorme | READMEs modulares      |
| **Cobertura tests** | 95%             | 87% (código 5x mayor)  |
| **Eficiencia**      | ~45%            | ~82% más eficiente     |
| **Benchmarks**      | ❌              | ✅ vs PyTorch          |
