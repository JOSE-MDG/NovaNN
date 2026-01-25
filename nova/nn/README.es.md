# Módulo `nn`

El directorio **`nn/`** contiene las **abstracciones de alto nivel para construcción de redes neuronales en NovaNN**.  
Este módulo proporciona capas, módulos, funciones de activación, pérdidas, inicialización de parámetros y utilidades para construir modelos de deep learning de forma modular y declarativa.

El diseño de `nn` sigue de cerca la filosofía de **PyTorch**, ofreciendo una API familiar y expresiva para definir arquitecturas complejas mediante composición de módulos simples.

## Ejemplo:

```python
import nova
import nova.nn as nn

# Definir un modelo simple
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = MLP()
x = nova.randn(32, 784)  # batch de 32 muestras
output = model(x)
print(output.shape)  # (32, 10)
```

## Estructura general

El módulo `nn/` está organizado en:

- **Archivos principales** en la raíz que definen la API pública (`Parameter`, `Buffer`, `functional`, `init`)
- **[`modules/`](#submódulo-modules)**: implementaciones de todas las capas y módulos (Linear, Conv, BatchNorm, etc.)
- **[`utils/`](#submódulo-utils)**: utilidades para gradient clipping y estandarización de parámetros

## Archivos principales

### `parameter.py`

Define las clases base para parámetros y buffers del modelo:

- **`Parameter`**: Tensor aprendible que se actualiza durante el entrenamiento. Tiene `requires_grad=True` por defecto.
- **`Buffer`**: Tensor no aprendible que forma parte del estado del modelo (ej: estadísticas corrientes en BatchNorm). No requiere gradientes.
- **`UninitializedParameter`** / **`UninitializedBuffer`**: Versiones lazy que se materializan automáticamente en el primer forward pass.
- **`UninitializedTensorMixin`**: Clase base común para todos los tensores no inicializados.
- **`is_lazy(param)`**: Función helper para detectar si un parámetro/buffer es lazy.

Los `Parameter` y `Buffer` son fundamentales para el sistema de módulos, ya que se registran automáticamente cuando se asignan como atributos de un `Module`.

## Ejemplo:

```python
import nova
from nova.nn import Parameter, Buffer, UninitializedParameter, is_lazy

# Crear parámetro aprendible
weight = Parameter(nova.randn(10, 5))
print(weight.requires_grad)  # True

# Crear buffer no aprendible
running_mean = Buffer(nova.zeros(10))
print(running_mean.requires_grad)  # False

# Parámetro lazy (se materializa después)
lazy_param = UninitializedParameter()
print(is_lazy(lazy_param))  # True
materialized = lazy_param.materialize((3, 3))
print(materialized.shape)  # (3, 3)
```

### `init.py`

Contiene **funciones de inicialización de pesos** para redes neuronales:

**Métodos principales:**

- **Xavier/Glorot**: `xavier_normal_()`, `xavier_uniform_()`
  - Diseñados para activaciones lineales, sigmoid y tanh
  - Mantienen la varianza de activaciones y gradientes estable
- **Kaiming/He**: `kaiming_normal_()`, `kaiming_uniform_()`
  - Optimizados para ReLU y variantes
  - Ajustan la varianza según el fan-in/fan-out
- **Básicos**: `uniform_()`, `normal_()`, `constant_()`, `zeros_()`, `ones_()`, `random_()`

## Ejemplo:

```python
import nova
from nova.nn import Parameter, init

# Inicialización Xavier para capas con tanh/sigmoid
weight = Parameter(nova.empty((64, 128)))
init.xavier_normal_(weight)

# Inicialización Kaiming para capas con ReLU
weight_relu = Parameter(nova.empty((128, 256)))
init.kaiming_normal_(weight_relu, nonlinearity='relu')

# Inicialización personalizada
bias = Parameter(nova.empty(64))
init.constant_(bias, 0.0)

# Obtener información de fan-in/fan-out
fan_in, fan_out = init.get_fans(weight, mode='both')
print(f"Fan-in: {fan_in}, Fan-out: {fan_out}")  # Fan-in: 128, Fan-out: 64
```

**Utilidades:**

- `calculate_gain(nonlinearity)`: Calcula el factor de ganancia recomendado para cada tipo de activación
- `get_fans(tensor, mode)`: Calcula fan-in y fan-out a partir de la forma del tensor

Todas las funciones operan **in-place** (sufijo `_`) y desactivan temporalmente el tracking de gradientes durante la inicialización.

### `functional.py`

Proporciona **versiones funcionales** de todas las operaciones de redes neuronales. Este módulo es análogo a `torch.nn.functional` y permite usar capas sin mantener estado.

**Categorías de funciones:**

#### Activaciones

- **Rectificadas**: `relu()`, `leaky_relu()`, `prelu()`
- **Suaves**: `sigmoid()`, `tanh()`, `gelu()`
- **Normalización**: `softmax()`, `log_softmax()`, `normalize()`

## Ejemplo:

```python
import nova.nn.functional as F

x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU
print(F.relu(x))  # tensor([0., 0., 0., 1., 2.])

# Softmax
logits = nova.tensor([[2.0, 1.0, 0.1], [1.0, 3.0, 0.5]])
probs = F.softmax(logits, dim=1)
print(probs.sum(dim=1))  # tensor([1., 1.]) (suma a 1 por fila)
```

#### Pérdidas (Loss Functions)

- **Regresión**: `mse_loss()`, `l1_loss()`, `smooth_l1_loss()`
- **Clasificación binaria**: `binary_cross_entropy()`, `binary_cross_entropy_with_logits()`
- **Clasificación multiclase**: `nll_loss()`, `cross_entropy()`
- **Otras**: `kl_div()` (divergencia KL para destilación y modelos generativos)

Todas las funciones de pérdida soportan:

- Reducción configurable (`'none'`, `'mean'`, `'sum'`, `'batchmean'`)
- Pesos por elemento o por clase
- Estabilidad numérica mediante implementaciones cuidadosas

### Ejemplo:

```python
import nova.nn.functional as F

# MSE Loss
predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
loss = F.mse_loss(predictions, targets)
print(loss)  # 0.375

# Cross Entropy
logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = nova.tensor([0, 1], dtype=nova.long)
loss = F.cross_entropy(logits, targets)
print(loss)  # scalar loss
```

#### Capas lineales y convolucionales

- **Linear**: `linear()` - transformación afín y = xW^T + b
- **Convoluciones**: `conv1d()`, `conv2d()`, `conv3d()`
  - Soporte para stride, padding, dilation
  - Múltiples modos de padding ('zeros', 'reflect', 'replicate', 'circular')
  - Implementación eficiente mediante im2col
- **Utilidades**: `flatten()` - aplana dimensiones para capas fully connected

### Ejemplo:

```python
import nova.nn.functional as F

# Conv2D
x = nova.randn(1, 3, 32, 32)  # (batch, channels, height, width)
weight = nova.randn(16, 3, 3, 3)  # (out_channels, in_channels, kh, kw)
output = F.conv2d(x, weight, kernel_size=3, padding=1)
print(output.shape)  # (1, 16, 32, 32)
```

#### Pooling

**Average Pooling:**

- `avg_pool1d()`, `avg_pool2d()`, `avg_pool3d()`
- `global_avg_pool1d()`, `global_avg_pool2d()`, `global_avg_pool3d()`

**Max Pooling:**

- `max_pool1d()`, `max_pool2d()`, `max_pool3d()`
- Soporte para dilation en ventanas de pooling

### Ejemplo:

```python
import nova.nn.functional as F

x = nova.randn(1, 64, 32, 32)

# Max Pooling
max_pooled = F.max_pool2d(x, kernel_size=2, stride=2)
print(max_pooled.shape)  # (1, 64, 16, 16)

# Global Average Pooling
global_pooled = F.global_avg_pool2d(x)
print(global_pooled.shape)  # (1, 64, 1, 1)
```

#### Normalización

- **`batch_norm()`**: Normalización por lotes
  - Calcula estadísticas por batch en training
  - Usa running statistics en eval
  - Actualiza running mean/var con momentum
- **`layer_norm()`**: Normalización por capas
  - Independiente del batch size
  - Común en Transformers

#### Regularización

- **`dropout()`**: Dropout estándar (apaga elementos aleatorios)
- **`dropout2d()`**: Dropout espacial (apaga canales completos en 2D)
- **`dropout3d()`**: Dropout espacial (apaga canales completos en 3D)

### Ejemplo:

```python
import nova.nn.functional as F

# Batch Norm
x = nova.randn(4, 3, 8, 8)
running_mean = nova.zeros(3)
running_var = nova.ones(3)
normalized = F.batch_norm(x, running_mean, running_var, training=True)
print(normalized.shape)  # (4, 3, 8, 8)

# Dropout
x = nova.ones((2, 4))
dropped = F.dropout(x, p=0.5, training=True)
print(dropped)  # algunos elementos en 0, otros escalados
```

Todas las funciones de dropout:

- Solo actúan en modo training
- Escalan valores restantes por 1/(1-p) para mantener la suma esperada

### `module.py`

Define la clase base **`Module`**, que es la abstracción fundamental para todos los componentes de redes neuronales en NovaNN.

**Características clave:**

**Sistema de registro automático:**

- Detecta y registra automáticamente `Parameter`, `Buffer` y sub-`Module` cuando se asignan como atributos
- Mantiene tres diccionarios internos: `_parameters`, `_buffers`, `_modules`
- Usa `__setattr__` para interceptar asignaciones

**Metaclase `ModuleMeta`:**

- Registra automáticamente todas las subclases de `Module` en el sistema de serialización
- Permite deserialización segura con `weights_only=True`
- No requiere decoradores `@registry_class` explícitos

**API de iteración:**

- `parameters(recurse=True)`: itera sobre parámetros aprendibles
- `buffers(recurse=True)`: itera sobre buffers no aprendibles
- `named_parameters()` / `named_buffers()`: versiones con nombres
- `named_modules()`: itera sobre toda la jerarquía del modelo

**Modos de entrenamiento:**

- `train(mode=True)`: activa modo entrenamiento (afecta Dropout, BatchNorm, etc.)
- `eval()`: activa modo evaluación
- `_training`: flag interno que se propaga recursivamente a submódulos

**Serialización:**

- `state_dict()`: exporta el estado completo (parámetros + buffers persistentes)
- `load_state_dict(state_dict)`: carga estado desde diccionario

**Representación:**

- `__repr__()`: genera representación legible de la jerarquía del modelo
- `extra_repr()`: método override para añadir información personalizada

**Método forward:**

- Debe ser implementado por todas las subclases
- Define la computación del forward pass
- Se invoca automáticamente al llamar `module(x)` gracias a `__call__()`

## Ejemplo:

```python
import nova
import nova.nn as nn

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.bn = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.global_avg_pool2d(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

model = CustomNet()

# Iterar sobre parámetros
for name, param in model.named_parameters():
    print(name, param.shape)

# Cambiar modo
model.train()  # modo entrenamiento
model.eval()   # modo evaluación

# Serialización
state = model.state_dict()
nova.save(state, 'model.pth')
```

## Submódulo `modules/`

Contiene todas las implementaciones concretas de capas y módulos de redes neuronales.

### `container.py`

**`Sequential`**: Contenedor que ejecuta módulos en secuencia.

Funcionalidades:

- Acepta módulos como argumentos posicionales o `OrderedDict`
- Indexación por entero o string
- Soporte para slicing
- Métodos `append()`, `extend()`, `insert()`, `pop()`
- Forward automático encadena todos los módulos

## Ejemplo:

```python
import nova.nn as nn

# Crear Sequential
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# Usar el modelo
x = nova.randn(32, 784)
output = model(x)
print(output.shape)  # (32, 10)

# Acceder a capas
print(model[0])  # Linear(784, 256)

# Añadir capas
model.append(nn.Softmax(dim=1))
```

### `activation.py`

Implementa capas de activación como módulos stateful:

- **`ReLU()`**: Rectified Linear Unit
- **`LeakyReLU(negative_slope)`**: ReLU con pendiente negativa configurable
- **`PReLU(num_parameters, init)`**: ReLU paramétrico con pendiente aprendible
- **`GELU()`**: Gaussian Error Linear Unit
- **`Sigmoid()`**: Función sigmoide
- **`Tanh()`**: Tangente hiperbólica
- **`Softmax(dim)`**: Normalización exponencial

Todas heredan de `Module` y delegan la computación a `functional`.

## Ejemplo:

```python
import nova
import nova.nn as nn

# Activaciones como módulos
relu = nn.ReLU()
leaky_relu = nn.LeakyReLU(negative_slope=0.2)
prelu = nn.PReLU(num_parameters=1, init=0.25)
gelu = nn.GELU()

x = nova.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
print(relu(x))  # tensor([0., 0., 0., 1., 2.])
print(leaky_relu(x))  # tensor([-0.4, -0.2, 0., 1., 2.])
print(prelu(x)) # tensor([-0.5, -0.25, 0.0, 1.0, 2.0], requires_grad=True)
print(gelu(x)) # tensor([-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977])
```

### `linear.py`

**`Linear(in_features, out_features, bias=True)`**: Capa fully connected (transformación afín).

**Variante lazy:**

- **`LazyLinear(out_features, bias=True)`**: Infiere `in_features` automáticamente en el primer forward

Características:

- Inicialización Kaiming uniform por defecto
- Parámetros: `weight` (out_features, in_features), `bias` opcional
- Soporte para inferencia automática de dimensiones de entrada

## Ejemplo:

```python
import nova
import nova.nn as nn

# Linear normal
linear = nn.Linear(10, 5, bias=True)
x = nova.randn(3, 10)
output = linear(x)
print(output.shape)  # (3, 5)

# LazyLinear (infiere in_features automáticamente)
lazy_linear = nn.LazyLinear(5)
x = nova.randn(3, 10)
output = lazy_linear(x)  # Materializa in_features=10
print(output.shape)  # (3, 5)
```

### `conv.py`

Implementa capas convolucionales 1D, 2D y 3D:

- **`Conv1d`**, **`Conv2d`**, **`Conv3d`**
- Parámetros: `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`, `dilation`, `bias`, `padding_mode`

**Variantes lazy:**

- **`LazyConv1d`**, **`LazyConv2d`**, **`LazyConv3d`**
- Clase base: `_LazyConvXdMixin`
- Infieren `in_channels` en el primer forward

Características:

- Inicialización Kaiming uniform
- Soporte para múltiples modos de padding
- Implementación eficiente mediante im2col/as_strided

## Ejemplo:

```python
import nova
import nova.nn as nn

# Conv2d normal
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
x = nova.randn(1, 3, 32, 32)
output = conv(x)
print(output.shape)  # (1, 64, 32, 32)

# LazyConv2d (infiere in_channels automáticamente)
lazy_conv = nn.LazyConv2d(out_channels=64, kernel_size=3, padding=1)
x = nova.randn(1, 3, 32, 32)
output = lazy_conv(x)  # Materializa in_channels=3
print(output.shape)  # (1, 64, 32, 32)
```

### `batchnorm.py`

Implementa normalización por lotes en 1D, 2D y 3D:

- **`BatchNorm1d`**, **`BatchNorm2d`**, **`BatchNorm3d`**
- Clase base: **`_BatchNorm`**

**Variantes lazy:**

- **`LazyBatchNorm1d`**, **`LazyBatchNorm2d`**, **`LazyBatchNorm3d`**
- Clase base: **`_LazyNormBase`**

Características:

- Parámetros aprendibles: `weight` (gamma), `bias` (beta)
- Buffers: `running_mean`, `running_var`, `num_batches_tracked`
- Comportamiento diferente en train/eval
- Momentum configurable para actualización de estadísticas

## Ejemplo:

```python
import nova
import nova.nn as nn

# BatchNorm2d
bn = nn.BatchNorm2d(num_features=64)
x = nova.randn(4, 64, 32, 32)

# Training mode
bn.train()
output_train = bn(x)  # Usa estadísticas del batch
print(output_train.shape)  # (4, 64, 32, 32)

# Eval mode
bn.eval()
output_eval = bn(x)  # Usa running statistics
print(output_eval.shape)  # (4, 64, 32, 32)

# LazyBatchNorm2d
lazy_bn = nn.LazyBatchNorm2d()
output = lazy_bn(x)  # Materializa num_features=64
```

### `layernorm.py`

**`LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`**: Normalización por capas.

Características:

- Normaliza sobre las últimas dimensiones especificadas
- Independiente del batch size
- Común en arquitecturas Transformer

## Ejemplo:

```python
import nova
import nova.nn as nn

# LayerNorm para secuencias (ej: Transformers)
ln = nn.LayerNorm(normalized_shape=(512,))
x = nova.randn(32, 128, 512)  # (batch, seq_len, features)
output = ln(x)
print(output.shape)  # (32, 128, 512)

# LayerNorm para imágenes
ln_2d = nn.LayerNorm(normalized_shape=(3, 32, 32))
x = nova.randn(4, 3, 32, 32)
output = ln_2d(x)
print(output.shape)  # (4, 3, 32, 32)
```

### `pooling.py`

Implementa operaciones de pooling:

**Average Pooling:**

- **`AvgPool1d`**, **`AvgPool2d`**, **`AvgPool3d`**
- **`GlobalAvgPool1d`**, **`GlobalAvgPool2d`**, **`GlobalAvgPool3d`**

**Max Pooling:**

- **`MaxPool1d`**, **`MaxPool2d`**, **`MaxPool3d`**

Características:

- Soporte para kernel_size, stride, padding, dilation
- GlobalAvgPool colapsa dimensiones espaciales completamente

## Ejemplo:

```python
import nova
import nova.nn as nn

# MaxPool2d
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
x = nova.randn(1, 64, 32, 32)
output = max_pool(x)
print(output.shape)  # (1, 64, 16, 16)

# GlobalAvgPool2d
global_pool = nn.GlobalAvgPool2d()
x = nova.randn(1, 64, 32, 32)
output = global_pool(x)
print(output.shape)  # (1, 64, 1, 1)
```

### `dropout.py`

Implementa regularización mediante dropout:

- **`Dropout(p=0.5)`**: Dropout estándar
- **`Dropout2d(p=0.5)`**: Dropout espacial para feature maps 2D
- **`Dropout3d(p=0.5)`**: Dropout espacial para feature maps 3D

Características:

- Solo activo en modo training
- Escala automática de valores restantes
- Dropout2d/3d apagan canales completos para preservar correlación espacial

## Ejemplo:

```python
import nova
import nova.nn as nn

# Dropout estándar
dropout = nn.Dropout(p=0.5)
x = nova.ones(4, 10)

dropout.train()
output_train = dropout(x)  # Algunos elementos en 0
print(output_train)

dropout.eval()
output_eval = dropout(x)  # Todos los elementos intactos
print(output_eval)

# Dropout2d (para CNNs)
dropout2d = nn.Dropout2d(p=0.3)
x = nova.randn(2, 64, 8, 8)
output = dropout2d(x)  # Apaga canales completos
```

### `flatten.py`

**`Flatten(start_dim=1, end_dim=-1)`**: Aplana un rango de dimensiones.

Uso común: preparar salida de capas convolucionales para capas fully connected.

```python
import nova
import nova.nn as nn

flatten = nn.Flatten(start_dim=1)
x = nova.randn(2, 3, 4, 4)
output = flatten(x)
print(output.shape)  # (2, 48) - aplana todas excepto batch
```

### `loss.py`

Implementa funciones de pérdida como módulos:

**Regresión:**

- **`MSELoss(reduction='mean')`**: Mean Squared Error
- **`L1Loss(reduction='mean')`**: Mean Absolute Error
- **`SmoothL1Loss(beta=1.0, reduction='mean')`**: Huber loss

**Clasificación:**

- **`BCELoss(weight=None, reduction='mean')`**: Binary Cross Entropy
- **`BCEWithLogitsLoss(weight=None, pos_weight=None, reduction='mean')`**: BCE con logits (numéricamente estable)
- **`NLLLoss(weight=None, reduction='mean')`**: Negative Log Likelihood
- **`CrossEntropyLoss(weight=None)`**: Cross Entropy (combina log_softmax + NLL)
- **`KLDivLoss(log_target=False, reduction='mean')`**: Divergencia Kullback-Leibler

Características:

- Todas heredan de `Module`
- Soportan reducción configurable
- Pesos por elemento o por clase
- Delegan cómputo a `functional`

## Ejemplo:

```python
import nova
import nova.nn as nn

# MSELoss
criterion = nn.MSELoss()
predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
targets = nova.tensor([3.0, -0.5, 2.0, 7.0])
loss = criterion(predictions, targets)
print(loss.item())  # 0.375

# CrossEntropyLoss
criterion = nn.CrossEntropyLoss()
logits = nova.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
targets = nova.tensor([0, 1], dtype=nova.long)
loss = criterion(logits, targets)
print(loss.item())  # scalar loss
```

### `lazy.py`

**`LazyModuleMixin`**: Clase base para todas las variantes lazy de módulos.

Características:

- Método abstracto `initialize_parameters()` que debe implementar cada subclase
- Maneja la materialización automática de parámetros no inicializados
- Se integra con `UninitializedParameter` y `UninitializedBuffer`
- Permite construcción de modelos sin conocer dimensiones de entrada a priori

## Submódulo `utils/`

Contiene utilidades auxiliares para el módulo `nn`.

### `clip_grad.py`

Funciones para gradient clipping (prevención de explosión de gradientes):

- **`clip_grad_norm_(parameters, max_norm, get_norm=False)`**: Clipea la norma global del gradiente
- **`clip_grad_value_(parameters, clip_value)`**: Clipea valores individuales del gradiente

Uso común en entrenamiento de RNNs y Transformers.

## Ejemplo:

```python
import nova.nn as nn
from nova.nn.utils import clip_grad_norm_, clip_grad_value_

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 5)
)

# Clipear norma global del gradiente
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Norma total: {total_norm}")

# Clipear valores individuales
clip_grad_value_(model.parameters(), clip_value=0.5)
```

### `tensor_utils.py`

Funciones helper para estandarización de parámetros de capas:

- **`_single(x)`**: Asegura que valor siempre sea un int
- **`_pair(x)`**: Convierte int a tupla de 2 elementos
- **`_triple(x)`**: Convierte int a tupla de 3 elementos
- **`add_padding(input, padding, padding_mode)`**: Añade padding simétrico a tensores 3D/4D/5D según su dimensionalidad
- **`calculate_out_size(H, W, kernel_size, padding, stride, dilation)`**: Calcula las dimensiones espaciales de salida después de aplicar convoluciones o pooling, soportando operaciones 1D, 2D y 3D con o sin dilation

Estas funciones facilitan el manejo de parámetros como `kernel_size`, `stride`, `padding` y `dilation` que pueden especificarse como int, tuplas o strings (como "valid"), estandarizándolos al formato requerido internamente por las capas

## Diseño y filosofía

El módulo `nn` de NovaNN está diseñado siguiendo estos principios:

- **Composición sobre herencia**: Los modelos complejos se construyen componiendo módulos simples
- **Separación de concerns**:
  - `Module` maneja el estado y la jerarquía
  - `functional` proporciona operaciones sin estado
  - `Parameter`/`Buffer` encapsulan datos aprendibles/no aprendibles
- **Lazy initialization**: Permite definir arquitecturas sin conocer todas las dimensiones a priori
- **Consistencia con PyTorch**: API familiar para facilitar la transición y el aprendizaje
- **Extensibilidad**: Fácil añadir nuevas capas heredando de `Module` y siguiendo el patrón establecido

## Integración con otros módulos

El módulo `nn` se integra estrechamente con:

- **[`autograd/`](../autograd/README.es.md)**: Todas las operaciones soportan diferenciación automática
- **[`optim/`](../optim/README.es.md)**: Los optimizadores operan sobre `model.parameters()`
- **[`serialization/`](../serialization/README.es.md)**: `state_dict()` y `load_state_dict()` permiten guardar/cargar modelos
- **[`_internal/`](../_internal/README.es.md)**: Sistema de binding para operaciones de bajo nivel

---

> Para más detalles sobre operaciones específicas, consulta el código fuente en `modules/` y `functional.py`.
