# 🧪 Tests - NovaNN

[![Tests](https://img.shields.io/badge/tests-passing-success?style=flat-square)]()
[![Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen?style=flat-square)]()
[![Pytest](https://img.shields.io/badge/framework-pytest-orange?style=flat-square)]()

Suite completa de tests para validar todos los componentes de NovaNN, desde operaciones de autograd hasta capas de redes neuronales y optimizadores.

## Estructura de Tests

Los tests están organizados en dos niveles:

- **`tests/`** (raíz): Tests de componentes de alto nivel (módulos `nn`, optimizadores, serialización, API pública)
- **`nova/autograd/tests/`**: Tests específicos del sistema de autograd (ver [README de autograd](../nova/autograd/README.es.md))

## Tests de Funciones (`tests/functional/`)

### `test_activation.py`

Tests exhaustivos de funciones de activación:

- **ReLU**: forward (no negativos), backward (gradient checking)
- **LeakyReLU**: forward con negative slope, backward
- **GELU**: forward, backward (tolerancias relajadas por complejidad numérica)
- **PReLU**: forward con parámetros aprendibles, backward
- **Tanh**: forward (rango [-1, 1]), backward
- **Sigmoid**: forward (rango [0, 1]), backward
- **Softmax**: forward (suma a 1), backward
- **LogSoftmax**: forward (exp suma a 1), backward

Todos los tests validan:

- Shapes correctos
- Rangos de valores esperados
- Gradientes analíticos vs numéricos con `grad_check_wrt_inputs`

### `test_loss.py`

Suite completa de funciones de pérdida con múltiples clases de tests:

**MSELoss (Mean Squared Error):**

- Cálculos básicos con valores conocidos
- Modos de reducción (`none`, `mean`, `sum`)
- Pesos por elemento
- Validación de gradientes
- Edge cases (tensors multidimensionales, pérdida cero)

**L1Loss:**

- Cálculos básicos
- Modos de reducción
- Validación de gradientes

**SmoothL1Loss (Huber):**

- Cálculos con diferentes betas
- Convergencia a L1 para errores grandes
- Validación de gradientes

**BCELoss (Binary Cross Entropy):**

- Cálculos básicos
- Predicciones perfectas
- Estabilidad numérica (sin NaN/Inf)
- Gradientes

**BCEWithLogitsLoss:**

- Equivalencia con sigmoid + BCE
- Positive class weighting (`pos_weight`)
- Estabilidad con logits extremos

**NLLLoss (Negative Log Likelihood):**

- Cálculos básicos y por batch
- Class weights
- Modos de reducción

**CrossEntropyLoss:**

- Equivalencia con log_softmax + NLL
- Class weights
- Gradientes
- Predicciones perfectas

**KLDivLoss (Kullback-Leibler Divergence):**

- Cálculos básicos (≥ 0)
- Distribuciones idénticas (≈ 0)
- Modo `log_target`
- Modos de reducción incluyendo `batchmean`

**Tests comunes:**

- Función helper `_reduce` con todos los modos
- Edge cases (tensors vacíos, elemento único, batches grandes)
- Tests parametrizados para todos los modos de reducción

## Tests de Módulos NN (`tests/nn/`)

### `test_batchnorm.py`

Tests completos de BatchNormalization en todas sus variantes:

**Forward pass:**

- BatchNorm1d con input 2D (N, C) y 3D (N, C, L)
- BatchNorm2d con input 4D (N, C, H, W)
- BatchNorm3d con input 5D (N, C, D, H, W)
- Validación de normalización (mean ≈ 0, std ≈ 1)

**Backward pass:**

- Gradientes con cross entropy loss
- Gradient checking con diferencias finitas
- Tests para 1D, 2D y 3D

**Running statistics:**

- Actualización durante training con momentum exponencial
- No actualización durante eval
- Uso de running stats en eval mode

**Parámetros afines:**

- Aplicación de weight y bias
- Comportamiento sin parámetros afines (`affine=False`)

**Validación de dimensiones:**

- Rechazo de dimensiones incorrectas
- Mensajes de error apropiados

**Edge cases:**

- `track_running_stats=False`
- `momentum=None` (cumulative average)
- `reset_parameters()`

### `test_container.py`

Tests exhaustivos de `Sequential`:

**Construcción:**

- Con lista de módulos
- Con `OrderedDict`
- Sequential vacío

**Forward pass:**

- Encadenamiento correcto
- Con capas Linear

**Indexing y slicing:**

- `__getitem__` por índice y slice
- `__setitem__`
- `__delitem__` (elemento y slice)
- Index out of range

**Métodos:**

- `append`, `insert`, `extend`, `pop`
- Índices negativos
- Pop con slices

**Iteración:**

- `__iter__`, `__len__`

**Representación:**

- `__repr__` simple y vacío
- Módulos repetidos

**Aritmética:**

- Suma (`+`, `+=`)
- Multiplicación (`*`, `*=`, `__rmul__`)
- Validación de tipos
- Multiplicación no positiva

### `test_conv.py`

Tests de capas convolucionales:

**Conv1d, Conv2d, Conv3d:**

- Shapes de salida correctos
- Gradientes (analytical vs numerical)
- Sin bias (`bias=False`)
- Stride y padding
- Kernels asimétricos (Conv2d)
- Preservación de dimensiones temporales (Conv3d)

### `test_dropout.py`

Tests de regularización con dropout:

**Dropout, Dropout2d, Dropout3d:**

- Preservación de shape
- Desactivación en eval mode
- Aplicación en train mode (algunos valores a 0)
- Probabilidad cero (sin dropout)
- Dropout de canales completos (2d y 3d)

### `test_flatten.py`

Tests de Flatten:

- Flatten por defecto (todas excepto batch)
- Gradientes
- Rango de dimensiones custom
- Flatten incluyendo batch
- Dimensión única

### `test_layernorm.py`

Tests de Layer Normalization:

- Shape correcto
- Propiedades de normalización (mean ≈ 0, var ≈ 1)
- Gradientes
- Sin parámetros afines
- Normalización multidimensional

### `test_lazy_variants.py`

Tests de variantes lazy (inferencia automática de dimensiones):

**LazyBatchNorm (1d, 2d, 3d):**

- Inferencia de `num_features`
- Parámetros no inicializados antes del primer forward
- Forwards subsecuentes

**LazyConv (1d, 2d, 3d):**

- Inferencia de `in_channels`
- Forwards subsecuentes

**LazyLinear:**

- Inferencia de `in_features`
- Input multidimensional

### `test_linear.py`

Tests de capa Linear:

- Forward shape
- Gradientes
- Sin bias
- Input multidimensional

### `test_pooling.py`

Tests de operaciones de pooling:

**MaxPool (1d, 2d, 3d):**

- Shapes
- Gradientes
- Padding
- Dilation
- Kernels no cuadrados
- Preservación de dimensiones

**AvgPool (1d, 2d, 3d):**

- Shapes
- Gradientes
- Padding

**GlobalAvgPool (1d, 2d, 3d):**

- Reducción a dimensiones unitarias
- Gradientes

## Tests de Optimizadores (`tests/optim/`)

### `test_sgd.py`

Tests de Stochastic Gradient Descent:

- Paso básico (descenso en dirección del gradiente)
- Acumulación de momentum
- Weight decay (L2)
- Convergencia en función cuadrática
- Múltiples parámetros
- Manejo de gradientes None

### `test_adam.py`

Tests de Adam optimizer:

- Paso básico
- Corrección de bias en pasos tempranos
- Learning rate adaptativo
- Convergencia en cuadrática
- Weight decay acoplado

### `test_adamw.py`

Tests de AdamW:

- Paso básico
- Weight decay desacoplado
- Diferencia vs Adam con weight decay
- Convergencia
- Corrección de bias

### `test_rmsprop.py`

Tests de RMSprop:

- Paso básico
- Adaptación a escala del gradiente
- Modo centrado
- Convergencia
- Momentum

### `test_common.py`

Tests comunes a todos los optimizadores:

- Parameter groups con diferentes learning rates
- Persistencia de estado entre steps
- Utilidad `zero_grad()`

### `test_schedulers.py`

Tests exhaustivos de schedulers de learning rate:

**StepLR:**

- Decaimiento en intervalos
- Convergencia
- Actualización de `last_epoch`
- Diferentes gammas
- State dict (save/load)

**CosineAnnealingLR:**

- Progresión coseno
- Convergencia a `eta_min`
- State dict

**OneCycleLR:**

- Progresión del ciclo
- Cycle momentum (inverso al LR)
- Fase de warm-up
- Fase de cool-down
- Compatibilidad con Adam (betas)

**Tests integrados:**

- Todos los schedulers con todos los optimizadores
- Persistencia de estado entre schedulers

## Tests de Serialización (`tests/serialization/`)

### `test_save.py`

Tests de la función `save`:

**Guardado básico:**

- Guardado a file path (string y Path)
- Guardado a buffer (BytesIO)
- Creación automática de directorios padres

**Tipos de objetos:**

- Módulos (Sequential, Linear)
- Tensors
- State dicts
- Diccionarios regulares
- Listas de tensors

**Protocolos pickle:**

- Tests con diferentes protocolos (0-4, HIGHEST_PROTOCOL)

**Manejo de errores:**

- Error al guardar None
- Error con tipo de archivo inválido
- Error en directorio read-only
- Error con objetos no serializables (lambdas)

### `test_load.py`

Tests de la función `load`:

**Carga básica:**

- Carga desde file path
- Carga desde buffer

**Seguridad:**

- Error al cargar clases no registradas con `weights_only=True`
- Carga exitosa de clases no registradas con `weights_only=False`
- Carga exitosa de clases registradas

**Manejo de errores:**

- Error con archivo no existente
- Error con archivo corrupto
- Error con archivo vacío

**Roundtrip:**

- Save/load de modelos
- Save/load de state dicts

## Tests de API Pública (`tests/`)

### `test_api.py`

Suite masiva que valida toda la API pública de NovaNN:

**Creación de tensors:**

- `tensor`, `zeros`, `ones`, `empty`, `full`, `eye`
- `arange`, `linspace`
- `zeros_like`, `ones_like`, `full_like`

**Funciones aleatorias:**

- `rand`, `randn`, `randint`, `randperm`
- `uniform`, `normal`
- `manual_seed` (reproducibilidad)

**Funciones matemáticas:**

- `abs`, `sqrt`, `exp`, `log`, `pow`
- `floor`, `ceil`, `sign`, `clamp`

**Funciones trigonométricas:**

- `sin`, `cos`, `tan`, `tanh`
- `arcsin`, `arccos`, `arctan`, `sec`

**Reducción:**

- `sum`, `mean`, `var`, `std`
- `max`, `min`, `maximum`, `minimum`
- `argmax`, `argmin`, `argsort`

**Álgebra lineal:**

- `dot`, `det`, `inv`, `trace`, `norm`

**Manipulación de forma:**

- `reshape`, `permute`, `flatten`
- `unsqueeze`, `split`, `tile`, `repeat_interleave`, `pad`

**Concatenación:**

- `cat`, `stack`

**Comparación y lógica:**

- `allclose`, `all`, `any`, `where`
- `isnan`, `isinf`, `argwhere`, `unique`

**Utilidades:**

- `one_hot`, `as_strided`

**Context managers:**

- `no_grad()`, `enable_grad()`, `is_grad_enabled()`

**Dtypes:**

- Disponibilidad de todos los dtypes
- Uso correcto en creación de tensors

**Metadatos:**

- `__version__` existe

### `test_binding_system.py`

Tests del sistema de binding dinámico:

**YAML loading:**

- Carga exitosa del archivo
- Estructura correcta
- Definición de operaciones comunes

**Generadores de funciones:**

- `make_forward_func`: operaciones binarias y unarias
- `make_reverse_func`: operaciones reverse (`__radd__`, etc.)
- `make_method`: métodos regulares (`.add()`, etc.)
- `make_inplace_func`: operaciones in-place con validaciones

**Bootstrapping:**

- Métodos correctamente enlazados a Tensor
- Dunder methods funcionan (`+`, `-`, `*`)
- Reverse methods funcionan (`5 + tensor`)
- Métodos regulares funcionan (`.add()`, `.mul()`)
- Métodos in-place funcionan (`.add_()`, `.mul_()`)
- Operaciones unarias (`-tensor`, `abs(tensor)`)
- Raw args (indexing mantiene tipos)

**Edge cases:**

- Métodos no enlazados dos veces
- Operaciones encadenadas
- Mezcla de scalars y tensors

### `test_conversion.py`

Tests de conversión de tipos con `ensure_tensor`:

**Tensor passthrough:**

- Sin cambios cuando no es necesario
- Copia cuando cambia dtype
- Copia cuando cambia requires_grad

**Conversión desde numpy:**

- Arrays básicos
- Con dtype especificado
- Con requires_grad

**Conversión desde Python:**

- int, float, bool
- Listas y listas anidadas

**Edge cases:**

- Arrays 0-dimensionales
- Arrays vacíos
- Dtypes complejos (manejo de error)

### `test_creation_and_casting.py`

Tests rápidos de creación y casting de dtypes:

- Preservación de dtypes en operaciones
- Math, trigonometría, reducción
- Indexing, álgebra lineal, concatenación

### `test_dataset.py`

Tests de la clase base `Dataset`:

- Longitud correcta
- Indexing simple, slice, lista, tensor
- Métodos abstractos lanzan NotImplementedError

### `test_loader.py`

Tests de `DataLoader`:

- Iteración básica sobre batches
- Último batch con tamaño correcto
- Shuffle produce orden diferente
- Sin shuffle produce mismo orden
- Longitud del loader
- Dataset vacío
- Múltiples epochs
- Integración con training loop

### `test_mnist_loader.py`

Tests exhaustivos del cargador de MNIST:

**TestMnistDataClass:**

- Inicialización de `MnistData` con tensors
- Método `__len__` retorna tamaño correcto
- Método `__getitem__` con índice simple y slicing

**TestLoadMnistData:**

- Carga básica como tensors (verificación de tipos, shapes, dtypes)
- Formato 4D con `tensor4d=True` (N, 1, 28, 28)
- Normalización (mean ≈ 0, std ≈ 1)
- Sin normalización (valores en rango [0, 255])
- Labels en rango válido [0, 9]
- Diferentes dtypes (float16, float32, float64)
- Salida como numpy arrays (`as_tensor=False`)
- Consistencia entre splits (train/test/val)
- Sin data leakage entre splits
- `requires_grad=False` por defecto
- Iteración por batches
- Normalización con numpy arrays
- Distribución de labels (todas las 10 clases presentes)
- Slicing del dataset (rangos, índices negativos)
- Eficiencia de memoria (sin copias excesivas)

**TestEdgeCases:**

- Manejo de paths inválidos
- Acceso a muestras individuales (1D para features, scalar para label)
- Acceso a muestras 4D individuales (shape (1, 28, 28))

**TestMnistMemoryUsage:**

- Uso de memoria en carga básica (<500MB)
- Overhead de tensors 4D vs 2D (ratio 0.8-1.5x)
- Impacto de normalización en memoria
- Tracking con decorator `@measure_memory`
- Comparación entre dtypes (float64 ≈ 2x float32)
- Memoria al acceder batches (grandes vs pequeños)
- Limpieza de memoria después de loading
- Numpy vs tensors (ratio <3.0x)

### `test_fashion_loader.py`

Tests completos del cargador de Fashion-MNIST:

**TestFashionDataClass:**

- Inicialización de `FashionData`
- Métodos `__len__` y `__getitem__`

**TestLoadFashionMnistData:**

- Carga básica como tensors con validación de tipos
- Formato 4D (N, 1, 28, 28)
- Normalización estadística (mean ≈ 0, std ≈ 1)
- Sin normalización (rango [0, 255])
- Labels válidos [0, 9] (10 clases de ropa)
- Diferentes dtypes (float16/32/64)
- Salida numpy
- Formato 4D con normalización combinados
- Consistencia dimensional entre splits
- Sin data leakage
- `requires_grad=False` por defecto
- Iteración por batches
- Normalización con numpy
- Distribución de todas las 10 clases
- Slicing avanzado
- Eficiencia de memoria
- Diferencia vs MNIST regular (contenido distinto)
- Tamaños relativos de splits (train > test, val)

**TestEdgeCases:**

- Paths inválidos (raises Exception)
- Acceso a muestra individual (1D features, scalar label)
- Muestra 4D individual (1, 28, 28)
- Slices vacíos (shape[0] == 0)

**TestFashionMemoryUsage:**

- Memoria en carga básica (<850MB)
- Overhead 4D vs 2D (ratio 0.5-1.5x)
- Impacto de normalización
- Decorator `@measure_memory`
- Comparación dtypes (float64 ≈ 2x float32)
- Acceso a batches (grandes usan más memoria)
- Limpieza post-loading
- Numpy vs tensors (ratio <3.0x)
- Similitud de memoria con MNIST (ratio 0.5-3.5x)
- Overhead de cargar 3 splits (<1000MB)
- Múltiples cargas sin memory leaks (varianza <30%)

### `test_hooks.py`

Tests del sistema de hooks:

**HooksHandle:**

- Creación y remoción
- Remociones múltiples (seguras)

**Tensor hooks:**

- `register_hook` en backward
- Remoción de hooks
- Múltiples hooks en mismo tensor

**Optimizer hooks:**

- Pre-step hooks
- Post-step hooks
- Orden de ejecución
- Remoción de hooks

### `test_clip_grad.py`

Tests de utilidades de gradient clipping:

**TestClipping:**

- **clip*grad_norm***: Normaliza gradientes a norma máxima
  - Entrenamiento de 20 epochs con modelo Sequential
  - Verificación de que todos los gradientes están bajo `max_norm=1.0`
  - Integración con optimizer SGD
- **clip*grad_value***: Recorta gradientes por valor absoluto
  - Clipping a threshold=0.5
  - Validación de que todos los gradientes están en [-threshold, +threshold]
  - Integración con training loop completo

Ambos tests verifican:

- Clipping correcto en todos los parámetros del modelo
- Compatibilidad con optimizadores
- No rompe el flujo backward-optimizer

### `test_metrics.py`

Tests de métricas de evaluación:

**Regresión:**

- **MSE/RMSE**: básico, con error, reset, múltiples batches
- **MAE**: básico, con error, robustez a outliers
- **R²Score**: fit perfecto (=1), baseline (≈0), buen fit

**Clasificación:**

- **Accuracy**: perfecta, parcial, cero
- **Precision/Recall/F1**: perfectos, con errores
- **ConfusionMatrix**: binaria, multiclass
- **ROCAUC**: perfecto (=1), random (≈0.5), buena separación

### `test_registry.py`

Tests del sistema de registro:

**`registry_class`:**

- Registro de clases simples
- Retorna clase original
- Registro idempotente
- `get_registered_classes` funciona
- Clase no registrada retorna None

### `test_memory_utils.py`

Tests del sistema de monitoreo de memoria:

**TestMemoryTracker:**

- Context manager básico
- Propiedades de memoria (peak_mb, current_mb, peak_kb, current_kb)

**TestQuickMemoryCheck:**

- Profiling de funciones simples
- Soporte de kwargs
- Retorno de resultados y estadísticas

**TestCompareMemory:**

- Comparación entre dos funciones
- Ratio de uso de memoria
- Validación de que función grande usa más memoria

**TestMemoryTrackerAdvanced:**

- `get_top_stats(n)` retorna top allocations
- Modo verbose imprime estadísticas

**TestMemoryContextBehavior:**

- Usos secuenciales del tracker
- Consistencia entre múltiples runs

### `test_timing_decorators.py`

Tests de herramientas de benchmarking y timing:

**TestBenchmark:**

- Benchmarking básico de funciones
- Soporte de kwargs
- Efecto de warmup iterations (no afectan timing)
- Consistencia de resultados (std bajo para funciones estables)
- Retorna: resultado, mean_time, std_time

**TestChronometer:**

- Decorator básico `@chronometer`
- Múltiples iteraciones con warmup
- Flag `return_time=True` retorna (result, elapsed)
- Modo `verbose=False` suprime output
- Validación de warmup iterations
- Combinación de return_time + n_iters retorna tiempo promedio

**`registry_op`:**

- Registro de Functions
- Retorna clase original
- Error con clases no-Function
- Registro idempotente
- Operaciones accesibles

**Integración:**

- Ambos decoradores juntos

## Estrategia de Testing

### Gradient Checking

Método principal de validación: **finite difference gradient checking**

```python
analytic, numeric = grad_check_wrt_inputs(operation, input, eps=1e-4)
assert nova.allclose(analytic[0], numeric[0], rtol=1e-2, atol=1e-3)
```

Compara:

- **Gradientes analíticos**: computados por `backward()`
- **Gradientes numéricos**: aproximados por diferencias finitas

### Tolerancias

Adaptadas según estabilidad numérica:

- Operaciones estables: `rtol=1e-3, atol=5e-3`
- Operaciones complejas: `rtol=1e-2, atol=5e-2`
- Operaciones muy sensibles: `rtol=0.1, atol=0.1`

### Fixtures y Parametrización

```python
@pytest.mark.parametrize("optimizer", [SGD, Adam, AdamW, RMSprop])
def test_with_all_optimizers(optimizer):
    # Test que se ejecuta con cada optimizador
```

### Reproducibilidad

Seed fijo en todos los tests:

```python
nova.manual_seed(8)  # o nova.manual_seed(42)
```

## Ejecución de Tests

```bash
# Todos los tests
poetry run pytest

# Tests verbosos
poetry run pytest tests/ -v

# Tests con cobertura
poetry run pytest --cov

# Test con reporte html
poetry run pytest --cov --cov-report=html
```

## Cobertura de Tests

**Coverage actual: 87%**

Los tests cubren:

- ✅ Forward pass (shapes, valores, propiedades)
- ✅ Backward pass (gradientes analíticos vs numéricos)
- ✅ Edge cases (dimensiones incorrectas, valores extremos)
- ✅ Configuraciones especiales (sin bias, lazy initialization)
- ✅ Modos de operación (train vs eval)
- ✅ Persistencia de estado (state_dict, load_state_dict)
- ✅ Compatibilidad entre componentes
- ✅ API pública completa
- ✅ Sistema de binding dinámico
- ✅ Serialización segura
- ✅ Métricas de evaluación
- ✅ Data loading

---

> Para tests específicos de autograd (operaciones, engine, Function), consulta [`nova/autograd/tests/`](../nova/autograd/tests/) y el [README de autograd](../nova/autograd/README.es.md).
