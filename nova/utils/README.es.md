# Módulo `utils`

El directorio **`utils/`** proporciona **utilidades generales y herramientas auxiliares** que dan soporte a todo el framework NovaNN.

Este módulo contiene funcionalidades transversales que no pertenecen a ninguna categoría específica pero que son esenciales para el funcionamiento, debugging, logging, manejo de datos y extensibilidad del framework.

## Estructura general

El módulo está organizado en:

- **Archivos principales** con utilidades de propósito general
- **[`decorators/`](#submódulo-decorators)**: Decoradores para registro, timing y otras funcionalidades
- **[`datasets/`](#submódulo-datasets)**: Loaders y utilidades para datasets comunes (MNIST, Fashion-MNIST)
- **[`data/`](#submódulo-data)**: Clases base y utilidades para manejo de datos (Dataset, DataLoader, preprocesamiento)

## Archivos principales

### `logger.py`

Implementa un **sistema de logging** singleton para NovaNN con soporte multi-nivel y múltiples outputs.

**Características:**

- **Patrón Singleton**: Una única instancia de logger en toda la aplicación
- **Multi-output**: Log a consola y archivo simultáneamente
- **Niveles configurables**: DEBUG, INFO, WARNING, ERROR
- **Formato personalizable**: Timestamps, niveles, nombres de función
- **Thread-safe**: Seguro para uso concurrente

**Clase `LoggerLevel`:**

Enum con los niveles de logging disponibles:

```python
class LoggerLevel(Enum):
    DEBUG = logging.DEBUG      # Información detallada de debugging
    INFO = logging.INFO        # Información general
    WARNING = logging.WARNING  # Advertencias
    ERROR = logging.ERROR      # Errores
```

**Clase `Logger`:**

Logger singleton con métodos para cada nivel:

**Métodos principales:**

- `info(msg, **kwargs)`: Log nivel INFO
- `debug(msg, **kwargs)`: Log nivel DEBUG
- `warning(msg, **kwargs)`: Log nivel WARNING
- `error(msg, **kwargs)`: Log nivel ERROR con traceback automático
- `set_level(level)`: Cambiar nivel dinámicamente

**Ejemplos de uso:**

```python
from nova.utils.logger import logger, LoggerLevel

# Logging básico
logger.info("Model training started")
logger.debug("Batch size: 32, Learning rate: 0.001")
logger.warning("Gradient norm is high")
logger.error("Failed to load checkpoint")

# Con kwargs adicionales
logger.info("Epoch completed", epoch=10, loss=0.123, acc=0.95)
# Output: ... | INFO | Epoch completed | epoch: 10, loss: 0.123, acc: 0.95

# Cambiar nivel dinámicamente
logger.set_level(LoggerLevel.WARNING)  # Solo muestra WARNING y ERROR
```

**Función `enable_file_logging()`:**

Función global para habilitar el logging a archivo después de la inicialización del logger:

**Parámetros:**

- `path` (Optional[Path | str]): Ruta donde se guardará el archivo de log. Si es None, usa `~/.novann/logs/nova.log`
- `level` (LoggerLevel): Nivel de logging para el file handler (por defecto: DEBUG)
- `replace_existing` (bool): Si es True, remueve los file handlers existentes antes de añadir uno nuevo (por defecto: True)

**Retorna:**

- `logging.Logger`: La instancia del logger configurada

**Excepciones:**

- `PermissionError`: Si el directorio de logs no tiene permisos de escritura
- `OSError`: Si falla la creación del directorio

**Características:**

- **Validación de directorio**: Verifica que el directorio de logs tenga permisos de escritura antes de crear el handler
- **Prevención de duplicados**: Puede reemplazar file handlers existentes para evitar logs duplicados
- **Auto-creación**: Crea directorios padres si no existen
- **Manejo de errores**: Excepciones claras para errores de permisos e IO

**Ejemplos de uso:**

```python
from nova.utils.logger import enable_file_logging, LoggerLevel

# Habilitar con ruta por defecto (~/.novann/logs/nova.log)
enable_file_logging()

# Habilitar con ruta personalizada
enable_file_logging(path="logs/training.log", level=LoggerLevel.INFO)

# Añadir file handler adicional sin reemplazar los existentes
enable_file_logging(path="logs/debug.log", replace_existing=False)

# Lanzará PermissionError si el directorio no es escribible
try:
    enable_file_logging(path="/root/cannot_write.log")
except PermissionError as e:
    print(f"No se puede habilitar file logging: {e}")
```

**Función `is_file_logging_enabled()`:**

Verifica si el file logging está actualmente activo:

**Retorna:**

- `bool`: True si hay algún FileHandler adjunto al logger

**Ejemplos de uso:**

```python
from nova.utils.logger import enable_file_logging, is_file_logging_enabled

# Verificar antes de habilitar
if not is_file_logging_enabled():
    enable_file_logging()

# Verificar después de habilitar
enable_file_logging()
assert is_file_logging_enabled() == True
```

**Función `get_log_file_path()`:**

Obtiene la ruta del archivo de log actual si el file logging está habilitado:

**Retorna:**

- `Optional[Path]`: Ruta al archivo de log, o None si no existe ningún file handler

**Ejemplos de uso:**

```python
from nova.utils.logger import enable_file_logging, get_log_file_path

# Obtener ruta del archivo de log actual
enable_file_logging("logs/app.log")
log_path = get_log_file_path()
print(f"Logging a: {log_path}")  # Logging a: logs/app.log

# Retorna None si no hay file handler
from nova.utils.logger import logger
# Solo console logging por defecto
assert get_log_file_path() is None
```

**Cuándo usar:**

- Tracking de progreso de entrenamiento
- Debugging de operaciones del framework
- Registro de errores y advertencias
- Auditoría de operaciones críticas
- Cuando necesitas añadir file logging después de la inicialización del logger
- Cuando necesitas verificar la configuración de logging programáticamente

### `memory.py`

Proporciona **`MemoryTracker`** y utilidades para profiling de uso de memoria durante la ejecución de código.

**Características:**

- **Context manager** para tracking automático de memoria
- **Baseline adjustment**: Resta memoria base para mediciones precisas
- **Peak y current memory**: Tracking de uso máximo y actual
- **Snapshots internos**: Permite análisis de top allocations
- **Múltiples unidades**: Propiedades para MB y KB
- **Garbage collection**: Limpieza automática antes de medir

**Clase `MemoryTracker`:**

Context manager que rastrea el uso de memoria usando `tracemalloc`.

**Atributos:**

- `verbose`: Si imprime estadísticas automáticamente al salir
- `baseline`: Memoria base en bytes (pre-tracking)
- `peak`: Memoria pico en bytes (ajustada por baseline)
- `current`: Memoria actual en bytes (ajustada por baseline)

**Métodos:**

- `__enter__()`: Inicia tracking con GC y baseline
- `__exit__()`: Detiene tracking, calcula stats, opcional verbose print
- `get_top_stats(limit=10)`: Retorna top N allocations
- Propiedades: `peak_mb`, `current_mb`, `peak_kb`, `current_kb`

**Ejemplos:**

```python
from nova.utils.memory import MemoryTracker

# Uso básico
with MemoryTracker() as mem:
    data = [i for i in range(1000000)]
print(f"Peak: {mem.peak_mb:.2f} MB")

# Verbose mode (auto-print)
with MemoryTracker(verbose=True) as mem:
    model = create_large_model()
# ==================================================
# Memory Usage Statistics
# ==================================================
# Peak memory:         125.43 MB
# Current memory:       98.21 MB
# ==================================================

# Analizar top allocations
with MemoryTracker() as mem:
    data = process_large_dataset()
top_stats = mem.get_top_stats(5)
for stat in top_stats:
    print(stat)
```

**Funciones auxiliares:**

#### `quick_memory_check(func, *args, **kwargs)`

Ejecuta una función mientras trackea memoria, retorna stats + resultado.

**Retorna:**

Dict con claves: `peak_mb`, `current_mb`, `peak_kb`, `current_kb`, `result`

**Ejemplos:**

```python
from nova.utils.memory import quick_memory_check

def create_list(n):
    return [i for i in range(n)]

stats = quick_memory_check(create_list, 1000000)
print(f"Peak: {stats['peak_mb']:.2f} MB")
print(f"Result length: {len(stats['result'])}")
```

#### `compare_memory(nova_func, torch_func, *args, verbose=True, **kwargs)`

Compara uso de memoria entre dos metodos.

**Retorna:**

Tupla `(nova_peak_mb, torch_peak_mb, ratio)`

**Ejemplos:**

```python
from nova.utils.memory import compare_memory

input_peak, torch_peak, ratio = compare_memory(
    input_forward,
    other_forward,
    x_nova, x_torch,
    verbose=True
)
# ==================================================
# Memory Comparison: NovaNN vs PyTorch
# ==================================================
# input_forward peak:       125.43 MB
# other_forward peak:       98.21 MB
# Ratio:              1.28x
# ==================================================
```

**Cuándo usar:**

- Profiling de memory footprint en operaciones costosas
- Debugging de memory leaks
- Benchmarking vs PyTorch
- Optimización de uso de memoria

### `memory_usage.py`

Proporciona **`@measure_memory`**, decorador para medir uso de memoria de funciones.

**Decorador `@measure_memory`:**

**Características:**

- **Dual-mode**: Con o sin parámetros
- **Verbose optional**: Auto-print de estadísticas
- **Return memory optional**: Retorna (result, (peak_mb, current_mb))
- **Wraps preserva metadata**: Usa `@wraps` internamente

**Parámetros:**

- `func`: Función a decorar (automático)
- `verbose`: Si imprime stats (default: False)
- `return_memory`: Si retorna tuple con stats (default: False)

**Ejemplos:**

```python
from nova.utils.decorators import measure_memory

# Uso básico (sin parámetros)
@measure_memory
def my_function():
    data = [i**2 for i in range(1000000)]
    return data

result = my_function()  # Solo retorna result

# Con verbose
@measure_memory(verbose=True)
def train_model():
    model = create_model()
    train(model)
    return model

# Con return_memory
@measure_memory(return_memory=True, verbose=False)
def compute():
    return heavy_computation()

result, (peak_mb, current_mb) = compute()
print(f"Used {peak_mb:.2f} MB peak")
```

**Cuándo usar:**

- Decorar funciones de entrenamiento para tracking
- Debugging de funciones sospechosas de memory leaks
- Profiling automático sin modificar código interno

### `hooks.py`

Define **`HooksHandle`**, un manejador para registrar y remover hooks de forma segura.

**Características:**

- **Gestión simplificada**: Registro y remoción de hooks
- **Prevención de duplicados**: Flag interno `_removed` evita remociones múltiples
- **Integración con Tensor y Optimizer**: Usado internamente por backward hooks y step hooks

**Clase `HooksHandle`:**

**Atributos:**

- `hooks_list`: Lista donde está registrado el hook
- `hooks_func`: Función del hook
- `_removed`: Flag de si ya fue removido

**Métodos:**

- `remove()`: Remueve el hook de la lista

**Ejemplos de uso:**

```python
import nova

# Uso con Tensor backward hooks
x = nova.tensor([1.0, 2.0], requires_grad=True)

def my_hook(grad):
    print(f"Gradient: {grad}")
    return grad * 2

# Registrar hook
handle = x.register_hook(my_hook)

# Usar el tensor normalmente
y = (x ** 2).sum()
y.backward()  # El hook se ejecuta aquí

# Remover hook cuando ya no se necesite
handle.remove()

# Ahora el hook no se ejecuta más
y.backward()
```

**Cuándo usar:**

- Al implementar custom backward hooks
- Para debugging temporal de gradientes
- En sistemas que necesitan hooks dinámicos

### `to_tensor.py`

Implementa **`ensure_tensor()`**, una función utilitaria para conversión robusta a Tensors.

**Características:**

- **Conversión automática**: Convierte arrays, escalares, listas a Tensors
- **Preservación condicional**: Si ya es Tensor y no hay cambios, retorna el original
- **Inferencia de dtype**: Infiere tipos apropiados según el input
- **Manejo de errores**: Logging detallado de excepciones

**Firma:**

```python
def ensure_tensor(
    obj: Any,
    dtype: Optional[Dtype] = None,
    requires_grad: Optional[bool] = None
) -> Tensor
```

**Casos de uso:**

**Caso 1: Ya es Tensor**

```python
t = nova.tensor([1.0, 2.0])
result = ensure_tensor(t)  # Retorna el mismo objeto
assert result is t
```

**Caso 2: Array de NumPy**

```python
arr = np.array([1.0, 2.0, 3.0])
t = ensure_tensor(arr, dtype=nova.float32, requires_grad=True)
# Convierte a Tensor con dtype y requires_grad especificados
```

**Caso 3: Escalares Python**

```python
# Inferencia automática de dtype
ensure_tensor(5)          # dtype=nova.long (int)
ensure_tensor(5.0)        # dtype=nova.float32 (float)
ensure_tensor(True)       # dtype=nova.bool (bool)
ensure_tensor([1, 2, 3])  # dtype=nova.float32 (lista de float32)
```

**Caso 4: Override de propiedades**

```python
t = nova.tensor([1.0], requires_grad=False)
new_t = ensure_tensor(t, requires_grad=True)
# Crea nuevo Tensor con requires_grad=True
```

**Cuándo usar:**

- En funciones que aceptan inputs flexibles (Tensor, array, scalar)
- Para normalizar inputs en operaciones del framework
- Cuando se necesita conversión segura con fallback

**Usado internamente en:**

- `nova.nn.functional` (todas las funciones normalizan inputs)
- funciones de creación
- Operaciones del autograd

### `grad_checking.py`

Proporciona **`grad_check_wrt_inputs()`** para verificación numérica de gradientes.

**Propósito:**

Compara gradientes analíticos (calculados por backprop) con gradientes numéricos (diferencias finitas) para detectar bugs en implementaciones de operaciones custom.

**Firma:**

```python
def grad_check_wrt_inputs(
    fn: Callable[[Tensor], Tensor],
    *args: Tensor,
    eps: float = 1e-4,
    zero_grads: bool = True,
    domain_bounds: Optional[tuple[float, float]] = None,
    **kwargs
) -> tuple[list[ndarray], list[ndarray]]
```

**Parámetros:**

- `fn`: Función a verificar (debe retornar Tensor)
- `*args`: Tensors de input con `requires_grad=True`
- `eps`: Perturbación para diferencias finitas
- `zero_grads`: Si limpia gradientes después
- `domain_bounds`: Límites para clamping (ej: (0, 1) para probabilidades)

**Retorna:**

- Tupla de `(analytic_grads, numerical_grads)` (listas de arrays)

**Método de diferencias finitas centrales:**

```
∂f/∂x ≈ (f(x + ε) - f(x - ε)) / (2ε)
```

**Ejemplos de uso:**

**Ejemplo 1: Verificar operación cuadrática**

```python
from nova.utils import grad_check_wrt_inputs

x = nova.tensor([1.0, 2.0, 3.0], requires_grad=True)

def square_sum(t):
    return (t ** 2).sum()

analytic, numeric = grad_check_wrt_inputs(square_sum, x)

# Comparar
diff = np.abs(analytic[0] - numeric[0])
print(f"Max difference: {diff.max()}")  # Debería ser muy pequeño (~1e-5)

# Verificación con allclose
assert nova.allclose(analytic[0], numeric[0], rtol=1e-3, atol=1e-5)
```

**Ejemplo 2: Verificar función sigmoid**

```python
def sigmoid_sum(t):
    return nova.sigmoid(t).sum()

x = nova.tensor([0.5, -0.5, 1.0], requires_grad=True)
analytic, numeric = grad_check_wrt_inputs(
    sigmoid_sum, x,
    domain_bounds=(0, 1)  # Clamping para estabilidad numérica
)

assert nova.allclose(analytic[0], numeric[0], rtol=1e-3)
```

**Ejemplo 3: Verificar operación custom**

```python
from nova.autograd.function import Function

class MyOp(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x ** 3

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        return grad_out * 3 * (x ** 2)  # ¿Correcto?

# Verificar
x = nova.tensor([2.0], requires_grad=True)
analytic, numeric = grad_check_wrt_inputs(lambda t: MyOp.apply(t).sum(), x)

if not nova.allclose(analytic[0], numeric[0], rtol=1e-3):
    print("BUG DETECTED: Gradients don't match!")
```

**Cuándo usar:**

- Al implementar nuevas operaciones en `autograd/_ops`
- Para debugging de gradientes incorrectos
- Testing de custom layers o módulos
- Validación de implementaciones matemáticas

**Limitaciones:**

- Lento para tensors grandes (complejidad O(N) en elementos)
- Puede tener problemas numéricos con funciones discontinuas
- Requiere funciones diferenciables

## Submódulo `decorators/`

Contiene decoradores reutilizables para funcionalidad transversal.

### `registry.py`

Implementa el **sistema de registro de clases** para serialización segura.

**Decoradores principales:**

#### `@registry_class`

Registra una clase para deserialización segura con `nova.load()`.

**Características:**

- **Registro automático**: Usa módulo + nombre como clave única
- **Idempotente**: Re-registrar la misma clase no causa error
- **Usado por Module**: Todas las subclases de `Module` se registran automáticamente via metaclass

**Ejemplo de uso:**

```python
from nova.utils import registry_class
import nova.nn as nn

@registry_class
class CustomLayer:
    def __init__(self, features):
        super().__init__()
        self.weight = nn.Parameter(nova.randn(features, features))

    def forward(self, x):
        return x @ self.weight

# Ahora CustomLayer puede ser guardada y cargada de forma segura
model = CustomLayer(10)
nova.save(model, "custom.pth")
loaded = nova.load("custom.pth", weights_only=True)  # ✅ OK
```

**Cuándo usar:**

- Al definir custom modules que se guardarán
- Para custom optimizers o schedulers
- Cualquier clase que se serializará

#### `@registry_op(op_name)`

Registra una operación del autograd para binding dinámico.

**Características:**

- Asocia un nombre público (ej: "add") con una clase `Function`
- Usado internamente por el sistema de binding
- Valida que solo se registren subclases de `Function`
- **Previene duplicados**: Solo registra si el nombre no existe ya

**Parámetros:**

- `op_name`: Nombre público de la operación (ej: "add", "relu")

**Retorna:**

Decorador que registra la clase `Function`

**Raises:**

- `ValueError`: Si se intenta registrar algo que no es subclase de `Function`

**Ejemplos:**

```python
from nova.utils import registry_op
from nova.autograd.function import Function

@registry_op("custom_op")
class CustomOp(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y + x

    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx.saved_tensors
        return grad_out * (y + 1), grad_out * x

# Ahora "custom_op" está registrado
# Y si lo incorporas en el yaml puede ser usado por el binding system
```

**Funciones auxiliares:**

- `get_registered_classes(module, name)`: Recupera clase registrada por (módulo, nombre)
- `_MODULES`: Dict global `{(module, name): class}` con clases registradas
- `_OPS_REGISTERED`: Dict global `{op_name: Function}` con operaciones registradas

**Cuándo usar:**

- Al implementar nuevas operaciones autograd custom
- Para operaciones que se usarán en el binding system
- Cuando se necesita deserialización segura de grafos computacionales

### `timing.py`

Proporciona utilidades para medir tiempo de ejecución, incluyendo **`@chronometer`** y **`benchmark`**.

#### `@chronometer`

Decorador para medir tiempo de ejecución de funciones con soporte de benchmarking.

**Características:**

- **Formateo inteligente**: Ajusta unidades según duración (ns, μs, ms, s, m, h)
- **Logging automático**: Usa el sistema de logging de NovaNN
- **No invasivo**: Retorna el resultado original sin modificarlo (a menos que `return_time=True`)
- **Emojis descriptivos**: ⚡ (rápido), ⏱️ (medio), 🐢 (lento)
- **Modo benchmarking**: Múltiples iteraciones con warmup
- **Dual return mode**: Puede retornar solo result o (result, avg_time)

**Parámetros:**

- `func`: Función a decorar (automático)
- `n_iters`: Número de iteraciones para promediar (default: 1)
- `warmup`: Iteraciones de calentamiento no contadas (default: 0)
- `return_time`: Si True, retorna `(result, avg_time)` (default: False)
- `verbose`: Si True, loggea el tiempo (default: True)

**Retorna:**

- Si `return_time=False`: resultado de la función
- Si `return_time=True`: `(resultado, tiempo_promedio_en_segundos)`

**Ejemplos:**

```python
from nova.utils.decorators import chronometer

# Uso básico (sin parámetros)
@chronometer
def train_step(model, batch):
    loss = model(batch)
    loss.backward()
    return loss

# ⚡ train_step: 234ms

# Con benchmarking
@chronometer(n_iters=50, warmup=10)
def forward_pass(model, x):
    return model(x)

# ⚡ forward_pass: 12.34ms (avg over 50 runs)

# Con return_time (sin verbose)
@chronometer(return_time=True, verbose=False)
def compute():
    return expensive_operation()

result, elapsed = compute()
print(f"Took {elapsed*1000:.2f}ms")

# Benchmarking silencioso
@chronometer(n_iters=100, warmup=10, return_time=True, verbose=False)
def benchmark_op():
    return matrix_multiply(A, B)

result, avg_time = benchmark_op()
```

**Rangos de formato:**

- `< 1μs`: nanosegundos (ns)
- `< 1ms`: microsegundos (μs)
- `< 1s`: milisegundos (ms)
- `< 1min`: segundos (s)
- `< 1h`: minutos + segundos (Xm Ys)
- `≥ 1h`: horas + minutos + segundos (Xh Ym Zs)

**Cuándo usar:**

- Profiling de funciones de entrenamiento
- Debugging de cuellos de botella
- Benchmarking de operaciones con múltiples iteraciones
- Tracking de performance con acceso programático a tiempos

#### `benchmark(func, *args, n_iters=100, warmup=10, **kwargs)`

Función para ejecutar benchmarks precisos sobre cualquier callable.

**Descripción:**

Ejecuta una función múltiples veces, omitiendo las iteraciones de calentamiento, y devuelve:

- El resultado de la función
- El tiempo medio de ejecución
- La desviación estándar del tiempo

**Parámetros:**

- `func`: Función a benchmarkear
- `*args`: Argumentos posicionales para la función
- `n_iters`: Número de iteraciones medidas (default: 100)
- `warmup`: Iteraciones de calentamiento (default: 10)
- `**kwargs`: Argumentos keyword para la función

**Retorna:**

Tupla `(result, mean_time, std_time)` donde:

- `result`: Valor retornado por la función (última iteración)
- `mean_time`: Tiempo promedio en segundos (float)
- `std_time`: Desviación estándar en segundos (float)

**Ejemplos:**

```python
from nova.utils import benchmark

def matmul(A, B):
    return A @ B

# Benchmark básico
result, mean, std = benchmark(matmul, A, B, n_iters=100)
print(f"Promedio: {mean*1000:.3f} ms ± {std*1000:.3f} ms")

# Con warmup personalizado
result, mean, std = benchmark(
    neural_net_forward,
    model, x,
    n_iters=50,
    warmup=5
)

# Comparación de implementaciones
nova_res, nova_mean, nova_std = benchmark(nova_conv, x, w)
torch_res, torch_mean, torch_std = benchmark(torch_conv, x, w)
speedup = torch_mean / nova_mean
print(f"Speedup: {speedup:.2f}x")
```

**Características:**

- **Calentamiento configurable** para estabilizar mediciones
- **Promedio y desviación estándar** con NumPy para análisis estadístico
- **Retorna resultado** además de stats para verificar correctitud
- Ideal para **comparaciones reproducibles** de performance

**Cuándo usar:**

- Comparaciones con implementaciones en PyTorch
- Evaluación de optimizaciones internas
- Medición reproducible en scripts de `benchmarks/`
- Cuando necesitas stats detalladas (mean + std) en lugar de solo timing

## Submódulo `data/`

Contiene abstracciones para manejo de datasets y dataloaders.

### `dataset.py`

Define la clase base abstracta **`Dataset`**.

**Clase `Dataset`:**

Contrato para todos los datasets en NovaNN.

**Métodos abstractos:**

- `__len__()`: Retorna número total de muestras
- `__getitem__(index)`: Retorna muestra(s) en el índice

**Tipo `Index`:**

```python
type Index = slice | int | tuple | Tensor | ndarray
```

Soporta múltiples tipos de indexing:

- Entero: `dataset[0]` → single sample
- Slice: `dataset[0:10]` → batch
- Lista/tuple: `dataset[[1,5,9]]` → fancy indexing
- Tensor: `dataset[nova.tensor([0,2,4])]`
- Array: `dataset[np.array([1,3,5])]`

**Ejemplo de implementación:**

```python
from nova.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Uso
dataset = MyDataset(nova.randn(100, 10), nova.randint(0, 2, (100,)))
print(len(dataset))  # 100
x, y = dataset[0]    # Primera muestra
batch_x, batch_y = dataset[0:32]  # Primer batch
```

**Cuándo heredar:**

- Al crear custom datasets
- Para datasets de imágenes, texto, audio, etc.
- Cuando se necesita lazy loading o augmentation

### `dataloader.py`

Implementa **`DataLoader`**, un iterador que produce batches de un Dataset.

**Características:**

- **Batching automático**: Divide dataset en batches del tamaño especificado
- **Shuffling configurable**: Mezcla índices al inicio de cada época
- **Último batch variable**: Maneja automáticamente último batch más pequeño
- **Iteración eficiente**: No carga todo el dataset en memoria

**Clase `DataLoader`:**

**Parámetros:**

- `dataset`: Instancia de Dataset
- `batch_size`: Tamaño de batch (default: 64)
- `shuffle`: Si mezclar índices (default: True)

**Métodos:**

- `__iter__()`: Retorna iterador para una época
- `__len__()`: Retorna número de batches
- `batch_size` (property): Acceso read-only al batch size

**Clase interna `_Iter`:**

Iterador que mantiene el estado de una época completa.

**Ejemplos de uso:**

**Ejemplo 1: Training loop básico**

```python
from nova.utils.data import DataLoader

# Crear dataset y loader
dataset = MyDataset(nova.randn(1000, 784), nova.randint(0, 10, (1000,)))
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
model.train()
for epoch in range(10):
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
```

**Ejemplo 2: Evaluation sin shuffling**

```python
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model.eval()
total_correct = 0
with nova.no_grad():
    for xb, yb in test_loader:
        pred = model(xb)
        total_correct += (pred.argmax(dim=1) == yb).sum()

accuracy = total_correct / len(test_dataset)
```

**Ejemplo 3: Múltiples épocas con shuffling independiente**

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(5):
    # Cada época tiene shuffling diferente
    for batch_idx, (xb, yb) in enumerate(loader):
        # training...
        pass
    print(f"Epoch {epoch}: {len(loader)} batches")
```

**Cuándo usar:**

- En todos los loops de entrenamiento/evaluación
- Para iterar eficientemente sobre datasets grandes
- Cuando se necesita batching y shuffling automático

### `preprocessing.py`

Utilidades de preprocesamiento para la normalización de datos, división, guardado y descarga de conjuntos de datos.

**Funciones**:

- `normalize(x_data, x_mean, x_std)`: Normaliza los datos utilizando la media y la desviación estándar. Soporta arrays de NumPy y Tensores de Nova. Incluye una protección de épsilon (`1e-8`) para evitar la división por cero.

- `split_features_and_labels(df, label_column, dtype)`: Divide un DataFrame en arrays de características (features) y etiquetas (labels). Si `label_column` no existe, se utiliza la primera columna como etiquetas. Las características por defecto son `float32`, las etiquetas siempre son `int64`.

- `split_validation_subset(x, y, factor, shuffle, stratify, random_state)`:
  Divide arrays o Tensores en subconjuntos de entrenamiento y validación. Si las entradas son Tensores de Nova, se convierten internamente y se devuelven como Tensores. Lanza un `ValueError` si el `factor` no está en el rango `(0, 1)`.

- `split_validation_dataset(dataset, label, factor, root, save_method, ...)`: Divide un DataFrame en conjuntos de entrenamiento y validación y los guarda en el disco. Soporta formatos `csv`, `parquet` y `excel`.

- `save_to_csv(df, root)` / `save_to_parquet(df, root)` / `save_to_excel(df, root)`: Guarda un DataFrame en el formato especificado. Valida el DataFrame antes de escribir, crea directorios si es necesario y limpia archivos parciales en caso de fallo.

- `download_dataset(dataset, root, format, force_redownload, validate)`: Descarga MNIST o Fashion-MNIST desde sus servidores oficiales y los convierte a formato tabular. Cada imagen se aplana a 784 columnas de píxeles. Se omiten los archivos ya convertidos a menos que `force_redownload=True`.

**Ejemplos de uso:**

**Ejemplo 1: normalizar datos**

```python
from nova.utils.data import normalize
import numpy as np

x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x_normalized = normalize(x, x_mean=np.mean(x), x_std=np.std(x))
```

**Ejemplo 2: dividir características y etiquetas**

```python
from nova.utils.data import split_features_and_labels
import pandas as pd

df = pd.DataFrame({'label': [0, 1], 'pixel0': [128, 255], 'pixel1': [64, 32]})
x, y = split_features_and_labels(df)
# x.shape -> (2, 2), dtype float32
# y.shape -> (2,),   dtype int64
```

**Ejemplo 3: dividir subconjunto de validación**

```python
from nova.utils.data import split_validation_subset
import numpy as np

x = np.random.rand(1000, 784)
y = np.random.randint(0, 10, 1000)

x_train, y_train, x_val, y_val = split_validation_subset(
    x, y, factor=0.2, stratify=True, random_state=8
)
# x_train.shape -> (800, 784)
# x_val.shape   -> (200, 784)
```

**Ejemplo 4: dividir y guardar conjunto de datos de validación**

```python
from nova.utils.data import split_validation_dataset
import pandas as pd

df = pd.read_parquet("data/mnist_train.parquet")

train, val = split_validation_dataset(
    df,
    label="label",
    factor=0.16,
    root="data/Mnist",
    save_method="parquet",
    set_name="mnist_train_e",
    val_name="mnist_validation",
    random_state=8,
    stratify=True,
)
```

**Ejemplo 5: guardar DataFrame**

```python
from nova.utils.data import save_to_csv, save_to_parquet, save_to_excel
import pandas as pd

df = pd.DataFrame({'label': [0, 1], 'pixel0': [128, 255]})

save_to_csv(df, root="output/data.csv")
save_to_parquet(df, root="output/data.parquet")
save_to_excel(df, root="output/data.xlsx")
```

**Ejemplo 6: descargar conjuntos de datos desde sitios web**

```python
from nova.utils.data import download_dataset

# Descargar como parquet (recomendado)
download_dataset("mnist", root="~/.novann/datasets", format="parquet")

# Forzar la redescarga
download_dataset("fashion-mnist", root="~/.novann/datasets", format="parquet", force_redownload=True)
```

## Submódulo `datasets/`

Cargadores preconfigurados para conjuntos de datos comunes.

### `mnist.py` y `fashion.py`

Proporcionan funciones para cargar **MNIST** y **Fashion-MNIST** desde archivos `.parquet`.

**Funciones:**

- `load_mnist_data(...)`: Carga MNIST.
- `load_mnist_defatul()`: Carga MNIST con argumentos por defecto.
- `load_fashion_mnist_data(...)`: Carga Fashion-MNIST.
- `load_fashion_mnist_data()`: Carga Fashion-MNIST con argumentos por defecto.

**Parámetros comunes:**

- `tensor4d`: Si es `True`, cambia la forma (reshape) a (N, 1, 28, 28) para redes neuronales convolucionales (CNNs).
- `as_tensor`: Si es `True`, convierte los datos a `nova.Tensor`.
- `do_normalize`: Si es `True`, normaliza utilizando las estadísticas del conjunto de entrenamiento.
- `dtype`: Tipo de dato para las características (features).
- `train_path`, `test_path`, `val_path`: Rutas para guardar los archivos. Por defecto es `~/.novann/datasets`.
  **Retorna:**

Tupla de 3 datasets: `(train, test, val)`, cada uno es instancia de `MnistData`/`FashionData` (subclases de `Dataset`).

**Ejemplos de uso:**

**Ejemplo 1: Cargar MNIST para MLP**

```python
from nova.utils.datasets import mnist

train, test, val = mnist.load_mnist_data(
    tensor4d=False,  # (N, 784) para MLP
    as_tensor=True,
    do_normalize=True,
    dtype=nova.float32
)

print(len(train))  # ~15000
print(train[0][0].shape)  # (784,)
```

**Ejemplo 2: Cargar Fashion-MNIST para CNN**

```python
from nova.utils.datasets import fashion

train, test, val = fashion.load_fashion_mnist_data(
    tensor4d=True,  # (N, 1, 28, 28) para CNN
    as_tensor=True,
    do_normalize=True,
    dtype=nova.float32
)

print(train[0][0].shape)  # (1, 28, 28)
```

**Ejemplo 3: Pipeline completo**

```python
from nova.utils.datasets import mnist
from nova.utils.data import DataLoader

# Cargar datos
train, test, val = mnist.load_mnist_data(dtype=nova.float32)

# Crear loaders
train_loader = DataLoader(train, batch_size=64, shuffle=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False)

# Training loop
for epoch in range(10):
    for xb, yb in train_loader:
        # xb: (64, 784), yb: (64,)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

## Integración con otros módulos

El módulo `utils` se integra con:

- **[`serialization/`](../serialization/README.es.md)**: `registry_class` registra clases para carga segura
- **[`autograd/`](../autograd/README.es.md)**: `registry_op` registra operaciones, `grad_checking` verifica gradientes
- **[`nn/`](../nn/README.es.md)**: `Dataset` y `DataLoader` son fundamentales para entrenamiento
- **Todo el framework**: `logger` se usa globalmente, `ensure_tensor` normaliza inputs

## Diseño y filosofía

El módulo `utils` sigue estos principios:

- **Utilidades transversales**: Funcionalidad que beneficia a múltiples módulos
- **Mínima dependencia**: Utils no dependen de componentes complejos
- **Extensibilidad**: Decoradores y clases base facilitan extensión
- **Robustez**: Manejo de errores y logging detallado
- **Performance**: Decoradores como `@chronometer` para profiling

---

> Para más detalles sobre componentes específicos, consulta el código fuente en los archivos correspondientes.
