# Módulo `autograd`

El directorio **`autograd/`** implementa el **sistema de diferenciación automática de NovaNN**.  
Este es el motor que permite calcular gradientes automáticamente durante el entrenamiento, construyendo dinámicamente el grafo computacional y ejecutando la retropropagación (backpropagation).

El autograd de NovaNN sigue un diseño similar al de **PyTorch**: cada operación construye un nodo en el grafo de cómputo, y al llamar `.backward()` los gradientes fluyen en orden inverso hasta las hojas del grafo.

**Construcción Dinámica del Grafo Computacional**

<p align="center">
  <img src="../../images/graph.png" width="970" height="500" alt="Computatinoal graph">
</p>

## Estructura general

El módulo está organizado en:

- **Archivos principales** en la raíz que definen la API pública del autograd
- **[`engine/`](#submódulo-engine)**: motor de backpropagation y construcción del grafo computacional
- **[`_ops/`](#submódulo-_ops)**: todas las operaciones diferenciables organizadas por categoría
- **[`utils/`](#submódulo-utils)**: utilidades internas para procesamiento de argumentos y tipos
- **[`tests/`](#submódulo-tests)**: suite de tests para validar el sistema de gradientes

## Archivos principales

### `function.py`

Define la clase base **`Function`**, que es la abstracción fundamental para todas las operaciones diferenciables en NovaNN.

Cada operación (suma, multiplicación, ReLU, etc.) hereda de `Function` e implementa:

- **`forward(ctx, \*args, **kwargs)`\*\*: cómputo hacia adelante (recibe numpy arrays)
- **`backward(ctx, grad_output)`**: cómputo de gradientes (retropropagación)
- **`apply(*args)`**: método de clase que orquesta todo el proceso

El método `apply()` es el punto de entrada que:

1. Crea un `Context` para guardar valores intermedios
2. Convierte Tensors a arrays de NumPy
3. Ejecuta `forward()` con los arrays
4. Construye el nodo del grafo si `requires_grad=True`
5. Retorna un nuevo `Tensor` con `grad_fn` adjuntado

### `grad.py`

Implementa la función **`grad()`**, que permite calcular gradientes de salidas con respecto a entradas de forma explícita.

Esta función es útil para:

- Cálculo de gradientes sin modificar `.grad` de los tensores
- Derivadas de orden superior (`create_graph=True`)
- Gradientes parciales o condicionales

### `grad_mode.py`

Proporciona **context managers** para controlar el comportamiento del autograd:

- **`no_grad()`**: desactiva el tracking de gradientes (útil para inferencia)
- **`enable_grad()`**: reactiva el tracking explícitamente
- **`is_grad_enabled()`**: consulta el estado actual

Estos mecanismos son **thread-safe** mediante `threading.local()`.

## Submódulo `engine/`

Contiene el **núcleo del motor de backpropagation**.

### `context.py`

Define la clase **`Context`**, un contenedor simple para guardar valores intermedios durante `forward()` que serán necesarios en `backward()`.

### `engine.py`

Implementa las funciones internas que ejecutan la retropropagación:

- **`_build_topo(tensor)`**: construye el orden topológico del grafo computacional usando DFS iterativo
- **`_backward(tensor, gradient, retain_graph)`**: ejecuta el backward pass completo

**Fases del backward:**

1. Construcción del orden topológico (output → inputs)
2. Limpieza de gradientes intermedios previos
3. Propagación de gradientes en orden inverso
4. Aplicación de hooks
5. Limpieza del grafo (si `retain_graph=False`)

## Submódulo `_ops/`

Contiene **todas las operaciones diferenciables** organizadas por categoría funcional.

Cada operación hereda de `Function` e implementa su propio `forward()` y `backward()`.

### Estructura del directorio `_ops/`

El directorio está organizado de la siguiente forma:

```
_ops/
├── __init__.py
├── _activation.py        # Funciones de activación
├── _arithmetic.py        # Operaciones aritméticas básicas
├── _comparison.py        # Operaciones de comparación
├── _convolution.py       # Operaciones de convolución optimizadas
├── _creation.py          # Funciones de creación de tensores
├── _indexing.py          # Operaciones de indexing
├── _linalg.py            # Álgebra lineal
├── _linear.py            # Capa lineal
├── _loss.py              # Funciones de pérdida
├── _manipulation.py      # Manipulación de forma y estructura
├── _normalization.py     # Operaciones de normalización
├── _random.py            # Generación aleatoria
├── _reduction.py         # Operaciones de reducción
├── _trigonometric.py     # Funciones trigonométricas
├── _view.py              # Operaciones de vistas
├── utils.py              # Utilidades internas
└── native/
    └── native_functions.yaml  # Registro de operaciones
```

### Categorías de operaciones

#### `_activation.py`

Funciones de activación diferenciables:

- **ReLU**: Rectified Linear Unit (`max(0, x)`)
  - Gradiente: 1 si x > 0, 0 en caso contrario
- **LeakyReLU**: ReLU con pendiente negativa
  - Forward: `x if x > 0 else alpha * x`
  - Gradiente: 1 si x > 0, alpha en caso contrario
- **PReLU**: Parametric ReLU con weight aprendible
  - Forward: `max(0, x) + weight * min(0, x)`
  - Computa gradientes tanto para input como para weight
- **GELU**: Gaussian Error Linear Unit
  - Usa aproximación con tanh: `0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`
  - Activación suave con derivada continua
- **Sigmoid**: Función sigmoide
  - Forward: `1 / (1 + exp(-x))`
  - Gradiente: `out * (1 - out)` (derivada eficiente usando output pre-computado)

#### `_arithmetic.py`

Operaciones aritméticas fundamentales con soporte para broadcasting:

**Operaciones binarias:**

- **Add**, **Sub**, **Mul**, **Div**: aritmética básica elemento a elemento
  - Soportan broadcasting de NumPy
  - Gradientes ajustados con `unbroadcasting()` para formas originales
- **DivInt**: división entera (`//`) - no diferenciable
- **Mod**: operación módulo (`%`)
  - Gradiente simplificado: ∂(a % b)/∂a = 1
- **Pow**: exponenciación (`a^b`)
  - Gradiente respecto a base: `b * a^(b-1)`
  - Gradiente respecto a exponente: `a^b * ln(a)` (con máscara para a > 0)

**Operaciones unarias:**

- **Exp**: exponencial (`e^x`)
  - Gradiente: `e^x` (la derivada es la función misma)
- **Log**: logaritmo natural
  - Gradiente: `1/x` (con epsilon para estabilidad)
- **Sqrt**: raíz cuadrada
  - Gradiente: `1/(2√x)`
- **Neg**: negación (`-x`)
  - Gradiente: -1
- **Abs**: valor absoluto
  - Gradiente: `sign(x)`
- **Floor**: redondeo hacia abajo - no diferenciable
- **Ceil**: redondeo hacia arriba - no diferenciable
- **Clamp**: limitación de valores a rango [min, max]
  - Gradiente: pasa solo donde `min <= input <= max`

#### `_comparison.py`

Operaciones de comparación y selección:

- **Maximum**: máximo elemento a elemento
  - Gradiente: distribuido entre inputs (0.5 para empates)
- **Minimum**: mínimo elemento a elemento
  - Gradiente: distribuido entre inputs (0.5 para empates)
- **Where**: selección condicional (`condition ? x : y`)
  - Gradiente fluye solo por la rama seleccionada
- **Sign**: función signo (-1, 0, +1) - no diferenciable

#### `_convolution.py`

Operaciones de convolución optimizadas basadas en im2col:

- **ConvMatMul1d**: multiplicación matricial optimizada para convoluciones 1D
  - Fusiona matmul y reshaping en operación única
  - Forward: `(weight @ col).reshape(...).transpose(...)`
  - Soporta bias opcional
  - Gradientes eficientes usando reglas de matmul
- **ConvMatMul2d**: multiplicación matricial optimizada para convoluciones 2D
  - Implementación análoga para inputs 4D (N, C, H, W)
  - Utilizada por `nn.Conv2d`
- **ConvMatMul3d**: multiplicación matricial optimizada para convoluciones 3D
  - Para inputs 5D (N, C, D, H, W)
  - Soporta convoluciones volumétricas

Todas estas operaciones:

- Implementan el algoritmo im2col implícitamente
- Optimizan memoria al evitar materializar la matriz col explícitamente
- Calculan gradientes respecto a weight, bias y col (para backprop a través de im2col)

#### `_creation.py`

Funciones de creación y generación de tensores. Este archivo contiene **funciones auxiliares de alto nivel** que crean tensores, no operaciones diferenciables de `Function`:

**Creación básica:**

- `zeros()`, `ones()`, `full()`, `empty()`: tensores con valores constantes
- `eye()`: matriz identidad
- `arange()`: secuencia de valores con step
- `linspace()`: valores equiespaciados

**Variantes condicionales:**

- `zeros_like()`, `ones_like()`, `full_like()`: basados en forma de otro tensor

**Utilidades:**

- `one_hot()`: codificación one-hot para labels
- `unique()`: valores únicos en tensor
- `as_strided()`: vistas con strides personalizados

**Selección e indexing:**

- `argmin()`, `argmax()`: índices de valores extremos
- `argsort()`: índices de ordenamiento
- `argwhere()`: índices de elementos no-cero

**Operaciones lógicas:**

- `any()`, `all()`: reducción lógica
- `allclose()`: comparación con tolerancia
- `isnan()`, `isinf()`: detección de valores especiales

**Funciones de reducción estadística:**

- `mean()`, `var()`, `std()`: estadísticas básicas
- `min()`, `max()`, `sum()`: agregación
- `norm()`: normas vectoriales/matriciales

**Manipulación:**

- `reshape()`, `permute()`, `flatten()`: cambio de forma
- `squeeze()`, `unsqueeze()`: dimensiones unitarias
- `cat()`, `stack()`, `split()`: concatenación/división
- `tile()`, `repeat_interleave()`: replicación
- `pad()`: padding con diferentes modos
- `clamp()`: limitación de valores

**Operaciones matemáticas:**

- `sqrt()`, `exp()`, `log()`: funciones elementales
- `abs()`, `sign()`, `floor()`, `ceil()`: operaciones unarias
- `pow()`, `maximum()`, `minimum()`: operaciones binarias
- `where()`: selección condicional

**Álgebra lineal:**

- `dot()`: producto punto
- `det()`, `inv()`, `trace()`: operaciones matriciales

**Trigonométricas:**

- `sin()`, `cos()`, `tan()`, `tanh()`: directas
- `sinh()`, `cosh()`: hiperbólicas directas
- `arcsin()`, `arccos()`, `arctan()`: inversas
- `asinh()`, `acosh()`, `atanh()`: hiperbólicas inversas
- `sec()`, `csc()`, `cot()`: secante, cosecante y cotangente
- `arcsec()`, `arccsc()`, `arccot()`: inversas de secante, cosecante y cotangente
- `atan2()`: arctangente de dos argumentos (y/x) con selección correcta de cuadrante

Estas funciones son **wrappers de alto nivel** que delegan a las operaciones diferenciables correspondientes en otros archivos de `_ops/`, proporcionando una interfaz funcional consistente con PyTorch.

#### `_indexing.py`

Operaciones de indexing avanzado:

- **GetItem**: implementa `tensor[index]`
  - Soporta slicing, fancy indexing, indexing booleano
  - Gradiente: acumula grad_output en posiciones indexadas usando `np.add.at()`
  - Sanitiza índices (convierte floats a int64, maneja booleanos)
- **SetItem**: implementa `tensor[index] = value` (in-place)
  - Gradiente: copia grad_output con ceros en posiciones asignadas
  - Usado internamente, no recomendado con autograd activo

#### `_linalg.py`

Operaciones de álgebra lineal:

- **MatMul**: multiplicación de matrices (`@`)
  - Gradiente: `grad_input = grad_output @ other.T`, `grad_other = input.T @ grad_output`
- **Dot**: producto punto
  - Maneja vectores y matrices apropiadamente
  - Gradiente: producto con transpuesta según dimensionalidad
- **Det**: determinante de matriz cuadrada
  - Gradiente: `det(A) * (A⁻¹)ᵀ * grad_output` (usando adjunta)
- **Inv**: inversa de matriz
  - Gradiente: `-(A⁻¹)ᵀ @ grad_output @ (A⁻¹)ᵀ`
- **Trace**: traza de matriz (suma de diagonal)
  - Gradiente: `grad_output * I` (matriz identidad escalada)
- **Norm**: normas vectoriales y matriciales
  - Implementa norma L2 por defecto (`ord=2`)
  - Gradiente: `grad_output * (input / ||input||)` con protección contra división por cero
- **Diag**: extracción/construcción de diagonal
  - Si input es 1D: construye matriz diagonal
  - Si input es 2D: extrae diagonal
  - Soporta parámetro `diagonal` para diagonales offset

#### `_linear.py`

Capa lineal (fully connected) optimizada:

- **Dense**: Transformación lineal con bias opcional
  - Forward: `Y = X @ W.T + b`
    - Pre-asigna buffer de salida para eficiencia
    - Soporta término de bias opcional
  - Backward:
    - Gradiente respecto a input: `grad_output @ weight`
    - Gradiente respecto a weight: `grad_output.T @ input`
    - Gradiente respecto a bias: `Σ(grad_output)` (suma sobre dimensión de batch)
  - Usa multiplicación matricial eficiente con buffers pre-asignados
  - Base para la capa `nn.Linear`

#### `_loss.py`

Funciones de pérdida implementadas como operaciones atómicas para mayor estabilidad numérica y eficiencia computacional:

- **MSELoss**: Mean Squared Error / L2 Loss
  - Computa `(input - target)²` como operación atómica
  - Más estable numéricamente que separar resta y potencia
  - Soporta pesos opcionales por elemento
  - Gradiente: `∂L/∂input = 2 * (input - target)`

- **BCELoss**: Binary Cross Entropy Loss
  - Computa `-[target * log(input + ε) + (1 - target) * log(1 - input + ε)]`
  - Requiere inputs en rango [0, 1] (post-sigmoid)
  - Soporta pesos opcionales por elemento
  - Gradiente: `∂L/∂input = (1 - target)/(1 - input + ε) - target/(input + ε)`

- **BCEWithLogitsLoss**: BCE con logits (numéricamente estable)
  - Combina sigmoid y BCE en operación única
  - Usa formulación estable: `max(x, 0) - x*y + log(1 + exp(-|x|))`
  - Soporta `pos_weight` para balancear clases positivas
  - Gradiente: `∂L/∂input = sigmoid(x) - target`

Todas las funciones de pérdida soportan tres modos de reducción:

- `'mean'`: promedio sobre todos los elementos
- `'sum'`: suma sobre todos los elementos
- `'none'`: retorna pérdida elemento a elemento sin reducción

**Utilidades:** El módulo incluye `reduce()` para aplicar el modo de reducción especificado.

#### `_manipulation.py`

Operaciones de manipulación de forma y estructura:

- **Reshape**: cambio de forma con copia si es necesario
  - Gradiente: reshape inverso a forma original
- **View**: cambio de forma sin copia (vista)
  - Más eficiente que reshape cuando es posible
  - Gradiente: reshape a forma original
- **Permute**: permutación de dimensiones
  - Gradiente: permutación inversa (usando `np.argsort()`)
- **Squeeze**: eliminación de dimensiones de tamaño 1
  - Gradiente: reshape a forma original
- **Unsqueeze**: adición de dimensión de tamaño 1
  - Gradiente: reshape a forma original
- **Stack**: apilamiento de tensors en nueva dimensión
  - Gradiente: split y squeeze de cada componente
- **Concat**: concatenación en dimensión existente
  - Gradiente: split usando offsets calculados de formas originales
- **Split**: división en múltiples chunks
  - Gradiente: concatenación de gradientes de outputs
- **Tile**: replicación a lo largo de dimensiones
  - Gradiente: suma sobre bloques replicados
- **Repeat**: repetición de elementos a lo largo de dimensión
  - Gradiente: suma de gradientes de elementos repetidos
  - Soporta `dim=None` para flatten + repeat
- **Pad**: padding con diferentes modos
  - Soporta: constant, edge, reflect, wrap
  - Gradiente: slicing que remueve regiones padded
- **Clone**: copia profunda con tracking de gradientes
  - Gradiente: pasa directamente

#### `_normalization.py`

Operaciones de normalización para estabilidad en entrenamiento:

- **BatchNorm**: Normalización por lotes con transformación afín
  - Normaliza sobre dimensiones de batch y espaciales
  - Forward: `(x - μ) / √(σ² + ε) * weight + bias`
    - Modo entrenamiento: usa estadísticas del batch (μ, σ²) y actualiza promedios corrientes con momentum
    - Modo evaluación: usa estadísticas corrientes pre-computadas
    - Aplica corrección de Bessel para estimación insesgada de varianza
  - Backward: `∂L/∂input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]`
    - Formulación eficiente que considera dependencia de μ y σ² en el input
    - Computa gradientes para weight, bias e input simultáneamente
  - Estadísticas corrientes actualizadas mediante promedio móvil exponencial:
    - `running_mean = (1 - momentum) * running_mean + momentum * batch_mean`
    - `running_var = (1 - momentum) * running_var + momentum * batch_var`
- **LayerNorm**: Normalización de capa
  - Normaliza sobre las últimas N dimensiones (independiente del batch)
  - Forward: `(x - μ) / √(σ² + ε) * weight + bias`
    - Estadísticas computadas sobre dimensiones de `normalized_shape`
    - Comúnmente usado en Transformers por independencia del tamaño de batch
  - Backward: usa la misma formulación eficiente que BatchNorm
    - `∂L/∂input = (1/(m*σ)) * [m*dout - Σ(dout) - x_hat*Σ(dout*x_hat)]`
  - Pre-asigna buffers para media, varianza y cómputos intermedios
  - Soporta transformación afín opcional (weight y bias)

Ambas operaciones de normalización:

- Usan buffers pre-asignados para minimizar asignaciones de memoria
- Implementan cómputo de gradientes numéricamente estable
- Soportan parámetros afines aprendibles opcionales
- Son base para `nn.BatchNorm1d`, `nn.BatchNorm2d` y `nn.LayerNorm`

#### `_random.py`

Generación aleatoria de tensores. Este archivo contiene **utilidades de generación aleatoria**, no operaciones diferenciables:

- **Generator**: wrapper alrededor de `np.random.Generator`
  - Permite control de semilla para reproducibilidad
  - Método `manual_seed()` para resetear semilla

**Funciones de generación:**

- `rand()`: distribución uniforme [0, 1)
- `randn()`: distribución normal estándar (μ=0, σ=1)
- `randint()`: enteros aleatorios en rango [low, high)
- `randperm()`: permutación aleatoria de enteros [0, n)
- `normal()`: distribución normal con μ y σ personalizados
- `uniform()`: distribución uniforme en rango [low, high)

**Utilidades:**

- `manual_seed()`: establece semilla del generador por defecto
- Todas las funciones soportan parámetro `generator` opcional para control independiente
- Soportan `requires_grad` y `dtype` como parámetros

Estas funciones crean tensors con datos aleatorios pero **no son parte del grafo de autograd** (no tienen gradiente).

#### `_reduction.py`

Operaciones de reducción sobre dimensiones:

- **Sum**: suma de elementos
  - Gradiente: broadcast de grad_output a forma original
  - Soporta reducción sobre ejes específicos o total
- **Mean**: promedio de elementos
  - Gradiente: broadcast dividido por número de elementos reducidos
- **Var**: varianza
  - Gradiente: `(2/N) * (input - mean) * grad_output`
  - Guarda diferencias pre-computadas para eficiencia
- **Max**: valor máximo
  - Gradiente: distribuido equitativamente entre elementos máximos
  - Usa máscara para identificar posiciones del máximo
- **Min**: valor mínimo
  - Gradiente: distribuido equitativamente entre elementos mínimos
  - Implementación análoga a Max

Todas las operaciones de reducción:

- Normalizan `dim` a tupla para consistencia
- Soportan `keepdims` para mantener dimensiones reducidas
- Manejan correctamente broadcasting inverso en backward

#### `_trigonometric.py`

Funciones trigonométricas e hiperbólicas completas:

**Funciones trigonométricas directas:**

- **Sin**: `sin(x)` - Gradiente: `cos(x)`
- **Cos**: `cos(x)` - Gradiente: `-sin(x)`
- **Tan**: `tan(x)` - Gradiente: `1 + tan²(x) = sec²(x)`
- **Cot**: `1/tan(x)` - Gradiente: `-1/sin²(x)`
- **Sec**: `1/cos(x)` - Gradiente: `sec(x) * tan(x)`
- **Csc**: `1/sin(x)` - Gradiente: `-csc(x) * cot(x)`

**Funciones trigonométricas inversas:**

- **Arcsin**: `arcsin(x)` - Gradiente: `1/√(1 - x²)`
  - Input clamped a [-1, 1] para estabilidad
- **Arccos**: `arccos(x)` - Gradiente: `-1/√(1 - x²)`
  - Input clamped a [-1, 1]
- **Arctan**: `arctan(x)` - Gradiente: `1/(1 + x²)`
- **Atan2**: `atan2(y, x)` - Gradientes: `x/(x² + y²)` y `-y/(x² + y²)`
- **Arccot**: `arctan(1/x)` - Gradiente: `-1/(1 + x²)`
- **Arcsec**: `arccos(1/x)` - Gradiente: `1/(|x| * √(x² - 1))`
- **Arccsc**: `arcsin(1/x)` - Gradiente: `-1/(|x| * √(x² - 1))`

**Funciones hiperbólicas:**

- **Sinh**: `sinh(x)` - Gradiente: `cosh(x)`
- **Cosh**: `cosh(x)` - Gradiente: `sinh(x)`
- **Tanh**: `tanh(x)` - Gradiente: `1 - tanh²(x)`

**Funciones hiperbólicas inversas:**

- **Asinh**: `asinh(x)` - Gradiente: `1/√(x² + 1)`
- **Acosh**: `acosh(x)` - Gradiente: `1/√(x² - 1)`
  - Input clamped a ≥1
- **Atanh**: `atanh(x)` - Gradiente: `1/(1 - x²)`
  - Input clamped a (-1, 1)

Todas las funciones implementan:

- Clamping de inputs donde sea necesario para evitar valores inválidos
- Gradientes estables numéricamente
- Guardan inputs o outputs pre-computados cuando es eficiente

#### `_view.py`

Operaciones de vistas sin copia de datos:

- **AsStrided**: construcción de vistas con strides personalizados
  - Permite crear vistas arbitrarias especificando `shape` y `strides`
  - Validación de bounds: verifica que max_offset < nbytes
  - Usado internamente para im2col en convoluciones
  - Gradiente: acumula usando `np.add.at()` en posiciones calculadas por offsets
- **View**: cambio de forma sin copia
  - Wrapper eficiente sobre `np.reshape()`
  - Gradiente: reshape inverso a forma original
- **Extend**: broadcasting a nueva forma
  - Wrapper sobre `np.broadcast_to()`
  - Gradiente: `unbroadcasting()` para sumar sobre dimensiones expandidas

### `utils.py`

Utilidades internas para operaciones de autograd que permiten la propagación correcta de gradientes y computación eficiente en memoria:

- **`unbroadcasting(grad, shape)`**: revierte broadcasting de gradientes a forma original
  - Remueve dimensiones leading extra (cuando `grad.ndim > len(shape)`)
  - Suma sobre ejes donde ocurrió broadcasting (size 1 en original)
  - Crucial para garantizar que gradientes tengan la forma correcta después de operaciones con broadcasting
  - Usado extensivamente en operaciones binarias (Add, Mul, Div, etc.)
  - Maneja tanto expansión de dimensiones como broadcasting de tamaño-1

- **`ensure_casting(dest, src)`**: asegura casting seguro de dtype para operaciones in-place
  - Verifica si arrays origen y destino tienen dtypes compatibles
  - Convierte automáticamente el array origen para coincidir con dtype del destino si es necesario
  - Crítico para prevenir errores de tipo en operaciones in-place
  - Retorna tupla `(dest, src)` donde `src` puede ser una nueva copia convertida
  - Usado internamente por `write_to_buffer()`

- **`write_to_buffer(dest, src)`**: realiza copia de memoria in-place
  - Maneja operación de copia de memoria de bajo nivel usando `np.copyto()`
  - Asegura que datos del origen se escriban físicamente en memoria del destino
  - Maneja automáticamente casting de dtype vía `ensure_casting()`
  - El array destino debe ser mutable (no read-only)
  - Retorna el array destino actualizado
  - Fundamento de todas las operaciones in-place (`add_()`, `mul_()`, etc.)

- **`dispatch_output(destination, src)`**: enruta resultados de computación a salida correcta
  - Actúa como dispatcher para operaciones con parámetro `out` opcional
  - Si `destination` se proporciona: copia resultado in-place vía `write_to_buffer()`
  - Si `destination` es None: retorna array origen directamente (sin copia)
  - Permite reutilización eficiente de memoria en operaciones como `torch.add(x, y, out=z)`
  - Usado en todas las operaciones de autograd para soportar buffers de salida opcionales

- **`accelerated_conv_backward(weight_shape, grad_output, col, w_col, dims)`**: backward pass optimizado para convoluciones
  - Computa gradientes tanto para pesos como para columnas im2col en un solo paso
  - Usa buffers pre-asignados para minimizar asignaciones de memoria
  - Asegura contigüidad de memoria con `np.ascontiguousarray()` para optimización BLAS
  - Aprovecha multiplicación matricial eficiente para cómputo de gradientes
  - Retorna tupla `(grad_weight, grad_col)` con formas correctas
  - Optimización de rendimiento crítica para capas convolucionales
  - Usado por operaciones ConvMatMul1d, ConvMatMul2d y ConvMatMul3d
  - Alcanza rendimiento cercano a BLAS mediante disposición cuidadosa de memoria

### `native/`

Contiene el archivo **`native_functions.yaml`**, que define el **registro de operaciones** para el sistema de binding dinámico.

Este archivo especifica cómo cada operación se enlaza a la clase `Tensor`:

- **Métodos dunder**: `__add__`, `__mul__`, `__matmul__`, etc.
- **Métodos reverse**: `__radd__`, `__rmul__`, `__rmatmul__`, etc.
- **Métodos regulares**: `add()`, `mul()`, `relu()`, etc.
- **Variantes in-place**: `add_()`, `mul_()`, `relu_()`, etc.

**Flags especiales:**

- `is_unary: true`: operaciones unarias (ReLU, sin, exp, etc.)
- `raw_args: true`: mantiene argumentos como valores raw sin convertir a Tensor (usado para índices, alpha en LeakyReLU, etc.)

**Ejemplo de entrada:**

```yaml
- name: add
  tensor:
    dunder: __add__
    reverse: __radd__
    method: add
    inplace:
      method: add_
      dunder: __iadd__
```

Este sistema permite que NovaNN genere automáticamente los métodos apropiados sin repetir código manualmente. El módulo `_internal/_binding.py` parsea este archivo y genera dinámicamente todos los bindings.

## Submódulo `utils/`

Contiene utilidades internas para el procesamiento de argumentos y determinación de tipos:

- **`ArgumentProcessor`**: convierte argumentos mixtos (Tensors, scalars, arrays) a numpy arrays
- **`determine_base_dtype()`**: determina el dtype base para consistencia numérica

Estas utilidades aseguran que las operaciones manejen correctamente:

- Conversión automática de tipos
- Broadcasting de NumPy
- Propagación correcta de gradientes con formas diferentes

## Submódulo `tests/`

Suite completa de tests para validar el sistema de autograd. Los tests están organizados en varios archivos especializados:

### `op_signatures.py`

Módulo de utilidades para testing que define:

- **`OpCategory`**: enum que clasifica operaciones por signatura (UNARY, BINARY, REDUCTION, SHAPE, SPECIAL)
- **`OPERATIONS`**: diccionario que agrupa todas las operaciones por categoría
- **`OP_TO_CATEGORY`**: mapeo inverso para lookup rápido
- **`SKIP_GRAD_CHECK`**: conjunto de operaciones que no deben validarse con gradient checking
- **`make_test_input()`**: genera inputs apropiados según la operación (valores positivos para `log`/`sqrt`, matrices cuadradas para `det`/`inv`, etc.)
- **`create_op_wrapper()`**: crea wrappers que invocan operaciones correctamente según su categoría
- **`ALL_TESTABLE_OPS`**: lista de todas las operaciones que pueden validarse con gradient checking

Este módulo es la base para tests parametrizados que verifican automáticamente todas las operaciones.

### `test_backward.py`

Tests del motor de backpropagation:

- **`test_topo_order()`**: verifica que `_build_topo()` construya el orden topológico correcto
- **`test_parents_of_tensors()`**: valida que los tensors almacenen correctamente sus inputs
- **`test_backward_pass_exceptions()`**: prueba que se lancen excepciones apropiadas (backward sin `requires_grad`, operaciones in-place, etc.)
- **`test_backward_pass()`**: test básico de propagación de gradientes
- **`test_retain_graph_simple()`**: verifica que `retain_graph=True` permita múltiples backwards
- **`test_gradient_accumulation()`**: valida acumulación de gradientes en múltiples backward passes
- **`test_shared_computation_graph()`**: prueba correctitud con nodos compartidos en el grafo
- **`test_no_retain_graph_fails()`**: verifica que backward falle sin `retain_graph` tras primer uso
- **`test_retain_graph_with_zero_grad()`**: combina `retain_graph` con `zero_grad()`
- **`test_mean_plus_sum_accumulation()`**: valida acumulación con diferentes operaciones de reducción

### `test_function.py`

Tests de la clase `Function` base:

- **`MockAdd`**: implementación mock de `Function` para testing
- **`test_forward_output_type()`**: verifica tipos de salida correctos
- **`test_no_grad_required()`**: valida comportamiento cuando ningún input tiene `requires_grad=True`
- **`test_dtype_coercion_and_casting()`**: prueba coerción de tipos (float, int, long)
- **`test_process_containers_and_index_like()`**: valida `ArgumentProcessor` con containers y tipos mixtos

### `test_gradients.py`

Tests de gradient checking con operaciones complejas:

- **`test_gradient_wrt_inputs()`**: valida gradientes analíticos vs numéricos en operación compuesta
- **`test_gradient_wrt_layer_op()`**: prueba gradient checking con capas de `nn` (Linear, BatchNorm1d, LayerNorm)
- **`test_retain_grad()`**: verifica que `retain_grad()` permita guardar gradientes en nodos intermedios

### `test_operations.py`

Suite exhaustiva de tests para todas las operaciones:

- **`test_operation_gradients()`**: test parametrizado que ejecuta gradient checking en **todas** las operaciones testables, con tolerancias adaptativas según complejidad numérica
- **`test_unary_operations()`**: tests específicos para operaciones unarias (validación de shape, presencia de gradientes no-cero)
- **`test_reduction_operations()`**: tests para reductions con diferentes configuraciones (`dim`, `keepdims`)
- **`test_operations_with_no_useful_gradients()`**: valida operaciones como `sign()` y `ceil()` cuyas derivadas son casi siempre cero
- **`test_trace_operation()`**: test específico para `trace()` con validación de gradiente esperado (debe ser `eye()`)

**Estrategia de testing:**

Los tests utilizan `grad_check_wrt_inputs()` que implementa **finite difference gradient checking**: compara gradientes analíticos (computados por backward) con gradientes numéricos (diferencias finitas). Esto garantiza que todas las implementaciones de `backward()` sean matemáticamente correctas.

Las tolerancias se ajustan según la estabilidad numérica de cada operación:

- Operaciones estables: `rtol=1e-2, atol=1e-3`
- Operaciones numéricamente sensibles (`exp`, `inv`, `det`, etc.): `rtol=1e-1, atol=1e-2`

## Integración con `Tensor`

El autograd se integra con la clase `Tensor` mediante:

1. **Atributo `grad_fn`**: referencia a la clase `Function` que creó el tensor
2. **Atributo `_inputs`**: lista de tensors de entrada (para backprop)
3. **Atributo `_ctx`**: instancia de `Context` con valores guardados
4. **Método `backward()`**: invoca `_backward()` del engine

Cuando un `Tensor` con `requires_grad=True` participa en una operación, automáticamente:

- Se registra en el grafo computacional
- Se adjunta su `grad_fn`
- Se guardan referencias a sus inputs

## Diseño y filosofía

El autograd de NovaNN está diseñado siguiendo estos principios:

- **Explícito sobre implícito**: el grafo se construye dinámicamente, pero cada paso es trazable
- **Separación de concerns**: `Function` define qué, `Context` guarda estado, `engine` ejecuta cómo
- **Extensibilidad**: añadir nuevas operaciones solo requiere heredar de `Function` y registrarlas
- **Performance consciente**: uso de NumPy vectorizado y liberación de grafos tras backward
- **Testing exhaustivo**: gradient checking automático sobre todas las operaciones para garantizar correctitud matemática

---

> Para más detalles sobre operaciones específicas, consulta el código fuente en `_ops/` o los tests en `tests/`.
