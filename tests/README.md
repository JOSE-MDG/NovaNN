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

### `📂 tests/`

**Suite de tests unitarios para verificar la correcta implementación de todos los componentes de NovaNN**

Contiene tests organizados por módulos que verifican funcionalidad, gradientes y comportamiento en diferentes modos de todas las capas, optimizadores, inicializadores y utilidades del framework.

#### `📂 tests/📂 dataloader/`

##### `test_dataloader.py`

- **Propósito**: Verifica el comportamiento correcto del `DataLoader`, especialmente el manejo del último batch cuando el tamaño del dataset no es múltiplo del batch size
- **Pruebas principales**:
  - `test_last_batch_size()`: Asegura que el último batch tenga el tamaño correcto (2 muestras cuando batch_size=4 y dataset de 10 muestros)
- **Metodología**:
  - Crea dataset sintético de 10 muestras con batch_size=4
  - Verifica que se produzcan 3 batches (4, 4, 2 muestras)
  - Confirma que el último batch tenga exactamente 2 muestras

#### `📂 tests/📂 initializers/`

##### `test_init.py`

- **Propósito**: Verifica las funciones de inicialización de pesos (`kaiming_normal_`, `kaiming_uniform_`, `xavier_normal_`, `xavier_uniform_`, `random_init_`)
- **Pruebas principales**:
  - `test_kaiming_normal_distribution()`: Verifica media ≈0 y desviación estándar correcta para Kaiming normal
  - `test_kaiming_uniform_distribution()`: Verifica que valores estén dentro de límites uniformes calculados
  - `test_xavier_normal_distribution()`: Verifica media ≈0 y varianza correcta para Xavier normal
  - `test_xavier_uniform_distribution()`: Verifica límites uniformes para Xavier uniform
  - `test_random_initializer()`: Verifica media ≈0 para inicialización aleatoria pequeña
  - `test_exceptions_of_init_methods()`: Verifica que nonlinearities no soportadas levanten `ValueError`
- **Metodología**:
  - Prueba con múltiples formas tensoriales (2D a 5D)
  - Compara estadísticas muestrales con valores teóricos esperados
  - Usa `calculate_gain` y `shape_validation` de `novann.core.init`
  - Tolerancias empíricas (0.1) para estadísticas

#### `📂 tests/📂 layers/📂 activations/`

**Tests para verificar la correcta implementación de las funciones de activación**

##### `test_relu.py`

- **Propósito**: Verifica la capa `ReLU` (Rectified Linear Unit)
- **Pruebas**:
  - `test_relu_forward_backward_and_numeric()`: Comprueba forward (no negatividad), backward (máscara de gradiente) y gradiente numérico para entradas no cero
- **Metodología**:
  - Forward: Verifica forma y propiedad `max(0, x)`
  - Backward: Compara con máscara `(x > 0)`
  - Gradiente numérico: Usa `numeric_grad_elementwise` para validar gradientes analíticos
  - Excluye `x = 0` donde la derivada no está definida

##### `test_leaky_relu.py`

- **Propósito**: Verifica la capa `LeakyReLU` con pendiente negativa configurable
- **Pruebas**:
  - `test_leaky_relu_forward_backward_and_numeric()`: Comprueba forward (comportamiento piecewise), backward (gradiente piecewise) y gradiente numérico
- **Metodología**:
  - Forward: Verifica `x` si `x ≥ 0`, `slope * x` si `x < 0`
  - Backward: Compara con `1` (x ≥ 0) y `slope` (x < 0)
  - Gradiente numérico: Validación con diferencias finitas para entradas no cero

##### `test_sigmoid.py`

- **Propósito**: Verifica la capa `Sigmoid`
- **Pruebas**:
  - `test_sigmoid_forward_backward_and_numeric()`: Comprueba forward (rango (0,1)), backward (gradiente) y gradiente numérico
- **Metodología**:
  - Forward: Verifica forma y rango `0 < σ(x) < 1`
  - Backward: Compara con fórmula analítica `σ(x) * (1 - σ(x))`
  - Gradiente numérico: Validación completa con `numeric_grad_elementwise`

##### `test_softmax.py`

- **Propósito**: Verifica la capa `Softmax` con estabilidad numérica y propiedades de probabilidad
- **Pruebas**:
  - `test_softmax_forward_properties_and_shift_invariance_columnwise()`: Verifica propiedades forward (suma a 1 por fila, no negatividad, invariancia a desplazamiento)
  - `test_softmax_backward_numeric_columnwise()`: Verifica backward usando producto Jacobiano-vector y gradiente numérico
- **Metodología**:
  - Forward: Suma a 1, no negatividad, invariancia a constante aditiva
  - Backward: Compara gradiente analítico con aproximación numérica usando `numeric_grad_scalar_from_softmax`

##### `test_tanh.py`

- **Propósito**: Verifica la capa `Tanh` (tangente hiperbólica)
- **Pruebas**:
  - `test_tanh_forward_backward_and_numeric()`: Comprueba forward (rango (-1,1), propiedad de función impar), backward (gradiente) y gradiente numérico
- **Metodología**:
  - Forward: Verifica forma, rango `-1 < tanh(x) < 1` y propiedad `tanh(-x) = -tanh(x)`
  - Backward: Compara con fórmula analítica `1 - tanh²(x)`
  - Gradiente numérico: Validación con `numeric_grad_elementwise`

#### `📂 tests/📂 layers/📂 batch_norm/`

**Tests para verificar las implementaciones de Batch Normalization en 1D y 2D**

##### `test_batchnorm1d.py`

- **Propósito**: Verifica la capa `BatchNorm1d` para normalización por lotes en entradas 1D/2D
- **Pruebas**:
  - `test_batchnorm1d_forward_train_mode()`: Verifica forward en modo entrenamiento (centrado y normalización por características, actualización de estadísticas móviles)
  - `test_batchnorm1d_forward_eval_mode()`: Verifica forward en modo evaluación (uso de estadísticas móviles, sin actualización)
  - `test_batchnorm1d_backward_gradient_check()`: Verifica gradientes analíticos vs numéricos para parámetros `gamma` y `beta`
  - `test_batchnorm1d_momentum_and_eps()`: Verifica parámetros de momentum y épsilon personalizados
  - `test_batchnorm1d_parameters()`: Verifica que el método `parameters()` retorne los parámetros correctos
- **Metodología**:
  - Modo entrenamiento: Verifica media ≈0 y varianza ≈1 por característica después de normalización
  - Modo evaluación: Verifica uso de estadísticas móviles y estabilidad numérica
  - Gradientes: Usa `numeric_grad_wrt_param` para comparar gradientes analíticos y numéricos de `gamma` y `beta`
  - Parámetros: Verifica formas de estadísticas móviles y listas de parámetros

##### `test_batchnorm2d.py`

- **Propósito**: Verifica la capa `BatchNorm2d` para normalización por lotes en entradas 2D convolucionales (4D)
- **Pruebas**:
  - `test_batchnorm2d_forward_train_mode()`: Verifica forward en modo entrenamiento para datos 4D (normalización por canal sobre dimensiones espaciales)
  - `test_batchnorm2d_forward_eval_mode()`: Verifica forward en modo evaluación con estadísticas móviles
  - `test_batchnorm2d_backward_gradient_check()`: Verifica gradientes de `gamma` y `beta` con gradientes numéricos
  - `test_batchnorm2d_momentum_and_eps()`: Verifica parámetros de momentum y épsilon
  - `test_batchnorm2d_different_spatial_sizes()`: Verifica comportamiento con diferentes tamaños espaciales
  - `test_batchnorm2d_parameters()`: Verifica método `parameters()`
- **Metodología**:
  - Modo entrenamiento: Verifica media ≈0 y varianza ≈1 por canal (reducción sobre ejes batch, height, width)
  - Modo evaluación: Verifica uso de estadísticas móviles sin actualización
  - Gradientes: Compara gradientes analíticos de `gamma` y `beta` con aproximaciones numéricas
  - Tamaños espaciales: Prueba con diferentes alturas y anchos, verificando conservación de forma
  - Estadísticas móviles: Verifica formas `(1, C, 1, 1)` para broadcasting

#### `📂 tests/📂 layers/📂 conv/`

**Tests para capas convolucionales 1D y 2D**

##### `test_conv1d.py`

- **Propósito**: Verificar la capa `Conv1d` (convolución 1D para procesamiento de secuencias)
- **Pruebas principales**:
  - `test_conv1d_forward_shape()`: Verifica la forma de salida en forward pass con diferentes configuraciones
  - `test_conv1d_forward_no_bias()`: Verifica forward sin término de bias
  - `test_conv1d_backward_gradient_check()`: Verifica gradientes de pesos y bias mediante comparación con gradientes numéricos
  - `test_conv1d_padding_modes()`: Prueba diferentes modos de padding (zeros, reflect, replicate, circular)
  - `test_conv1d_parameters()`: Verifica que el método `parameters()` retorne los parámetros correctos
- **Metodología**:
  - Usa RNG determinístico para reproducibilidad
  - Calcula formas esperadas usando fórmulas: $L_{out} = \lfloor\frac{L_{in} + 2 \times \text{padding} - K}{\text{stride}}\rfloor + 1$
  - Para verificación de gradientes: compara gradientes analíticos (`layer.weight.grad`, `layer.bias.grad`) con aproximaciones numéricas usando `numeric_grad_wrt_param`
  - Tolerancia `THRESHOLD=5e-3` para diferencias máximas

##### `test_conv2d.py`

- **Propósito**: Verificar la capa `Conv2d` (convolución 2D para procesamiento de imágenes)
- **Pruebas principales**:
  - `test_conv2d_forward_shape()`: Verifica formas de salida en 4D
  - `test_conv2d_forward_no_bias()`: Verifica forward sin bias
  - `test_conv2d_backward_gradient_check_small()`: Verifica gradientes con entradas pequeñas para eficiencia
  - `test_conv2d_different_kernel_stride_padding()`: Prueba combinaciones de kernel, stride y padding (incluyendo tuplas para dimensiones separadas)
  - `test_conv2d_padding_modes()`: Prueba diferentes modos de padding
  - `test_conv2d_parameters()`: Verifica método `parameters()`
- **Metodología**:
  - Calcula dimensiones esperadas: $H_{out} = \lfloor\frac{H_{in} + 2 \times p_h - K_h}{s_h}\rfloor + 1$, similar para ancho
  - Verificación de gradientes con entradas reducidas (`6x6`) para mantener tiempos de ejecución manejables
  - Misma tolerancia `THRESHOLD=5e-3` para comparaciones
  - Soporte para configuraciones asimétricas (kernels, strides, paddings como tuplas)

#### `📂 tests/📂 layers/📂 linear/`

**Tests para capas lineales (fully connected)**

##### `test_linear.py`

- **Propósito**: Verificar la capa `Linear` (transformación lineal completamente conectada)
- **Pruebas principales**:
  - `test_linear_forward_shape()`: Verifica forma de salida `(batch, out_features)`
  - `test_linear_forward_no_bias()`: Verifica forward sin término de bias
  - `test_linear_backward_gradient_check()`: Verifica gradientes de pesos y bias con gradientes numéricos
- **Metodología**:
  - Usa RNG determinístico
  - Verifica formas y tipos de datos (`dtype=np.float32`)
  - Compara gradientes analíticos vs numéricos usando `numeric_grad_wrt_param` para ambos parámetros (weight, bias)
  - Tolerancia `THRESHOLD=5e-3`

#### `📂 tests/📂 layers/📂 pooling/`

**Tests para capas de pooling (reducción dimensional)**

##### `tests/layers/pooling/gap/`

**Tests para Global Average Pooling**

##### `test_gap1d.py`

- **Propósito**: Verificar la capa `GlobalAvgPool1d` (pooling global promedio en 1D)
- **Pruebas principales**:
  - `test_global_avg_pool1d_forward_shape()`: Verifica que colapse dimensión de longitud a 1
  - `test_global_avg_pool1d_forward_values()`: Verifica cálculo correcto del promedio con valores constantes
  - `test_global_avg_pool1d_backward_gradient()`: Verifica gradiente con comparación numérica
  - `test_global_avg_pool1d_uniform_gradient()`: Verifica distribución uniforme del gradiente (cada elemento recibe $1/L$)
- **Metodología**:
  - Forward: verifica forma `(batch, channels, 1)` y valores de promedio
  - Backward: usa `numeric_grad_scalar_wrt_x` para comparación numérica
  - Distribución uniforme: verifica que gradiente sea $1/L$ donde $L$ es la longitud original

##### `test_gap2d.py`

- **Propósito**: Verificar la capa `GlobalAvgPool2d` (pooling global promedio en 2D)
- **Pruebas principales**:
  - `test_global_avg_pool2d_forward_shape()`: Verifica colapso de dimensiones espaciales a `1x1`
  - `test_global_avg_pool2d_forward_values()`: Verifica cálculo de promedio con valores constantes
  - `test_global_avg_pool2d_backward_gradient()`: Verifica gradiente con comparación numérica
  - `test_global_avg_pool2d_uniform_gradient()`: Verifica distribución uniforme del gradiente (cada elemento recibe $1/(H \times W)$)
- **Metodología**:
  - Similar a `test_gap1d.py` pero para 4D tensores
  - Verifica forma `(batch, channels, 1, 1)`
  - Distribución uniforme sobre área espacial

#### `📂 tests/📂 layers/📂 pooling/📂 maxpool/`

**Tests para Max Pooling**

##### `test_maxpooling1d.py`

- **Propósito**: Verificar la capa `MaxPool1d` (pooling máximo en 1D)
- **Pruebas principales**:
  - `test_maxpool1d_forward_shape()`: Verifica forma de salida con kernel=2, stride=2
  - `test_maxpool1d_forward_padding()`: Verifica forma con padding
  - `test_maxpool1d_backward_gradient()`: Verifica gradiente con comparación numérica
  - `test_maxpool1d_stride_different()`: Verifica con stride diferente al kernel
- **Metodología**:
  - Calcula dimensiones esperadas usando fórmula de convolución
  - Backward: comparación con `numeric_grad_scalar_wrt_x`
  - Tolerancia `THRESHOLD=5e-3`

##### `test_maxpooling2d.py`

- **Propósito**: Verificar la capa `MaxPool2d` (pooling máximo en 2D)
- **Pruebas principales**:
  - `test_maxpool2d_forward_shape()`: Verifica forma de salida con kernel=2, stride=2
  - `test_maxpool2d_forward_padding()`: Verifica forma con padding
  - `test_maxpool2d_backward_gradient()`: Verifica gradiente con comparación numérica
- **Metodología**:
  - Similar a `test_maxpooling1d.py` pero para 2D
  - Verifica formas 4D
  - Misma tolerancia para comparación de gradientes

#### `📂 tests/📂 layers/📂 regularization/`

**Tests para capas de regularización**

##### `test_dropout.py`

- **Propósito**: Verificar la capa `Dropout` (regularización por apagado aleatorio de neuronas)
- **Pruebas principales**:
  - `test_dropout_eval_mode()`: Verifica que en modo evaluación no se aplique dropout (la entrada pasa sin cambios)
  - `test_dropout_train_mode()`: Verifica que en modo entrenamiento se aplique máscara aleatoria y escalado correcto
  - `test_dropout_zero_probability()`: Verifica que probabilidades inválidas (p=0.0) levanten `ValueError`
- **Metodología**:
  - Modo evaluación: Comprueba que entrada y salida sean idénticas, y que los gradientes pasen sin cambios
  - Modo entrenamiento: Verifica que aproximadamente `(1-p)` fracción de elementos se conserven, que los valores conservados escalen por `1/(1-p)`, y que los gradientes se enmascaren y escalen de la misma manera
  - Validación de parámetros: Comprueba que solo se acepten probabilidades en el rango `(0, 1)`
- **Detalles**:
  - Usa tensores de prueba grandes (`100x100`) para obtener estadísticas confiables
  - Tolerancia del 5% para variación aleatoria en la proporción de elementos conservados
  - Verifica coherencia entre forward y backward (misma máscara y escalado)

#### `📂 tests/📂 optimizers/`

**Tests para optimizadores**

##### `test_adam.py`

- **Propósito**: Verificar el optimizador `Adam` (Adaptive Moment Estimation)
- **Pruebas principales**:
  - `test_adam_basic_update()`: Verifica que Adam actualice parámetros de una capa `Linear`
  - `test_adam_with_conv_layer()`: Verifica que Adam funcione con capas convolucionales
  - `test_adam_bias_correction()`: Verifica el mecanismo de corrección de bias en pasos tempranos
- **Metodología**:
  - Comprueba que los parámetros cambien después de `step()`
  - Verifica que el contador de pasos (`t`) se incremente
  - Para la corrección de bias, ejecuta múltiples pasos y verifica que todas las actualizaciones sean no nulas
  - Usa capas reales (`Linear`, `Conv2d`) con forward/backward simulados
- **Integración**: Depende de `Adam` de `novann/optim/` y de capas del framework

##### `test_adamw.py`

- **Propósito**: Verificar el optimizador `AdamW` (Adam con weight decay desacoplado)
- **Pruebas principales**:
  - `test_adamw_updates_parameters()`: Verifica que AdamW actualice parámetros correctamente y que el contador de pasos (`t`) se incremente
  - `test_adamw_decoupled_weight_decay()`: Verifica que el weight decay se aplique **separadamente** de la actualización del gradiente (característica distintiva de AdamW vs Adam)
  - `test_adamw_excludes_batchnorm_from_weight_decay()`: Verifica que AdamW **no** aplique weight decay a parámetros `gamma` y `beta` de BatchNorm
- **Metodología**:
  - **Actualización básica**: Genera gradientes sintéticos, ejecuta `step()` y verifica cambios en parámetros
  - **Weight decay desacoplado**: Compara dos modelos idénticos (uno con `weight_decay=0.5`, otro con `weight_decay=0.0`) tras un paso de optimización. Verifica que la magnitud de actualización con decay sea **menor** que sin decay, confirmando el efecto de regularización desacoplada
  - **Exclusión de BatchNorm**: Crea un modelo con capas `Conv2d` (debe recibir decay) y `BatchNorm2d` (no debe recibir decay). Asigna nombres `"gamma"` y `"beta"` a los parámetros de BatchNorm. Tras `step()`, verifica que:
    - Los pesos de Conv cambien (gradiente + weight decay)
    - Los parámetros de BatchNorm cambien solo por el gradiente (sin amplificación de decay)
- **Integración**: Depende de `AdamW` de `novann/optim/` y de capas `Linear`, `Conv2d`, `BatchNorm2d` del framework

##### `test_rmsprop.py`

- **Propósito**: Verificar el optimizador `RMSprop` (Root Mean Square Propagation)
- **Pruebas principales**:
  - `test_rmsprop_basic_update()`: Verifica actualización básica de parámetros
  - `test_rmsprop_with_weight_decay()`: Verifica el efecto de weight decay (L2) en la magnitud de parámetros
  - `test_rmsprop_zero_grad()`: Verifica que `zero_grad()` limpie los gradientes
- **Metodología**:
  - Compara parámetros antes y después de `step()` para confirmar actualización
  - Para weight decay: compara dos modelos idénticos (con y sin decay) tras un paso de optimización
  - Para `zero_grad()`: verifica que todos los gradientes se pongan a cero
- **Nota**: El test de weight decay actualmente verifica que las normas sean iguales (con tolerancia), lo cual podría refinarse para verificar que la norma con decay sea menor.

##### `test_sgd.py`

- **Propósito**: Verificar el optimizador `SGD` (Stochastic Gradient Descent) con momentum y gradient clipping
- **Pruebas principales**:
  - `test_sgd_basic_update()`: Verifica actualización básica en un modelo `Sequential` con múltiples capas
  - `test_sgd_with_momentum()`: Verifica el efecto de momentum en actualizaciones consecutivas
  - `test_sgd_gradient_clipping()`: Verifica que los gradientes se recorten correctamente al `max_grad_norm` especificado mediante clipping global
  - `test_sgd_zero_grad()`: Verifica que `zero_grad()` limpie gradientes
- **Metodología**:
  - Usa un modelo `Sequential` con dos capas `Linear` para prueba integral
  - Para momentum: ejecuta dos pasos con el mismo gradiente y verifica que el segundo paso tenga mayor magnitud (acumulación de velocidad)
  - **Para gradient clipping**: Crea un gradiente artificialmente grande (`100.0` en todos los elementos), configura `max_grad_norm=1.0`, ejecuta `step()` y verifica que la norma L2 del gradiente resultante sea aproximadamente `1.0` (dentro de tolerancia `1e-5`), confirmando que el clipping global funcionó correctamente
  - Para `zero_grad()`: verifica que gradientes existan antes y sean cero después
  
#### `📂 tests/📂 sequential/`

**Tests para el contenedor Sequential (apilado de capas)**

##### `test_sequential.py`

- **Propósito**: Verificar el contenedor `Sequential`, que permite apilar múltiples capas y ejecutarlas en secuencia, tanto en forward como en backward pass, incluyendo manejo de modos (train/eval) y utilidades de inicialización.
- **Pruebas principales**:
  - `test_sequential_linear_activation()`: Verifica secuencias con capas lineales y funciones de activación variadas (ReLU, LeakyReLU, Sigmoid, Tanh), comprobando formas de salida y rangos esperados.
  - `test_sequential_conv_pooling()`: Verifica secuencias con capas convolucionales (Conv1d, Conv2d) y de pooling (MaxPool, GlobalAvgPool) para procesamiento 1D y 2D.
  - `test_sequential_mixed_layers()`: Verifica secuencias complejas con mezcla de capas (Conv, Dropout, Flatten, Linear, Softmax) y comportamiento diferenciado en modos train vs eval.
  - `test_sequential_backward()`: Verifica la propagación backward completa a través de múltiples capas, comprobando formas de gradientes y existencia de gradientes en todos los parámetros.
  - `test_sequential_initialization_helpers()`: Verifica métodos internos `_find_next_activation` y `_find_last_activation` usados para inicialización inteligente de pesos.
  - `test_sequential_parameters_and_zero_grad()`: Verifica que `parameters()` retorne todos los parámetros de las capas contenidas y que `zero_grad()` limpie correctamente los gradientes.
- **Metodología**:
  - Crea modelos `Sequential` con arquitecturas variadas (MLP, CNN).
  - En forward: pasa tensores de entrada sintéticos y verifica formas, rangos de salida y propiedades (ej. suma a 1 con Softmax).
  - En backward: calcula gradientes respecto a salidas aleatorias y verifica propagación correcta a través de todas las capas.
  - Modos train/eval: alterna entre modos y verifica comportamientos específicos (ej. Dropout activo solo en train).
  - Parámetros: cuenta y verifica acceso a todos los `weight` y `bias` de capas internas.
  - Utilidades de inicialización: simula búsqueda de funciones de activación adyacentes a capas lineales para inicialización adecuada (Kaiming/Xavier).