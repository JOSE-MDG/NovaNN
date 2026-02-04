# Módulo `_typing`

El directorio **`_typing/`** contiene definiciones de **tipos, protocolos y estructuras de datos** que se utilizan en todo NovaNN.  
Estas definiciones ayudan a **tipar correctamente tensores, operaciones, optimizadores, schedulers, datasets y módulos**, mejorando la seguridad de tipo y la autocompletación en editores.

## Tipos principales

- **Core Tensor Types**: `Size`, `Dtype`, `Dim`, `TensorOrArray`, `Inputs`, `Gradients`.  
  Representan formas, tipos de datos, entradas flexibles para operaciones tensoriales y tuplas de gradientes.

- **Autograd Types**: `Hook`, `StepHook`, `Hooks`, `HooksList`, `Closure`.  
  Facilitan la tipificación de gradientes, hooks de backward, hooks de optimizador y closures.

- **Dataset Types**: `Mnist`, `Fashion`.  
  Tipos para representar los datasets cargados por las utilidades de NovaNN.

- **Module and Parameter Types**: `Modules`, `ModuleTypes`.  
  Representan cualquier objeto tipo módulo, parámetro o buffer dentro del framework.

- **Convolution y Pooling Types**: `KernelSize`, `Stride`, `Dilation`, `Padding`, `PaddingMode`.  
  Tipos para operaciones de convolución y pooling en 1D, 2D o 3D.

- **Optimizer Types**: `Defaults`, `Group`, `ParamGroups`, `State`, `OptimizerStateDict`.  
  Tipos que tipan los optimizadores, grupos de parámetros y estados serializables.

- **Scheduler Types**: `SchedulerStateDict`.  
  Tipos para serializar estados de schedulers de tasa de aprendizaje.

- **Loss Function Types**: `LossReduction`.  
  Tipos que definen modos de reducción de funciones de pérdida.

- **Metrics Types**: `Average`.  
  Tipos para estrategias de promediado en métricas (micro, macro, weighted).

- **YAML Configuration Types**: `InplaceInfo`, `TensorInfo`, `OperationInfo`, `YAMLFile`.  
  Tipos que describen la estructura del archivo YAML de operaciones nativas, incluyendo métodos in-place y dunder.

- **Binding Types**: `UnaryMethod`, `BinaryMethod`, `ReverseBinaryMethod`, `VariadicMethod`, `InplaceUnaryMethod`, `InplaceBinaryMethod`, `InplaceVariadicMethod`.  
  Protocolos que definen la firma de los métodos generados dinámicamente para tensores, ya sea unary, binary, reverse o in-place.

## Descripciones Detalladas de Tipos

### Core Tensor Types

- `Size`: Tupla de enteros representando la forma del tensor
- `Dtype`: Unión de todos los tipos de datos numéricos soportados (uint8, int32, float32, etc.)
- `Dim`: Dimensión única o tupla de dimensiones para operaciones
- `TensorOrArray`: Entrada flexible que acepta tensores, arrays de numpy, o listas/tuplas de tensores
- `Inputs`: Tipo de entrada general para operaciones (tensores, escalares, o cualquier valor)
- `Gradients`: Tupla de arrays de gradientes o valores None de backward passes

### Autograd Types

- `Hook`: Función hook de backward que recibe y opcionalmente modifica gradientes
- `StepHook`: Hook de optimizador llamado después de actualizaciones de parámetros
- `Hooks`: Unión de todos los tipos de hooks
- `HooksList`: Lista conteniendo cualquier tipo de hooks
- `Closure`: Función closure opcional para optimizadores que re-evalúan modelos

### Dataset Types

- `Mnist`: Tipo de retorno para la función de carga del dataset MNIST (train, test, validation)
- `Fashion`: Tipo de retorno para la función de carga del dataset Fashion-MNIST

### Module and Parameter Types

- `Modules`: Unión de todos los objetos tipo módulo (Tensor, Optimizer, Parameter, Buffer, Module, etc.)
- `ModuleTypes`: Objetos tipo para clases tipo módulo

### Convolution y Pooling Types

- `KernelSize`: Entero o tupla para tamaños de kernel 1D, 2D o 3D
- `Stride`: Especificación opcional de stride para operaciones
- `Dilation`: Tasa de dilatación para convoluciones dilatadas
- `Padding`: Especificación de padding (entero, tupla, o modos 'valid'/'same')
- `PaddingMode`: Modo de relleno de padding ('zeros', 'reflect', 'replicate', 'circular')

### Optimizer Types

- `Defaults`: Diccionario de hiperparámetros por defecto
- `Group`: Grupo de parámetros con hiperparámetros asociados
- `ParamGroups`: Lista de grupos de parámetros
- `State`: Diccionario de estado del optimizador mapeando parámetros a su estado
- `OptimizerStateDict`: Estado completo del optimizador para serialización

### Scheduler Types

- `SchedulerStateDict`: Estado de serialización del scheduler de tasa de aprendizaje

### Loss Function Types

- `LossReduction`: Modos de reducción para funciones de pérdida ('none', 'mean', 'sum', 'batchmean')

### Metrics Types

- `Average`: Estrategias de promediado para métricas ('micro', 'macro', 'weighted', None)

### YAML Configuration Types

- `InplaceInfo`: Configuración para variantes de operaciones in-place
- `TensorInfo`: Configuración para binding de operaciones a la clase Tensor
- `OperationInfo`: Definición completa de operación para configuración YAML
- `YAMLFile`: Estructura raíz de native_functions.yaml

### Binding Types

- `UnaryMethod`: Protocolo para métodos unarios de tensor (`__neg__`, `__abs__`, `relu()`)
- `BinaryMethod`: Protocolo para métodos binarios de tensor (`__add__`, `__mul__`)
- `ReverseBinaryMethod`: Protocolo para métodos binarios reversos (`__radd__`, `__rmul__`)
- `VariadicMethod`: Protocolo para métodos variádicos (`sum(dim=...)`, `reshape(...)`)
- `InplaceUnaryMethod`: Protocolo para métodos unarios in-place (`abs_()`, `relu_()`)
- `InplaceBinaryMethod`: Protocolo para métodos binarios in-place (`add_()`, `mul_()`)
- `InplaceVariadicMethod`: Protocolo para métodos variádicos in-place (`clamp_()`)

---

> `_typing/` proporciona la **infraestructura de tipado y contratos** que asegura consistencia entre operaciones, métodos de Tensor, optimizadores, módulos y datasets, sin afectar la API pública. Todos los tipos están diseñados para funcionar perfectamente con los type checkers estáticos de Python y la autocompletación de IDEs.
