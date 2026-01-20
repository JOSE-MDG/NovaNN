# Módulo `_typing`

El directorio **`_typing/`** contiene definiciones de **tipos, protocolos y estructuras de datos** que se utilizan en todo NovaNN.  
Estas definiciones ayudan a **tipar correctamente tensores, operaciones, optimizadores, schedulers, datasets y módulos**, mejorando la seguridad de tipo y la autocompletación en editores.

## Tipos principales

- **Core Tensor Types**: `Size`, `Dtype`, `Dim`, `TensorOrArray`, `Inputs`.  
  Representan formas, tipos de datos y entradas flexibles para operaciones tensoriales.

- **Autograd Types**: `Hook`, `StepHook`, `HooksList`, `Closure`.  
  Facilitan la tipificación de gradientes, hooks de backward y closures para optimizadores.

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

- **Loss Function Types**: `LossReducton`.  
  Tipos que definen modos de reducción de funciones de pérdida.

- **YAML Configuration Types**: `InplaceInfo`, `TensorInfo`, `OperationInfo`, `YAMLFile`.  
  Tipos que describen la estructura del archivo YAML de operaciones nativas, incluyendo métodos in-place y dunder.

- **Binding Types**: `UnaryMethod`, `BinaryMethod`, `ReverseBinaryMethod`, `VariadicMethod`, `InplaceUnaryMethod`, `InplaceBinaryMethod`, `InplaceVariadicMethod`.  
  Protocolos que definen la firma de los métodos generados dinámicamente para tensores, ya sea unary, binary, reverse o in-place.

---

> `_typing/` proporciona la **infraestructura de tipado y contratos** que asegura consistencia entre operaciones, métodos de Tensor, optimizadores, módulos y datasets, sin afectar la API pública.
