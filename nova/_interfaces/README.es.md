# Módulo `_interfaces`

El directorio **`_interfaces/`** define las **clases base abstractas** que establecen contratos y comportamientos compartidos para componentes clave del framework.

Estas interfaces no están pensadas para ser instanciadas directamente, sino para ser heredadas por implementaciones concretas, asegurando consistencia en la API y facilitando la extensibilidad del framework.

## Propósito

Las interfaces en NovaNN cumplen varios roles:

- **Contratos de comportamiento**: Definen qué métodos debe implementar cada componente
- **Tipado estático**: Mejoran la experiencia en IDEs y herramientas de análisis
- **Reutilización de código**: Implementan lógica compartida entre subclases
- **Documentación viva**: Sirven como especificación de la API esperada

## Archivos principales

### `_optimizer.py`

Define la clase base **`Optimizer`**, que es el contrato para todos los optimizadores en NovaNN.

**Responsabilidades:**

- **Gestión de parameter groups**: Permite aplicar diferentes hiperparámetros a distintos subconjuntos de parámetros
- **Estado del optimizador**: Mantiene momentum, estadísticas adaptativas, etc.
- **Sistema de hooks**: Soporta pre-step y post-step hooks para logging, gradient clipping, etc.
- **Serialización**: Guarda y carga el estado completo del optimizador

**Atributos principales:**

- `param_groups`: Lista de grupos de parámetros, cada uno con sus hiperparámetros
- `state`: Diccionario que mapea parámetros a su estado (momentum, velocidad, etc.)
- `defaults`: Hiperparámetros por defecto (lr, weight_decay, etc.)

**Métodos clave:**

- `_step_impl(closure)`: **Método abstracto** que debe ser implementado por subclases. Define la regla de actualización específica del optimizador.
- `step(closure)`: Ejecuta un paso de optimización completo (hooks → actualización → hooks)
- `zero_grad(set_to_none)`: Limpia gradientes de todos los parámetros
- `add_param_group(group)`: Añade un nuevo grupo de parámetros con hiperparámetros específicos
- `state_dict()` / `load_state_dict()`: Serialización del estado

**Ejemplo de uso (herencia):**

```python
from nova._interfaces._optimizer import Optimizer

class MyOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, custom_param=0.5):
        defaults = {'lr': lr, 'custom_param': custom_param}
        super().__init__(params, defaults)

    def _step_impl(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            custom = group['custom_param']

            for param in group['params']:
                if param.grad is None:
                    continue

                # Regla de actualización personalizada
                param.data -= lr * param.grad * custom

        return loss
```

**Cuándo heredar de `Optimizer`:**

- Al implementar un nuevo algoritmo de optimización (Adagrad, Adadelta, etc.)
- Cuando se necesita gestión automática de parameter groups
- Para aprovechar el sistema de hooks y serialización

### `_lr_scheduler.py`

Define la clase base **`_LRScheduler`**, que es el contrato para todos los planificadores de learning rate.

**Responsabilidades:**

- **Ajuste dinámico del LR**: Modifica la tasa de aprendizaje según un calendario predefinido
- **Sincronización con el optimizador**: Actualiza directamente los `param_groups` del optimizador
- **Tracking de estado**: Mantiene el registro de la época/paso actual
- **Serialización**: Permite guardar y restaurar el estado del scheduler

**Atributos principales:**

- `optimizer`: Referencia al optimizador cuyo LR se va a ajustar
- `last_epoch`: Índice de la última época/paso ejecutado
- `base_lrs`: Learning rates iniciales de cada parameter group

**Métodos clave:**

- `get_lr()`: **Método abstracto** que debe retornar una lista de learning rates (uno por parameter group)
- `step()`: Avanza el scheduler un paso y actualiza los LRs del optimizador
- `get_last_lr()`: Retorna los últimos learning rates aplicados
- `state_dict()` / `load_state_dict()`: Serialización del estado

**Ejemplo de uso (herencia):**

```python
from nova._interfaces._lr_scheduler import _LRScheduler

class LinearDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, last_epoch=-1):
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # Decaimiento lineal de base_lr a 0
        progress = self.last_epoch / self.total_steps
        factor = 1.0 - progress
        return [base_lr * factor for base_lr in self.base_lrs]

# Uso
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = LinearDecayLR(optimizer, total_steps=100)

for epoch in range(100):
    train(...)
    scheduler.step()  # LR decrece linealmente
```

**Cuándo heredar de `_LRScheduler`:**

- Al implementar una nueva estrategia de ajuste de learning rate
- Cuando se necesita sincronización automática con el optimizador
- Para aprovechar la serialización de estado

**Schedulers implementados en NovaNN:**

NovaNN incluye tres schedulers concretos en [`optim/lr_scheduler.py`](../optim/README.es.md):

- **StepLR**: Decae el LR cada N épocas por un factor gamma
- **CosineAnnealingLR**: Decae el LR siguiendo una curva coseno
- **OneCycleLR**: Implementa el método "1cycle" (aumenta y luego decrece el LR)

### `_base_tensor.py`

Define la clase base **`TensorBase`**, que proporciona propiedades y metadatos fundamentales para todos los tensores.

**Propósito:**

- Exponer una interfaz consistente para acceder a atributos del tensor
- Separar la lógica de metadatos de las operaciones matemáticas
- Proveer propiedades read-only y computed properties

**Propiedades principales:**

- `data`: Array de NumPy subyacente (getter/setter con validación)
- `shape`: Forma del tensor (tuple de dimensiones)
- `dtype`: Tipo de dato de los elementos
- `ndim` / `dim()`: Número de dimensiones
- `strides`: Strides en bytes de cada dimensión
- `T`: Transpuesta (alias de `permute()`)
- `is_leaf`: Si es un nodo hoja en el grafo computacional
- `device`: Siempre retorna `'cpu'` (NovaNN no aun soporta GPU)
- `is_cuda`: Siempre retorna `False`

**Métodos principales:**

- `size(dim)`: Retorna la forma completa o el tamaño de una dimensión específica
- `numel()`: Número total de elementos
- `itemsize`: Tamaño en bytes de cada elemento
- `nbytes`: Bytes totales consumidos

**Características:**

- Usa `__slots__ = []` para eficiencia de memoria
- El setter de `data` maneja conversión automática de arrays y tensors
- Soporta indexación negativa en `size(dim)`

**Ejemplo de implementación:**

```python
import nova

x = nova.randn(3, 4, 5)

# Propiedades de TensorBase
print(x.shape)      # (3, 4, 5)
print(x.dtype)      # float32
print(x.ndim)       # 3
print(x.numel())    # 60
print(x.size(0))    # 3
print(x.size(-1))   # 5
print(x.strides)    # Strides en bytes
print(x.T.shape)    # (5, 4, 3) - transpuesta
```

**Cuándo heredar de `TensorBase`:**

- Al implementar una nueva clase de tensor con comportamiento personalizado
- Para mantener consistencia con la API de propiedades del tensor
- Generalmente no se hereda directamente por usuarios finales

## Integración con otros módulos

El módulo `_interfaces` se integra con:

- **[`optim/`](../optim/README.es.md)**: Todos los optimizadores (SGD, Adam, AdamW, RMSprop) heredan de `Optimizer`
- **[`optim/lr_scheduler.py`](../optim/README.es.md)**: Todos los schedulers (StepLR, CosineAnnealingLR, OneCycleLR) heredan de `_LRScheduler`
- **`_tensor.py`**: La clase `Tensor` usa `TensorBase` como base para propiedades

## Diseño y filosofía

El módulo `_interfaces` sigue estos principios:

- **Contratos claros**: Métodos abstractos (`_step_impl`, `get_lr`) definen explícitamente qué debe implementarse
- **Funcionalidad compartida**: Métodos como `step()`, `zero_grad()`, `state_dict()` se implementan una vez
- **Extensibilidad**: Añadir nuevos optimizadores/schedulers solo requiere implementar el método abstracto
- **Tipado fuerte**: Usa type hints y stubs (`.pyi`) para mejor experiencia en IDEs
- **Registro automático**: Usa `@registry_class` para serialización segura

---

> Para implementaciones concretas de estas interfaces, consulta [`optim/`](../optim/README.es.md) para optimizadores y schedulers.
