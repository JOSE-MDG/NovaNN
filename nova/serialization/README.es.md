# Módulo `serialization`

El directorio **`serialization/`** implementa funcionalidades para **guardar y cargar objetos de NovaNN** de forma segura y reproducible.

Este módulo permite persistir modelos, pesos, estados de optimizadores y cualquier objeto serializable, utilizando pickle como backend pero con capas adicionales de seguridad para prevenir la ejecución de código arbitrario durante la deserialización.

## Estructura general

El módulo está organizado en:

- **`save.py`**: Función pública para guardar objetos
- **`load.py`**: Función pública para cargar objetos de forma segura
- **`_safe_load.py`**: Unpickler restringido para deserialización segura

## Archivos principales

### `save.py`

Implementa la función **`save()`**, que serializa objetos NovaNN a disco o buffers.

**Características:**

- **Serialización con pickle**: Usa el protocolo pickle de Python
- **Soporte multi-target**: Guarda a archivos (str/Path) o buffers (BytesIO)
- **Creación automática de directorios**: Si la ruta padre no existe, la crea
- **Logging**: Reporta éxito o errores durante el guardado
- **Manejo de errores**: Captura y reporta errores específicos (permisos, IO, pickle)

**Firma:**

```python
def save(
    obj: Any,
    f: str | Path | io.BufferedIOBase,
    protocol: int = pickle.HIGHEST_PROTOCOL
) -> None
```

**Parámetros:**

- `obj`: Objeto a serializar (Module, Tensor, state_dict, etc.)
- `f`: Ruta de archivo o buffer donde guardar
- `protocol`: Versión del protocolo pickle (por defecto usa el más reciente)

**Ejemplos de uso:**

```python
import nova
import nova.nn as nn
from pathlib import Path
import io

# 1. Guardar un modelo completo
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
nova.save(model, "model.pth")
# ✅ Saved successfully to model.pth

# 2. Guardar solo los pesos (state_dict)
state = model.state_dict()
nova.save(state, "weights.pth")

# 3. Guardar con Path object
checkpoint_dir = Path("checkpoints")
nova.save(model, checkpoint_dir / "epoch_10.pth")

# 4. Guardar a un buffer en memoria
buffer = io.BytesIO()
nova.save(model, buffer)
# Útil para enviar por red o almacenar en base de datos
bytes_data = buffer.getvalue()

# 5. Guardar estado completo de entrenamiento
checkpoint = {
    'epoch': 42,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    'scheduler_state': scheduler.state_dict(),
    'loss': 0.123
}
nova.save(checkpoint, "training_checkpoint.pth")

# 6. Usar protocolo pickle específico
nova.save(model, "model_v4.pth", protocol=4)
```

**Excepciones:**

- `SaveError`: Error genérico durante serialización
- `PermissionError`: Sin permisos de escritura
- `TypeError`: Argumento `f` inválido

**Cuándo usar `save()`:**

- Guardar modelos entrenados para uso posterior
- Crear checkpoints durante entrenamiento para reanudar más tarde
- Exportar pesos para transferencia de aprendizaje
- Persistir configuraciones experimentales

### `load.py`

Implementa la función **`load()`**, que deserializa objetos de forma **segura por defecto**.

**Características principales:**

- **Carga segura por defecto**: `weights_only=True` previene ejecución de código arbitrario
- **Unpickler restringido**: Solo permite clases registradas explícitamente
- **Fallback inseguro opcional**: `weights_only=False` para compatibilidad (no recomendado)
- **Soporte multi-source**: Carga desde archivos o buffers
- **Validación de rutas**: Verifica existencia antes de intentar cargar
- **Logging detallado**: Reporta éxito, advertencias y errores

**Firma:**

```python
def load(
    f: str | Path | io.BufferedIOBase,
    *,
    weights_only: bool = True
) -> Any
```

**Parámetros:**

- `f`: Ruta de archivo o buffer desde donde cargar
- `weights_only`: Si True, usa unpickler seguro (recomendado). Si False, usa pickle estándar (riesgo de seguridad)

**Ejemplos de uso:**

```python
import nova
from pathlib import Path
import io

# 1. Cargar un modelo guardado (seguro por defecto)
model = nova.load("model.pth")
# ✅ Successfully loaded from model.pth

# 2. Cargar state_dict y aplicarlo a un modelo
new_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
state = nova.load("weights.pth")
new_model.load_state_dict(state)

# 3. Cargar con Path object
checkpoint_path = Path("checkpoints/epoch_10.pth")
model = nova.load(checkpoint_path)

# 4. Cargar desde buffer en memoria
buffer = io.BytesIO(saved_bytes)
buffer.seek(0)  # Importante: volver al inicio
model = nova.load(buffer)

# 5. Cargar checkpoint completo de entrenamiento
checkpoint = nova.load("training_checkpoint.pth")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scheduler.load_state_dict(checkpoint['scheduler_state'])
start_epoch = checkpoint['epoch'] + 1
print(f"Resuming from epoch {start_epoch}")

# 6. Carga insegura (NO RECOMENDADO)
# Solo usar si confías completamente en la fuente
model = nova.load("model.pth", weights_only=False)
# ⚠️ Warning: Loading with weights_only=False is unsafe
```

**Excepciones:**

- `FileNotFoundError`: Archivo no existe
- `LoadError`: Error genérico durante deserialización
- `UnsafeLoadError`: Intento de cargar clase no registrada con `weights_only=True`
- `TypeError`: Argumento `f` inválido

**Cuándo usar `weights_only=True` (por defecto):**

- Cargar pesos de fuentes externas o no confiables
- Entornos de producción donde la seguridad es crítica
- Cuando solo necesitas cargar state_dicts o módulos registrados

**Cuándo usar `weights_only=False` (riesgo de seguridad):**

- Cargar objetos Python arbitrarios que confías completamente
- Debugging o desarrollo donde controlas la fuente
- Compatibilidad con checkpoints antiguos que contienen clases no registradas

**NUNCA uses `weights_only=False` con archivos de fuentes desconocidas o no confiables.**

### `_safe_load.py`

Implementa **`SafeUnpickler`**, un unpickler restringido que previene ejecución de código arbitrario.

**Propósito:**

La deserialización con pickle puede ejecutar código malicioso si un archivo contiene instrucciones para instanciar clases arbitrarias. `SafeUnpickler` resuelve esto implementando una lista blanca (allowlist) de módulos y clases permitidos.

**Listas blancas (allowlists):**

**Módulos permitidos:**

```python
ALLOWED_MODULES = {
    "numpy",
    "numpy.core.multiarray",
    "numpy.core.numeric",
    "numpy._core.numeric",
    "numpy._core.multiarray",
    "nova.dtypes",
}
```

**Tipos built-in permitidos:**

```python
ALLOWED_BUILTINS = {
    "dict", "list", "tuple", "set", "frozenset",
    "int", "float", "str", "bytes", "bool",
    "complex", "bytearray", "range", "slice",
    "type", "object", "NoneType"
}
```

**Clases NovaNN registradas:**

Cualquier clase decorada con `@registry_class` se permite automáticamente. Esto incluye:

- Tensores
- Todos los módulos de `nn` (Linear, Conv2d, ReLU, etc.)
- Optimizadores (SGD, Adam, AdamW, RMSprop)
- Schedulers (StepLR, CosineAnnealingLR, OneCycleLR)
- Funciones del autograd
- Métricas

**Método clave:**

```python
def find_class(self, module_name: str, global_name: str)
```

Este método se llama durante la deserialización para resolver cada clase. Implementa la lógica de allowlist:

1. Verifica si el módulo está en `ALLOWED_MODULES`
2. Permite internals específicos de NumPy necesarios para arrays
3. Permite tipos built-in seguros
4. Permite `OrderedDict` (usado en state_dicts)
5. Busca en el registro de clases NovaNN (`@registry_class`)
6. **Bloquea todo lo demás** y lanza `UnsafeLoadError`

**Ejemplo de clase registrada:**

```python
from nova.utils import registry_class

@registry_class
class MyCustomLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = Parameter(nova.randn(10, 10))

    def forward(self, x):
        return x @ self.weight

# Ahora MyCustomLayer puede ser cargada de forma segura
model = MyCustomLayer()
nova.save(model, "custom.pth")

# Esto funciona porque MyCustomLayer está registrada
loaded = nova.load("custom.pth", weights_only=True)  # ✅ OK
```

**Ejemplo de clase NO registrada:**

```python
# Sin @registry_class
class UnsafeLayer:
    def __init__(self):
        super().__init__()
        self.data = [1, 2, 3]

model = UnsafeLayer()
nova.save(model, "unsafe.pth")

# Esto falla porque UnsafeLayer no está registrada
try:
    loaded = nova.load("unsafe.pth", weights_only=True)
except UnsafeLoadError as e:
    print(e)
    # Blocked unpickling of unregistered class: __main__.UnsafeLayer
    # To fix this, either:
    #   1. Register the class using @registry_class decorator
    #   2. Load with weights_only=False (not recommended)
```

**Función auxiliar:**

```python
def _load_from_file(file: io.BufferedIOBase, weights_only: bool = True) -> Any
```

Helper interno usado por `load()` que decide entre `SafeUnpickler` y `pickle.load()` estándar según `weights_only`.

## Flujo de serialización completo

### Guardado (save):

1. Usuario llama `nova.save(obj, "model.pth")`
2. Se valida que `obj` no sea None
3. Se crea el directorio padre si no existe
4. Se serializa con `pickle.dump(obj, file, protocol)`
5. Se registra el éxito en logs

### Carga segura (load con weights_only=True):

1. Usuario llama `nova.load("model.pth")`
2. Se verifica que el archivo exista
3. Se abre el archivo en modo binario
4. Se crea instancia de `SafeUnpickler(file)`
5. Durante `unpickler.load()`:
   - Cada clase se valida contra las allowlists
   - Solo se permiten NumPy, builtins y clases registradas
   - Se bloquean clases arbitrarias
6. Se retorna el objeto deserializado

### Carga insegura (load con weights_only=False):

1. Usuario llama `nova.load("model.pth", weights_only=False)`
2. Se muestra advertencia de seguridad en logs
3. Se usa `pickle.load()` estándar (sin restricciones)
4. **Cualquier clase puede ser deserializada** (riesgo de seguridad)

## Integración con otros módulos

El módulo `serialization` se integra con:

- **[`nn/`](../nn/README.es.md)**: Todos los módulos soportan `state_dict()` / `load_state_dict()`
- **[`optim/`](../optim/README.es.md)**: Optimizadores y schedulers son serializables
- **[`utils/decorators/registry.py`](../utils/README.es.md)**: El decorador `@registry_class` registra clases para carga segura
- **`_tensor.py`**: Tensors son serializables directamente

## Diseño y filosofía

El módulo `serialization` sigue estos principios:

- **Seguridad por defecto**: `weights_only=True` previene ataques de deserialización
- **Opt-in explícito para riesgos**: `weights_only=False` requiere decisión consciente del usuario
- **Registro explícito**: Solo clases decoradas con `@registry_class` son confiables
- **Compatibilidad**: Soporta tanto rutas como buffers para flexibilidad
- **Logging claro**: Reporta éxitos, advertencias y errores de forma descriptiva
- **Separación de concerns**: `save` no conoce detalles de seguridad, `_safe_load` se encarga de eso

## Ejemplos avanzados

### Ejemplo 1: Checkpoint periódico durante entrenamiento

```python
import nova
import nova.nn as nn
from nova.optim import Adam
from nova.optim.lr_scheduler import CosineAnnealingLR

model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

best_loss = float('inf')

for epoch in range(100):
    # ... training loop ...

    # Guardar checkpoint cada 10 épocas
    if epoch % 10 == 0:
        checkpoint = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'loss': current_loss,
        }
        nova.save(checkpoint, f"checkpoints/epoch_{epoch}.pth")

    # Guardar mejor modelo
    if current_loss < best_loss:
        best_loss = current_loss
        nova.save(model.state_dict(), "best_model.pth")

print("Training completed!")
```

### Ejemplo 2: Reanudar entrenamiento desde checkpoint

```python
import nova
import nova.nn as nn
from nova.optim import Adam
from nova.optim.lr_scheduler import CosineAnnealingLR

# Crear modelo y optimizer
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# Cargar checkpoint
checkpoint = nova.load("checkpoints/epoch_40.pth")
model.load_state_dict(checkpoint['model_state'])
optimizer.load_state_dict(checkpoint['optimizer_state'])
scheduler.load_state_dict(checkpoint['scheduler_state'])
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['loss']

print(f"Resuming training from epoch {start_epoch}")

# Continuar entrenamiento
for epoch in range(start_epoch, 100):
    # ... training loop ...
    pass
```

### Ejemplo 3: Transferencia de aprendizaje (cargar pesos parciales)

```python
import nova
import nova.nn as nn

# Modelo preentrenado en ImageNet (simplificado)
pretrained = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Linear(128, 1000)  # 1000 clases ImageNet
)

# Guardar modelo preentrenado
nova.save(pretrained.state_dict(), "imagenet_weights.pth")

# Nuevo modelo para 10 clases (tu dataset)
model = nn.Sequential(
    nn.Conv2d(3, 64, 3),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3),
    nn.ReLU(),
    nn.Linear(128, 10)  # Solo 10 clases
)

# Cargar pesos parciales (solo capas convolucionales)
pretrained_weights = nova.load("imagenet_weights.pth")
model_weights = model.state_dict()

# Copiar solo capas compatibles
for name, param in pretrained_weights.items():
    if name in model_weights and param.shape == model_weights[name].shape:
        model_weights[name] = param

model.load_state_dict(model_weights)
print("Transferred convolutional layers from pretrained model!")
```

### Ejemplo 4: Enviar modelo por red

```python
import nova
import nova.nn as nn
import io
import socket

# Servidor: serializar y enviar
model = nn.Sequential(nn.Linear(10, 5), nn.ReLU())
buffer = io.BytesIO()
nova.save(model, buffer)
model_bytes = buffer.getvalue()

# Enviar por socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 12345))
sock.sendall(model_bytes)
sock.close()

# Cliente: recibir y deserializar
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(('localhost', 12345))
sock.listen(1)
conn, addr = sock.accept()
received_bytes = conn.recv(4096)

buffer = io.BytesIO(received_bytes)
buffer.seek(0)
model = nova.load(buffer)
print("Model received and loaded!")
```

### Ejemplo 5: Registro de clase personalizada para carga segura

```python
from nova.utils import registry_class
import nova.nn as nn

# Definir clase personalizada con registro
@registry_class # En si no hace falta por se registra automaticamente al heredar de nn.Module
class CustomAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = (Q @ K.T) / (Q.shape[-1] ** 0.5)
        return attn @ V

# Crear y guardar modelo con clase personalizada
model = nn.Sequential(
    CustomAttention(512),
    nn.ReLU(),
    nn.Linear(512, 10)
)
nova.save(model, "custom_model.pth")

# Cargar de forma segura (funciona porque CustomAttention está registrada)
loaded_model = nova.load("custom_model.pth", weights_only=True)  # ✅ OK
print("Custom model loaded safely!")
```

---

> Para más detalles sobre el sistema de registro de clases, consulta [`utils/decorators/registry.py`](../utils/README.es.md).
