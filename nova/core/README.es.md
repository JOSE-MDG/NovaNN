# Módulo `core`

El módulo **`core/`** contiene la configuración central y constantes del framework NovaNN. Gestiona variables de entorno, rutas de datasets y configuración del sistema de logging.

## Estructura

```
core/
├── __init__.py      # Exporta todas las constantes y configuración
└── constants.py     # Define constantes y carga variables de entorno
```

## Archivos

### `constants.py`

Carga variables de entorno usando `dotenv` y expone constantes configurables para el framework.

**Variables de entorno - Datasets:**

- **Fashion-MNIST:**
  - `FASHION_TRAIN_DATA_PATH`: Ruta al conjunto de entrenamiento
  - `EXPORTATION_FASHION_TRAIN_DATA_PATH`: Ruta para exportación de datos procesados
  - `FASHION_TEST_DATA_PATH`: Ruta al conjunto de prueba
  - `FASHION_VALIDATION_DATA_PATH`: Ruta al conjunto de validación

- **MNIST:**
  - `MNIST_TRAIN_DATA_PATH`: Ruta al conjunto de entrenamiento
  - `EXPORTATION_MNIST_TRAIN_DATA_PATH`: Ruta para exportación de datos procesados
  - `MNIST_TEST_DATA_PATH`: Ruta al conjunto de prueba
  - `MNIST_VALIDATION_DATA_PATH`: Ruta al conjunto de validación

**Variables de entorno - Logging:**

- `LOG_FILE`: Ruta al archivo de logs
- `LOGGER_DEFAULT_FORMAT`: Formato de los mensajes de log
- `LOGGER_DEFAULT_LEVEL`: Nivel de logging por defecto (DEBUG, INFO, WARNING, ERROR)
- `LOGGER_DATE_FORMAT`: Formato de las fechas en los logs

**Variables de entorno - Sistema:**

- `NATIVE_YAML`: Ruta al archivo `native_functions.yaml` para el sistema de binding

### `__init__.py`

Exporta todas las constantes definidas en `constants.py`, proporcionando una API limpia para acceder a la configuración desde cualquier parte del framework.

## Uso

### Configuración con archivo `.env`

Crea un archivo `.env` en la raíz del proyecto:

```bash
# Datasets
FASHION_TRAIN_DATA_PATH=/path/to/fashion_mnist/train
FASHION_TEST_DATA_PATH=/path/to/fashion_mnist/test
MNIST_TRAIN_DATA_PATH=/path/to/mnist/train
MNIST_TEST_DATA_PATH=/path/to/mnist/test

# Logging
LOG_FILE=logs/nova.log
LOGGER_DEFAULT_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s
LOGGER_DEFAULT_LEVEL=INFO
LOGGER_DATE_FORMAT=%Y-%m-%d %H:%M:%S

# Sistema
NATIVE_YAML=nova/autograd/_ops/native/native_functions.yaml
```

### Acceso a constantes

```python
from nova.core import (
    MNIST_TRAIN_DATA_PATH,
    LOG_FILE,
    LOGGER_DEFAULT_LEVEL
)

# Usar rutas de datasets
train_path = MNIST_TRAIN_DATA_PATH

# Configurar logging
import logging
logging.basicConfig(
    filename=LOG_FILE,
    level=LOGGER_DEFAULT_LEVEL,
    format=LOGGER_DEFAULT_FORMAT
)
```

### Integración con otros módulos

El módulo `core` es utilizado internamente por:

- **`utils.logger`**: Lee configuración de logging
- **`_internal._binding`**: Lee `YAML_FILE_PATH` para cargar `native_functions.yaml`
- **Scripts de entrenamiento**: Acceden a rutas de datasets

## Diseño

El módulo `core` sigue estos principios:

- **Centralización**: Todas las constantes y configuración en un solo lugar
- **Flexibilidad**: Variables configurables vía `.env` sin modificar código
- **Type hints**: Todas las constantes tipadas como `Optional[str]`
- **Separación de concerns**: `constants.py` define, `__init__.py` exporta
- **Defaults seguros**: Variables pueden ser `None` si no están definidas

## Variables opcionales

Todas las constantes son `Optional[str]`, lo que permite:

```python
from nova.core import MNIST_TRAIN_DATA_PATH

if MNIST_TRAIN_DATA_PATH is None:
    print("⚠️ MNIST path not configured. Using default.")
    MNIST_TRAIN_DATA_PATH = "./data/mnist/train"
```

## Buenas prácticas

1. **No hardcodear rutas**: Siempre usar constantes de `core`
2. **Validar antes de usar**: Verificar que las rutas existan
3. **Documentar variables nuevas**: Añadir comentarios en `constants.py`
4. **Mantener `.env` local**: Nunca commitear `.env` al repositorio

---

> El módulo `core` es la base de configuración de NovaNN. Para añadir nuevas constantes, editarlas en `constants.py` y exportarlas en `__init__.py`.
