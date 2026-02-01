# Módulo `core`

El módulo **`core/`** contiene la configuración central y las constantes del framework NovaNN. Gestiona las rutas de datasets, URLs de descarga y la configuración del sistema de logging.

## Estructura

```
core/
├── __init__.py      # Exporta todas las constantes y la configuración
└── constants.py     # Define las constantes usando pathlib
```

## Archivos

### `constants.py`

Define todas las constantes del framework usando `pathlib.Path`. Todas las rutas de datasets se resuelven en `~/.novann/datasets/`, garantizando una ubicación consistente independientemente del entorno.

**Estructura del proyecto:**

- `PROJECT_ROOT`: Ruta absoluta a la raíz del proyecto NovaNN, resuelta relativa a la ubicación de `constants.py`.
- `DATA_ROOT`: Directorio base para todos los datasets (`~/.novann/datasets/`).

**Rutas de datasets — MNIST:**

- `MNIST_DIR`: `~/.novann/datasets/Mnist/`
- `MNIST_TRAIN_DATA_PATH`: Ruta al conjunto de entrenamiento completo (`mnist_train.parquet`)
- `EXPORTATION_MNIST_TRAIN_DATA_PATH`: Ruta al conjunto de entrenamiento tras el split de validación (`mnist_train_e.parquet`)
- `MNIST_TEST_DATA_PATH`: Ruta al conjunto de prueba (`mnist_test.parquet`)
- `MNIST_VALIDATION_DATA_PATH`: Ruta al conjunto de validación (`mnist_validation.parquet`)

**Rutas de datasets — Fashion-MNIST:**

- `FASHION_DIR`: `~/.novann/datasets/FashionMnist/`
- `FASHION_TRAIN_DATA_PATH`: Ruta al conjunto de entrenamiento completo (`fashion-mnist_train.parquet`)
- `EXPORTATION_FASHION_TRAIN_DATA_PATH`: Ruta al conjunto de entrenamiento tras el split de validación (`fashion-mnist_train_e.parquet`)
- `FASHION_TEST_DATA_PATH`: Ruta al conjunto de prueba (`fashion-mnist_test.parquet`)
- `FASHION_VALIDATION_DATA_PATH`: Ruta al conjunto de validación (`fashion-mnist_validation.parquet`)

**URLs de datasets:**

- `MNIST_URLS`: URLs de descarga para los archivos IDX oficiales de MNIST (imágenes y etiquetas de entrenamiento y prueba).
- `FASHION_URLS`: URLs de descarga para los archivos IDX oficiales de Fashion-MNIST (imágenes y etiquetas de entrenamiento y prueba).

**Configuración del Logger:**

- `LOGGER_DEFAULT_FORMAT`: Formato de los mensajes de log
- `LOGGER_DATE_FORMAT`: Formato de fecha usado en los logs

**Otros:**

- `YAML_FILE_PATH`: Ruta a `native_functions.yaml`, utilizado por el sistema de bindings.

### `__init__.py`

Exporta todas las constantes definidas en `constants.py`, proporcionando una API limpia de importación desde cualquier parte del framework.

## Uso

### Acceder a las rutas de datasets

```python
from nova.core import MNIST_TRAIN_DATA_PATH, MNIST_TEST_DATA_PATH

print(MNIST_TRAIN_DATA_PATH)
# ~/.novann/datasets/Mnist/mnist_train.parquet

print(MNIST_TEST_DATA_PATH)
# ~/.novann/datasets/Mnist/mnist_test.parquet
```

### Acceder a la configuración del logger

```python
from nova.core import LOGGER_DEFAULT_FORMAT, LOGGER_DATE_FORMAT
import logging

logging.basicConfig(
    filename="ruta/al/logger_file.log",
    format=LOGGER_DEFAULT_FORMAT,
    datefmt=LOGGER_DATE_FORMAT,
)
```

### Integración con otros módulos

El módulo `core` es utilizado internamente por:

- **`utils.logger`**: Lee las constantes de configuración del logging.
- **`utils.datasets.mnist`**: Lee las constantes de rutas de MNIST para cargar o disparar descargas.
- **`utils.datasets.fashion_mnist`**: Lee las constantes de rutas de Fashion-MNIST de la misma manera.
- **`_internal._binding`**: Lee `YAML_FILE_PATH` para cargar `native_functions.yaml`.

## Diseño

- **Centralización**: Todas las rutas y la configuración viven en un solo archivo.
- **Predecibilidad**: Las rutas siempre se resuelven en `~/.novann/datasets/`, independientemente de dónde se instale el framework.
- **Sin dependencias externas**: La configuración se define directamente usando `pathlib.Path`. No se requieren archivos `.env` ni variables de entorno.
- **Separación de responsabilidades**: `constants.py` define, `__init__.py` exporta.

## Best practices

1. **No hardcodear rutas**: Siempre importar desde `nova.core`.
2. **Validar antes de usar**: Verificar que las rutas existan antes de leer archivos.
3. **Exportar nuevas constantes**: Cualquier constante añadida a `constants.py` debe exportarse también en `__init__.py`.

---

> El módulo `core` es la fundación de configuración de NovaNN. Para añadir nuevas constantes, defínelas en `constants.py` y exórtaelas en `__init__.py`.
