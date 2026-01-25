# Módulo `_internal`

El directorio **`_internal/`** contiene los componentes internos que permiten a NovaNN **generar y enlazar dinámicamente operaciones a la clase Tensor**.  
Estos archivos no forman parte de la API pública estable, pero son esenciales para que el framework funcione de manera modular y escalable.

## Archivos principales

### `_binding.py`

Este archivo es el **núcleo del sistema de binding dinámico**.  
Se encarga de:

- Cargar la configuración de operaciones desde un archivo YAML (`native_yaml`).
- Recuperar las funciones registradas en el sistema (`_OPS_REGISTERED`).
- Generar los métodos apropiados para cada operación:
  - **Dunder methods** (por ejemplo `__add__`)
  - **Reverse methods** (`__radd__`)
  - **Métodos regulares** (`add`)
  - **In-place variants** (`add_`, `__iadd__`)
- Adjuntar estas funciones dinámicamente a la clase `Tensor`.

Este mecanismo permite que NovaNN pueda **definir nuevas operaciones o modificar existentes sin cambiar la implementación central** de `Tensor`.

### `_generators.py`

Contiene las **funciones generadoras de métodos** que `_binding.py` usa para crear los métodos dinámicamente.  
Incluye:

- `make_forward_func`: genera métodos forward (como `__add__` o `__mul__`) para operaciones unary o binary.
- `make_reverse_func`: genera los métodos reverse (como `__radd__`) cuando la operación se llama desde el lado derecho.
- `make_method`: genera métodos regulares para llamadas directas (`tensor.add()`).
- `make_inplace_func`: genera métodos in-place (`tensor.add_()`) que modifican el tensor original.

Estas funciones se encargan de **validar argumentos, convertir automáticamente a tensores cuando hace falta, y asegurar la compatibilidad con operaciones que requieren o no gradientes**.

---

> En conjunto, `_internal/` proporciona la **infraestructura que permite a NovaNN conectar el sistema de operaciones con la clase Tensor** de manera flexible, manteniendo separadas las definiciones de operaciones, su registro y su integración en la API pública.
