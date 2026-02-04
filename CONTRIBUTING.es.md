# Guía de Contribución

¡Gracias por tu interés en contribuir a **NovaNN**! Este proyecto educativo de código abierto se beneficia enormemente de las contribuciones de la comunidad.

## 📋 Tabla de Contenidos

- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Configuración del Entorno de Desarrollo](#configuración-del-entorno-de-desarrollo)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Estándares de Código](#estándares-de-código)
- [Testing](#testing)
- [Proceso de Pull Request](#proceso-de-pull-request)
- [Reportar Bugs](#reportar-bugs)
- [Proponer Features](#proponer-features)

## Código de Conducta

Este proyecto busca ser un espacio acogedor y educativo. Se espera:

- **Respeto mutuo**: Trata a todos con cortesía y profesionalismo
- **Crítica constructiva**: Enfócate en el código, no en las personas
- **Colaboración**: Ayuda a otros a aprender y crecer
- **Paciencia**: Recuerda que todos estamos aprendiendo

## ¿Cómo Puedo Contribuir?

### Áreas Prioritarias

- 🐛 **Bugs**: Reportar o corregir errores encontrados
- 💡 **Features**: Nuevas capas, optimizadores o funcionalidades
- 📚 **Documentación**: Mejorar READMEs, docstrings, tutoriales
- 🧪 **Tests**: Aumentar cobertura y casos edge
- ⚡ **Performance**: Optimizaciones de código NumPy
- 🎓 **Tutoriales**: Ejemplos educativos y guías de uso

### Proceso General

1. **Fork** el repositorio en GitHub
2. **Clona** tu fork localmente
3. **Crea una rama** para tu cambio: `git checkout -b feat/nueva-funcionalidad`
4. **Haz tus cambios** siguiendo los estándares del proyecto
5. **Commit** con mensajes descriptivos: `feat(nn): add GroupNorm layer`
6. **Push** a tu fork: `git push origin feat/nueva-funcionalidad`
7. **Abre un Pull Request** en el repositorio principal

## Configuración del Entorno de Desarrollo

### 1. Requisitos Previos

- Python >= 3.12, < 4.0.0
- Poetry (gestor de dependencias)
- Git

### 2. Instalación

```bash
# With pip
pip install novann

# With poetry
poetry add novann
```

**[Mas detalles](README.md#📦-instalación)**

### 3. Verificar Instalación

```bash
# Correr tests
poetry run pytest tests/ -v

# Verificar cobertura
poetry run pytest --cov --cov-report=html
```

## Estructura del Proyecto

[Estructura completa de directorios](Tree.md)

```
NovaNN/
├── nova/              # Código fuente principal
│   ├── autograd/      # Sistema de diferenciación automática
│   ├── nn/            # Módulos de redes neuronales
│   ├── optim/         # Optimizadores y schedulers
│   ├── metrics/       # Métricas de evaluación
│   └── ...
├── tests/             # Tests unitarios
├── examples/          # Scripts de ejemplo
└── benchmarks/        # Benchmarks de rendimiento
```

### ¿Dónde Va Cada Cosa?

- **Nueva capa**: `nova/nn/modules/`
- **Nuevo optimizador**: `nova/optim/`
- **Nueva métrica**: `nova/metrics/`
- **Nueva operación autograd**: `nova/autograd/_ops/`
- **Tests**: `tests/` (espejando la estructura de `nova/`)
- **Ejemplos**: `examples/` (scripts standalone)

## Estándares de Código

### Estilo de Código

- **Formatter**: Usamos **Black** con configuración por defecto

```bash
  poetry run black nova/ tests/
```

- **Convenciones**: Sigue PEP 8 y el estilo existente en el proyecto
- **Type hints**: Usa anotaciones de tipo consistentemente

```python
  def forward(self, input: Tensor) -> Tensor:
      ...
```

### Naming Conventions

- **Clases**: `PascalCase` (`Linear`, `ReLU`, `SGD`)
- **Funciones/métodos**: `snake_case` (`forward`, `zero_grad`)
- **Constantes**: `UPPER_SNAKE_CASE` (`LOG_FILE`, `MNIST_PATH`)
- **Privado**: Prefijo `_` (`_step_impl`, `_calculate_fans`)

### Docstrings

Usa docstrings estilo Google/NumPy con descripción, Args, Returns, Examples:

```python
def kaiming_normal_(
    tensor: Parameter,
    a: float = 0.0,
    nonlinearity: str = "leaky_relu"
) -> None:
    """
    Initialize tensor using Kaiming normal initialization.

    Args:
        tensor: Parameter to initialize.
        a: Negative slope for leaky ReLU.
        nonlinearity: Activation function name.

    Examples:
        >>> weight = Parameter(nova.empty((64, 128)))
        >>> init.kaiming_normal_(weight, nonlinearity='relu')
    """
    ...
```

### Mensajes de Commit

Seguimos **Conventional Commits**:

```
<tipo>(<scope>): <descripción>

[cuerpo opcional]

[footer opcional]
```

**Tipos:**

- `feat`: Nueva funcionalidad
- `fix`: Corrección de bug
- `docs`: Solo cambios en documentación
- `style`: Formateo, punto y coma faltantes, etc.
- `refactor`: Refactorización sin cambiar funcionalidad
- `test`: Añadir o corregir tests
- `perf`: Mejora de rendimiento
- `chore`: Cambios en build, dependencias, etc.

**Ejemplos:**

```
feat(nn): add GroupNorm layer
fix(optim): correct AdamW weight decay calculation
docs(tutorials): add transformer example
test(autograd): increase coverage for backward ops
perf(autograd._ops._loss): improve backward performance in loss functions
```

## Testing

### Escribir Tests

- Usa **pytest** para todos los tests
- Un archivo de test por módulo: `test_<nombre_modulo>.py`
- Agrupa tests relacionados en clases: `TestLinear`, `TestSGD`
- Nombres descriptivos: `test_forward_with_bias`, `test_backward_without_grad`

**Ejemplo:**

```python
import pytest
import nova
import nova.nn as nn

class TestLinear:
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        layer = nn.Linear(10, 5)
        x = nova.randn(3, 10)
        output = layer(x)
        assert output.shape == (3, 5)

    def test_backward_updates_grad(self):
        """Test that backward pass computes gradients."""
        layer = nn.Linear(10, 5)
        x = nova.randn(3, 10)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        assert layer.weight.grad is not None
```

### Correr Tests

```bash
# Todos los tests
poetry run pytest

# Tests específicos
poetry run pytest tests/nn/test_linear.py -v

# Clase específica
poetry run pytest tests/nn/test_linear.py::TestLinear -v

# Test específico
poetry run pytest tests/nn/test_linear.py::TestLinear::test_forward_shape -v

# Con cobertura
poetry run pytest --cov --cov-report=html

# Ver reporte de cobertura
open htmlcov/index.html  # macOS/Linux
start htmlcov/index.html  # Windows
```

### Cobertura Mínima

- Mantener cobertura **≥ 85%** en código nuevo
- Archivos excluidos: `__init__.py`, `.pyi`, `_internal/`, `_typing/`, `examples/`, `benchmarks/`

## Proceso de Pull Request

### Antes de Abrir el PR

- [ ] Los tests pasan: `poetry run pytest`
- [ ] Código formateado: `poetry run black nova/ tests/`
- [ ] Cobertura mantenida o mejorada
- [ ] Documentación actualizada (docstrings, READMEs)
- [ ] Commits siguen Conventional Commits

### Template de PR

```markdown
## Descripción

[Descripción clara de los cambios]

## Tipo de Cambio

- [ ] Bug fix
- [ ] Nueva feature
- [ ] Cambio breaking
- [ ] Documentación

## Checklist

- [ ] Tests añadidos/actualizados
- [ ] Documentación actualizada
- [ ] Código formateado con Black
- [ ] Tests pasan localmente

## Testing

[Describe cómo testeaste los cambios]

## Notas Adicionales

[Cualquier contexto relevante]
```

### Proceso de Review

1. El mantenedor revisará tu PR en 1-3 días
2. Se pueden solicitar cambios o aclaraciones
3. Una vez aprobado, se mergeará a `main`
4. Tu contribución aparecerá en el siguiente release

## Reportar Bugs

### Antes de Reportar

- Busca en [Issues existentes](https://github.com/JOSE-MDG/NovaNN/issues) para evitar duplicados
- Asegúrate de que es un bug de NovaNN, no un problema de entorno

### Template de Issue

````markdown
**Descripción del Bug**
[Descripción clara y concisa]

**Pasos para Reproducir**

1. Código usado
2. Comando ejecutado
3. Error observado

**Comportamiento Esperado**
[Qué esperabas que sucediera]

**Comportamiento Actual**
[Qué sucedió realmente]

**Entorno**

- Python version: [ej. 3.14.0]
- NovaNN version: [ej. 4.0.0]
- OS: [ej. Ubuntu 22.04]
- NumPy version: [ej. 1.26.0]

**Código Mínimo Reproducible**

```python
import nova
# Tu código aquí
```

**Logs/Traceback**
````

[Pega el error completo aquí]

```
**Contexto Adicional**
[Cualquier información relevante]
```

## Proponer Features

### Template de Feature Request

```markdown
**¿El feature resuelve un problema?**
[Describe el problema que enfrentas]

**Describe la Solución Propuesta**
[Cómo te gustaría que funcionara]

**Alternativas Consideradas**
[Otras soluciones que consideraste]

**Implementación Sugerida**
[Si tienes ideas de cómo implementarlo]

**Contexto Adicional**
[Papers, referencias, ejemplos de otros frameworks]
```

## Posibles preguntas

### ¿Puedo contribuir siendo principiante?

¡Absolutamente! Issues etiquetados como `good first issue` son ideales para comenzar.

### ¿En qué idioma debo escribir código/docs?

- **Código**: Inglés (nombres de variables, funciones, comentarios)
- **Documentación**: Español e Inglés (ambos bienvenidos)
- **Issues/PRs**: Español o Inglés

### ¿Debo implementar tests para docs?

No es necesario para cambios solo de documentación, pero sí para código nuevo.

### ¿Cuánto tiempo toma la revisión?

Generalmente 1-3 días. Si pasa más tiempo, no dudes en hacer ping en el PR.

## Contacto

- **GitHub Issues**: Para bugs y features
- **Email**: josepemlengineer@gmail.com
- **Discussions**: Para preguntas generales y discusiones

---

¡Gracias por contribuir a NovaNN! 🚀 Tu aporte ayuda a que este proyecto educativo siga creciendo.
