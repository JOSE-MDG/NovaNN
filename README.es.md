![Banner](./images/NovaNN%20Banners.png)

![version](https://img.shields.io/badge/version-4.0.0-blue)
![python](https://img.shields.io/badge/python-v3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)
![coverage](https://img.shields.io/badge/coverage-87%25-success)

## 🌐 Idiomas disponibles

- 🇬🇧 [English](README.en.md)
- 🇪🇸 [Español](README.md)

## ¿Qué es NovaNN?

**NovaNN** es un framework de **[Deep Learning](https://www.ibm.com/think/topics/deep-learning)** desarrollado desde cero en **Python**, diseñado para construir, entrenar y evaluar redes neuronales de forma modular, clara y extensible.

El objetivo principal de NovaNN no es competir con frameworks industriales, sino **entender, implementar y demostrar** cómo funcionan internamente frameworks modernos como **[PyTorch](https://docs.pytorch.org/docs/stable/index.html)** o **[TensorFlow](https://www.tensorflow.org/api_docs)**, poniendo especial énfasis en la arquitectura de **PyTorch**, que sirvió como inspiración principal.

NovaNN permite definir modelos neuronales completos, gestionar el entrenamiento y realizar backpropagation automático mediante un **motor de autograd dinámico**, todo construido explícitamente y sin depender de motores de cómputo externos.

## Filosofía del proyecto

NovaNN nace con una idea clara:

> _No usar la magia de los frameworks existentes, sino construirla._

Cada componente del framework está diseñado para ser **legible, trazable y testeable**, priorizando la comprensión profunda de:

- Cómo se construyen los grafos computacionales
- Cómo fluye el gradiente durante el backward
- Cómo se estructuran frameworks escalables de ML
- Cómo se diseñan APIs limpias y extensibles

## Backend numérico

NovaNN utiliza **NumPy** como backend principal para el cálculo numérico, aprovechando:

- Operaciones vectorizadas eficientes
- Manipulación explícita de tensores
- Control total sobre las operaciones matemáticas

Esto permite centrarse en la **lógica del Deep Learning** (autograd, capas, optimización, entrenamiento) sin abstraer en exceso el comportamiento interno del sistema.

## Objetivo educativo y técnico

Este framework fue creado con fines **educativos y demostrativos**, con el propósito de evidenciar conocimientos sólidos en:

- **Machine Learning y Deep Learning**
- **Fundamentos matemáticos** (álgebra lineal, cálculo, optimización)
- **Autograd y backpropagation**
- **Diseño de sistemas y arquitectura de software**
- **Testing unitario y validación numérica**
- **Diseño modular y extensible**
- **Buenas prácticas de ingeniería de software**
- **Preprocesamiento de datos y entrenamiento de modelos**

NovaNN está pensado para personas que quieran **entender cómo funcionan realmente los frameworks de Deep Learning por dentro**, más allá de simplemente utilizarlos.

> ⚠️ **Nota**  
> NovaNN no pretende reemplazar frameworks como PyTorch o TensorFlow en entornos de producción.  
> Su propósito es servir como herramienta de aprendizaje avanzada y como demostración técnica de ingeniería aplicada al Deep Learning.

## Introducción

**NovaNN** adopta una **organización modular** inspirada en frameworks modernos de Deep Learning, con responsabilidades claramente separadas entre datos, modelos, entrenamiento y utilidades.
Esta estructura favorece tanto la extensibilidad como la claridad del flujo de trabajo.

### Organización del proyecto

- **`examples/`**  
  Contiene scripts funcionales que muestran el uso del framework en distintos escenarios:
  - Clasificación binaria
  - Clasificación multiclase
  - Regresión
  - Redes convolucionales

- **[`nova/`](./nova/README.es.md)**
  Contiene el **núcleo completo del framework NovaNN**.  
  Aquí se implementan los tensores, el motor de autograd, las operaciones matemáticas, los módulos de redes neuronales, optimizadores, métricas, serialización y utilidades internas.  
  Está organizado de forma modular para separar claramente los distintos niveles del sistema: bajo nivel (tensores y operaciones), autograd, APIs de alto nivel (`nn`, `optim`, `metrics`) y utilidades auxiliares.  
  Cada submódulo cuenta con su propia documentación para facilitar la navegación y el mantenimiento del código.

- **[`benchmarks/`](./benchmarks/README.es.md)**
  Incluye **benchmarks diseñados para evaluar el rendimiento de NovaNN** en distintos escenarios y compararlo con otros frameworks (principalmente PyTorch).  
   Los benchmarks se centran en:
  - operaciones elementales y reducción
  - coste del sistema de autograd
  - entrenamiento en CPU en modelos pequeños
  - uso de memoria y overhead computacional  
    Este directorio no forma parte del runtime del framework y está pensado exclusivamente para **análisis de rendimiento, validación técnica y estudios comparativos**.

## 🛠️ Tecnologías utilizadas

El framework **NovaNN** está construido utilizando las siguientes tecnologías y librerías principales:

- **Lenguaje**: Python >= 3.14, < 3.15
- **Gestión de dependencias**: Poetry (para manejo de paquetes y entornos virtuales)
- **Librerías principales**:
  - `numpy`: Operaciones numéricas eficientes y arrays multidimensionales
  - `pandas`: Manejo y análisis de datos tabulares (para carga de datasets)
  - `matplotlib`: Visualización de gráficos y resultados
  - `seaborn`: Mejora estética de visualizaciones estadísticas
  - `scikit-learn`: Herramientas de Machine Learning clásico y utilidades
  - `pyarrow`: Backend eficiente para DataFrames de pandas (reduce uso de memoria)
  - `pyyaml`: Para manipular archivos YAML
  - `requests`: Para hacer consultas web
  - `tqdm`: Para mostar barras de progreso
- **Herramientas de desarrollo**:
  - `pytest`: Framework de testing unitario
  - `pytest-cov`: Cobertura de código en tests
  - `ipykernel`: Kernel de Jupyter para notebooks
  - `black`: Formateador de código para mantener estilo consistente
- **Herramientas para benchmarks**
  - `torch`: Framework de deep learning
  - `torchvision`: Paquete extra de torch para tareas de vision

## 📦 Instalación

NovaNN utiliza **Poetry** para la gestión de dependencias y empaquetado. Sigue estos pasos para configurar el entorno:

### 1. Clonar el repositorio

```bash
git clone git@github.com:JOSE-MDG/NovaNN.git
cd NovaNN
```

### 2. Instalar Poetry (si no lo tienes instalado)

- Windows (PowerShell):

```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

- Linux/macOS:

```bash
# Con curl
curl -sSL https://install.python-poetry.org | python3 -

# Con pipx
pipx install poetry
```

#### Añadir Poetry al PATH:

- En Linux/macOS:

```bash
# Bash/Zsh (temporal)
export PATH="$HOME/.local/bin:$PATH"

# Bash (permanente - añadir al ~/.bashrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Zsh (permanente - añadir al ~/.zshrc)
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

- En Windows

```powershell
# PowerShell (temporal para la sesión actual)
$env:Path += ";$env:APPDATA\Python\Scripts"

# PowerShell (permanente - usuario actual)
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\Python\Scripts", "User")

# PowerShell (permanente - sistema)
[System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\Python\Scripts", "Machine")
```

```cmd
# Command Prompt (temporal)
set PATH=%PATH%;%APPDATA%\Python\Scripts

# Command Prompt (permanente)
setx PATH "%PATH%;%APPDATA%\Python\Scripts"
```

### 3. Añadir el proyecto al python path

- En Linux/macOS

```bash
# Temporal
export PYTHONPATH="/ruta/a/tu/proyecto:$PYTHONPATH"

# Permanente (añadir al ~/.bashrc o ~/.zshrc)
echo 'export PYTHONPATH="/ruta/a/tu/proyecto:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

- En Windows:

```powershell
# PowerShell (temporal)
$env:PYTHONPATH = "C:\ruta\a\tu\proyecto"

# PowerShell (permanente)
[System.Environment]::SetEnvironmentVariable("PYTHONPATH", "C:\ruta\a\tu\proyecto", "User")
```

```cmd
# Command Prompt (temporal)
set PYTHONPATH=C:\ruta\a\tu\proyecto

# Command Prompt (permanente)
setx PYTHONPATH "C:\ruta\a\tu\proyecto"
```

### 4. Instalar dependencias del proyecto

```bash
# Escribir el archvo lock
poetry lock

# Instalar todas las dependencias (incluyendo las de desarrollo)
poetry install
```

### 5. Activar el entorno virtual

```bash
# instalaer el plugin de shell
poetry self add poetry-plugin-shell

# # Activar el shell con el entorno virtual
poetry shell

# Alternativamente, ejecutar comandos directamente sin activar el shell:
poetry run python examples/binary_classification.py
```

### 6. Ejecutar ejemplos

```bash
# Clasificación binaria
poetry run python examples/binary_classification.py

# Clasificación multiclase
poetry run python examples/multiclass_classification.py

# Redes convolucionales
poetry run python examples/conv_example.py

# Regresión
poetry run python examples/regresion.py
```

## 🧪 Testing

El framework incluye una suite completa de tests unitarios en el directorio [`tests/`](./tests/) que verifican la correcta implementación de todos los componentes cubriendo un **87%** del modulo. Para más información vaya a [Tests unitarios](./tests/README.es.md)

### Ejecutar todos los tests

```bash
# Todos los tests
poetry run pytest

# Tests verbosos
poetry run pytest tests/ -v

# Tests con cobertura
poetry run pytest --cov

# Test con reporte html
poetry run pytest --cov --cov-report=html
```

## 🤝 Contribución

Para saber como contribuir a **NovaNN** puede a [contribuciones](./CONTRIBUTING.es.md)

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver el archivo [LICENCE](./LICENCE) para más detalles.

**Resumen de la licencia MIT:**

- Software libre para usar, copiar, modificar, fusionar, publicar, distribuir
- Se puede usar para fines comerciales
- La licencia incluye derechos de autor originales
- No hay garantía y los autores no son responsables de daños

## 👤 Autor y Mantenedor

**Juan José** - Developer & Machine Learning Enthusiast (16 años)

- GitHub: [https://github.com/JOSE-MDG](https://github.com/JOSE-MDG)
- Email: josepemlengineer@gmail.com

**Sobre mí**: Con solo 16 años, construí **NovaNN** desde cero como un proyecto educativo para demostrar mi pasión y comprensión profunda del deep learning. Este framework representa meses de estudio autodidacta, experimentación y dedicación, implementando cada algoritmo matemáticamente desde los papers originales.

**Agradecimientos:**

- Inspirado en PyTorch y otros frameworks de deep learning
- Comunidad de open source por herramientas y conocimientos compartidos
- Papers de investigación que fundamentan las implementaciones
