![Banner](./images/NovaNN%20Banners.png)

![version](https://img.shields.io/badge/version-3.0.0-blue)
![python](https://img.shields.io/badge/python-v3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)
![coverage](https://img.shields.io/badge/coverage-95%25-success)

## 🌐 Idiomas disponibles

- 🇬🇧 [English](README.en.md)
- 🇪🇸 [Español](README.md)

**NovaNN** es un framework **que** ofrece herramientas y ejemplos para la creación de redes neuronales **Fully Connected** y **convolucionales** junto con módulos que brindan soporte y mejoran el entrenamiento de la red. Este proyecto **demuestra** una comprensión profunda y dominio sobre cómo funcionan estas redes, inspirado en cómo lo hacen los frameworks de deep learning más populares como **PyTorch** y **TensorFlow**, especialmente **PyTorch**, que sirvió como inspiración principal para este proyecto

**Aclaración**: Este framework fue creado con fines educativos para tener un idea clara de que hacen los grandes frameworks de Deep Learning. **Objetivo**: Demostrar conocimientos sólidos en: **redes neuronales**, **Deep Learning**, **Machine Learning**, **matemáticas**, **ingeniería de software**, **Diseño de sistemas**, **buenas prácticas**, **tests unitarios**, **diseño ultra-modular** y **preprocesamiento de datos**.

## Introducción

**NovaNN** cuenta con una estructura completamente **modular diseñada** para que sea lo más parecido a un framework

El directorio `data/` está destinado a datasets como _Fashion-MNIST_ y _MNIST_. Dado que los archivos originales no se incluyen en el repositorio por su tamaño, puedes descargarlos desde **Kaggle** mediante los siguientes enlaces:

- [fasion-mnist-train](https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_train.csv)
- [fasion-mnist-test](https://www.kaggle.com/datasets/zalando-research/fashionmnist?select=fashion-mnist_test.csv)
- [mnist-train](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv)
- [mnist-test](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_test.csv)

El directorio `examples/` contiene scripts de ejemplos como **clasificación binaria**, **clasificación multiclase**, **regresión** y **capas convolucionales**.

En `notebooks/` encontrarás un cuaderno de Jupyter que prepara los datos de validación a partir de los datasets descargados.
**Nota importante**: Verifica la estructura de los datos antes de ejecutar el notebook, ya que variaciones en el formato pueden causar errores.

También es **necesario crear un archivo `.env`** con las siguientes variables de entorno:

- **FASHION_TRAIN_DATA_PATH**: Ruta de datos de entrenamiento
- **EXPORTATION_FASHION_TRAIN_DATA_PATH**: Ruta de datos de entrenamiento separado de los datos de validación.
- **FASHION_VALIDATION_DATA_PATH**: Ruta de datos de validación separados de los de entrenamiento.
- **FASHION_TEST_DATA_PATH**: Ruta de los datos de prueba

- **MNIST_TRAIN_DATA_PATH**: Ruta de datos de entrenamiento
- **EXPORTATION_MNIST_TRAIN_DATA_PATH**: Ruta de datos de entrenamiento separado de los datos de validación.
- **MNIST_VALIDATION_DATA_PATH**: Ruta de datos de validación separados de los de entrenamiento.
- **MNIST_TEST_DATA_PATH**: Ruta de los datos de prueba

- **LOG_FILE**: Ruta del archivo de logs
- **LOGGER_DEFAULT_FORMAT**: `%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s` <- Valor por defecto.
- **LOGGER_DATE_FORMAT** `%Y-%m-%d %H:%M:%S` <- Valor por defecto.

## 🛠️ Tecnologías utilizadas

El framework **NovaNN** está construido utilizando las siguientes tecnologías y librerías principales:

- **Lenguaje**: Python >= 3.14
- **Gestión de dependencias**: Poetry (para manejo de paquetes y entornos virtuales)
- **Librerías principales**:
  - `numpy`: Operaciones numéricas eficientes y arrays multidimensionales
  - `pandas`: Manejo y análisis de datos tabulares (para carga de datasets)
  - `matplotlib`: Visualización de gráficos y resultados
  - `seaborn`: Mejora estética de visualizaciones estadísticas
  - `scikit-learn`: Herramientas de Machine Learning clásico y utilidades
  - `pyarrow`: Backend eficiente para DataFrames de pandas (reduce uso de memoria)
- **Herramientas de desarrollo**:
  - `pytest`: Framework de testing unitario
  - `pytest-cov`: Cobertura de código en tests
  - `python-dotenv`: Manejo de variables de entorno desde archivos `.env`
  - `ipykernel`: Kernel de Jupyter para notebooks
  - `black`: Formateador de código para mantener estilo consistente

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

### 6. Configurar variables de entorno

Crea un archivo .env en la raíz del proyecto con las siguientes variables (ajusta las rutas según tu configuración):

```env
# Rutas para Fashion-MNIST
FASHION_TRAIN_DATA_PATH=<SU RUTA>/NovaNN/data/FashionMnist/fashion-mnist_train.csv
EXPORTATION_FASHION_TRAIN_DATA_PATH=<SU RUTA>/data/FashionMnist/fashion_train_ready.csv
FASHION_VALIDATION_DATA_PATH=<SU RUTA>/data/FashionMnist/fashion_validation_ready.csv
FASHION_TEST_DATA_PATH=<SU RUTA>/data/FashionMnist/fashion-mnist_test.csv

# Rutas para MNIST
MNIST_TRAIN_DATA_PATH=<SU RUTA>/data/Mnist/mnist_train.csv
EXPORTATION_MNIST_TRAIN_DATA_PATH=<SU RUTA>/data/Mnist/mnist_train_ready.csv
MNIST_VALIDATION_DATA_PATH=<SU RUTA>/data/Mnist/mnist_validation_ready.csv
MNIST_TEST_DATA_PATH=<SU RUTA>/data/Mnist/mnist_test.csv

# Configuración de logging
LOG_FILE=<SU RUTA>/logs/nova_nn.log
LOGGER_DEFAULT_FORMAT=%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s - %(message)s # Puede ser el que usted quiera
LOGGER_DATE_FORMAT=%Y-%m-%d %H:%M:%S
```

### 7. Ejecutar ejemplos

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

### 8. Ejecutar todos los tests

```bash
# Todos los tests
poetry run pytest tests/

# Tests específicos con cobertura
poetry run pytest tests/ --cov=novann --cov-report=term-missing

# Tests verbosos
poetry run pytest tests/ -v
```

## 🧪 Testing

El framework incluye una suite completa de tests unitarios en el directorio [`tests/`](./tests/) que verifican la correcta implementación de todos los componentes. Para más información vaya a [Tests unitarios](./tests/README.md)

## 🤝 Contribución

Para saber como contribuir a **NovaNN** puede a [contribuciones](./CONTRIBUTING.md)

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver el archivo [LICENCE](./LICENCE) para más detalles.

**Resumen de la licencia MIT:**

- Software libre para usar, copiar, modificar, fusionar, publicar, distribuir
- Se puede usar para fines comerciales
- La licencia incluye derechos de autor originales
- No hay garantía y los autores no son responsables de daños

## 👤 Autor y Mantenedor

**Juan José** - Developer & Machine Learning Engineer (16 años)

- GitHub: [https://github.com/JOSE-MDG](https://github.com/JOSE-MDG)
- Email: josepemlengineer@gmail.com

**Sobre mí**: Con solo 16 años, construí **NovaNN** desde cero como un proyecto educativo para demostrar mi pasión y comprensión profunda del deep learning. Este framework representa meses de estudio autodidacta, experimentación y dedicación, implementando cada algoritmo matemáticamente desde los papers originales.

**Agradecimientos:**

- Inspirado en PyTorch y otros frameworks de deep learning
- Comunidad de open source por herramientas y conocimientos compartidos
- Papers de investigación que fundamentan las implementaciones
