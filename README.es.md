![Banner](./images/NovaNN%20Banners.png)

![version](https://img.shields.io/badge/version-4.0.1-blue)
![python](https://img.shields.io/badge/python-v3.14-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
![tests](https://img.shields.io/badge/tests-pytest-orange)
![coverage](https://img.shields.io/badge/coverage-87%25-success)

## 🌐 Idiomas disponibles

- 🇬🇧 [English](README.md)
- 🇪🇸 [Español](README.es.md)

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

## 🚀 Inicio rápido

Construye y entrena modelos con una sintaxis que ya conoces. NovaNN se parece a PyTorch, pero se ejecuta en tu propio motor personalizado.

### 1. **Autograd y Grafos Computacionales**

Experimenta con el motor de diferenciación automática de NovaNN. Crea tensores, realiza operaciones y observa cómo fluyen los gradientes.

```python
import nova
import nova.nn as nn

# Crear tensores con seguimiento de gradientes
x = nova.tensor([[0.5, -0.2]], requires_grad=True)
w = nova.tensor([[1.0], [0.5]], requires_grad=True)
b = nova.tensor([0.1], requires_grad=True)

# Forward pass manual
y = x @ w + b
loss = (y ** 2).sum()

# Backward pass automático
loss.backward()

print(f"Loss: {loss.item()}")           # Loss: 0.25
print(f"Gradiente de x: {x.grad}")    # Gradientes calculados automáticamente
print(f"Gradiente de w: {w.grad}")
print(f"Gradiente de b: {b.grad}")

# O usando capas nn.Module
model = nn.Linear(2, 1)
output = model(x)
output.backward()
```

### 2. **Entrenamiento Completo de una Red Neuronal**

Entrena un clasificador binario simple con todas las funcionalidades del framework.

```python
import nova
import nova.nn as nn
import nova.optim as optim
from nova.nn import functional as F

# Definir el modelo
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

# Crear modelo y optimizador
model = BinaryClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# Datos de ejemplo
X_train = nova.randn(100, 10)  # 100 muestras, 10 características
y_train = nova.randint(0, 2, (100, 1))

# Loop de entrenamiento
for epoch in range(50):
    # Forward pass
    predictions = model(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Época {epoch+1}/50, Loss: {loss.item():.4f}")
```

### 3. **Arquitecturas CNN para Visión Computacional**

NovaNN admite módulos complejos como convoluciones 2D, normalización por lotes y capas diferidas (lazy).

```python
import nova
import nova.nn as nn
import nova.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Bloque convolucional 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=0.1)

        # Bloque convolucional 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, momentum=0.1)

        # Bloque convolucional 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.1)

        self.pool = nn.MaxPool2d(2, 2)

        # Capas fully connected (lazy para inferir dimensiones automáticamente)
        self.fc1 = nn.LazyLinear(256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x32 -> 16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 16x16 -> 8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 8x8 -> 4x4

        # Flatten
        x = x.view(x.size(0), -1)

        # Classification head
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Crear modelo y procesar imágenes
model = ConvNet(num_classes=10)
batch_images = nova.rand(8, 3, 32, 32)  # Batch de 8 imágenes RGB 32x32
logits = model(batch_images)

print(f"Salida: {logits.shape}")  # Shape: (8, 10)
```

### 4. **Transfer Learning - Congelar Capas**

Aprovecha modelos pre-entrenados congelando capas para usarlas como extractores de características fijos.

```python
import nova
import nova.nn as nn
import nova.optim as optim
from nova.nn import functional as F

# Supongamos que tenemos un modelo pre-entrenado
class PretrainedFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x

# Modelo completo con transfer learning
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Backbone pre-entrenado (congelado)
        self.backbone = PretrainedFeatureExtractor()

        # Nueva cabeza de clasificación (entrenable)
        self.classifier = nn.Sequential(
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Crear modelo
model = TransferLearningModel(num_classes=5)

# Congelar las capas del backbone
for param in model.backbone.parameters():
    param.requires_grad = False

# Solo entrenar el clasificador
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.001
)

# Verificar qué parámetros son entrenables
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Parámetros entrenables: {trainable}/{total}")
```

### 5. **Fine-Tuning con Learning Rates Diferenciados**

Entrena diferentes partes de la red con velocidades de aprendizaje distintas para un ajuste fino óptimo.

```python
import nova
import nova.nn as nn
import nova.optim as optim
from nova.nn import functional as F

class FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Capas base (pre-entrenadas)
        self.base_layers = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Capas intermedias
        self.mid_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Cabeza de clasificación (nueva)
        self.head = nn.Linear(64, 10)

    def forward(self, x):
        x = self.base_layers(x)
        x = self.mid_layers(x)
        x = self.head(x)
        return x

model = FineTuneModel()

# Configurar learning rates diferenciados por grupo de capas
optimizer = optim.Adam([
    {'params': list(model.base_layers.parameters()), 'lr': 1e-5},  # Muy bajo para capas base
    {'params': list(model.mid_layers.parameters()), 'lr': 1e-4},   # Intermedio
    {'params': list(model.head.parameters())}
], lr=1e-3) # Alto para nueva cabeza

# Datos de ejemplo
X = nova.randn(32, 100)
y = nova.randint(0, 10, (32,))

# Entrenamiento
for epoch in range(100):
    logits = model(X)
    loss = F.cross_entropy(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f"Época {epoch+1}: Loss = {loss.item().4f}")
```

### 6. **Guardado y Carga de Modelos**

Serializa tus modelos entrenados para reutilizarlos más tarde.

```python
import nova
import nova.nn as nn

# Entrenar modelo
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# Guardar modelo completo
nova.save(model, 'model.pt')

# Guardar solo los parámetros (state_dict)
nova.save(model.state_dict(), 'model_weights.pt')

# Cargar modelo completo
loaded_model = nova.load('model.pt')

# Cargar solo parámetros en un modelo nuevo
new_model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
new_model.load_state_dict(nova.load('model_weights.pt'))
```

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

NovaNN está disponible en **[PyPI](https://pypi.org/)** y puede instalarse fácilmente usando `pip` o `poetry`. También puedes instalarlo desde el código fuente si quieres contribuir o explorar el framework en profundidad.

### Opción 1: Instalación desde PyPI (Recomendado)

La forma más sencilla de instalar NovaNN es mediante pip o poetry:

```bash
# Con pip
pip install novann

# Con poetry
poetry add novann
```

#### Verificar la instalación

```python
import nova
print(nova.__version__)  # Debería mostrar: 4.0.1
```

#### Requisitos del sistema

- **Python**: >= 3.12, < 4.0.0
- **Sistema operativo**: Windows, Linux, macOS

### Opción 2: Instalación desde el código fuente

Si deseas contribuir al proyecto o explorar el código fuente, puedes clonar el repositorio e instalar usando Poetry.

#### 1. Clonar el repositorio

```bash
git clone git@github.com:JOSE-MDG/NovaNN.git
cd NovaNN
```

#### 2. Instalar Poetry (si no lo tienes instalado)

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

#### 3. Instalar dependencias del proyecto

```bash
# Escribir el archivo lock
poetry lock

# Instalar todas las dependencias (incluyendo las de desarrollo)
poetry install

# Solo dependencias de producción
poetry install --without dev,benchmark

# Con herramientas de benchmarking
poetry install --with benchmark
```

#### 4. Activar el entorno virtual

```bash
# Instalar el plugin de shell
poetry self add poetry-plugin-shell

# Activar el shell con el entorno virtual
poetry shell

# Alternativamente, ejecutar comandos directamente sin activar el shell:
poetry run python examples/binary_classification.py
```

### Ejecutar ejemplos

Una vez instalado NovaNN (desde PyPI o código fuente), puedes ejecutar los ejemplos incluidos:

```bash
# Si instalaste desde código fuente
poetry run python examples/binary_classification.py
poetry run python examples/multiclass_classification.py
poetry run python examples/conv_example.py
poetry run python examples/regresion.py

# Si instalaste desde PyPI, crea tus propios scripts
python mi_red_neuronal.py
```

### Desinstalación

```bash
# Si instalaste desde PyPI con pip
pip uninstall novann

# Si instalaste con poetry
poetry remove novann

# Si instalaste desde código fuente
# Simplemente elimina el entorno virtual de Poetry
poetry env remove python
```

### Solución de problemas

#### Error: "Python version not compatible"

NovaNN requiere Python `>=3.12, <4.0.0` Verifica tu versión:

```bash
python --version
```

Si tienes múltiples versiones de Python, usa:

```bash
poetry env use python3.14 # o python3.12/13
```

#### Error: "Module nova not found"

Si instalaste desde el código fuente, asegúrate de estar en el entorno virtual de Poetry:

```bash
poetry shell
```

O usa `poetry run` antes de tus comandos.

#### Problemas con NumPy o dependencias

Si tienes conflictos de dependencias, intenta:

```bash
pip install --upgrade pip
pip install novann --no-cache-dir
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
