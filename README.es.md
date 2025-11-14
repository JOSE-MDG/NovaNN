# ⭐ MiniNN Framework

## 🌐 Available languages / Idiomas disponibles

- 🇬🇧 [English](README.en.md)
- 🇪🇸 [Español](README.es.md)

Este mini framework ofrece herramientas y ejemplos para la creación de redes neuronales
**MLP** junto con modulos que brindan soporte y mejoran el entrenamiento de la red.
Este proyecto intenta reflejar una buena compresión y dominio sobre funcionan estas redes
inspirado en como lo hacen los frameworks de deep learning más populares como **Pytorch** y **Tensorflow**,
espcialmente **Pytorch** que fue la base en la que se inspiro este proyecto.

**Aclaracion**: Es este mini framework busca demostrar solidas bases y conocimientos sobre como funciona:
las redes neuronales, Deep Leaning, Machine Learning, matematícas, ingenieria de software, buenas practicas,
tests unitarios, Diseño de modular y preprocesamiento de datos.

## Introducción

- Este proyecto tiene una estrcutura completamente **modular**; incluye un directorio
  llamado `examples/` con ejemplos de **Clasificación binaria**, **Clasificación multiclase**
  y **Regresion** de como se puede utilizar las herramientas que posee este mini framework.

- El directorio `data/` posee datasets como _fashion-mnist_ y _mnist_ donde _fashion-mnist_
  fue utilizado para comparar el performance del proyecto con otro framework y _mnist_ para realizar un ejemplo de uso normal de clasificación en el directorio `examples/`

- Se realizo una revisión, preprocesamiento y división previa de datos en `notebooks/exploration.ipynb`
  donde se visualizarion los datasets y se particiono en ambos el set de validación.

- El modulo `src/` es el modulo principal que contiene todas las partes y/o herramientas que conforman este
  mini framework. Este posee una estructura centralizadaa en donde `core/config.py` alamcena
  y carga las los valores de las variables de entorno para que puedan ser asequibles por el resto de modulos,
  y así no tener que cargar en cada script las variables de entorno que se vayan a utilizar.

- Se evaluo el performance de **MiniNN Framework** con el popular framework de deep learning **Pytorch**
  en una tarea de clasificación con el dataset de _fashion-mnist_, en la cual se utilizo exactamente el
  mismo dataset e hiperpatametros para ambas pruebas. Para sacar los resultados del cotejo se guardo en
  formato `json` metricas como el accuracy y la perdida.

  - **[main.py](main.py)**: Este archivo inplmenta el código de entrenamiento y la estrcutura de la red que se va a utilizar
    para el cotejo.
  - **[pytorch_comparison](https://colab.research.google.com/drive/1APfspox9ONmDWL0jFXmndHZ70UPjr9Mn?usp=sharing)**: En
    el notebook está el código de entrenamiento version Pytorch, que realiza el mismo procedimiento que el script.

### Resultados del cotejo:

Ya obtenidos los resultados, se hizo un script ([visualization.py](visualization.py)) para graficar los resultados
de una manera más presentable.

![comparison](images/comparison.png)

## 📂 Estructura del proyecto

[Structure file](FileTree_NeuralNetwork.md)

```
📁 Neural Networks
├── 📁 data
│   ├── 📁 FashionMnist
│   └── 📁 Mnist
├── 📁 examples
│   ├── 🐍 binary_classification.py
│   ├── 🐍 multiclass_classification.py
│   └── 🐍 regresion.py
├── 📁 images
│   └── 🖼️ comparison.png
├── 📁 logs
├── 📁 notebooks
│   └── 📄 exploration.ipynb
├── 📁 src
│   ├── 📁 core
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 config.py
│   │   ├── 🐍 dataloader.py
│   │   ├── 🐍 init.py
│   │   └── 🐍 logger.py
│   ├── 📁 layers
│   │   ├── 📁 activations
│   │   │   ├── 🐍 __init__.py
│   │   │   ├── 🐍 activations.py
│   │   │   ├── 🐍 relu.py
│   │   │   ├── 🐍 sigmoid.py
│   │   │   ├── 🐍 softmax.py
│   │   │   └── 🐍 tanh.py
│   │   ├── 📁 bn
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 batch_normalization.py
│   │   ├── 📁 linear
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 linear.py
│   │   ├── 📁 regularization
│   │   │   ├── 🐍 __init__.py
│   │   │   └── 🐍 dropout.py
│   │   └── 🐍 __init__.py
│   ├── 📁 losses
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 functional.py
│   ├── 📁 metrics
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 metrics.py
│   ├── 📁 model
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 nn.py
│   ├── 📁 module
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 layer.py
│   │   └── 🐍 module.py
│   ├── 📁 optim
│   │   ├── 🐍 __init__.py
│   │   ├── 🐍 adam.py
│   │   ├── 🐍 rmsprop.py
│   │   └── 🐍 sgd.py
│   ├── 🐍 __init__.py
│   └── 🐍 utils.py
├── 📁 tests
│   ├── 📁 activations
│   │   ├── 🐍 test_leaky_relu.py
│   │   ├── 🐍 test_relu.py
│   │   ├── 🐍 test_sigmoid.py
│   │   ├── 🐍 test_softmax.py
│   │   └── 🐍 test_tanh.py
│   ├── 📁 batch_norm
│   │   └── 🐍 test_batch_norm.py
│   ├── 📁 dataloader
│   │   └── 🐍 test_dataloader.py
│   ├── 📁 initializers
│   │   └── 🐍 test_init.py
│   ├── 🐍 test_dropout_regularization.py
│   ├── 🐍 test_linear_layer.py
│   └── 🐍 test_sequential_module.py
├── ⚙️ .gitignore
├── 📝 FileTree_NeuralNetwork.md
├── 📝 README.en.md
├── 📝 README.es.md
├── 🐍 main.py
├── 📄 requirements.txt
└── 🐍 visualization.py
```

## Estructura del modulo `src/` y sub directorios

Aquí que se va a explicar a detalle que hace cada submodulo y sus archivos

### `core/`

**Centraliza las funciones ecencialies del proyecto**

- `config.py`: Contiene las **configuraciones globales** del proyecto. Ej. Variables de entorno
  y diccionarios con los distintos **metodos de inicialización** de parametros para las capas lineales.

  Hay dos diccionarios: `DEFAULT_NORMAL_INIT_MAP` (se usa por defecto en todo el código) que contiene las funciones
  de inicialización con distribuciones normales (`xavier_normal_`/`kaiming_normal`) del archivo `init.py`  
  y `DEFAULT_UNIFORM_INIT_MAP` contiene las funciones con distribuciones uniformes
  (`xavier_uniform`/`kaiming_uniform`). Como clave de estos diccionarios está el nombre de la función de
  activación que le corresponde a dicha inicialiación.

- `dataloder.py`

  - `DataLoader`: Esta clase recibe dos arrays de numpy (x, y) y va a retornar un objeto iterador, el
    iterador va a regresar una tupla con dos arrays (B,features) según el tamaño del `batch_size`, barajandolos
    aleatoriamente si `shuffle` es True.

- `init.py`: Este script que contiene las funciones de inicialización de parametros `xavier_normal_`, `xavier_uniform_`,
  `kaiming_normal_`, `kaiming_uniform_` y `random_init_`.

- `logger.py`:
  - `Logger`: Permite la creación de un logger de diferentes niveles con una facil configuración. Permite agregar
    _kwargs_ para hacer anotaciones adicionales sobre algo en concreto

### `layers/`

- `activations/`

  - `activations.py`

    - `Activation()`: Clase que hereda de `Layer`, esta clase tiene como atributos: `name`, que almacena
      en _lower case_ el nombre de sus sub clases y `affect_init` que ayuda a indicar en si la
      función deberia influir en la inicialización por defecto, basicamente si esta en False indicaria que su nombre
      no esta en el diccionario de inicializaciones, por lo que se inicializaria con el metodo por defecto `random_init_`

  - `relu.py`

    - `ReLU()`: Clase que hereda de `Activation`, la identifica como un una función de activación, lo que yuda a
      identificación en en modulo `Sequential` para saber que metodo de inicialización utilizar averiguando su atributo
      `name` y si tiene o no parametros.
      En su metodo `forward(x)` aplica la función relu `(max(0,x))`.
      En su metodo `backward(grad)` retropropaga el gradiente: ∂L/∂a \* σ(x)

    - `LeakyReLU()`: Clase que hereda de `Activation`, la identifica como un función de activación. Esta consta de un
      parametros llamado `negative_slope` que lo que hace es proveer una pequeña pendiente pequeña a los valos negtivos
      para evitar las neuronas muertas.
      En su metodo `forward(x)` lo que hace es pasar los valores positivos, y proveer una pequeña pendiente a los
      negativos `(si x >= 0; x, si no α * x)`.
      En su metodo `backward(grad)` lo que hace es `(si x >= 0, 1.0, si no α)`

  - `sigmoid.py`
    - `Sigmoid()`:...
  - `softmax.py`
    - `Softmax()`:...
  - `tanh.py`
    - `Tanh()`:...

- `bn/`

  - `batch_normalization.py`
    - `BatchNormalization()`:...

- `linear/`

  - `linear.py`
    - `Linear()`:...

- `regularization/`
  - `dropout.py`
    - `Dropout()`:...

### `losses/`

- `functional.py`
  - `CrossEntropyLoss()`:...
  - `MSE()`:...
  - `MAE()`:...
  - `BinaryCrossEntropy()`:...

### `metrics/`

- `metrics.py`
  - `accuracy(...)`:...
  - `binaty_accuracy(...)`:...
  - `r2_score(...)`:...

### `model/`

- `nn.py`
  - `Sequential()`:...

### `module/`

- `layer.py`
  - `Layer()`:...
- `module.py`
  - `Module()`:...

### `optim/`

- `adam.py`
  - `Adam()`:...
- `rmsprop.py`
  - `RMSprop()`:...
- `sgd.py` - `SGD()`:...
  ...

### `utils.py`:...

1. `numeric_grad_elementwise(...)`:...
2. `numeric_grad_scalar_from_softmax(...)`:...
3. `numeric_grad_scalar_wrt_x(...)`:...
4. `numeric_grad_wrt_param(...)`:...
5. `load_fashion_mnist_data(...)`:...
6. `load_mnist_data(...)`:...

## 🛠️ Tecnologías utilizadas

- Lenguajes: Python 3.14.0 🐍

- Herramientas de desarrollo: Estenciones: `Black Formatter`, `FileTree Pro`

- Principales Librerias utilizadas:

1.  **`numpy`**: Arrays con opreaciones vectorizadas optimizadas en memoria.
2.  **`pandas`**: Manejo de datos tabulares. Ej. Datasets como _fashion-mnist_ y _mnist_.
3.  **`matplotlib`**: Visualización de datos y graficos estadísticos
4.  **`seaborn`**: Mejorar el estilo de las visualizaciones
5.  **`scikit-learn`**: Libreria de Machine Learning clasico.
6.  **`pyarrow`**: Reducir el gran uso de memoría de pandas con datasets grandes
7.  **`pytest`**: Tests unitarios.

## 📦 Instalación

Instrucciones para instalar dependecias y preparar el entrono

1. **Clonar repositorio**

```bash
git clone https://github.com/JOSE-MDG/mini-nn-framework

# Acceder al directorio
cd "mini-nn-framework"
```

2. **Crear y activar un entorno virtual**

- Windows:

```bash
python -m venv .venv

# Activar entorno virtual
.\\.venv\\Scripts\\activate

# Si el anterior da problemas
.\\.venv\\Scripts\\Activate.Ps1
```

- Linux/MacOS:

```bash
python -m venv .venv

# Activar entorno virtual
source .venv/bin/activate
```

3. **Instalar los requerimientos (`requirements.txt `)**

```bash
pip install -r requirements.txt
```

## 🧪 Testing (`tests/`)

...

## 🤝 Contribución

...

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENCE) para más detalles.

## 👤 Author

Juan José - Developer, Machine & Deep Learning Enthusiast.
GitHub: https://github.com/JOSE-MDG
