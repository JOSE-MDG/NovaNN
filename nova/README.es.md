# Módulo `nova`

El directorio **`nova/`** contiene el **núcleo del framework NovaNN**.  
Aquí se implementan todas las abstracciones fundamentales necesarias para construir, entrenar y analizar modelos de aprendizaje profundo, así como la infraestructura interna que permite que el sistema sea extensible, coherente y eficiente.

Este módulo define tanto la **API pública** que utilizan los usuarios como los **mecanismos internos** que hacen posible el funcionamiento del framework.

## Estructura general

El módulo `nova/` está organizado de forma modular, separando claramente responsabilidades entre:

- representación de datos
- diferenciación automática
- capas y modelos
- optimización
- métricas
- serialización
- utilidades internas

Cada submódulo cuenta con su propia documentación detallada.

## Submódulos principales

- **[`autograd/`](./autograd/README.es.md)**  
  Implementa el sistema de diferenciación automática de NovaNN.  
  Incluye la construcción del grafo computacional, la definición de funciones diferenciables, el cálculo de gradientes y el control del modo de gradiente.

- **[`nn/`](./nn/README.es.md)**  
  Contiene las abstracciones de alto nivel para redes neuronales: módulos, capas, funciones de activación, pérdidas y utilidades relacionadas con el entrenamiento.

- **[`optim/`](./optim/README.es.md)**  
  Implementa optimizadores y planificadores de tasa de aprendizaje utilizados durante el entrenamiento de modelos.

- **[`metrics/`](./metrics/README.es.md)**  
  Proporciona métricas para tareas de clasificación y regresión, diseñadas para integrarse de forma natural con `Tensor`.

- **[`serialization/`](./serialization/README.es.md)**  
  Permite guardar y cargar modelos, tensores y estados de entrenamiento de forma segura y reproducible.

- **[`core/`](./core/README.es.md)**  
  Define configuraciones globales, constantes y parámetros base utilizados en todo el framework.

- **[`utils/`](./utils/README.es.md)**  
  Incluye utilidades generales como registros, hooks, validaciones, logging y herramientas auxiliares.

## Archivos y módulos internos (`_`)

NovaNN utiliza el prefijo `_` para indicar **componentes internos** que no forman parte de la API pública estable.

Estos módulos existen para **soportar la arquitectura interna del framework** y no están pensados para ser utilizados directamente por el usuario final.

### `_internal/`

- **[`_internal/`](./_internal/README.es.md)**  
  Contiene la infraestructura que permite que NovaNN genere y conecte dinámicamente operaciones al sistema de tensores.  
  Este módulo actúa como el **motor de ensamblaje interno** del framework y es clave para mantener separadas la definición de operaciones, su registro y su integración en la API pública.

### `_interfaces/`

- **[`_interfaces/`](./_interfaces/README.es.md)**  
  Define contratos y abstracciones base para componentes como optimizadores y planificadores de tasa de aprendizaje.  
  Facilita la consistencia entre implementaciones y el tipado estático.

### `_typing/`

- **[`_typing/`](./_typing/README.es.md)**  
  Proporciona definiciones de tipos auxiliares y anotaciones utilizadas en todo el proyecto,
  para mejorar la experiencia en editores y herramientas de análisis estático.

## Ejemplos de uso de la API de NovaNN

### Crear un tensor y hacer operaciones básicas

```python
import nova

# Crear tensores
x = nova.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = nova.tensor([[2.0, 0.0], [1.0, 3.0]])

# Operaciones
z = x * y + x
loss = z.sum()
loss.backward()

print(x.grad)  # Gradientes calculados automáticamente

# array([[3. 1.]
#        [2. 4.]])
```

### Definir una read neuronal simple

```python
import nova.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = SimpleNet()
print(model)

# SimpleNet(
#   (fc1): Linear(in_features=2, out_features=4, bias=True)
#   (relu): ReLU()
#   (fc2): Linear(in_features=4, out_features=1, bias=True)
# )
```

### Definir optimizador y entrenamiento básico

```python
import nova.nn.functional as F
from nova import optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = F.mse_loss(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss={loss.item()}")

```

### Guardar y cargar un modelo

```python
import nova

nova.save(model.state_dict(), "mimodelo.pt")
loaded_model.load_state_dict(nova.load("mimodelo.pt"))
```

### Uso de metricas

```python
from nova.metrics import Accuracy

metric = Accuracy(num_classes=2)
metric.reset()

preds = loaded_model(x)
labels = nova.tensor([[1.0], [0.0]])
metric.update(pred, labels)
acc = metric.comput().item()
print("Accuracy:", acc)
```

---

> Para más detalles sobre cada componente, consulta la documentación específica de cada submódulo.
