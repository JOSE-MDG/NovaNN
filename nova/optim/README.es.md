# Módulo `optim`

El directorio **`optim/`** implementa **optimizadores y planificadores de tasa de aprendizaje (learning rate schedulers)** para el entrenamiento de redes neuronales en NovaNN.

Este módulo proporciona algoritmos de optimización modernos que actualizan los parámetros del modelo basándose en sus gradientes, así como estrategias para ajustar dinámicamente la tasa de aprendizaje durante el entrenamiento.

El diseño sigue de cerca la API de **PyTorch**, facilitando la transición entre frameworks y proporcionando una interfaz consistente y familiar.

## Estructura general

El módulo `optim/` está organizado en:

- **Optimizadores**: algoritmos para actualización de parámetros (`SGD`, `Adam`, `AdamW`, `RMSprop`)
- **Schedulers**: estrategias para ajuste dinámico de learning rate (`StepLR`, `CosineAnnealingLR`, `OneCycleLR`)

Todos los optimizadores heredan de la clase base **`Optimizer`** definida en [`_interfaces/_optimizer.py`](../_interfaces/README.es.md), que proporciona la estructura común y el manejo de estado.

## Optimizadores

### `sgd.py`

**`SGD(parameters, lr, momentum=0.0, weight_decay=0.0)`**: Stochastic Gradient Descent con momentum opcional.

**Características:**

- **SGD básico**: Actualización simple θ = θ - lr \* ∇θ
- **Momentum**: Acumula velocidad para suavizar actualizaciones y acelerar convergencia
  - v = momentum \* v + ∇θ
  - θ = θ - lr \* v
- **Weight decay**: Regularización L2 aplicada al gradiente
- **Excepción para BatchNorm**: No aplica weight decay a parámetros marcados con `is_bn_param=True`

**Cuándo usar:**

- Baseline simple y robusto
- Datasets pequeños o medianos
- Cuando se busca control fino del proceso de optimización
- Problemas convexos o casi convexos

**Estado interno por parámetro:**

- `velocity`: acumulador de momentum (si momentum > 0)

### `adam.py`

**`Adam(parameters, lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)`**: Optimizador Adam estándar con weight decay acoplado.

**Características:**

- **Momentos adaptativos**: Mantiene promedios móviles exponenciales del gradiente (primer momento) y su cuadrado (segundo momento)
  - m = β₁ _ m + (1 - β₁) _ ∇θ
  - v = β₂ _ v + (1 - β₂) _ ∇θ²
- **Bias correction**: Corrige el sesgo de inicialización en cero
  - m̂ = m / (1 - β₁ᵗ)
  - v̂ = v / (1 - β₂ᵗ)
- **Actualización adaptativa**: Escala el learning rate por parámetro según el historial de gradientes
  - θ = θ - lr \* m̂ / (√v̂ + ε)
- **Weight decay acoplado**: Se aplica al gradiente antes de calcular momentos

**Cuándo usar:**

- Problemas de deep learning en general
- Datasets grandes
- Cuando el learning rate es difícil de ajustar manualmente
- Entrenamiento de redes profundas con gradientes ruidosos

**Estado interno por parámetro:**

- `step`: contador de pasos (para bias correction)
- `exp_avg`: primer momento (media móvil del gradiente)
- `exp_avg_sq`: segundo momento (media móvil del gradiente al cuadrado)

### `adamw.py`

**`AdamW(parameters, lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)`**: Adam con weight decay desacoplado.

**Características:**

- **Weight decay desacoplado**: A diferencia de Adam estándar, el weight decay se aplica **después** de calcular los momentos adaptativos
  - θ = θ - lr _ wd _ θ (decay desacoplado)
  - θ = θ - lr \* m̂ / (√v̂ + ε) (actualización Adam)
- **Mejor regularización**: El decay desacoplado funciona mejor con learning rates adaptativos
- **Mismo algoritmo Adam**: Idéntico a Adam excepto por el orden del weight decay

**Cuándo usar:**

- Entrenamiento de Transformers y modelos grandes
- Cuando se necesita regularización fuerte
- Transfer learning y fine-tuning
- Alternativa superior a Adam con weight decay

**Diferencia clave Adam vs AdamW:**

```
Adam:   grad = grad + wd * param  →  actualización adaptativa
AdamW:  actualización adaptativa  →  param = param - lr * wd * param
```

**Estado interno por parámetro:**

- `step`: contador de pasos
- `exp_avg`: primer momento
- `exp_avg_sq`: segundo momento

### `rmsprop.py`

**`RMSprop(parameters, lr, alpha=0.99, weight_decay=0.0, momentum=0.0, centered=True, eps=1e-8)`**: Root Mean Square Propagation.

**Características:**

- **Media móvil de gradientes al cuadrado**: Normaliza por la raíz cuadrada de la media móvil del cuadrado del gradiente
  - E[g²] = α _ E[g²] + (1 - α) _ g²
  - θ = θ - lr \* g / (√E[g²] + ε)
- **Centered variant**: Opción para normalizar por la varianza centrada (más estable)
  - E[g] = α _ E[g] + (1 - α) _ g
  - Var[g] = E[g²] - E[g]²
  - θ = θ - lr \* g / (√Var[g] + ε)
- **Momentum opcional**: Puede combinarse con momentum para mejor convergencia
- **Diseñado para RNNs**: Originalmente creado para problemas con gradientes no estacionarios

**Cuándo usar:**

- Entrenamiento de RNNs y LSTMs
- Problemas con gradientes muy variables
- Alternativa a Adam cuando se busca simplicidad
- Datasets con ruido significativo

**Estado interno por parámetro:**

- `step`: contador de pasos
- `exp_avg_sq`: segundo momento (E[g²])
- `exp_avg`: primer momento (E[g], solo si centered=True)
- `velocity`: acumulador de momentum (si momentum > 0)

## Planificadores de Learning Rate

### `lr_scheduler.py`

Contiene tres estrategias para ajustar dinámicamente el learning rate durante el entrenamiento.

#### `StepLR(optimizer, step_size, gamma=1.0, last_epoch=-1)`

Reduce el learning rate multiplicativamente cada `step_size` épocas.

**Fórmula:**

```
lr = initial_lr * gamma^(epoch // step_size)
```

**Características:**

- Simple y predecible
- Decaimiento escalonado (step decay)
- Útil cuando se conoce aproximadamente cuándo el entrenamiento se estanca

**Cuándo usar:**

- Entrenamiento estándar de CNNs
- Cuando tienes un presupuesto fijo de épocas
- Para ajustar fino después de preentrenamiento

#### `CosineAnnealingLR(optimizer, T_max, eta_min=0.0, last_epoch=-1)`

Reduce el learning rate siguiendo una curva coseno desde `base_lr` hasta `eta_min`.

**Fórmula:**

```
lr = eta_min + (base_lr - eta_min) * (1 + cos(π * epoch / T_max)) / 2
```

**Características:**

- Decaimiento suave y continuo
- Evita caídas bruscas que pueden desestabilizar el entrenamiento
- Ampliamente usado en entrenamiento de ImageNet

**Cuándo usar:**

- Entrenamiento largo de modelos grandes
- Cuando se busca convergencia suave
- Competiciones de deep learning (muy popular)
- Como parte de ciclos de warm restarts

#### `OneCycleLR(optimizer, max_lr, total_steps, pct_start=0.3, div_factor=25.0, final_div_factor=1e4, cycle_momentum=True, max_momentum=0.95, last_epoch=-1)`

Implementa la política de 1cycle: learning rate crece linealmente hasta `max_lr`, luego decae con coseno annealing.

**Fases:**

1. **Warmup** (pct_start \* total_steps): lr crece de `initial_lr` a `max_lr`
2. **Annealing** (resto): lr decae con coseno de `max_lr` a `final_lr`

**Características:**

- **Ciclo de momentum inverso**: Momentum alto → bajo → alto (opcional)
- **Super-convergencia**: Permite entrenar con learning rates mucho mayores
- **Regularización implícita**: El ciclo actúa como regularizador
- Basado en el paper "Super-Convergence" de Leslie Smith

**Cuándo usar:**

- Entrenamiento rápido con menos épocas
- Cuando se quiere maximizar el learning rate sin divergir
- Transfer learning y fine-tuning
- Entrenamiento en tiempo limitado

**Configuración de momentum:**

- Detecta automáticamente si el optimizador tiene `momentum` o `betas`
- Ajusta inversamente: cuando lr sube, momentum baja (y viceversa)

## Clase base `Optimizer`

Todos los optimizadores heredan de **`Optimizer`** (definida en [`_interfaces/_optimizer.py`](../_interfaces/README.es.md)).

**Estructura común:**

- **`param_groups`**: Lista de diccionarios, cada uno con parámetros y sus hiperparámetros
  - Permite diferentes learning rates por grupo de parámetros
- **`state`**: Diccionario que mapea parámetros a su estado interno (momentos, velocidades, etc.)
- **`defaults`**: Hiperparámetros por defecto para todos los grupos

**Métodos principales:**

- **`step(closure=None)`**: Ejecuta un paso de optimización
  - Llama internamente a `_step_impl()` que cada optimizador implementa
  - Opcionalmente acepta closure para reevaluar la función de pérdida
- **`zero_grad(set_to_none=False)`**: Limpia los gradientes de todos los parámetros
  - `set_to_none=True`: Libera memoria estableciendo gradientes a None
  - `set_to_none=False`: Pone gradientes a cero (más rápido para siguientes backward)
- **`add_param_group(param_group)`**: Añade un nuevo grupo de parámetros
  - Útil para fine-tuning con diferentes learning rates por capa

## Clase base `_LRScheduler`

Todos los schedulers heredan de **`_LRScheduler`** (definida en [`_interfaces/_lr_scheduler.py`](../_interfaces/README.es.md)).

**Estructura común:**

- **`optimizer`**: Referencia al optimizador que se está ajustando
- **`base_lrs`**: Learning rates iniciales de cada param_group
- **`last_epoch`**: Contador de épocas/steps

**Métodos principales:**

- **`step()`**: Avanza el scheduler y actualiza el learning rate del optimizador
- **`get_lr()`**: Método abstracto que cada scheduler implementa
  - Calcula el nuevo learning rate basado en `last_epoch`
- **`get_last_lr()`**: Devuelve el último learning rate aplicado

## Diseño y filosofía

El módulo `optim` de NovaNN está diseñado siguiendo estos principios:

- **Separación de concerns**: Los optimizadores solo se encargan de actualizar parámetros, los schedulers solo de ajustar el learning rate
- **Flexibilidad**: Sistema de param_groups permite configuración granular por capa
- **Consistencia con PyTorch**: API familiar para facilitar el aprendizaje y la portabilidad de código
- **Estado explícito**: Cada optimizador mantiene su estado interno de forma clara y accesible
- **Weight decay consciente**: Diferenciación entre weight decay acoplado (Adam) y desacoplado (AdamW)
- **BatchNorm awareness**: Optimizadores detectan parámetros de BatchNorm para no aplicarles weight decay

## Integración con otros módulos

El módulo `optim` se integra estrechamente con:

- **[`nn/`](../nn/README.es.md)**: Opera sobre `model.parameters()` para actualizar pesos
- **[`autograd/`](../autograd/README.es.md)**: Lee gradientes calculados automáticamente en `.grad`
- **[`_interfaces/`](../_interfaces/README.es.md)**: Hereda de clases base `Optimizer` y `_LRScheduler`

## Comparación de optimizadores

| Optimizador | Memoria | Velocidad | Mejor para                    | Hiperparámetros clave |
| ----------- | ------- | --------- | ----------------------------- | --------------------- |
| **SGD**     | Baja    | Rápido    | Convexos, baselines           | lr, momentum          |
| **Adam**    | Alta    | Medio     | Deep learning general         | lr, betas             |
| **AdamW**   | Alta    | Medio     | Transformers, grandes modelos | lr, weight_decay      |
| **RMSprop** | Media   | Medio     | RNNs, gradientes ruidosos     | lr, alpha             |

## Ejemplos de uso

### Ejemplo 1: Ciclo de entrenamiento completo con SGD y StepLR

```python
import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import SGD
from nova.optim.lr_scheduler import StepLR

# Definir modelo
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializar modelo, optimizador y scheduler
model = SimpleNet()
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Ciclo de entrenamiento
num_epochs = 30
for epoch in range(num_epochs):
    model.train()

    # Forward pass
    inputs = nova.randn(32, 784)
    targets = nova.randint(0, 10, (32,))

    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Actualizar learning rate
    scheduler.step()

    if epoch % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, LR={current_lr:.6f}")

# Evaluación
model.eval()
test_inputs = nova.randn(10, 784)
with nova.no_grad():
    predictions = model(test_inputs)
    print(f"Predicciones shape: {predictions.shape}")
```

### Ejemplo 2: Transfer learning con AdamW y diferentes learning rates por capa

```python
import nova
import nova.nn as nn
from nova.optim import AdamW
from nova.optim.lr_scheduler import CosineAnnealingLR

# Modelo preentrenado (simulado)
class PretrainedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extractor "congelado"
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Clasificador nuevo
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = PretrainedCNN()

# Configurar diferentes learning rates: features más bajo, classifier más alto
optimizer = AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters()}
], lr=1e-3, weight_decay=0.01)

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
criterion = nn.CrossEntropyLoss()

# Fine-tuning
num_epochs = 50
for epoch in range(num_epochs):
    model.train()

    # Batch de imágenes
    images = nova.randn(16, 3, 32, 32)
    labels = nova.randint(0, 10, (16,))

    # Forward + backward
    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if epoch % 10 == 0:
        lr_features = optimizer.param_groups[0]['lr']
        lr_classifier = optimizer.param_groups[1]['lr']
        print(f"Epoch {epoch}: Loss={loss.item():.4f}")
        print(f"  LR features: {lr_features:.8f}, LR classifier: {lr_classifier:.8f}")

print("\nEntrenamiento completado!")
```

### Ejemplo 3: Entrenamiento rápido con OneCycleLR y gradient clipping

```python
import nova
import nova.nn as nn
import nova.nn.functional as F
from nova.optim import Adam
from nova.optim.lr_scheduler import OneCycleLR
from nova.nn.utils import clip_grad_norm_

# Modelo más profundo
class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x)

model = DeepNet()

# Configurar optimizador y OneCycleLR
total_steps = 1000
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = OneCycleLR(
    optimizer,
    max_lr=0.01,
    total_steps=total_steps,
    pct_start=0.3,
    cycle_momentum=True
)
criterion = nn.CrossEntropyLoss()

# Entrenamiento con super-convergencia
for step in range(total_steps):
    model.train()

    # Simular mini-batch
    inputs = nova.randn(64, 784)
    targets = nova.randint(0, 10, (64,))

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass con gradient clipping
    optimizer.zero_grad()
    loss.backward()

    # Clipear gradientes para estabilidad
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0, get_norm=True)

    optimizer.step()
    scheduler.step()

    # Logging periódico
    if step % 100 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        current_momentum = optimizer.param_groups[0]['betas'][0]
        print(f"Step {step}/{total_steps}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  LR: {current_lr:.6f}, Momentum: {current_momentum:.4f}")
        print(f"  Grad norm: {grad_norm:.4f}")

# Evaluación final
model.eval()
with nova.no_grad():
    test_inputs = nova.randn(100, 784)
    test_outputs = model(test_inputs)
    predictions = test_outputs.argmax(dim=1)
    print(f"\nEvaluación: {predictions.shape[0]} muestras procesadas")
```

### Ejemplo 4: Comparación de Adam vs AdamW con weight decay

```python
import nova
import nova.nn as nn
from nova.optim import Adam, AdamW

# Modelo simple para comparar
class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Entrenar con Adam (weight decay acoplado)
print("=== Entrenando con Adam ===")
model_adam = TinyNet()
optimizer_adam = Adam(model_adam.parameters(), lr=0.01, weight_decay=0.1)

for step in range(5):
    x = nova.randn(8, 10)
    y = nova.randn(8, 1)

    pred = model_adam(x)
    loss = F.mse_loss(pred, y)

    optimizer_adam.zero_grad()
    loss.backward()
    optimizer_adam.step()

    print(f"Step {step}: Loss={loss.item():.4f}, "
          f"Weight norm={nova.norm(model_adam.fc.weight, ord=2).item():.4f}")

# Entrenar con AdamW (weight decay desacoplado)
print("\n=== Entrenando con AdamW ===")
model_adamw = TinyNet()
optimizer_adamw = AdamW(model_adamw.parameters(), lr=0.01, weight_decay=0.1)

for step in range(5):
    x = nova.randn(8, 10)
    y = nova.randn(8, 1)

    pred = model_adamw(x)
    loss = F.mse_loss(pred, y)

    optimizer_adamw.zero_grad()
    loss.backward()
    optimizer_adamw.step()

    print(f"Step {step}: Loss={loss.item():.4f}, "
          f"Weight norm={nova.norm(model_adamw.fc.weight, ord=2).item():.4f}")

print("\nNota: AdamW típicamente produce mejor regularización en modelos grandes")
```

### Ejemplo 5: Guardado y carga de estado del optimizador

```python
import nova
import nova.nn as nn
from nova.optim import SGD
from nova.optim.lr_scheduler import StepLR

# Crear modelo y optimizador
model = nn.Linear(10, 5)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Entrenar por algunos pasos
for step in range(3):
    x = nova.randn(4, 10)
    y = nova.randn(4, 5)

    pred = model(x)
    loss = F.mse_loss(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    print(f"Step {step}: Loss={loss.item():.4f}")

# Guardar checkpoint completo
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state,
    'scheduler_last_epoch': scheduler.last_epoch,
    'epoch': 3
}
nova.save(checkpoint, 'checkpoint.pth')

# Crear nuevo modelo y optimizador
new_model = nn.Linear(10, 5)
new_optimizer = SGD(new_model.parameters(), lr=0.1, momentum=0.9)
new_scheduler = StepLR(new_optimizer, step_size=5, gamma=0.5)

# Cargar checkpoint
loaded = nova.load('checkpoint.pth')
new_model.load_state_dict(loaded['model_state_dict'])
new_optimizer.state = loaded['optimizer_state_dict']
new_scheduler.last_epoch = loaded['scheduler_last_epoch']

# Continuar entrenamiento
for step in range(3, 6):
    x = nova.randn(4, 10)
    y = nova.randn(4, 5)

    pred = new_model(x)
    loss = F.mse_loss(pred, y)

    new_optimizer.zero_grad()
    loss.backward()
    new_optimizer.step()
    new_scheduler.step()

    print(f"Step {step}: Loss={loss.item():.4f}")
```

---

> Para más detalles sobre implementaciones específicas, consulta el código fuente de cada optimizador y scheduler.
