# Módulo `metrics`

El directorio **`metrics/`** implementa **métricas para evaluación de modelos** en tareas de clasificación y regresión.

A diferencia de las funciones de pérdida (loss functions), las métricas **no se usan para backpropagation** y están diseñadas para **acumular estadísticas a través de múltiples batches**, proporcionando una evaluación completa del rendimiento del modelo durante el entrenamiento y validación.

Todas las métricas siguen un patrón consistente de tres pasos: `reset()` → `update()` → `compute()`.

## Estructura general

El módulo `metrics/` está organizado en:

- **`metric.py`**: Clase base abstracta `Metric` que define la interfaz común
- **[`classification/`](#submódulo-classification)**: Métricas para tareas de clasificación
- **[`regression/`](#submódulo-regression)**: Métricas para tareas de regresión

## Clase base `Metric`

Define la interfaz abstracta que todas las métricas deben implementar.

**Métodos obligatorios:**

- **`reset()`**: Limpia el estado interno acumulado
  - Se llama al inicio de cada época para empezar desde cero
  - Inicializa contadores, sumas y estadísticas a sus valores por defecto

- **`update(preds, targets)`**: Acumula estadísticas de un batch
  - Recibe predicciones y targets del batch actual
  - Actualiza contadores internos sin calcular el valor final
  - Se llama una vez por batch durante entrenamiento/evaluación

- **`compute()`**: Calcula y retorna el valor final de la métrica
  - Usa las estadísticas acumuladas de todos los `update()`
  - Se llama al final de la época para obtener el resultado
  - No resetea el estado (hay que llamar `reset()` manualmente)

**Utilidades:**

- **`_check_dims(preds, targets)`**: Valida que las formas coincidan

**Patrón de uso típico:**

```python
metric = SomeMetric()

for epoch in range(num_epochs):
    metric.reset()  # Limpiar para nueva época

    for x, y in loader:
        out = model(x)
        metric.update(out, y)  # Acumular

    score = metric.compute().item()  # Calcular resultado final
    print(f"Epoch {epoch}: {score}")
```

## Submódulo `classification/`

Contiene métricas para evaluar modelos de clasificación.

### `_confusion.py`

**`ConfusionMatrix(num_classes)`**: Matriz de confusión multi-clase.

**Características:**

- **Estructura**: Matriz (num_classes, num_classes)
  - Fila i = clase verdadera
  - Columna j = clase predicha
  - Diagonal = predicciones correctas (TP)
  - Fuera de diagonal = errores
- **Implementación eficiente**: Usa `np.bincount` para conteo rápido
- **Auto-argmax**: Si recibe probabilidades/logits (N, C), aplica argmax automáticamente
- **Validación**: Filtra índices inválidos para evitar errores

**Fórmula:**

```
C[i,j] = count(y_true == i and y_pred == j)
```

### `_stat.py`

Contiene métricas derivadas de la matriz de confusión.

**Clase base `ClassificationStat(num_classes, average)`:**

Todas las métricas de clasificación heredan de esta clase que:

- Mantiene una `ConfusionMatrix` interna
- Calcula TP, FP, TN, FN automáticamente
- Soporta diferentes estrategias de averaging:
  - **`'micro'`**: Calcula métricas globalmente (suma todos los TP, FP, FN)
  - **`'macro'`**: Calcula por clase y promedia sin pesos
  - **`'weighted'`**: Calcula por clase y promedia ponderado por soporte
  - **`None`**: Retorna score por cada clase (array de tamaño num_classes)

#### `Accuracy(num_classes, average='micro')`

Proporción de predicciones correctas.

**Fórmula:**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Interpretación:**

- "¿Qué porcentaje de todas las predicciones fueron correctas?"
- Valor entre 0.0 y 1.0
- Puede ser engañosa en datasets desbalanceados

#### `Precision(num_classes, average='macro')`

Proporción de predicciones positivas que fueron correctas.

**Fórmula:**

```
Precision = TP / (TP + FP)
```

**Interpretación:**

- "De todas las muestras que predije como Positivas, ¿cuántas realmente lo eran?"
- Precision alta = pocos falsos positivos
- Importante cuando el costo de FP es alto

#### `Recall(num_classes, average='macro')`

Proporción de positivos reales que fueron encontrados.

**Fórmula:**

```
Recall = TP / (TP + FN)
```

**Interpretación:**

- "De todas las muestras que realmente eran Positivas, ¿cuántas encontré?"
- Recall alto = pocos falsos negativos
- Importante cuando el costo de FN es alto (ej: detección de enfermedades)

#### `F1Score(num_classes, average='macro')`

Media armónica de Precision y Recall.

**Fórmula:**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

**Interpretación:**

- Balance entre Precision y Recall
- Útil cuando se necesita un solo número que resuma ambas métricas
- Valor entre 0.0 y 1.0
- F1 alto requiere tanto precision como recall altos

### `_roc_auc.py`

**`ROCAUC(num_classes=2)`**: Área bajo la curva ROC.

**Características:**

- **Almacena todos los datos**: A diferencia de otras métricas, guarda todas las predicciones y targets en memoria
- **Curva ROC**: Gráfica de TPR (True Positive Rate) vs FPR (False Positive Rate) para diferentes umbrales
- **AUC**: Área bajo la curva ROC (integral usando regla trapezoidal)
- **Detach automático**: Libera gradientes para evitar leaks de memoria
- **Binario principalmente**: Optimizado para clasificación binaria

**Interpretación:**

- AUC = 1.0: Clasificador perfecto
- AUC = 0.5: Clasificador aleatorio (no mejor que lanzar una moneda)
- AUC < 0.5: Clasificador peor que aleatorio

**Advertencia:**

- Consume mucha memoria en datasets grandes (almacena todo)
- No recomendado para datasets extremadamente grandes

## Submódulo `regression/`

Contiene métricas para evaluar modelos de regresión.

### `_error.py`

#### `MeanSquaredError(squared=True)`

Error cuadrático medio (MSE) o raíz del error cuadrático medio (RMSE).

**Fórmulas:**

```
MSE = (1/N) * Σ(y_true - y_pred)²
RMSE = √MSE
```

**Características:**

- **MSE** (`squared=True`): Penaliza fuertemente errores grandes (por el cuadrado)
- **RMSE** (`squared=False`): En las mismas unidades que el target (más interpretable)
- Sensible a outliers
- Diferenciable (se usa como loss también)

**Cuándo usar:**

- MSE: Cuando errores grandes deben penalizarse mucho
- RMSE: Cuando se necesita interpretabilidad en unidades originales

#### `MeanAbsoluteError()`

Error absoluto medio (MAE).

**Fórmula:**

```
MAE = (1/N) * Σ|y_true - y_pred|
```

**Características:**

- Más robusto a outliers que MSE
- Todos los errores se ponderan igual
- En las mismas unidades que el target
- Fácil de interpretar

**Cuándo usar:**

- Cuando los outliers no deben dominar la métrica
- Cuando se prefiere tratamiento uniforme de errores

### `_r2.py`

**`R2Score()`**: Coeficiente de determinación (R²).

**Fórmula:**

```
R² = 1 - (SS_res / SS_tot)
   = 1 - (Σ(y - ŷ)² / Σ(y - ȳ)²)

donde:
- SS_res = suma de cuadrados residual (errores del modelo)
- SS_tot = suma de cuadrados total (varianza en los datos)
- ȳ = media de los targets
```

**Interpretación:**

- **R² = 1.0**: Modelo perfecto (explica toda la varianza)
- **R² = 0.0**: Modelo tan bueno como predecir la media
- **R² < 0.0**: Modelo peor que predecir la media

**Características:**

- Mide qué proporción de la varianza es explicada por el modelo
- Normalizado (no depende de las unidades)
- Puede ser negativo si el modelo es muy malo
- Sensible a outliers (como MSE)

## Diseño y filosofía

El módulo `metrics` de NovaNN está diseñado siguiendo estos principios:

- **Patrón consistente**: Todas las métricas siguen reset() → update() → compute()
- **Acumulación eficiente**: Mantienen solo estadísticas necesarias, no todos los datos (excepto ROCAUC)
- **Separación de concerns**: Las métricas no calculan gradientes, solo evalúan
- **Flexibilidad**: Soportan averaging strategies para multi-clase
- **Integración con Tensor**: Trabajan directamente con objetos Tensor de NovaNN

## Integración con otros módulos

El módulo `metrics` se integra con:

- **[`nn/`](../nn/README.md)**: Evalúa outputs de modelos durante training/validation
- **[`autograd/`](../autograd/README.md)**: Las métricas hacen `.detach()` internamente para evitar consumir memoria del grafo
- **Tensores**: Todas las métricas operan sobre objetos `Tensor`

## Ejemplos de uso

### Ejemplo 1: Accuracy en clasificación binaria

```python
from nova.metrics import Accuracy

acc = Accuracy(num_classes=2, average='micro')
preds = nova.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])  # Probabilidades
targets = nova.tensor([0, 1, 0])  # Clases verdaderas
acc.update(preds, targets)
print(f"Accuracy: {acc.compute():.2%}")  # 100%
```

### Ejemplo 2: Precision, Recall, F1 en multi-clase

```python
from nova.metrics import Precision, Recall, F1Score

precision = Precision(num_classes=3, average='macro')
recall = Recall(num_classes=3, average='macro')
f1 = F1Score(num_classes=3, average='macro')

# Simular predicciones de un batch
logits = nova.randn(16, 3)  # 16 muestras, 3 clases
targets = nova.randint(0, 3, (16,))

precision.update(logits, targets)
recall.update(logits, targets)
f1.update(logits, targets)

print(f"Precision: {precision.compute():.4f}")
print(f"Recall: {recall.compute():.4f}")
print(f"F1: {f1.compute():.4f}")
```

### Ejemplo 3: ROC-AUC para clasificación binaria

```python
from nova.metrics import ROCAUC

auc = ROCAUC(num_classes=2)

# Probabilidades para clase positiva

probs = nova.tensor([[0.1, 0.9], [0.4, 0.6], [0.35, 0.65], [0.8, 0.2]])
targets = nova.tensor([1, 1, 1, 0])
auc.update(probs, targets)
print(f"AUC: {auc.compute():.4f}")
```

### Ejemplo 4: MSE y MAE para regresión

```python
from nova.metrics import MSE, MAE

mse = MSE(squared=True)
rmse = MSE(squared=False)
mae = MAE()

predictions = nova.tensor([2.5, 0.0, 2.0, 8.0])
targets = nova.tensor([3.0, -0.5, 2.0, 7.0])

mse.update(predictions, targets)
rmse.update(predictions, targets)
mae.update(predictions, targets)

print(f"MSE: {mse.compute():.4f}")
print(f"RMSE: {rmse.compute():.4f}")
print(f"MAE: {mae.compute():.4f}")
```

### Ejemplo 5: R² Score

```python
from nova.metrics import R2Score

r2 = R2Score()
preds = nova.tensor([3.0, 2.5, 4.0, 5.5])
targets = nova.tensor([3.2, 2.4, 4.1, 5.0])
r2.update(preds, targets)
print(f"R² Score: {r2.compute():.4f}") # Cerca de 1.0 = buen ajuste
```

## Ejemplo 6: Loop de entrenamiento completo con múltiples métricas

```python
import nova.nn as nn
from nova.metrics import Accuracy, F1Score, Precision, Recall

model = nn.Sequential(nn.Linear(784, 10))
criterion = nn.CrossEntropyLoss()

# Métricas

train_acc = Accuracy(num_classes=10)
val_acc = Accuracy(num_classes=10)
val_f1 = F1Score(num_classes=10, average='weighted')

for epoch in range(10): # Training
model.train()
train_acc.reset()
for batch in train_loader:
preds = model(batch['input'])
loss = criterion(preds, batch['target']) # ... backward ...
train_acc.update(preds, batch['target'])

    # Validation
    model.eval()
    val_acc.reset()
    val_f1.reset()
    with nova.no_grad():
        for batch in val_loader:
            preds = model(batch['input'])
            val_acc.update(preds, batch['target'])
            val_f1.update(preds, batch['target'])

    print(f"Epoch {epoch}:")
    print(f"  Train Acc: {train_acc.compute():.2%}")
    print(f"  Val Acc: {val_acc.compute():.2%}")
    print(f"  Val F1: {val_f1.compute():.4f}")
```

## Ejemplo 7: Matriz de confusión

```python
from nova.metrics import ConfusionMatrix

cm = ConfusionMatrix(num_classes=3)
preds = nova.tensor([[0.1, 0.8, 0.1], [0.9, 0.05, 0.05], [0.2, 0.3, 0.5]])
targets = nova.tensor([1, 0, 2])
cm.update(preds, targets)
matrix = cm.compute()
print("Confusion Matrix:")
print(matrix)
```

---

> Para más detalles sobre implementaciones específicas, consulta el código fuente en `classification/` y `regression/`.
