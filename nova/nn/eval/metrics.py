from __future__ import annotations
import nova
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nova import Tensor
    from nova.nn import Module
    from nova.utils.data import DataLoader


def r2_score(
    model: Module,
    data_loader: DataLoader,
) -> float:

    all_y_true = []
    all_y_pred = []

    with nova.no_grad():
        for X_batch, y_batch in data_loader:
            y_pred = model(X_batch)

            all_y_pred.append(y_pred.reshape(-1))
            all_y_true.append(y_batch.reshape(-1))

        y_true = nova.cat(all_y_true)
        y_pred = nova.cat(all_y_pred)

        sse = nova.sum((y_true - y_pred) ** 2)

        sst = nova.sum((y_true - nova.mean(y_true)) ** 2).item()

        if sst == 0:
            return 1.0

        r2 = (1 - (sse / sst)).item()
    return r2


def accuracy(model: Module, loader: DataLoader) -> float:
    model.eval()
    total = 0
    correct = 0

    with nova.no_grad():
        for input, target in loader:
            output = model(input)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
        acc = correct / total
    return acc
