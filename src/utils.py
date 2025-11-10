import numpy as np


import numpy as np


import numpy as np


def accuracy(model, data_loader):

    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in data_loader:

        y_pred = model.forward(X_batch)

        pred_classes = np.argmax(y_pred, axis=1)

        total_correct += np.sum(pred_classes == y_batch)
        total_samples += y_batch.shape[0]

    return total_correct / total_samples


def numeric_grad_elementwise(act_forward, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = act_forward(x).copy()
        x[idx] = orig - eps
        f_minus = act_forward(x).copy()
        grad[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def numeric_grad_scalar_from_softmax(softmax_forward, x, G, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        L_plus = np.sum(softmax_forward(x) * G)
        x[idx] = orig - eps
        L_minus = np.sum(softmax_forward(x) * G)
        grad[idx] = (L_plus - L_minus) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def numeric_grad_scalar_wrt_x(forward_fn, x, G, eps=1e-6):
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        L_plus = np.sum(forward_fn(x) * G)
        x[idx] = orig - eps
        L_minus = np.sum(forward_fn(x) * G)
        grad[idx] = (L_plus - L_minus) / (2 * eps)
        x[idx] = orig
        it.iternext()
    return grad


def numeric_grad_wrt_param(layer, param_attr, x, G, eps=1e-6):
    p = getattr(layer, param_attr)
    shape = p.data.shape
    grad = np.zeros_like(p.data, dtype=float)
    it = np.nditer(p.data, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = p.data[idx]
        p.data[idx] = orig + eps
        L_plus = np.sum(layer.forward(x) * G)
        p.data[idx] = orig - eps
        L_minus = np.sum(layer.forward(x) * G)
        grad[idx] = (L_plus - L_minus) / (2 * eps)
        p.data[idx] = orig
        it.iternext()
    return grad
