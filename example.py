from src.layers import Flatten
import numpy as np


a = np.random.randn(4, 4, 4)
f = Flatten()
a_flat = f.forward(a)
print(f"from {a.shape} dims to {a_flat.shape} dims")
a_normmal = f.backward(a_flat)
print(f"from {a_flat.shape} dim to {a_normmal.shape} dim")
s
