import numpy as np
from src.layers import BatchNorm1d, Conv2d, Linear
from src.core.config import DEFAULT_UNIFORM_INIT_MAP


in_channel = 3
out_channel = 8
kernel_size = 3
init = DEFAULT_UNIFORM_INIT_MAP["relu"]

img = np.random.randn(5, 3, 10, 10)
conv1 = Conv2d(in_channel, out_channel, kernel_size, initializer=init)
out = conv1.forward(img)
print(out.shape)
