import numpy as np
from tinygrad.tensor import Tensor

x = Tensor(np.ones((4, 5)), requires_grad=True)
loss = x.sum() * 0.25
loss.backward()
print("unique grad:", np.unique(x.grad))
