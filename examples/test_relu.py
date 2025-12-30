import numpy as np
from tinygrad.tensor import Tensor

x = Tensor(np.array([-2.0, -1.0, 0.0, 3.0]), requires_grad=True)
y = x.relu()
loss = y.sum()
loss.backward()

print("y:", y.data)
print("x.grad", x.grad)