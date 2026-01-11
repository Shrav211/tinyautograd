import numpy as np
from tinygrad.tensor import Tensor

x = Tensor(np.ones((4, 5)), requires_grad=True)
b = Tensor(np.ones((5,)), requires_grad=True)

y = x + b          # b broadcast to (4,5)
loss = y.sum()
loss.backward()

print("x.grad unique:", np.unique(x.grad))   # should be [1.]
print("b.grad:", b.grad)                     # should be [4. 4. 4. 4. 4.]
