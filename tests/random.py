import cupy as cp
from tinygrad.tensor import Tensor

x = Tensor(cp.ones((2,2)), requires_grad=True)
y = (x + 2).sum()
y.backward()

assert isinstance(x.grad, cp.ndarray)
print("x.grad type:", type(x.grad), "shape:", x.grad.shape)
