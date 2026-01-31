import cupy as cp
from tinygrad.tensor import Tensor

x = Tensor(cp.ones((2,2)), requires_grad=True)
y = x + 2
assert isinstance(y.data, cp.ndarray)
