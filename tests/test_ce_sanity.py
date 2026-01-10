import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.losses import cross_entropy_logits  

logits = Tensor(np.array([[10.0, -10.0, -10.0],
                          [-10.0, 10.0, -10.0]]), requires_grad=True)
y = np.array([0, 1], dtype=int)

loss = cross_entropy_logits(logits, y)
loss.backward()

print("loss:", float(loss.data))     # should be tiny
print("grad:", logits.grad)          # should be near 0 on correct class, ~softmax elsewhere
