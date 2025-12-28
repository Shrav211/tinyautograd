import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.optim import SGD

xs = np.array([1., 2., 3., 4.])
ys = np.array([5., 8., 11., 14.])  # 3x + 2

model = Linear(w_init=0.0, b_init=0.0)
opt = SGD(model.parameters(), lr=0.05)

steps = 1000
for step in range(steps):
    opt.zero_grad()

    x = Tensor(xs)
    y = Tensor(ys)

    pred = model(x)              # broadcast: (,) * (4,) + (,) -> (4,)
    err = pred - y               # (4,)
    loss = (err ** 2).sum() * (1.0 / err.data.size)  # scalar

    loss.backward()
    print("loss", loss.data, "w.grad", model.w.grad, "b.grad", model.b.grad)
    # break
    opt.step()

    if step % 100 == 0:
        print(step, float(loss.data), float(model.w.data), float(model.b.data))
