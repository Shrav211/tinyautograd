import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP
from tinygrad.optim import SGD

#XOR Dataset
X = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.]
])
Y = np.array([
    [0.],
    [1.],
    [1.],
    [0.]
])

model = MLP(in_dim=2, hidden_dim=8, out_dim=1)
opt = SGD(model.parameters(), lr=0.1)

def mse(pred, y):
    err = pred - y
    return (err ** 2).sum() * (1.0 / err.data.size)

def bce_with_logits(logits: Tensor, target: Tensor):
    z = logits
    return (z.relu() - z * target + (1 + (-z.abs()).exp()).log()).sum() * (1.0 / target.data.size)

steps = 5000
for step in range(steps):
    opt.zero_grad()

    x = Tensor(X)
    y = Tensor(Y)

    logits = model(x)          # raw scores
    loss = bce_with_logits(logits, y)

    loss.backward()
    opt.step()

    if step % 500 == 0:
        print(step, float(loss.data))

pred = model(Tensor(X)).sigmoid()
print("pred:", pred.data)
print("W1", model.l1.W.data)
print("b1", model.l1.b.data)
print("W2", model.l2.W.data)
print("b2", model.l2.b.data)
