from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.optim import SGD

xs = [1.0, 2.0, 3.0, 4.0]
ys = [5.0, 8.0, 11.0, 14.0]

model = Linear(w_init=0.0, b_init=0.0)
opt = SGD(model.parameters(), lr=0.01)

steps = 2000
for step in range(steps):
    opt.zero_grad()

    loss = Tensor(0.0)
    for x, y in zip(xs, ys):
        x = Tensor(x)
        y = Tensor(y)

        pred = model(x)
        loss = loss + (pred - y) ** 2

    loss = loss * (1.0 / len(xs))
    loss.backward()
    opt.step()

    if step % 200 == 0:
        print(step, loss.data, model.w.data, model.b.data)
