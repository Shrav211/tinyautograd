from tinygrad.tensor import Tensor  # adjust import path

# synthetic data
xs = [1.0, 2.0, 3.0, 4.0]
ys = [5.0, 8.0, 11.0, 14.0]  # y = 3x + 2

# parameters
w = Tensor(0.0, requires_grad=True)
b = Tensor(0.0, requires_grad=True)

lr = 0.01
steps = 2000

for step in range(steps):
    # reset grads
    w.zero_grad()
    b.zero_grad()

    # compute loss (mean squared error)
    loss = Tensor(0.0, requires_grad=False)
    for x, y in zip(xs, ys):
        x = Tensor(x, requires_grad=False)
        y = Tensor(y, requires_grad=False)

        pred = w * x + b
        err = pred - y
        loss = loss + (err ** 2)

    # optional: mean
    loss = loss * (1.0 / len(xs))

    # backprop
    loss.backward()

    # SGD update
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    if step % 200 == 0:
        print(step, loss.data, w.data, b.data)
