import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP, LayerNorm
from tinygrad.optim import SGD, AdamW

def iterate_minibatches(X, Y, batch_size=4, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, N, batch_size):
        batch_idx = idx[start:start+batch_size]
        yield X[batch_idx], Y[batch_idx]

def mse(pred, y):
    err = pred - y
    return (err ** 2).sum() * (1.0 / err.data.size)

def bce_with_logits(logits: Tensor, target: Tensor):
    z = logits
    loss = (z.relu() - z * target + (1 + (-z.abs()).exp()).log())
    return loss.sum() * (1.0 / target.data.size)

X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
Y = np.array([[0.],[1.],[1.],[0.]])

model = MLP(2, 8, 1)

# opt = SGD(model.parameters(), lr=0.1)
opt = AdamW(model.parameters(), lr=0.01, weight_decay=1e-2)  # AdamW likes smaller LR than SGD

accum_steps = 4
micro_batch_size = 1

# for epoch in range(2000):
#     for xb, yb in iterate_minibatches(X, Y, batch_size=2, shuffle=True):
#         x = Tensor(xb, requires_grad=False)
#         y = Tensor(yb, requires_grad=False)

#         # 1) clear grads
#         opt.zero_grad()

#         # 2) forward + loss
#         logits = model(x)
#         loss = bce_with_logits(logits, y)

#         # 3) backward
#         loss.backward()

#         # 4) update
#         opt.step()

#     if epoch % 200 == 0:
#         print(epoch, float(loss.data))

for epoch in range(2000):
    opt.zero_grad()
    step_in_accum = 0
    loss_sum = 0.0  # track true loss over epoch
    num_samples = 0

    for xb, yb in iterate_minibatches(X, Y, batch_size=micro_batch_size, shuffle=True):
        x = Tensor(xb, requires_grad=False)
        y = Tensor(yb, requires_grad=False)

        logits = model(x)
        loss_raw = bce_with_logits(logits, y)   # true loss for this microbatch
        loss_sum += float(loss_raw.data)

        loss_scaled = loss_raw * (1.0 / accum_steps)
        loss_scaled.backward()
        num_samples += y.data.shape[0]

        step_in_accum += 1
        if step_in_accum == accum_steps:
            
            if epoch % 200 == 0:
                # print("W1 grad std:", model.l1.W.grad.std())
                print("||W1||:", np.linalg.norm(model.l1.W.data))

            opt.step()
            opt.zero_grad()
            step_in_accum = 0

    # handle leftovers (general case)
    if step_in_accum != 0:
        opt.step()
        opt.zero_grad()

    if epoch % 200 == 0:
        preds = model(Tensor(X)).sigmoid().data
        acc = np.mean((preds > 0.5) == Y)
        print(epoch, loss_sum / num_samples, "acc", acc)

# x = Tensor(np.random.randn(128, 2), requires_grad=False)
# h = model.l1(x).data
# print("h mean/std:", h.mean(), h.std())

x = Tensor(np.random.randn(2, 3), requires_grad=True)
y = x.sum()        # axis=None
y.backward()
print(x.grad.shape, x.grad)


