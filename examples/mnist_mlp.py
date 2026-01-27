# examples/mnist_mlp.py
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.optim import AdamW
from tinygrad.data import DataLoader
from tinygrad.datasets import MNIST
from tinygrad.nn import Module, Linear, cross_entropy_logits

class MNIST_MLP(Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.l1 = Linear(784, hidden, init="he")
        self.l2 = Linear(hidden, 10, init="xavier")

    def __call__(self, x: Tensor) -> Tensor:
        # x: (N, 784)
        h = self.l1(x).relu()
        logits = self.l2(h)  # (N, 10)
        return logits

def accuracy_from_logits(logits_np, y_np):
    # logits_np: (N, C), y_np: (N,)
    pred = np.argmax(logits_np, axis=1)
    return float(np.mean(pred == y_np))

def main():
    # data
    train_ds = MNIST(root="data/mnist", train=True, normalize=True, flatten=True)
    test_ds  = MNIST(root="data/mnist", train=False, normalize=True, flatten=True)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, drop_last=True, seed=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, drop_last=False)

    # model + opt
    model = MNIST_MLP(hidden=128)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    steps = 1000
    eval_every = 100

    for step, (xb, yb) in enumerate(train_loader):
        if step >= steps:
            break

        x = Tensor(xb, requires_grad=False)              # (B,784)
        y = np.array(yb, dtype=int).reshape(-1)         # (B,)

        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy_logits(logits, y)          # scalar Tensor
        loss.backward()
        opt.step()

        if step % eval_every == 0:
            model.eval()
            # quick eval on a couple batches (fast sanity)
            tr_logits = logits.data
            tr_acc = accuracy_from_logits(tr_logits, y)

            # full test accuracy (still cheap on MNIST)
            all_acc = []
            for xb2, yb2 in test_loader:
                x2 = Tensor(xb2, requires_grad=False)
                y2 = np.array(yb2, dtype=int).reshape(-1)
                out = model(x2).data
                all_acc.append(accuracy_from_logits(out, y2))
            te_acc = float(np.mean(all_acc))
            model.train()

            print(f"step {step:4d}  loss {float(loss.data):.4f}  tr_acc {tr_acc:.3f}  te_acc {te_acc:.3f}")

    # final eval
    model.eval()
    all_acc = []
    for xb2, yb2 in test_loader:
        x2 = Tensor(xb2, requires_grad=False)
        y2 = np.array(yb2, dtype=int).reshape(-1)
        out = model(x2).data
        all_acc.append(accuracy_from_logits(out, y2))
    te_acc = float(np.mean(all_acc))
    print("FINAL test acc:", te_acc)

if __name__ == "__main__":
    main()
