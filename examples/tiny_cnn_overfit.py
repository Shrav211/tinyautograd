import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d, MaxPool2d, Flatten, Linear, Module, cross_entropy_logits
from tinygrad.optim import AdamW

def make_bar_images(n=256, H=8, W=8, seed=0):
    rng = np.random.RandomState(seed)
    X = np.zeros((n, 1, H, W), dtype=np.float32)
    y = np.zeros((n,), dtype=int)

    for i in range(n):
        cls = rng.randint(0, 2)
        y[i] = cls
        img = rng.randn(H, W).astype(np.float32) * 0.2

        if cls == 0:
            c = rng.randint(1, W-1)
            img[:, c] += 2.0
        else:
            r = rng.randint(1, H-1)
            img[r, :] += 2.0

        X[i, 0] = img

    # shuffle and split
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    ntr = int(0.8 * n)
    return (X[:ntr], y[:ntr]), (X[ntr:], y[ntr:])

class TinyCNN(Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2d(1, 8, 3, padding=1)
        self.p1 = MaxPool2d(2, 2)
        self.c2 = Conv2d(8, 16, 3, padding=1)
        self.pool = MaxPool2d(2, 2)
        self.flat = Flatten(1)
        self.fc = Linear(16 * 2 * 2, 2, init="xavier")  # after 8x8 -> pool -> 4x4 -> pool -> 2x2

    def __call__(self, x: Tensor) -> Tensor:
        x = self.c1(x).relu()
        x = self.p1(x)
        x = self.c2(x).relu()
        x = self.pool(x)
        x = self.flat(x)
        return self.fc(x)

def accuracy(logits, y):
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def main():
    (Xtr, ytr), (Xva, yva) = make_bar_images(n=512)

    model = TinyCNN()
    opt = AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)

    for step in range(1000):
        opt.zero_grad()
        logits = model(Tensor(Xtr, requires_grad=False))
        loss = cross_entropy_logits(logits, ytr)
        loss.backward()
        opt.step()

        if step % 100 == 0:
            model.eval()
            tr_logits = model(Tensor(Xtr, requires_grad=False)).data
            va_logits = model(Tensor(Xva, requires_grad=False)).data
            tr_acc = accuracy(tr_logits, ytr)
            va_acc = accuracy(va_logits, yva)
            model.train()
            print(f"step {step:4d} loss {float(loss.data):.4f} tr_acc {tr_acc:.3f} va_acc {va_acc:.3f}")

    print("FINAL val_acc:", accuracy(model(Tensor(Xva, requires_grad=False)).data, yva))

if __name__ == "__main__":
    main()
