import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Module
from tinygrad.optim import AdamW
from tinygrad.losses import cross_entropy_logits
import matplotlib.pyplot as plt

def make_blobs(n_per_class=400, seed=0):
    rng = np.random.RandomState(seed)

    centers = np.array([
        [-2.0, 0.0],
        [2.0, 0.0],
        [0.0, 2.5],
    ])

    Xs, ys = [], []
    for k, c in enumerate(centers):
        Xk = c + 1.0 * rng.randn(n_per_class, 2)
        yk = np.full((n_per_class, ), k, dtype=int)
        Xs.append(Xk)
        ys.append(yk)

    X = np.vstack(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(int)

    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def train_val_split(X, y, val_frac=0.2, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    n_val = int(len(y) * val_frac)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def iterate_minibatches(X, y, batch_size=64, shuffle=True, seed=0):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)

    for start in range(0, N, batch_size):
        b = idx[start:start+batch_size]
        yield X[b], y[b]


def accuracy_from_logits(logits_np, y_np):
    pred = np.argmax(logits_np, axis=1)
    return np.mean(pred == y_np)

# Simple 2-layer MLP for multiclass
class MLP3(Module):
    def __init__(self, in_dim=2, hidden=32, out_dim=3):
        self.l1 = Linear(in_dim, hidden)
        self.l2 = Linear(hidden, out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        return self.l2(self.l1(x).relu())

def main():
    X, y = make_blobs(n_per_class=400, seed=0)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_frac=0.2, seed=1)

    model = MLP3(in_dim=2, hidden=32, out_dim=3)
    opt = AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)

    steps = 2000
    batch_size = 64

    for step in range(steps):
        # one epoch-ish minibatch stream; keep it simple
        xb, yb = next(iterate_minibatches(Xtr, ytr, batch_size=batch_size, shuffle=True, seed=step))

        x = Tensor(xb, requires_grad=False)
        # y stays numpy int array for CE
        opt.zero_grad()

        logits = model(x)  # (B,3)
        loss = cross_entropy_logits(logits, yb)

        loss.backward()
        opt.step()

        if step % 200 == 0:
            # eval
            model.eval()
            tr_logits = model(Tensor(Xtr, requires_grad=False)).data
            va_logits = model(Tensor(Xva, requires_grad=False)).data
            model.train()

            tr_acc = accuracy_from_logits(tr_logits, ytr)
            va_acc = accuracy_from_logits(va_logits, yva)

            print(f"step {step:4d}  loss {float(loss.data):.4f}  tr_acc {tr_acc:.3f}  va_acc {va_acc:.3f}")

    # final
    model.eval()
    va_logits = model(Tensor(Xva, requires_grad=False)).data
    va_acc = accuracy_from_logits(va_logits, yva)
    print("FINAL val_acc:", va_acc)
    plot_decision_boundary(model, Xva, yva, title="Val decision boundary")
    for name, m in model.named_modules():
        print(name, type(m).__name__)
    for name, p in model.named_parameters():
        print(name, p.data.shape)
    for name, p in model.named_parameters():
        gstd = float(p.grad.std()) if hasattr(p.grad, "std") else float(np.std(p.grad))
        print(name, gstd)

def plot_decision_boundary(model, X, y, title=""):
    # grid
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    model.eval()
    logits = model(Tensor(grid, requires_grad=False)).data
    model.train()

    pred = np.argmax(logits, axis=1).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, pred, alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y, s=10)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()
    
    
