import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn import Module, Linear, LayerNorm, BatchNorm1d, Dropout, MLP_LN, MLP_BN
from tinygrad.optim import AdamW


# ---------- utils ----------
def iterate_minibatches(X, Y, batch_size=32, shuffle=True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], Y[batch_idx]


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def bce_with_logits(logits: Tensor, target: Tensor):
    # stable BCE with logits:
    # loss = relu(z) - z*y + log(1 + exp(-abs(z)))
    z = logits
    loss = (z.relu() - z * target + (1 + (-z.abs()).exp()).log())
    return loss.sum() * (1.0 / target.data.size)


def accuracy_from_logits(logits_np, y_np):
    probs = sigmoid_np(logits_np)
    preds = (probs > 0.5).astype(y_np.dtype)
    return float(np.mean(preds == y_np))


# ---------- datasets ----------
def make_xor():
    X = np.array([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]], dtype=float)
    Y = np.array([[0.],
                  [1.],
                  [1.],
                  [0.]], dtype=float)
    return X, Y


def make_moons(n=512, noise=0.12, seed=0):
    """
    Simple 'two moons' generator (no sklearn).
    Returns X: (n,2), Y: (n,1) in {0,1}.
    """
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1

    t1 = rng.random(n1) * np.pi
    t2 = rng.random(n2) * np.pi

    # first moon
    x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)

    # second moon (shifted + flipped)
    x2 = np.stack([1 - np.cos(t2), 1 - np.sin(t2) - 0.5], axis=1)

    X = np.concatenate([x1, x2], axis=0)
    X += rng.normal(scale=noise, size=X.shape)

    Y = np.concatenate([np.zeros((n1, 1)), np.ones((n2, 1))], axis=0)

    # shuffle
    idx = rng.permutation(n)
    return X[idx].astype(float), Y[idx].astype(float)


def train_val_split(X, Y, val_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = rng.permutation(N)
    n_val = max(1, int(N * val_frac))
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], Y[tr_idx], X[val_idx], Y[val_idx]


# ---------- models ----------
# class MLP_LN(Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout_p=0.0):
#         super().__init__()
#         self.l1 = Linear(in_dim, hidden_dim)
#         self.norm = LayerNorm(hidden_dim)
#         self.drop = Dropout(dropout_p)
#         self.l2 = Linear(hidden_dim, out_dim)

#     def __call__(self, x: Tensor) -> Tensor:
#         h = self.l1(x)
#         h = self.norm(h)
#         h = h.relu()
#         h = self.drop(h)
#         return self.l2(h)


# class MLP_BN(Module):
#     def __init__(self, in_dim, hidden_dim, out_dim, dropout_p=0.0):
#         super().__init__()
#         self.l1 = Linear(in_dim, hidden_dim)
#         self.norm = BatchNorm1d(hidden_dim)
#         self.drop = Dropout(dropout_p)
#         self.l2 = Linear(hidden_dim, out_dim)

#     def __call__(self, x: Tensor) -> Tensor:
#         h = self.l1(x)
#         h = self.norm(h)
#         h = h.relu()
#         h = self.drop(h)
#         return self.l2(h)


# ---------- training ----------
def run_experiment(name, X, Y, batch_sizes=(1, 4, 32), steps=2000, lr=1e-2, wd=1e-2, hidden=32, dropout_p=0.0):
    print("\n" + "=" * 80)
    print(f"DATASET: {name}  (N={X.shape[0]})")
    print("=" * 80)

    Xtr, Ytr, Xva, Yva = train_val_split(X, Y, val_frac=0.2, seed=42)

    for bs in batch_sizes:
        print("\n--- batch_size =", bs, "---")

        # fresh models per batch size for fair comparison
        models = {
            "LN": MLP_LN(X.shape[1], hidden, 1, dropout_p=dropout_p),
            "BN": MLP_BN(X.shape[1], hidden, 1, dropout_p=dropout_p),
        }

        for tag, model in models.items():
            opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)

            # training
            for step in range(steps):
                model.train()

                # one epoch-ish pass: for small datasets this is fine
                loss_sum = 0.0
                n_seen = 0

                for xb, yb in iterate_minibatches(Xtr, Ytr, batch_size=bs, shuffle=True):
                    x = Tensor(xb, requires_grad=False)
                    y = Tensor(yb, requires_grad=False)

                    opt.zero_grad()
                    logits = model(x)
                    loss = bce_with_logits(logits, y)
                    loss.backward()
                    opt.step()

                    loss_sum += float(loss.data) * xb.shape[0]
                    n_seen += xb.shape[0]

                # periodic eval
                if step % max(1, steps // 10) == 0:
                    model.eval()

                    # train accuracy (eval mode!) and val accuracy
                    tr_logits = model(Tensor(Xtr, requires_grad=False)).data
                    va_logits = model(Tensor(Xva, requires_grad=False)).data

                    tr_acc = accuracy_from_logits(tr_logits, Ytr)
                    va_acc = accuracy_from_logits(va_logits, Yva)

                    avg_loss = loss_sum / max(1, n_seen)

                    print(f"{tag} step {step:4d}  loss {avg_loss:.4f}  train_acc {tr_acc:.3f}  val_acc {va_acc:.3f}")

            # final eval summary
            model.eval()
            va_logits = model(Tensor(Xva, requires_grad=False)).data
            va_acc = accuracy_from_logits(va_logits, Yva)

            print(f"{tag} FINAL val_acc {va_acc:.3f}")


def main():
    # XOR (tiny; mostly correctness)
    X_xor, Y_xor = make_xor()
    run_experiment(
        name="XOR",
        X=X_xor,
        Y=Y_xor,
        batch_sizes=(1, 4),
        steps=300,          # XOR converges fast; keep this short
        lr=1e-2,
        wd=1e-2,
        hidden=16,
        dropout_p=0.0
    )

    # Moons (where BN vs LN differences show up)
    X_m, Y_m = make_moons(n=1024, noise=0.15, seed=0)
    run_experiment(
        name="TwoMoons",
        X=X_m,
        Y=Y_m,
        batch_sizes=(1, 4, 32),
        steps=200,          # treat each "step" as one epoch-ish pass
        lr=1e-2,
        wd=1e-2,
        hidden=64,
        dropout_p=0.1
    )


if __name__ == "__main__":
    main()
