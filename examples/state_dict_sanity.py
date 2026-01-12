import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP_BN, MLP_LN, cross_entropy_logits
from tinygrad.optim import AdamW

def make_blobs(n_per_class=50, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[-2,0],[2,0],[0,2.5]], dtype=np.float32)
    Xs, ys = [], []
    for k,c in enumerate(centers):
        Xs.append(c + 0.6*rng.randn(n_per_class,2).astype(np.float32))
        ys.append(np.full((n_per_class,), k, dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]

def train_a_bit(model, X, y, steps=200):
    opt = AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for s in range(steps):
        opt.zero_grad()
        logits = model(Tensor(X, requires_grad=False))
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()
    return float(loss.data)

def main():
    X, y = make_blobs()

    model = MLP_BN(2, 32, 3, dropout_p=0.0)
    model.train()
    L = train_a_bit(model, X, y, steps=200)

    # IMPORTANT: eval mode for deterministic compare (BN + dropout)
    model.eval()
    out1 = model(Tensor(X, requires_grad=False)).data

    sd = model.state_dict()

    model2 = MLP_BN(2, 32, 3, dropout_p=0.0)
    model2.load_state_dict(sd, strict=True)
    model2.eval()
    out2 = model2(Tensor(X, requires_grad=False)).data

    print("train_loss:", L)
    print("max_abs_diff:", np.max(np.abs(out1 - out2)))

if __name__ == "__main__":
    main()
