import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.nn import MLP_BN, cross_entropy_logits
from tinygrad.optim import AdamW
from tinygrad.serialization import save_state_dict, load_state_dict


def make_blobs(n_per_class=50, seed=0):
    rng = np.random.RandomState(seed)
    centers = np.array([[-2, 0], [2, 0], [0, 2.5]], dtype=np.float32)
    Xs, ys = [], []
    for k, c in enumerate(centers):
        Xs.append(c + 0.6 * rng.randn(n_per_class, 2).astype(np.float32))
        ys.append(np.full((n_per_class,), k, dtype=int))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def train_a_bit(model, X, y, steps=200):
    opt = AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(Tensor(X, requires_grad=False))
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()
    return float(loss.data)


def main():
    X, y = make_blobs()

    # -------------------------
    # 1) Train + save a checkpoint for hidden_dim=32
    # -------------------------
    model_src = MLP_BN(2, 32, 3, dropout_p=0.0)
    model_src.train()
    L = train_a_bit(model_src, X, y, steps=200)

    ckpt_path = "ckpt_blobs_bn32.npz"
    save_state_dict(model_src.state_dict(), ckpt_path)

    # -------------------------
    # 2) Create a different model: hidden_dim=64
    # -------------------------
    model_tgt = MLP_BN(2, 64, 3, dropout_p=0.0)

    # Load with strict=False (partial load)
    sd = load_state_dict(ckpt_path)
    report = model_tgt.load_state_dict(sd, strict=False)

    # -------------------------
    # 3) Show what happened + prove model runs
    # -------------------------
    model_tgt.eval()
    out = model_tgt(Tensor(X, requires_grad=False)).data

    print("src_train_loss:", L)
    print("tgt_out_shape:", out.shape)

    print("\n=== PARTIAL LOAD REPORT ===")
    print("missing:", report["missing"][:10], "... total:", len(report["missing"]))
    print("unexpected:", report["unexpected"][:10], "... total:", len(report["unexpected"]))
    print("skipped (shape mismatches / errors):")
    for item in report["skipped"][:10]:
        print("  ", item)
    if len(report["skipped"]) > 10:
        print("  ... total skipped:", len(report["skipped"]))


if __name__ == "__main__":
    main()
