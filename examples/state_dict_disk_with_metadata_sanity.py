# examples/state_dict_disk_with_metadata_sanity.py
import numpy as np
import time
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP_BN, cross_entropy_logits
from tinygrad.optim import AdamW
from tinygrad.serialization import save_checkpoint, load_checkpoint  # <- adjust to your names

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

def train_steps(model, X, y, steps=200, lr=0.02, wd=1e-3):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for s in range(steps):
        opt.zero_grad()
        logits = model(Tensor(X, requires_grad=False))
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()
    return float(loss.data), opt

def main():
    seed = 0
    np.random.seed(seed)

    X, y = make_blobs(seed=seed)

    model = MLP_BN(2, 32, 3, dropout_p=0.0)
    model.train()
    train_loss, opt = train_steps(model, X, y, steps=200, lr=0.02, wd=1e-3)

    model.eval()
    out1 = model(Tensor(X, requires_grad=False)).data

    ckpt_path = "checkpoints/blob_mlp_bn_ckpt.npz"  # or .pkl/.json, your choice

    meta = {
        "timestamp": time.time(),
        "seed": seed,
        "steps": 200,
        "model": "MLP_BN(2,32,3,dropout_p=0.0)",
        "optimizer": "AdamW",
        "lr": 0.02,
        "weight_decay": 1e-3,
        # add anything you want:
        # "git_commit": "...",
        # "dataset": "blobs_v1",
    }

    # Save
    sd = model.state_dict()
    save_checkpoint(ckpt_path, sd, meta)

    # Load into fresh model
    model2 = MLP_BN(2, 32, 3, dropout_p=0.0)
    ckpt = load_checkpoint(ckpt_path)
    sd2 = ckpt["state_dict"]
    meta2 = ckpt.get("meta", {})

    report = model2.load_state_dict(sd2, strict=True)
    model2.eval()
    out2 = model2(Tensor(X, requires_grad=False)).data

    print("train_loss:", train_loss)
    print("max_abs_diff:", float(np.max(np.abs(out1 - out2))))
    print("load_report:", report)
    print("loaded_meta:", meta2)

if __name__ == "__main__":
    main()
