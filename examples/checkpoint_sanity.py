# examples/checkpoint_sanity.py
import os
import numpy as np

from tinygrad.tensor import Tensor, no_grad
from tinygrad.nn import MLP_BN, cross_entropy_logits
from tinygrad.optim import AdamW
from tinygrad.serialization import save_checkpoint, load_checkpoint


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


def one_step(model, opt, X, y):
    opt.zero_grad()
    logits = model(Tensor(X, requires_grad=False))
    loss = cross_entropy_logits(logits, y)
    loss.backward()
    opt.step()
    return float(loss.data)


def eval_logits(model, X):
    model.eval()
    with no_grad():
        out = model(Tensor(X, requires_grad=False)).data.copy()
    model.train()
    return out


def main():
    np.random.seed(0)

    # small dataset
    X, y = make_blobs(n_per_class=30, seed=1)

    # model + opt
    model = MLP_BN(2, 32, 3, dropout_p=0.0)
    model.train()
    opt = AdamW(model.parameters(), lr=0.02, weight_decay=1e-3)

    # train a bit
    for _ in range(50):
        one_step(model, opt, X, y)

    # deterministic compare: eval mode (BN uses running stats)
    out_before = eval_logits(model, X)

    # save checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    path = "checkpoints/ckpt_sanity.npz"
    save_checkpoint(path, model, opt, meta={"note": "checkpoint sanity"})

    # create fresh model + opt (different random init)
    model2 = MLP_BN(2, 32, 3, dropout_p=0.0)
    model2.train()
    opt2 = AdamW(model2.parameters(), lr=0.02, weight_decay=1e-3)

    # load checkpoint
    ckpt = load_checkpoint(path, model2, opt2, strict=True)

    # check outputs match exactly after load
    out_after = eval_logits(model2, X)
    max_abs_diff = float(np.max(np.abs(out_before - out_after)))
    print("max_abs_diff(after load):", max_abs_diff)
    assert max_abs_diff == 0.0, "Model restore failed: outputs differ after load"

    # now do ONE more identical step on both (checks optimizer state restore)
    loss1 = one_step(model, opt, X, y)
    loss2 = one_step(model2, opt2, X, y)

    out1 = eval_logits(model, X)
    out2 = eval_logits(model2, X)

    max_abs_diff2 = float(np.max(np.abs(out1 - out2)))
    print("loss_original:", loss1)
    print("loss_loaded:  ", loss2)
    print("max_abs_diff(after 1 more step):", max_abs_diff2)

    # loss might be float-close, but outputs should remain identical if everything restored properly
    assert max_abs_diff2 == 0.0, "Optimizer restore failed: models diverged after one more step"

    print("[OK] checkpoint sanity passed")
    print("loaded meta:", ckpt.get("meta", {}))


if __name__ == "__main__":
    main()
