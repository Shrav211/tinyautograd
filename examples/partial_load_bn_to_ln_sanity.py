# examples/partial_load_bn_to_ln_sanity.py
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

def train_steps(model, X, y, steps=200, lr=0.02, wd=1e-3):
    opt = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        logits = model(Tensor(X, requires_grad=False))
        loss = cross_entropy_logits(logits, y)
        loss.backward()
        opt.step()
    return float(loss.data)

def summarize_report(report, show=20):
    missing = report.get("missing", [])
    unexpected = report.get("unexpected", [])
    skipped = report.get("skipped", [])

    print("\n=== BN -> LN PARTIAL LOAD REPORT ===")
    print(f"missing: {len(missing)}")
    for k in missing[:show]:
        print("  ", k)
    if len(missing) > show: print("  ...")

    print(f"unexpected: {len(unexpected)}")
    for k in unexpected[:show]:
        print("  ", k)
    if len(unexpected) > show: print("  ...")

    print(f"skipped: {len(skipped)}")
    for item in skipped[:show]:
        print("  ", item)
    if len(skipped) > show: print("  ...")

def main():
    X, y = make_blobs(n_per_class=50, seed=0)

    # 1) Train BN model
    bn = MLP_BN(2, 32, 3, dropout_p=0.0)
    bn.train()
    src_loss = train_steps(bn, X, y, steps=200)
    bn.eval()
    out_bn = bn(Tensor(X, requires_grad=False)).data

    # 2) Get checkpoint
    sd_bn = bn.state_dict()

    # 3) Create LN model (same dims!)
    ln = MLP_LN(2, 32, 3, dropout_p=0.0)

    # 4) Strict load should fail (BN buffers/params != LN params)
    print("src_train_loss:", src_loss)
    try:
        ln.load_state_dict(sd_bn, strict=True)
        print("ERROR: strict=True unexpectedly succeeded (should not).")
    except Exception as e:
        print("strict=True failed as expected:")
        print(" ", repr(e))

    # 5) Partial load
    # Depending on your API:
    # - If you have strict=False and it returns report, great.
    # - If you require allow_partial=True, set it.
    report = ln.load_state_dict(sd_bn, strict=False)  # or allow_partial=True
    ln.eval()
    out_ln = ln(Tensor(X, requires_grad=False)).data

    print("tgt_out_shape:", out_ln.shape)
    summarize_report(report)

    # Optional: show that shared weights load affected LN outputs (not equal to BN outputs usually)
    print("max_abs_diff(BN_out, LN_out):", float(np.max(np.abs(out_bn - out_ln))))

if __name__ == "__main__":
    main()
