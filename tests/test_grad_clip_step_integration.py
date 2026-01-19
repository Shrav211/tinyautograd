import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.optim import AdamW

def main():
    # big gradient scenario
    w = Tensor(np.array([10.0, -10.0]), requires_grad=True)

    # loss = 1000 * sum(w) -> grad is huge constant
    loss = (w * 1000.0).sum()
    loss.backward()

    opt = AdamW([w], lr=0.1, weight_decay=0.0)

    # record before
    w0 = w.data.copy()

    # clip then step
    opt.clip_grad_norm_(max_norm=1.0)
    opt.step()

    delta = np.linalg.norm(w.data - w0)
    assert delta < 1.0, delta  # should not jump crazily

    print("[OK] grad clip + step integration works")

if __name__ == "__main__":
    main()
