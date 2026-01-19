import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.optim import SGD, AdamW

def main():
    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    y = (x * 1000.0).sum()
    y.backward()

    # raw norm should be huge
    raw_norm = float(np.sqrt(np.sum(x.grad * x.grad)))
    assert raw_norm > 10.0

    opt = SGD([x], lr=0.1)
    n = opt.clip_grad_norm_(max_norm=1.0)

    clipped_norm = float(np.sqrt(np.sum(x.grad * x.grad)))
    assert abs(clipped_norm - 1.0) < 1e-6, (n, clipped_norm)

    print("[OK] grad norm clipping works")

if __name__ == "__main__":
    main()
