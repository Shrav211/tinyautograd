import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None

from tinygrad.tensor import Tensor
from tinygrad.nn import cross_entropy_logits

def main():
    if cp is None:
        print("[SKIP] cupy not installed")
        return

    N, C = 64, 10
    logits = Tensor(cp.random.randn(N, C).astype(cp.float32), requires_grad=True)

    # labels on CPU numpy (this is the common case)
    y = np.random.randint(0, C, size=(N,), dtype=np.int64)

    loss = cross_entropy_logits(logits, y)
    assert loss.data.shape == (), f"loss should be scalar, got {loss.data.shape}"

    loss.backward()
    assert logits.grad is not None
    assert isinstance(logits.grad, cp.ndarray), "grad must be on GPU (cupy)"
    assert logits.grad.shape == (N, C)

    print("[OK] GPU smoke: CE gather forward/backward")

if __name__ == "__main__":
    main()
