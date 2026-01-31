# tests/test_gpu_smoke_maxpool2d.py
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from tinygrad.tensor import Tensor


def _skip_if_no_cupy():
    if cp is None:
        print("[SKIP] cupy not installed")
        return True
    return False


def test_maxpool2d_forward_backward_backend():
    # pick H,W divisible-ish
    x = Tensor(cp.random.randn(2, 3, 7, 7).astype(cp.float32), requires_grad=True)

    y = x.maxpool2d(kernel_size=2, stride=2)
    assert isinstance(y.data, cp.ndarray), "maxpool2d output must be cupy"
    assert y.data.shape == (2, 3, 3, 3), f"unexpected output shape {y.data.shape}"

    loss = y.sum()
    loss.backward()

    assert x.grad is not None and isinstance(x.grad, cp.ndarray), "x.grad must be cupy"
    assert x.grad.shape == x.data.shape


def main():
    if _skip_if_no_cupy():
        return

    test_maxpool2d_forward_backward_backend()
    print("[OK] GPU smoke: maxpool2d")


if __name__ == "__main__":
    main()
