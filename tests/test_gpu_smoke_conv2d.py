# tests/test_gpu_smoke_conv2d.py
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


def test_conv2d_forward_backward_backend():
    # small but non-trivial
    N, C, H, W = 2, 3, 8, 8
    F, kH, kW = 4, 3, 3

    x = Tensor(cp.random.randn(N, C, H, W).astype(cp.float32), requires_grad=True)
    w = Tensor(cp.random.randn(F, C, kH, kW).astype(cp.float32), requires_grad=True)
    b = Tensor(cp.random.randn(F).astype(cp.float32), requires_grad=True)

    y = x.conv2d(w, b, stride=1, padding=1)
    assert isinstance(y.data, cp.ndarray), "conv2d output must be cupy"
    assert y.data.shape == (N, F, H, W), f"unexpected output shape {y.data.shape}"

    loss = y.sum()
    loss.backward()

    assert x.grad is not None and isinstance(x.grad, cp.ndarray), "x.grad must be cupy"
    assert w.grad is not None and isinstance(w.grad, cp.ndarray), "w.grad must be cupy"
    assert b.grad is not None and isinstance(b.grad, cp.ndarray), "b.grad must be cupy"

    assert x.grad.shape == x.data.shape
    assert w.grad.shape == w.data.shape
    assert b.grad.shape == b.data.shape


def main():
    if _skip_if_no_cupy():
        return

    test_conv2d_forward_backward_backend()
    print("[OK] GPU smoke: conv2d")


if __name__ == "__main__":
    main()
