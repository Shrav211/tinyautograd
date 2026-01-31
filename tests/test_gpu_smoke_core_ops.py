# tests/test_gpu_smoke_core_ops.py
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from tinygrad.tensor import Tensor, no_grad

def _skip_if_no_cupy():
    if cp is None:
        print("[SKIP] cupy not installed")
        return True
    return False

def test_scalar_constant_follows_backend():
    x = Tensor(cp.ones((2, 2), dtype=cp.float32), requires_grad=True)
    y = x + 2.0
    assert isinstance(y.data, cp.ndarray), "scalar constant should follow x backend (cupy)"
    z = 3.0 * x
    assert isinstance(z.data, cp.ndarray), "scalar constant should follow x backend (cupy)"

def test_matmul_backward_on_gpu():
    x = Tensor(cp.random.randn(4, 3).astype(cp.float32), requires_grad=True)
    w = Tensor(cp.random.randn(3, 5).astype(cp.float32), requires_grad=True)

    y = x @ w
    assert isinstance(y.data, cp.ndarray)

    loss = y.sum()
    loss.backward()

    assert x.grad is not None and isinstance(x.grad, cp.ndarray), "x.grad should be cupy"
    assert w.grad is not None and isinstance(w.grad, cp.ndarray), "w.grad should be cupy"

def test_reductions_and_shape_ops_on_gpu():
    x = Tensor(cp.random.randn(2, 3, 4).astype(cp.float32), requires_grad=True)

    s = x.sum(axis=2)              # (2,3)
    m = x.mean(axis=(1, 2))        # (2,)
    r = x.reshape((2, 12))
    t = x.transpose((0, 2, 1))     # (2,4,3)

    assert isinstance(s.data, cp.ndarray)
    assert isinstance(m.data, cp.ndarray)
    assert isinstance(r.data, cp.ndarray)
    assert isinstance(t.data, cp.ndarray)

    # connect them into one scalar so backward covers all
    loss = (s.sum() + m.sum() + r.sum() + t.sum())
    loss.backward()

    assert isinstance(x.grad, cp.ndarray), "x.grad should be cupy"

def test_no_grad_produces_no_graph_gpu():
    x = Tensor(cp.random.randn(3, 3).astype(cp.float32), requires_grad=True)
    with no_grad():
        y = (x @ x).sum()
        # should not require grad / should not have prev
        assert y.requires_grad is False
        assert len(getattr(y, "_prev", [])) == 0

    # x.grad should remain None since no backward called and graph not built
    assert x.grad is None

def main():
    if _skip_if_no_cupy():
        return

    test_scalar_constant_follows_backend()
    test_matmul_backward_on_gpu()
    test_reductions_and_shape_ops_on_gpu()
    test_no_grad_produces_no_graph_gpu()
    print("[OK] GPU smoke: core ops")

if __name__ == "__main__":
    main()
