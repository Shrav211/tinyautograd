# tests/test_ce_gpu_no_onehot.py
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from tinygrad.tensor import Tensor, no_grad
from tinygrad.nn import cross_entropy_logits

def _skip_if_no_cupy():
    if cp is None:
        print("[SKIP] cupy not installed")
        return True
    return False

def test_ce_logits_gpu_y_cupy():
    N, C = 32, 10
    logits = Tensor(cp.random.randn(N, C).astype(cp.float32), requires_grad=True)
    y_cp = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    loss = cross_entropy_logits(logits, y_cp)
    assert isinstance(loss.data, cp.ndarray), "loss must stay on GPU (cupy)"
    assert loss.data.shape == (), f"loss should be scalar, got {loss.data.shape}"

    loss.backward()
    assert logits.grad is not None
    assert isinstance(logits.grad, cp.ndarray), "logits.grad must be cupy"

def test_ce_logits_gpu_y_numpy():
    N, C = 32, 10
    logits = Tensor(cp.random.randn(N, C).astype(cp.float32), requires_grad=True)
    y_np = np.random.randint(0, C, size=(N,), dtype=np.int64)

    loss = cross_entropy_logits(logits, y_np)
    assert isinstance(loss.data, cp.ndarray), "loss must stay on GPU (cupy)"
    loss.backward()
    assert logits.grad is not None
    assert isinstance(logits.grad, cp.ndarray), "logits.grad must be cupy"

def test_ce_eval_no_grad_gpu():
    N, C = 16, 10
    logits = Tensor(cp.random.randn(N, C).astype(cp.float32), requires_grad=True)
    y_cp = cp.random.randint(0, C, size=(N,), dtype=cp.int32)

    with no_grad():
        loss = cross_entropy_logits(logits, y_cp)
        assert loss.requires_grad is False, "no_grad should disable graph"
        assert len(getattr(loss, "_prev", [])) == 0, "no_grad loss should have no graph"

def main():
    if _skip_if_no_cupy():
        return

    test_ce_logits_gpu_y_cupy()
    test_ce_logits_gpu_y_numpy()
    test_ce_eval_no_grad_gpu()
    print("[OK] GPU CE no-onehot tests passed")

if __name__ == "__main__":
    main()
