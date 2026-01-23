import numpy as np
from tinygrad.tensor import Tensor

def rel_error(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))

def test_reshape_backward():
    x = Tensor(np.arange(6.0).reshape(2,3), requires_grad=True)
    y = x.reshape(6)
    loss = y.sum()
    loss.backward()
    assert x.grad.shape == (2,3)
    assert np.all(x.grad == 1.0)

def test_transpose_backward():
    np.random.seed(0)
    x = Tensor(np.random.randn(2,3,4), requires_grad=True)
    y = x.transpose(0,2,1)     # (2,4,3)
    loss = y.sum()
    loss.backward()
    assert x.grad.shape == x.data.shape
    assert np.allclose(x.grad, np.ones_like(x.data))

def main():
    test_reshape_backward()
    test_transpose_backward()
    print("[OK] reshape/transpose sanity passed")

if __name__ == "__main__":
    main()
