# tests/test_gradcheck_gather.py
import numpy as np
from tinygrad.tensor import Tensor

def rel_err(a, b, eps=1e-12):
    return np.abs(a - b) / (np.maximum(eps, np.abs(a) + np.abs(b)))

def test_gather_gradcheck_cpu():
    np.random.seed(0)

    N, C = 4, 5
    x = np.random.randn(N, C).astype(np.float64)

    # integer indices per row (gather along axis=1)
    y = np.array([0, 3, 1, 4], dtype=np.int64).reshape(N, 1)

    # autograd grad
    X = Tensor(x.copy(), requires_grad=True)
    Y = Tensor(y, requires_grad=False)
    out = X.gather(Y, axis=1)     # (N,1)
    loss = out.sum()              # scalar
    loss.backward()
    g_auto = X.grad.copy()

    # numeric grad (finite differences)
    eps = 1e-6
    g_num = np.zeros_like(x)

    def f(x_arr):
        # sum of gathered entries
        return float(np.sum(x_arr[np.arange(N), y.reshape(-1)]))

    for i in range(N):
        for j in range(C):
            xp = x.copy()
            xm = x.copy()
            xp[i, j] += eps
            xm[i, j] -= eps
            g_num[i, j] = (f(xp) - f(xm)) / (2 * eps)

    max_err = float(np.max(rel_err(g_auto, g_num)))
    print("max_rel_err:", max_err)
    assert max_err < 1e-6, f"gather gradcheck failed: max_rel_err={max_err}"

def main():
    test_gather_gradcheck_cpu()
    print("[OK] gather gradcheck passed")

if __name__ == "__main__":
    main()
