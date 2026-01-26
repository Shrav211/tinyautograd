import numpy as np
from tinygrad.tensor import Tensor

def rel_error(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))

def numeric_grad(param: Tensor, compute_loss, eps=1e-6):
    g = np.zeros_like(param.data, dtype=float)
    it = np.nditer(param.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        old = param.data[idx]

        param.data[idx] = old + eps
        L_pos = float(compute_loss().data)

        param.data[idx] = old - eps
        L_neg = float(compute_loss().data)

        param.data[idx] = old
        g[idx] = (L_pos - L_neg) / (2 * eps)
        it.iternext()
    return g

def main():
    np.random.seed(0)

    x = Tensor(np.random.randn(1, 2, 5, 5), requires_grad=True)

    def compute_loss():
        y = x.maxpool2d(kernel_size=2, stride=2, padding=0)   # (1,2,2,2)
        return y.sum()

    x.zero_grad()
    L = compute_loss()
    L.backward()
    g_auto = x.grad.copy()
    g_num = numeric_grad(x, compute_loss, eps=1e-6)

    err = rel_error(g_auto, g_num)
    print("rel_err x:", err)
    assert err < 1e-5, f"maxpool2d gradcheck failed: {err}"
    print("[OK] maxpool2d gradcheck passed")

if __name__ == "__main__":
    main()
