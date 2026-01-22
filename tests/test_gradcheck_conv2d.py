import numpy as np
from tinygrad.tensor import Tensor

def rel_error(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))

def numeric_grad(param: Tensor, compute_loss, eps=1e-5):
    g = np.zeros_like(param.data, dtype=float)
    it = np.nditer(param.data, flags=["multi_index"], op_flags=["readwrite"])
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

    # small shapes so numeric grad is not too slow
    x = Tensor(np.random.randn(1, 2, 5, 5), requires_grad=True)
    w = Tensor(np.random.randn(3, 2, 3, 3), requires_grad=True)
    b = Tensor(np.random.randn(3,), requires_grad=True)

    def compute_loss():
        y = x.conv2d(w, b, stride=1, padding=1)
        return y.sum()  # scalar

    x.zero_grad(); w.zero_grad(); b.zero_grad()
    L = compute_loss()
    L.backward()

    gx_auto = x.grad.copy()
    gw_auto = w.grad.copy()
    gb_auto = b.grad.copy()

    gx_num = numeric_grad(x, compute_loss, eps=1e-5)
    gw_num = numeric_grad(w, compute_loss, eps=1e-5)
    gb_num = numeric_grad(b, compute_loss, eps=1e-5)

    ex = rel_error(gx_auto, gx_num)
    ew = rel_error(gw_auto, gw_num)
    eb = rel_error(gb_auto, gb_num)

    print("rel_err x:", ex)
    print("rel_err w:", ew)
    print("rel_err b:", eb)

    assert ex < 1e-4
    assert ew < 1e-4
    assert eb < 1e-4
    print("[OK] conv2d gradcheck passed")

if __name__ == "__main__":
    main()