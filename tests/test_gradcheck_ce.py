import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.losses import cross_entropy_logits

def rel_error(a, b, eps=1e-12):
    return np.abs(a - b) / max(eps, (np.abs(a) + np.abs(b)))

def numeric_grad_single_entry(logits: Tensor, y: np.ndarray, idx, eps=1e-6):
    old = float(logits.data[idx])

    logits.data[idx] = old + eps
    L_pos = float(cross_entropy_logits(logits, y).data)

    logits.data[idx] = old - eps
    L_neg = float(cross_entropy_logits(logits, y).data)

    logits.data[idx] = old
    return (L_pos - L_neg) / (2 * eps)

def test_cross_entropy_gradcheck():
    np.random.seed(0)

    N, C = 4, 5
    logits = Tensor(np.random.randn(N, C) * 0.5, requires_grad=True)

    y = np.array([0, 3, 1, 4], dtype=int)

    logits.zero_grad()
    loss = cross_entropy_logits(logits, y)
    loss.backward()

    g_auto = logits.grad.copy()

    check_indices = [(0, 0), (1, 3), (2, 1), (3, 4), (0, 2), (2, 4)]
    eps = 1e-6

    max_err = 0.0
    for idx in check_indices:
        g_num = numeric_grad_single_entry(logits, y, idx, eps=eps)
        g_a = float(g_auto[idx])
        err = rel_error(g_a, g_num)

        max_err = max(max_err, err)
        print(f"idx={idx} g_auto={g_a:.6e} g_num={g_num:.6e} rel_err={err:.3e}")

    print("max_rel_err:", max_err)
    assert max_err < 1e-4, f"Gradcheck has failed: max_rel_err={max_err}"

if __name__ == "__main__":
    test_cross_entropy_gradcheck()
