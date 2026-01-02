import numpy as np
from tinygrad.tensor import Tensor

def rel_error(a, b, eps=1e-12):
    return np.max(np.abs(a - b) / np.maximum(eps, np.abs(a) + np.abs(b)))

def loss_fn(A: Tensor, B: Tensor):
    # simple scalar loss that uses matmul + sum
    # shapes of A: (N, D), B: (D, H) and out: (N, H)
    out = A @ B
    loss = out.sum()
    return loss

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

def test_matmul_gradcheck():
    np.random.seed(0)

    N, D, H = 2, 3, 4

    A = Tensor(np.random.randn(N, D), requires_grad=True)
    B = Tensor(np.random.randn(D, H), requires_grad=True)

    def compute_loss():
        return loss_fn(A, B)
    
    A.zero_grad(); B.zero_grad()
    L = compute_loss()
    L.backward()

    gA_auto = A.grad.copy()
    gB_auto = B.grad.copy()

    gA_num = numeric_grad(A, compute_loss, eps=1e-6)
    gB_num = numeric_grad(B, compute_loss, eps=1e-6)

    errA = rel_error(gA_auto, gA_num)
    errB = rel_error(gB_auto, gB_num)

    print("errA:", errA)
    print("errB:", errB)

    assert errA < 1e-5
    assert errB < 1e-5

if __name__ == "__main__":
    test_matmul_gradcheck()
