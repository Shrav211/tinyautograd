import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP

def rel_error(a, b, eps=1e-12):
    return abs(a - b) / max(eps, abs(a) + abs(b))

def  bce_with_logits(logits: Tensor, target: Tensor):
    z = logits
    loss = (z.relu() - z * target + (1 + (-z.abs()).exp()).log())
    return loss.sum() * (1.0 / target.data.size)

def numeric_grad_one(param: Tensor, idx, compute_loss, eps=1e-5):
    old = param.data[idx]

    param.data[idx] = old + eps
    L_pos = float(compute_loss().data)

    param.data[idx] = old - eps
    L_neg = float(compute_loss().data)

    param.data[idx] = old
    return (L_pos - L_neg) / (2 * eps)

def test_bce_with_logits_end_to_end():
    np.random.seed(0)

    X = np.array([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ])
    Y = np.array([
        [0.],
        [1.],
        [1.],
        [0.]
    ])

    model = MLP(in_dim=2, hidden_dim=8, out_dim=1)

    params = model.parameters()
    W1 = model.l1.W
    b1 = model.l1.b
    W2 = model.l2.W
    b2 = model.l2.b

    model.l1.b.data = 0.01 * np.random.randn(*model.l1.b.data.shape)
    model.l2.b.data = 0.01 * np.random.randn(*model.l2.b.data.shape)

    x = Tensor(X, requires_grad=True)
    y = Tensor(Y, requires_grad=True)

    def compute_loss():
        logits = model(x)
        return bce_with_logits(logits, y)
    
    for p in params:
        p.zero_grad()

    L = compute_loss()
    L.backward()

    tests = [
        (W1, (0, 0)),
        (W1, (1, 3)),
        (b1, (2,)),
        (W2, (4, 0)),
        (b2, (0,))
    ]

    for p, idx in tests:
        g_auto = float(p.grad[idx])
        g_num = float(numeric_grad_one(p, idx, compute_loss, eps=1e-5))
        err = rel_error(g_auto, g_num)

        print(F"{p=} idx={idx} g_auto={g_auto:.0e} g_num={g_num:.0e} rel_error={err:.3e}")

        assert err < 1e-4, f"Gradcheck failed at {idx}: auto={g_auto}, num={g_num}, err={err}"

if __name__ == "__main__":
    test_bce_with_logits_end_to_end()