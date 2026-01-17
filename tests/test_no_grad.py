import numpy as np

from tinygrad.tensor import Tensor, no_grad


def test_no_grad_builds_no_graph():
    x = Tensor(2.0, requires_grad=True)

    with no_grad():
        y = x * 3.0 + 1.0

    # In no_grad, outputs should not require grad and should have no history
    assert y.requires_grad is False
    assert len(y._prev) == 0, f"_prev not empty: {y._prev}"
    assert y.grad is None


def test_detach_cuts_graph():
    x = Tensor(2.0, requires_grad=True)
    y = x * 3.0
    z = y.detach() * 4.0

    # z is scalar -> backward allowed
    z.backward()

    # Since we detached, gradients must NOT flow to x
    assert x.grad is None, f"Expected x.grad=None, got {x.grad}"


def test_no_grad_matmul_has_no_graph():
    A = Tensor(np.random.randn(2, 3), requires_grad=True)
    B = Tensor(np.random.randn(3, 4), requires_grad=True)

    with no_grad():
        print("grad enabled?", Tensor._grad_enabled)
        C = A @ B
        print("C.requires_grad:", C.requires_grad)

    assert C.requires_grad is False
    assert len(C._prev) == 0, f"_prev not empty: {C._prev}"


def main():
    # simple runner without pytest
    test_no_grad_builds_no_graph()
    test_detach_cuts_graph()
    test_no_grad_matmul_has_no_graph()
    print("[OK] all no_grad/detach tests passed.")
    
if __name__ == "__main__":
    main()
