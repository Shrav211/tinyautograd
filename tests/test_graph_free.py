import numpy as np
from tinygrad.tensor import Tensor


def test_graph_freed_by_default():
    """
    After backward(), the computational graph should be freed:
    - _prev cleared
    - _backward replaced
    """
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y + x      # z = 2*3 + 2 = 8
    loss = z * 2       # loss = 16

    # graph exists before backward
    assert len(loss._prev) > 0

    loss.backward()

    # gradients correct
    # loss = 2*(x*y + x) = 2xy + 2x
    # d/dx = 2y + 2 = 8
    # d/dy = 2x = 4
    assert np.isclose(x.grad, 8.0)
    assert np.isclose(y.grad, 4.0)

    # graph freed
    assert loss._prev == set()
    assert loss._op == ""


def test_retain_graph_keeps_graph():
    """
    retain_graph=True should preserve _prev so backward can be reused
    """
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)

    z = x * y
    loss = z.sum()

    loss.backward(retain_graph=True)

    # graph still exists
    assert len(loss._prev) > 0

    # gradients correct
    assert np.isclose(x.grad, 3.0)
    assert np.isclose(y.grad, 2.0)


def test_parameters_survive_graph_free():
    """
    Freeing the graph should NOT delete parameter gradients
    """
    W = Tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    x = Tensor(np.array([[3.0], [4.0]]), requires_grad=False)

    y = W @ x
    loss = y.sum()

    loss.backward()

    # gradient correct
    assert W.grad.shape == W.data.shape
    assert np.allclose(W.grad, [[3.0, 4.0]])

    # graph freed
    assert loss._prev == set()


def main():
    test_graph_freed_by_default()
    test_retain_graph_keeps_graph()
    test_parameters_survive_graph_free()
    print("[OK] graph freeing after backward works correctly")


if __name__ == "__main__":
    main()
