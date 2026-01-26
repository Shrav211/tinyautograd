import numpy as np
from tinygrad.tensor import Tensor

def main():
    x = Tensor(np.array([[1.0, 5.0, 5.0],
                         [2.0, 3.0, 0.0]]), requires_grad=True)
    y = x.max(axis=1)        # shape (2,)
    L = y.sum()              # scalar
    L.backward()

    # Row0 has tie at idx1,idx2 => grad split: 0.5,0.5
    # Row1 max at idx1 => grad 1 there
    expected = np.array([[0.0, 0.5, 0.5],
                         [0.0, 1.0, 0.0]])
    assert np.allclose(x.grad, expected), (x.grad, expected)
    print("[OK] max backward sanity passed")

if __name__ == "__main__":
    main()
