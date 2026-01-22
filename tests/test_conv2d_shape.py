import numpy as np
from tinygrad.tensor import Tensor

def main():
    np.random.seed(0)

    x = Tensor(np.random.randn(2, 3, 7, 7), requires_grad=True)
    w = Tensor(np.random.randn(4, 3, 3, 3), requires_grad=True)
    b = Tensor(np.random.randn(4,), requires_grad=True)

    y = x.conv2d(w, b, stride=2, padding=1)
    assert y.data.shape == (2, 4, 4, 4), y.data.shape
    print("[OK] conv2d forward shape:", y.data.shape)

if __name__ == "__main__":
    main()
