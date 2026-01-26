import numpy as np
from tinygrad.tensor import Tensor

def main():
    np.random.seed(0)
    x = Tensor(np.random.randn(2, 3, 7, 7), requires_grad=False)
    y = x.maxpool2d(kernel_size=2, stride=2, padding=0)
    assert y.data.shape == (2, 3, 3, 3), y.data.shape
    print("[OK] maxpool2d forward shape:", y.data.shape)

if __name__ == "__main__":
    main()
