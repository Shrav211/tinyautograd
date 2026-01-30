# tests/test_device_sanity.py
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.device import cp  # if available

def main():
    x = Tensor(np.random.randn(2,3), requires_grad=True)
    y = (x + 2).sum()
    y.backward()
    assert x.grad is not None

    if cp is not None:
        x2 = Tensor(np.random.randn(2,3), requires_grad=True).to("cuda")
        y2 = (x2 + 2).sum()
        y2.backward()
        assert x2.grad is not None
        assert type(x2.data).__module__.split(".")[0] == "cupy"
        assert type(x2.grad).__module__.split(".")[0] == "cupy"

    print("[OK] device sanity")

if __name__ == "__main__":
    main()
