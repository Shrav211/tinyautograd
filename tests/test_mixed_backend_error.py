import numpy as np
from tinygrad.tensor import Tensor

def main():
    try:
        import cupy as cp
    except Exception:
        print("[SKIP] cupy not installed")
        return

    a = Tensor(np.ones((2,2)), requires_grad=True)   # numpy
    b = Tensor(cp.ones((2,2)), requires_grad=True)   # cupy

    try:
        _ = a + b
        raise AssertionError("Expected mixed-backend add to fail")
    except ValueError:
        pass

    print("[OK] mixed-backend guard works")

if __name__ == "__main__":
    main()
