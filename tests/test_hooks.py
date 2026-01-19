import numpy as np
from tinygrad.tensor import Tensor

def test_hook_clips_grad():
    x = Tensor(2.0, requires_grad=True)
    y = x * 10.0          # dy/dx = 10
    y.register_hook(lambda g: np.clip(g, -3.0, 3.0))  # clip at y
    loss = y.sum()
    loss.backward()
    # Without hook: x.grad = 10
    # With hook on y: y.grad gets clipped to 3 before flowing to x => x.grad = 3*10? no:
    # Careful: hook runs on the tensor being accumulated INTO.
    # We hooked y, so it clips y.grad (which is upstream from loss). For loss=y, upstream is 1 so clip does nothing.
    # Better: hook x directly.

def test_hook_on_x():
    x = Tensor(2.0, requires_grad=True)
    x.register_hook(lambda g: np.clip(g, -3.0, 3.0))
    loss = (x * 10.0).sum()
    loss.backward()
    assert np.isclose(x.grad, 3.0), f"expected 3.0, got {x.grad}"

def test_hook_scaling():
    x = Tensor(2.0, requires_grad=True)
    x.register_hook(lambda g: 0.5 * g)
    loss = (x * 10.0).sum()   # base grad = 10
    loss.backward()
    assert np.isclose(x.grad, 5.0), f"expected 5.0, got {x.grad}"

def main():
    test_hook_on_x()
    test_hook_scaling()
    print("[OK] hooks work")

if __name__ == "__main__":
    main()
