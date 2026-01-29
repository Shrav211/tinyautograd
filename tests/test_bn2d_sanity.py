# tests/test_bn2d_sanity.py
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import BatchNorm2d

def test_bn2d_shape_and_eval_determinism():
    np.random.seed(0)
    bn = BatchNorm2d(num_features=3)
    x = Tensor(np.random.randn(4, 3, 5, 5), requires_grad=False)

    bn.train()
    y1 = bn(x).data
    assert y1.shape == (4, 3, 5, 5)

    # switch to eval: should be deterministic across calls
    bn.eval()
    y2 = bn(x).data
    y3 = bn(x).data
    assert np.max(np.abs(y2 - y3)) == 0.0

def test_bn2d_running_stats_update():
    np.random.seed(1)
    bn = BatchNorm2d(num_features=2, momentum=0.1)
    x = Tensor(np.random.randn(8, 2, 4, 4), requires_grad=False)

    rm0 = bn.running_mean.copy()
    rv0 = bn.running_var.copy()

    bn.train()
    _ = bn(x)

    assert np.max(np.abs(bn.running_mean - rm0)) > 0.0
    assert np.max(np.abs(bn.running_var - rv0)) > 0.0

def test_bn2d_backward_grads():
    np.random.seed(2)
    bn = BatchNorm2d(num_features=4)
    bn.train()

    x = Tensor(np.random.randn(2, 4, 3, 3), requires_grad=True)
    y = bn(x).sum()
    y.backward()

    assert x.grad is not None
    assert bn.gamma.grad is not None
    assert bn.beta.grad is not None
    assert np.isfinite(x.grad).all()
    assert np.isfinite(bn.gamma.grad).all()
    assert np.isfinite(bn.beta.grad).all()

def main():
    test_bn2d_shape_and_eval_determinism()
    test_bn2d_running_stats_update()
    test_bn2d_backward_grads()
    print("[OK] BatchNorm2d sanity passed")

if __name__ == "__main__":
    main()
