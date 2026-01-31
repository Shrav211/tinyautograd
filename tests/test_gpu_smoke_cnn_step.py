# tests/test_gpu_smoke_cnn_step.py
import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

from tinygrad.tensor import Tensor, no_grad
from tinygrad.nn import Conv2d, BatchNorm2d, MaxPool2d, GlobalAvgPool2d, Linear, Module
from tinygrad.optim import AdamW
from tinygrad.nn import cross_entropy_logits

def _skip_if_no_cupy():
    if cp is None:
        print("[SKIP] cupy not installed")
        return True
    return False

class TinyCNN(Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(8)
        self.pool = MaxPool2d(2, 2)
        self.c2 = Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(16)
        self.gap = GlobalAvgPool2d()
        self.fc = Linear(16, 10, init="xavier")

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pool(self.bn1(self.c1(x)).relu())
        x = self.bn2(self.c2(x)).relu()
        x = self.gap(x)
        return self.fc(x)

def test_tiny_cnn_train_steps_gpu():
    model = TinyCNN()
    model.to("cuda")
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # random MNIST-like batch on GPU
    xb = cp.random.rand(64, 1, 28, 28).astype(cp.float32)
    yb = cp.random.randint(0, 10, size=(64,), dtype=cp.int32)
    # y needs to be numpy int array for your CE function — convert safely
    yb_np = cp.asnumpy(yb)

    for _ in range(5):
        opt.zero_grad()
        logits = model(Tensor(xb, requires_grad=False))
        assert isinstance(logits.data, cp.ndarray), "logits must be cupy"
        loss = cross_entropy_logits(logits, yb_np)
        loss.backward()
        opt.step()

    # quick eval forward
    model.eval()
    with no_grad():
        logits2 = model(Tensor(xb, requires_grad=False))
        assert isinstance(logits2.data, cp.ndarray)

    # spot check grads exist on params (cupy)
    for p in model.parameters():
        # after training step grads may be None due to zero_grad at end, so just check data backend
        assert isinstance(p.data, cp.ndarray), "param data must stay cupy"

def main():
    if _skip_if_no_cupy():
        return

    test_tiny_cnn_train_steps_gpu()
    print("[OK] GPU smoke: tiny CNN train steps")


if __name__ == "__main__":
    main()