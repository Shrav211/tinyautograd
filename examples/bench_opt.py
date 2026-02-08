# examples/bench.py
import time
import json
import math
from dataclasses import dataclass, asdict
from typing import List

import numpy as np

from tinygrad.tensor import Tensor, no_grad
from tinygrad.optim import AdamW
from tinygrad.amp import GradScaler
from tinygrad.datasets.mnist import MNIST
from tinygrad.data import DataLoader
from tinygrad.nn import (
    MLP_BN,
    cross_entropy_logits,
    BatchNorm2d, MaxPool2d, GlobalAvgPool2d, Linear, Module
)

# Try to import optimized conv
try:
    from tinygrad.nn_cupy_optimized import Conv2dCuPyOptimized
    OPTIMIZED_CONV_AVAILABLE = True
    print("✓ Using optimized CuPy convolution")
except ImportError:
    from tinygrad.nn import Conv2d as Conv2dCuPyOptimized
    OPTIMIZED_CONV_AVAILABLE = False
    print("⚠ Using regular convolution (optimized version not available)")

try:
    import cupy as cp
except Exception:
    cp = None


# -------------------------
# Models
# -------------------------

class MNIST_CNN(Module):
    def __init__(self, use_optimized=True):
        super().__init__()
        
        # Choose conv implementation
        if use_optimized and OPTIMIZED_CONV_AVAILABLE:
            Conv = Conv2dCuPyOptimized
        else:
            from tinygrad.nn import Conv2d
            Conv = Conv2d
        
        self.c1 = Conv(1, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = BatchNorm2d(16)
        self.pool = MaxPool2d(2, 2)

        self.c2 = Conv(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(32)

        self.gap = GlobalAvgPool2d()
        self.fc = Linear(32, 10, init="xavier")

    def __call__(self, x: Tensor) -> Tensor:
        x = self.pool(self.bn1(self.c1(x)).relu())
        x = self.bn2(self.c2(x)).relu()
        x = self.gap(x)
        return self.fc(x)


# -------------------------
# Utilities
# -------------------------

def _sync(device: str):
    if device == "cuda" and cp is not None:
        cp.cuda.Stream.null.synchronize()


def infinite_batches(loader):
    while True:
        for b in loader:
            yield b


def accuracy_from_logits(logits, y):
    pred = np.argmax(logits, axis=1)
    return float(np.mean(pred == y))


def assert_finite(x, name):
    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"Non-finite detected in {name}")


# -------------------------
# Benchmark record
# -------------------------

@dataclass
class BenchResult:
    model: str
    device: str
    mode: str
    steps: int
    batch_size: int
    steps_per_sec: float
    samples_per_sec: float
    loss_start: float
    loss_end: float
    optimized: bool = False


# -------------------------
# Core benchmark
# -------------------------

def run_benchmark(
    model_name: str,
    model: Module,
    device: str,
    mode: str,
    steps: int,
    batch_size: int,
    warmup: int = 20,
    optimized: bool = False,
) -> BenchResult:

    # Dataset
    flatten = (model_name == "MLP")
    train_ds = MNIST(root="data/mnist", train=True, normalize=True, flatten=flatten)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, seed=0)
    stream = infinite_batches(loader)

    # Model / optimizer
    model.train()
    model.to(device)
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = GradScaler() if mode == "amp" else None

    # -----------------
    # Warmup
    # -----------------
    for _ in range(warmup):
        xb, yb = next(stream)

        if device == "cuda":
            xb = cp.asarray(xb)
        
        if model_name == "CNN" and xb.ndim == 3:
            xb = xb.reshape(xb.shape[0], 1, 28, 28)

        x = Tensor(
            xb.astype(np.float16 if mode == "amp" else np.float32),
            requires_grad=False,
        )

        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy_logits(logits, yb)

        if mode == "amp":
            scaled = scaler.scale_loss(loss)
            scaled.backward()
            scaler.unscale_(model.parameters())
            if not scaler.found_inf(model.parameters()):
                opt.step()
            scaler.update(False)
        else:
            loss.backward()
            opt.step()

    _sync(device)

    # -----------------
    # Timed run
    # -----------------
    losses: List[float] = []
    n_seen = 0

    t0 = time.perf_counter()

    for _ in range(steps):
        xb, yb = next(stream)

        if device == "cuda":
            xb = cp.asarray(xb)

        if model_name == "CNN" and xb.ndim == 3:
            xb = xb.reshape(xb.shape[0], 1, 28, 28)

        x = Tensor(
            xb.astype(np.float16 if mode == "amp" else np.float32),
            requires_grad=False,
        )

        opt.zero_grad()
        logits = model(x)
        loss = cross_entropy_logits(logits, yb)

        losses.append(float(loss.data))
        assert_finite(loss.data, "loss")

        if mode == "amp":
            scaled = scaler.scale_loss(loss)
            scaled.backward()
            scaler.unscale_(model.parameters())
            if not scaler.found_inf(model.parameters()):
                opt.step()
            scaler.update(False)
        else:
            loss.backward()
            opt.step()

        n_seen += batch_size

    _sync(device)
    t1 = time.perf_counter()

    dt = t1 - t0
    steps_per_sec = steps / dt
    samples_per_sec = n_seen / dt

    # Numerical sanity
    loss_start = losses[0]
    loss_end = losses[-1]
    if not (loss_end < loss_start or math.isclose(loss_end, loss_start)):
        raise RuntimeError("Loss did not decrease — numerical sanity failed")

    return BenchResult(
        model=model_name,
        device=device,
        mode=mode,
        steps=steps,
        batch_size=batch_size,
        steps_per_sec=steps_per_sec,
        samples_per_sec=samples_per_sec,
        loss_start=loss_start,
        loss_end=loss_end,
        optimized=optimized,
    )


# -------------------------
# Main
# -------------------------

def main():
    results: List[BenchResult] = []

    # Test both regular and optimized CNNs
    configs = [
        ("MLP", lambda: MLP_BN(784, 256, 10, dropout_p=0.0), False),
        ("CNN", lambda: MNIST_CNN(use_optimized=False), False),
        ("CNN-OPT", lambda: MNIST_CNN(use_optimized=True), True),
    ]

    devices = ["cpu"]
    if cp is not None:
        devices.append("cuda")

    modes = ["fp32"]
    if cp is not None:
        modes.append("amp")

    for model_name, model_fn, optimized in configs:
        for device in devices:
            for mode in modes:
                if device == "cpu" and mode == "amp":
                    continue
                
                # Skip optimized on CPU (only works on GPU)
                if optimized and device == "cpu":
                    continue

                print(f"\nRunning: model={model_name} device={device} mode={mode}")
                model = model_fn()

                res = run_benchmark(
                    model_name=model_name,
                    model=model,
                    device=device,
                    mode=mode,
                    steps=200,
                    batch_size=128,
                    optimized=optimized,
                )
                results.append(res)

    # -----------------
    # Print table
    # -----------------
    print("\n" + "=" * 100)
    print(
        f"{'MODEL':<10} {'DEVICE':<6} {'MODE':<6} "
        f"{'STEPS/S':>10} {'SAMPLES/S':>12} "
        f"{'LOSS_START':>12} {'LOSS_END':>12}"
    )
    print("=" * 100)

    for r in results:
        print(
            f"{r.model:<10} {r.device:<6} {r.mode:<6} "
            f"{r.steps_per_sec:>10.2f} {r.samples_per_sec:>12.2f} "
            f"{r.loss_start:>12.4f} {r.loss_end:>12.4f}"
        )

    print("=" * 100)
    
    # Show speedup comparison
    if cp is not None:
        print("\n" + "=" * 100)
        print("SPEEDUP ANALYSIS (GPU only)")
        print("=" * 100)
        
        # Find regular and optimized CNN results on GPU
        cnn_regular = next((r for r in results if r.model == "CNN" and r.device == "cuda" and r.mode == "fp32"), None)
        cnn_optimized = next((r for r in results if r.model == "CNN-OPT" and r.device == "cuda" and r.mode == "fp32"), None)
        
        if cnn_regular and cnn_optimized:
            speedup = cnn_optimized.steps_per_sec / cnn_regular.steps_per_sec
            print(f"CNN (regular):    {cnn_regular.steps_per_sec:.2f} steps/sec")
            print(f"CNN (optimized):  {cnn_optimized.steps_per_sec:.2f} steps/sec")
            print(f"Speedup:          {speedup:.2f}×")
        
        print("=" * 100)

    # -----------------
    # Save JSON
    # -----------------
    with open("bench_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    print("\nSaved benchmark results to bench_results.json")


if __name__ == "__main__":
    main()