# tests/test_cudnn.py
import time
import numpy as np
try:
    import cupy as cp
    import cupy.cuda as cudnn
    CUDNN_AVAILABLE = True
except Exception:
    CUDNN_AVAILABLE = False
    cp = None
    cudnn = None

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d

if not CUDNN_AVAILABLE:
    print("CuPy not available. Skipping test.")
    exit(0)

# Create test input on GPU
x_data = cp.random.randn(128, 64, 32, 32).astype(cp.float32)
x = Tensor(x_data, requires_grad=False)

print("="*60)
print("Testing Convolution Performance")
print("="*60)
print(f"Input shape: {x.data.shape}")
print(f"Input device: {'GPU (CuPy)' if isinstance(x.data, cp.ndarray) else 'CPU (NumPy)'}")
print()

# Test regular Conv2d
print("1. Regular Conv2d (im2col approach)")
print("-"*60)
conv_regular = Conv2d(64, 128, kernel_size=3, padding=1, bias=False)

# Move conv weights to GPU
print("Moving weights to GPU...")
conv_regular.W.data = cp.asarray(conv_regular.W.data)

print("Running warmup (10 iterations)...")
for _ in range(10):
    out_regular = conv_regular(x)
    cp.cuda.Stream.null.synchronize()

print("Running benchmark (100 iterations)...")
start = time.time()
for _ in range(100):
    out_regular = conv_regular(x)
    cp.cuda.Stream.null.synchronize()
regular_time = time.time() - start

print(f"✓ Regular Conv2d: {regular_time:.3f}s for 100 iterations")
print(f"  Average per iteration: {regular_time/100*1000:.1f}ms")
print(f"  Throughput: {100/regular_time:.1f} iterations/sec")
print()

# Try CuPy optimized version if available
try:
    from tinygrad.nn_cudnn import Conv2dCuDNN
    print("[nn_cudnn] cudnn module:", cudnn.__name__, getattr(cudnn, "__file__", None))

    print("2. cuDNN")
    print("-"*60)
    conv_cupy = Conv2dCuDNN(64, 128, kernel_size=3, padding=1, bias=False)
    
    print("Running warmup (10 iterations)...")
    for _ in range(10):
        out_cupy = conv_cupy(x)
        cp.cuda.Stream.null.synchronize()
    
    print("Running benchmark (100 iterations)...")
    start = time.time()
    for _ in range(100):
        out_cupy = conv_cupy(x)
        cp.cuda.Stream.null.synchronize()
    cupy_time = time.time() - start
    
    print(f"✓ CuPy Conv2d: {cupy_time:.3f}s for 100 iterations")
    print(f"  Average per iteration: {cupy_time/100*1000:.1f}ms")
    print(f"  Throughput: {100/cupy_time:.1f} iterations/sec")
    print()
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Regular Conv2d:  {regular_time:.3f}s")
    print(f"CuPy Conv2d:     {cupy_time:.3f}s")
    print(f"Speedup:         {regular_time/cupy_time:.2f}×")
    print("="*60)
    
except ImportError as e:
    print("CuPy optimized Conv2d not available")
    print(f"Error: {e}")
    print()
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Regular Conv2d:  {regular_time:.3f}s")
    print("="*60)