# tests/test_cupy_conv.py
import time
import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d

if not CUPY_AVAILABLE:
    print("CuPy not available. Skipping test.")
    exit(0)

# Create test input on GPU
x_data = cp.random.randn(128, 64, 32, 32).astype(cp.float32)
x = Tensor(x_data, requires_grad=False)

print("="*70)
print("CONVOLUTION PERFORMANCE COMPARISON")
print("="*70)
print(f"Input shape: {x.data.shape}")
print(f"Configuration: 64 → 128 channels, 3×3 kernel, padding=1")
print()

# ============================================
# Test 1: Regular Conv2d (your current im2col)
# ============================================
print("Test 1: Regular Conv2d (manual im2col)")
print("-"*70)

conv_regular = Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
conv_regular.to('cuda')  # Move to GPU

print("Warmup...")
for _ in range(10):
    _ = conv_regular(x)
    cp.cuda.Stream.null.synchronize()

print("Benchmarking (100 iterations)...")
start = time.time()
for _ in range(100):
    _ = conv_regular(x)
    cp.cuda.Stream.null.synchronize()
time_regular = time.time() - start

print(f"✓ Time: {time_regular:.3f}s")
print(f"  Per iteration: {time_regular/100*1000:.1f}ms")
print(f"  Throughput: {100/time_regular:.1f} iters/sec")
print()

# ============================================
# Test 2: CuPy Optimized Conv2d
# ============================================
try:
    from tinygrad.nn_cupy_optimized import Conv2dCuPyOptimized
    
    print("Test 2: CuPy Optimized Conv2d (optimized stride_tricks + cuBLAS)")
    print("-"*70)
    
    conv_cupy = Conv2dCuPyOptimized(64, 128, kernel_size=3, padding=1, bias=False)
    
    print("Warmup...")
    for _ in range(10):
        _ = conv_cupy(x)
        cp.cuda.Stream.null.synchronize()
    
    print("Benchmarking (100 iterations)...")
    start = time.time()
    for _ in range(100):
        _ = conv_cupy(x)
        cp.cuda.Stream.null.synchronize()
    time_cupy = time.time() - start
    
    print(f"✓ Time: {time_cupy:.3f}s")
    print(f"  Per iteration: {time_cupy/100*1000:.1f}ms")
    print(f"  Throughput: {100/time_cupy:.1f} iters/sec")
    print()
    
    # Results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Regular Conv2d:      {time_regular:.3f}s  ({time_regular/100*1000:.1f}ms/iter)")
    print(f"CuPy Optimized:      {time_cupy:.3f}s  ({time_cupy/100*1000:.1f}ms/iter)")
    print(f"Speedup:             {time_regular/time_cupy:.2f}×")
    print("="*70)
    
    if time_regular/time_cupy > 3:
        print("✅ Significant speedup achieved!")
    elif time_regular/time_cupy > 1.5:
        print("✓ Moderate speedup")
    else:
        print("⚠️ Minor speedup - may need further optimization")
    
except ImportError as e:
    print(f"CuPy Optimized Conv2d not available: {e}")
    print()
    print("="*70)
    print("RESULTS")
    print("="*70)
    print(f"Regular Conv2d only: {time_regular:.3f}s")
    print("="*70)