import numpy as np

try:
    import cupy as cp
except Exception:
    cp = None

def get_xp_from_array(x):
    # works for numpy and cuda ndarrays
    mod = type(x).__module__.split(".")[0]
    if mod == "cupy":
        return cp
    return np

def get_xp(device: str):
    if device in ("cpu", "np", "numpy"):
        return np
    if device in ("gpu", "cuda", "cupy"):
        if cp is None:
            raise ImportError("cupy not installed")
        return cp
    raise ValueError(f"unknown device: {device}")

def to_numpy(x):
    # converting for saving or printing
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x