import numpy as np
try:
    import cupy as cp
except Exception:
    cp = None

from tinygrad.tensor import im2col

def main():
    x = np.random.randn(2, 3, 7, 7).astype(np.float32)
    a, oh, ow, shp = im2col(x, 3, 3, stride=2, padding=1)
    assert a.shape == (2*oh*ow, 3*3*3)

    if cp is not None:
        xg = cp.asarray(x)
        ag, oh2, ow2, shp2 = im2col(xg, 3, 3, stride=2, padding=1)
        assert (oh, ow) == (oh2, ow2)
        assert ag.shape == a.shape
    print("[OK] im2col shape + backend sanity")

if __name__ == "__main__":
    main()
