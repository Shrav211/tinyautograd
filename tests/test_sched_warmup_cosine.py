import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.optim import SGD
from tinygrad.sched import WarmupCosineLR

def main():
    w = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([w], lr=0.1)
    sch = WarmupCosineLR(opt, warmup_steps=5, total_steps=20, eta_min=0.01)

    lrs = []
    for _ in range(25):
        sch.step()
        lrs.append(opt.lr)

    # warmup increases
    assert lrs[1] > lrs[0]
    # later should be lower than base eventually
    assert min(lrs) >= 0.01 - 1e-9
    print("[OK] WarmupCosine looks sane. first:", lrs[:6], "last:", lrs[-5:])

if __name__ == "__main__":
    main()
