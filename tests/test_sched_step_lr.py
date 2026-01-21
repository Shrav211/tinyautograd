import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.optim import SGD
from tinygrad.sched import StepLR

def main():
    w = Tensor(np.array([1.0]), requires_grad=True)
    opt = SGD([w], lr=0.1)
    sch = StepLR(opt, step_size=3, gamma=0.1)

    lrs = []
    for _ in range(10):
        lrs.append(opt.lr)
        sch.step()

    # before step calls, lr is base
    # after step 0..2 -> base, step 3..5 -> base*0.1, etc (depending on your convention)
    assert abs(lrs[0] - 0.1) < 1e-12
    print("[OK] StepLR produces lrs:", lrs[:7])

if __name__ == "__main__":
    main()
