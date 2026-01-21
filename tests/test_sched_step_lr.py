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
        sch.step()          # ← Step first
        lrs.append(opt.lr)  # ← Then record
    
    print("LR schedule:", lrs)
    
    # Test 1: First 3 steps should be base_lr
    assert abs(lrs[0] - 0.1) < 1e-12, f"Step 0: {lrs[0]} should be 0.1"
    assert abs(lrs[1] - 0.1) < 1e-12, f"Step 1: {lrs[1]} should be 0.1"
    assert abs(lrs[2] - 0.1) < 1e-12, f"Step 2: {lrs[2]} should be 0.1"
    
    # Test 2: Steps 3-5 should be base_lr * gamma (0.01)
    assert abs(lrs[3] - 0.01) < 1e-12, f"Step 3: {lrs[3]} should be 0.01"
    assert abs(lrs[4] - 0.01) < 1e-12, f"Step 4: {lrs[4]} should be 0.01"
    assert abs(lrs[5] - 0.01) < 1e-12, f"Step 5: {lrs[5]} should be 0.01"
    
    # Test 3: Steps 6-8 should be base_lr * gamma^2 (0.001)
    assert abs(lrs[6] - 0.001) < 1e-12, f"Step 6: {lrs[6]} should be 0.001"
    assert abs(lrs[7] - 0.001) < 1e-12, f"Step 7: {lrs[7]} should be 0.001"
    assert abs(lrs[8] - 0.001) < 1e-12, f"Step 8: {lrs[8]} should be 0.001"
    
    # Test 4: Step 9 should be base_lr * gamma^3 (0.0001)
    assert abs(lrs[9] - 0.0001) < 1e-12, f"Step 9: {lrs[9]} should be 0.0001"
    
    # Test 5: Verify drops happen at right times
    assert lrs[3] < lrs[2], "Should drop at step 3"
    assert lrs[6] < lrs[5], "Should drop at step 6"
    assert lrs[9] < lrs[8], "Should drop at step 9"
    
    print("[OK] StepLR schedule correct:")
    print(f"  Steps 0-2:  {lrs[0:3]} (base_lr)")
    print(f"  Steps 3-5:  {lrs[3:6]} (base_lr * gamma)")
    print(f"  Steps 6-8:  {lrs[6:9]} (base_lr * gamma^2)")
    print(f"  Step 9:     {lrs[9]} (base_lr * gamma^3)")

if __name__ == "__main__":
    main()