import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Dropout

def test_dropout_train_eval():
    x = Tensor(np.ones((2, 5)), requires_grad=True)
    d = Dropout(p=0.5)

    # Train mode: randomness
    d.train()
    y1 = d(x).data
    y2 = d(x).data
    assert not np.allclose(y1, y2), "Dropout should be random in train mode"

    # Eval mode: deterministic
    d.eval()
    y3 = d(x).data
    y4 = d(x).data
    assert np.allclose(y3, y4), "Dropout should be deterministic in eval mode"
    assert np.allclose(y3, x.data), "Dropout should be identity in eval mode"

    # Backward masking test
    d.train()
    y = d(x).sum()
    x.zero_grad()
    y.backward()
    assert np.any(x.grad == 0), "Some gradients should be masked"

if __name__ == "__main__":
    test_dropout_train_eval()
    print("Dropout sanity test passed.")
