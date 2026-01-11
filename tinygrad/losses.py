import numpy as np
from tinygrad.tensor import Tensor

def cross_entropy_logits(logits: Tensor, y: np.ndarray):
    """
    logits: Tensor (N, C)
    y: numpy int array (N,) with values in [0, C-1]
    returns scalar Tensor
    """
    N, C = logits.data.shape
    assert y.shape == (N,)

    # one-hot targets as constant tensor
    Y = np.zeros((N, C), dtype=float)
    Y[np.arange(N), y] = 1.0
    Y = Tensor(Y, requires_grad=False)

    # logsumexp per sample: (N,1)
    lse = logits.logsumexp(axis=1, keepdims=True)

    # z_y per sample: (N,1)
    z_y = (logits * Y).sum(axis=1, keepdims=True)

    # loss per sample then mean
    loss = (lse - z_y).sum() * (1.0 / N)                      # scalar
    return loss
