import numpy as np

def iterate_minibatches(X, Y, batch_size=4, shuffle=True):
    N = X.shape()
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, N, batch_size):
        batch_idx = idx[start:start+batch_size]
        yield X[batch_idx], Y[batch_idx]
        