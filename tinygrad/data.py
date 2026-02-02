import numpy as np

class Dataset:
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
def default_collate(batch):
    # batch is the list of items from dataset, typically tuples

    if isinstance(batch[0], tuple):
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        return np.stack(xs, axis=0), np.stack(ys, axis=0)
    else:
        return np.stack(batch, axis=0)

def mnist_cnn_collate(batch):
    """Collate function that adds channel dimension for CNNs."""
    xs = [b[0] for b in batch]
    ys = [b[1] for b in batch]
    
    X = np.stack(xs, axis=0)  # (N, 28, 28)
    
    # Add channel dimension
    if X.ndim == 3:  # (N, H, W)
        X = X[:, None, :, :]  # (N, 1, H, W)
    
    Y = np.array(ys)
    
    return X, Y

def cifar10_collate(batch):
    # batch: list of (x: (3,32,32) float32, y: int)
    xs = np.stack([b[0] for b in batch], axis=0).astype(np.float32)  # (B,3,32,32)
    ys = np.array([b[1] for b in batch], dtype=np.int64)            # (B,)
    return xs, ys

class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True, drop_last=False, collate_fn=default_collate, seed=0):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.collate_fn = collate_fn
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        n = len(self.dataset)
        idx = np.arange(n)
        if self.shuffle:
            self.rng.shuffle(idx)

        bs = self.batch_size
        for start in range(0, n, bs):
            end = start + bs
            if end > n and self.drop_last:
                break
            batch_idx = idx[start:end]
            batch = [self.dataset[i] for i in batch_idx]
            yield self.collate_fn(batch)

        def __len__(self):
            n = len(self.dataset)
            bs = self.batch_size
            if self.drop_last:
                return n // bs
            return (n + bs - 1) // bs 
        