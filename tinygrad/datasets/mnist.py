# tinygrad/datasets/mnist.py
import os, gzip, struct, urllib.request
import numpy as np

from ..data import Dataset

MNIST_URLS = {
    "train_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
        "https://yann.lecun.org/exdb/mnist/train-images-idx3-ubyte.gz",
    ],
    "train_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
        "https://yann.lecun.org/exdb/mnist/train-labels-idx1-ubyte.gz",
    ],
    "test_images": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
        "https://yann.lecun.org/exdb/mnist/t10k-images-idx3-ubyte.gz",
    ],
    "test_labels": [
        "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
        "https://yann.lecun.org/exdb/mnist/t10k-labels-idx1-ubyte.gz",
    ],
}

def _download_any(urls, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return

    last_err = None
    for url in urls:
        try:
            print(f"Downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path)
            return
        except Exception as e:
            last_err = e
            print(f"  failed: {e}")
    raise RuntimeError(f"All download mirrors failed for {path}. Last error: {last_err}")

# then wherever you did _download(url, path), do:
# _download_any(MNIST_URLS["train_images"], train_images_path)

def _read_idx_gz(path):
    """
    Reads IDX format (gzipped) into numpy array.
    """
    with gzip.open(path, "rb") as f:
        magic = f.read(4)
        if len(magic) != 4:
            raise ValueError("Bad IDX file (too short)")
        zero, data_type, dims = struct.unpack(">HBB", magic)
        if zero != 0:
            raise ValueError("Bad IDX file (bad magic prefix)")

        shape = []
        for _ in range(dims):
            (d,) = struct.unpack(">I", f.read(4))
            shape.append(d)

        # data_type per IDX spec:
        # 0x08 = unsigned byte
        # 0x09 = signed byte
        # 0x0B = short
        # 0x0C = int
        # 0x0D = float
        # 0x0E = double
        dtype_map = {
            0x08: np.uint8,
            0x09: np.int8,
            0x0B: np.int16,
            0x0C: np.int32,
            0x0D: np.float32,
            0x0E: np.float64,
        }
        if data_type not in dtype_map:
            raise ValueError(f"Unsupported IDX dtype code: {data_type}")

        data = f.read()
        arr = np.frombuffer(data, dtype=dtype_map[data_type])
        return arr.reshape(shape)

def load_mnist(root="data/mnist"):
    """
    Returns:
      (X_train, y_train), (X_test, y_test)
    X arrays: uint8 (N, 28, 28)
    y arrays: uint8 (N,)
    """
    paths = {}
    for k, urls in MNIST_URLS.items():
        # Use first URL to get filename
        filename = os.path.basename(urls[0])
        paths[k] = os.path.join(root, filename)

    for k, url in MNIST_URLS.items():
        _download_any(url, paths[k])

    X_train = _read_idx_gz(paths["train_images"])
    y_train = _read_idx_gz(paths["train_labels"])
    X_test  = _read_idx_gz(paths["test_images"])
    y_test  = _read_idx_gz(paths["test_labels"])

    return (X_train, y_train), (X_test, y_test)

class MNIST(Dataset):
    def __init__(self, root="data/mnist", train=True, normalize=True, flatten=True):
        (Xtr, ytr), (Xte, yte) = load_mnist(root=root)
        if train:
            self.X = Xtr
            self.y = ytr
        else:
            self.X = Xte
            self.y = yte

        self.normalize = normalize
        self.flatten = flatten

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (28,28) uint8
        y = self.y[idx]  # scalar uint8

        x = x.astype(np.float32)
        if self.normalize:
            x = x / 255.0

        if self.flatten:
            x = x.reshape(-1)  # (784,)

        y = int(y)  # integer label
        return x, y
