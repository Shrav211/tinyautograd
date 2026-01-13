from .tensor import Tensor
import numpy as np

class Module:
    def __init__(self):
        self.training = True

    def train(self):
        self.training = True
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.train()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.train()
            elif isinstance(v, dict):
                for item in v.values():
                    if isinstance(item, Module):
                        item.train()
        return self

    def eval(self):
        self.training = False
        for v in self.__dict__.values():
            if isinstance(v, Module):
                v.eval()
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        item.eval()
            elif isinstance(v, dict):
                for item in v.values():
                    if isinstance(item, Module):
                        item.eval()
        return self

    def parameters(self):
        params = []

        def collect(obj):
            if isinstance(obj, Tensor):
                if obj.requires_grad:
                    params.append(obj)
            elif isinstance(obj, Module):
                for vv in obj.__dict__.values():
                    collect(vv)
            elif isinstance(obj, (list, tuple)):
                for vv in obj:
                    collect(vv)
            elif isinstance(obj, dict):
                for vv in obj.values():
                    collect(vv)

        for v in self.__dict__.values():
            collect(v)

        return params
    
    def _named_state(self, prefix=""):
        for k, v in self.__dict__.items():
            if k == "training":
                continue

            name = f"{prefix}{k}" if prefix == "" else f"{prefix}.{k}"

            # learnable parameters
            if isinstance(v, Tensor):
                if v.requires_grad:
                    yield (name, v, "param")

            # Buffers (non-learnable numpy arrays)
            elif isinstance(v, np.ndarray):
                # store BN running stats etc.
                yield (name, v, "buffer")

            # Submodules
            elif isinstance(v, Module):
                yield from v._named_state(prefix=name)

            # Containers
            elif isinstance(v, (list, tuple)):
                for i, item in enumerate(v):
                    subname = f"{name}.{i}"
                    if isinstance(item, Module):
                        yield from item._named_state(prefix=subname)
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield (subname, item, "param")
                    elif isinstance(item, np.ndarray):
                        yield (subname, item, "buffer")

            elif isinstance(v, dict):
                for kk, item in v.items():
                    subname = f"{name}.{kk}"
                    if isinstance(item, Module):
                        yield from item._named_state(prefix=subname)
                    elif isinstance(item, Tensor) and item.requires_grad:
                        yield (subname, item, "param")
                    elif isinstance(item, np.ndarray):
                        yield (subname, item, "buffer")
        
    def state_dict(self):
        sd = {}
        for name, obj, kind in self._named_state(prefix=""):
            if kind == "param":
                sd[name] = obj.data.copy()
            elif kind == "buffer":
                sd[name] = obj.copy()
        return sd
        
    def load_state_dict(self, sd, strict=True):
        missing = []
        unexpected = set(sd.keys())

        for name, obj, kind in self._named_state(prefix=""):
            if name not in sd:
                missing.append(name)
                continue

            unexpected.discard(name)
            val = sd[name]

            if kind == "param":
                # Tensor parameter
                if obj.data.shape != val.shape:
                    raise ValueError(f"Shape mismatch for {name}: {obj.data.shape} vs {val.shape}")
                obj.data = val.copy()

            elif kind == "buffer":
                # numpy buffer
                if obj.shape != val.shape:
                    raise ValueError(f"Shape mismatch for {name}: {obj.shape} vs {val.shape}")
                # assign back into attribute (since obj is a numpy array, rebinding is easiest)
                self._set_attr_by_name(name, val.copy())

        if strict:
            if missing:
                raise KeyError(f"Missing keys in state_dict: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected keys in state_dict: {sorted(unexpected)}")

        return {"missing": missing, "unexpected": sorted(unexpected)}

    def _set_attr_by_name(self, name, value):
        """
        Sets nested attribute by dotted path, e.g. "norm.running_mean".
        Supports numeric list indices too, e.g. "layers.0.W" if you ever use lists.
        """
        parts = name.split(".")
        obj = self

        for p in parts[:-1]:
            # list index?
            if p.isdigit():
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)

        last = parts[-1]
        if last.isdigit():
            obj[int(last)] = value
        else:
            setattr(obj, last, value)

    def named_modules(self, prefix="", include_self=True):
        # Yields (name, module) for this module and all submodules
        if include_self:
            yield (prefix, self)

        def walk(obj, name):
            if isinstance(obj, Module):
                yield (name, obj)
                for k, v in obj.__dict__.items():
                    if k == "training":
                        continue
                    child_name = f"{name}.{k}" if name else k
                    yield from walk(v, child_name)

            elif isinstance(obj, (list, tuple)):
                for i, item in enumerate(obj):
                    child_name = f"{name}.{i}" if name else str(i)
                    yield from walk(item, child_name)

            elif isinstance(obj, dict):
                for kk, item in obj.items():
                    child_name = f"{name}.{kk}" if name else str(kk)
                    yield from walk(item, child_name)

        for k, v in self.__dict__.items():
            if k == "training":
                continue
            child_name = f"{prefix}.{k}" if prefix else k
            yield from walk(v, child_name)
        
    def named_parameters(self, prefix=""):
        # Yield (name, Tensor) for all parameters
        for name, obj, kind in self._named_state(prefix=prefix):
            if kind == "param":
                yield (name, obj)

    def named_buffers(self, prefix=""):
        #yields (name, np.ndarray)
        for name, obj, kind in self._named_state(prefix=prefix):
            if kind == "buffer":
                yield (name, obj)

class Linear(Module):
    #Scalar Linear Layer
    def __init__(self, in_dim, out_dim, init="he"):
        super().__init__()

        if init == "he":
            scale = np.sqrt(2.0 / in_dim)
        elif init == "xavier":
            scale = np.sqrt(2.0 / (in_dim + out_dim))
        else:
            raise ValueError(f"Unknown init: {init}")
        
        W = np.random.randn(in_dim, out_dim) * scale
        b = np.zeros((out_dim,), dtype=float)
        
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.W) + self.b
    
class MLP_LN(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_p=0.0):
        super().__init__()
        self.l1 = Linear(in_dim, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.drop = Dropout(dropout_p)
        self.l2 = Linear(hidden_dim, out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        h = self.l1(x)
        h = self.norm(h)
        h = h.relu()
        h = self.drop(h)
        return self.l2(h)
    
class MLP_BN(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_p=0.0):
        super().__init__()
        self.l1 = Linear(in_dim, hidden_dim)
        self.norm = BatchNorm1d(hidden_dim)
        self.drop = Dropout(dropout_p)
        self.l2 = Linear(hidden_dim, out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        h = self.l1(x)
        h = self.norm(h)
        h = h.relu()
        h = self.drop(h)
        return self.l2(h)
    
class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps

        #Learnable parameters
        self.gamma = Tensor(np.ones((1, dim), dtype=float), requires_grad=True)
        self.beta = Tensor(np.zeros((1, dim), dtype=float), requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        #Normalize over last dimensions (features)
        mu = x.mean(axis=-1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)

        inv_std = (var + self.eps) ** -0.5
        xhat = (x - mu) * inv_std

        return xhat * self.gamma + self.beta

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        assert 0.0 <= p < 1.0
        self.p = p

    def __call__(self, x: Tensor) -> Tensor:
        if (not self.training) or self.p == 0.0:
            return x
        
        q = 1.0 - self.p
        mask = (np.random.rand(*x.data.shape) < q).astype(x.data.dtype)
        # inverted dropout: scale so expectation matches
        return x * Tensor(mask / q, requires_grad=False)
    
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        #learnable params
        self.gamma = Tensor(np.ones((1, dim), dtype=float), requires_grad=True)
        self.beta = Tensor(np.zeros((1, dim), dtype=float), requires_grad=True)

        #running stats not for learning
        self.running_mean = np.zeros((1, dim), dtype=float)
        self.running_var = np.ones((1, dim), dtype=float)

    def __call__(self, x: Tensor) -> Tensor:
        if self.training:
            # batch stats over N dimension
            mu = x.mean(axis=0, keepdims=True) # (1, D)
            var = ((x - mu) ** 2).mean(axis=0, keepdims=True) # (1, D)

            #update running stats (EMA)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data

            xhat = (x - mu) * ((var + self.eps) ** -0.5)
        else:
            # use stored running stats
            mu = Tensor(self.running_mean, requires_grad=False)
            var = Tensor(self.running_var, requires_grad=False)
            xhat = (x - mu) * ((var + self.eps) ** -0.5)

        return xhat * self.gamma + self.beta

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
    loss = (lse - z_y).mean()
    return loss
