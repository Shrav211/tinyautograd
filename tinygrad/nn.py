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
        skipped = []

        for name, obj, kind in self._named_state(prefix=""):
            if name not in sd:
                missing.append(name)
                continue

            unexpected.discard(name)
            val = sd[name]

            try:
                if kind == "param":
                    if obj.data.shape != val.shape:
                        skipped.append((name, obj.data.shape, val.shape))
                        continue
                    obj.data = val.copy()

                elif kind == "buffer":
                    if obj.shape != val.shape:
                        skipped.append((name, obj.shape, val.shape))
                        continue
                    self._set_attr_by_name(name, val.copy())
            except Exception as e:
                skipped.append((name, "error", str(e)))
                continue

        report = {
            "missing": missing,
            "unexpected": sorted(unexpected),
            "skipped": skipped,
        }

        if strict:
            if missing:
                raise KeyError(f"Missing keys in state_dict: {missing}")
            if unexpected:
                raise KeyError(f"Unexpected keys in state_dict: {sorted(unexpected)}")
            if skipped:
                raise ValueError(f"Some keys had shape mismatch or load errors: {skipped}")

        return report
    
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

    def label_parameters_(self, prefix=""):
        """
        Mutates parameter tensors in-place by setting tensor._name = full_name.
        This is purely for visualization/debugging.
        """
        for name, p in self.named_parameters(prefix=prefix):
            # only stamp if it’s a Tensor and trainable
            try:
                p._name = name
            except Exception:
                pass
        return self

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

class BatchNorm2d(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        # One gamma/beta per channel
        self.gamma = Tensor(np.ones((1, num_features, 1, 1)), requires_grad=True)
        self.beta = Tensor(np.zeros((1, num_features, 1, 1)), requires_grad=True)
        
        # Running stats per channel
        self.running_mean = np.zeros((1, num_features, 1, 1), dtype=np.float32)
        self.running_var  = np.ones((1, num_features, 1, 1), dtype=np.float32)
    
    def __call__(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        
        if self.training:
            # Compute stats over (N, H, W), keep C dimension
            mu = x.mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
            var = ((x - mu) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
            
            xhat = (x - mu) * ((var + self.eps) ** -0.5)
        else:
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
    loss = (lse - z_y).mean()  # Changed this line!
    return loss

class Conv2d(Module):
    """
    Conv2d layer:
      input  x: (N, Cin, H, W)
      weight W: (Cout, Cin, kH, kW)
      bias   b: (Cout,) or None
      output y: (N, Cout, out_h, out_w)
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, init="he"):
        super().__init__()

        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = (kH, kW)
        self.stride       = stride
        self.padding      = padding

        # fan_in for conv = Cin * kH * kW
        fan_in = in_channels * kH * kW

        if init == "he":
            scale = np.sqrt(2.0 / fan_in)
        elif init == "xavier":
            # a common conv-xavier: 2/(fan_in + fan_out), fan_out = Cout*kH*kW
            fan_out = out_channels * kH * kW
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Unknown init: {init}")

        W = np.random.randn(out_channels, in_channels, kH, kW).astype(float) * scale
        self.W = Tensor(W, requires_grad=True)
        self.W._name = "W"  # optional; you’ll rename with named_parameters later

        if bias:
            b0 = np.zeros((out_channels,), dtype=float)
            self.b = Tensor(b0, requires_grad=True)
            self.b._name = "b"
        else:
            self.b = None

    def __call__(self, x: Tensor) -> Tensor:
        return x.conv2d(self.W, self.b, stride=self.stride, padding=self.padding)

class MaxPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        # self.padding = padding

    def __call__(self, x: Tensor) -> Tensor:
        return x.maxpool2d(kernel_size=self.kernel_size, stride=self.stride)

class Conv2d(Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, init="he"):
        super().__init__()
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size

        self.stride = stride
        self.padding = padding

        # He init for ReLU conv: Var(W)=2/fan_in where fan_in = in_ch*kH*kW
        fan_in = in_ch * kH * kW
        if init == "he":
            scale = np.sqrt(2.0 / fan_in)
        elif init == "xavier":
            fan_out = out_ch * kH * kW
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        else:
            raise ValueError(f"Unknown init: {init}")

        W = np.random.randn(out_ch, in_ch, kH, kW) * scale
        self.W = Tensor(W, requires_grad=True)

        if bias:
            self.b = Tensor(np.zeros((out_ch,), dtype=float), requires_grad=True)
        else:
            self.b = None

    def __call__(self, x: Tensor) -> Tensor:
        return x.conv2d(self.W, self.b, stride=self.stride, padding=self.padding)

class Flatten(Module):
    def __init__(self, start_dim=1):
        super().__init__()
        self.start_dim = start_dim

    def __call__(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=self.start_dim)
    
class GlobalAvgPool2d(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        # x: (N,C,H,W) -> (N,C)
        return x.mean(axis=(2, 3), keepdims=False)

# class MNIST_CNN(Module):
#     def __init__(self):
#         super().__init__()
#         self.c1 = Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
#         self.bn1 = BatchNorm2d(16)
#         self.pool = MaxPool2d(2, 2)

#         self.c2 = Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
#         self.bn2 = BatchNorm2d(32)

#         self.gap = GlobalAvgPool2d()
#         self.fc = Linear(32, 10, init="xavier")  # logits

#     def __call__(self, x: Tensor) -> Tensor:
#         x = self.c1(x)
#         x = self.bn1(x)
#         x = x.relu()
#         x = self.pool(x)

#         x = self.c2(x)
#         x = self.bn2(x)
#         x = x.relu()

#         x = self.gap(x)          # (N,32)
#         x = self.fc(x)           # (N,10)
#         return x

class MNIST_CNN(Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 1 input channel!
        self.bn1 = BatchNorm2d(16)
        self.pool = MaxPool2d(2, 2)
        self.c2 = Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(32)
        self.gap = GlobalAvgPool2d()
        self.fc = Linear(32, 10, init="xavier")
    
    def __call__(self, x: Tensor) -> Tensor:
        x = self.c1(x)
        x = self.bn1(x)
        x = x.relu()
        x = self.pool(x)
        
        x = self.c2(x)
        x = self.bn2(x)
        x = x.relu()
        
        x = self.gap(x)
        x = self.fc(x)
        
        return x