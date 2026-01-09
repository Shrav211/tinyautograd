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

