from .tensor import Tensor
import numpy as np

class Module:
    def parameters(self):
        params = []

        def collect(obj):
            if isinstance(obj, Tensor):
                if obj.requires_grad:
                    params.append(obj)
            elif isinstance(obj, Module):
                for v in obj.__dict__.values():
                    collect(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    collect(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)

        for v in self.__dict__.values():
            collect(v)

        return params
    
class Linear(Module):
    #Scalar Linear Layer
    def __init__(self, in_dim, out_dim):
        W = np.random.randn(in_dim, out_dim) * 0.1
        b = np.zeros((out_dim,))
        
        self.W = Tensor(W, requires_grad=True)
        self.b = Tensor(b, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return (x @ self.W) + self.b
    
class MLP(Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        self.l1 = Linear(in_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, out_dim)

    def __call__(self, x: Tensor) -> Tensor:
        h = self.l1(x).relu()
        y = self.l2(h)
        return y
