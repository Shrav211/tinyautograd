from .tensor import Tensor

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
    def __init__(self, w_init=0.0, b_init=0.0):
        self.w = Tensor(w_init, requires_grad=True)
        self.b = Tensor(b_init, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w * x + self.b
    