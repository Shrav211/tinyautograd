from .tensor import Tensor

class Module:
    def parameters(self):
        return []
    
class Linear(Module):
    #Scalar Linear Layer
    def __init__(self, w_init=0.0, b_init=0.0):
        self.w = Tensor(w_init, requires_grad=True)
        self.b = Tensor(b_init, requires_grad=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.w * x + self.b
    
    def parameters(self):
        return [self.w, self.b]