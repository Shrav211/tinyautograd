import numpy as np
from contextlib import contextmanager

def _unbroadcast(grad, shape):
    """
    Reduce grad to match `shape` by summing over broadcasted dimensions.
    grad: numpy array
    shape: tuple (original tensor shape)
    """
    # 1) If grad has extra leading dims, sum them away
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)

    # 2) Sum over axes where original shape had size 1
    for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad

@contextmanager
def no_grad():
    prev = Tensor._grad_enabled
    Tensor._grad_enabled = False
    try:
        yield
    finally:
        Tensor._grad_enabled = prev

#tensor should contain value, gradient, who created it and the how to push the gradients backwards
class Tensor:
    #basically a tensor should represent a value or node in a graph
    _grad_enabled = True

    def __init__(self, data, requires_grad=False):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = set()
        self._backward = lambda: None
        self._op = ""
        self._name = None

    @property
    def _track_grad(self):
        # i want to track this only if this tensor wants grad AND grad tracking is enabled globally
        return self.requires_grad and Tensor._grad_enabled

    def detach(self):
        # Will return a new tensor that shares the same data, but no autograd history
        out = Tensor(self.data, requires_grad=False)
        out._op = "detach" #useful for viz
        return out

    def backward(self):
        # Implement the backward pass to compute gradients
        if self.data.shape != () and self.data.size != 1:
            raise ValueError(f"backward() can only be called on a scalar loss, got shape{self.data.shape}")
        
        self.grad = np.ones_like(self.data)
        
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        for v in reversed(topo):
            v._backward()

    def __init_grad(self):
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

    def __add__(self, other):

        other = other if isinstance(other, Tensor) else Tensor(other)

        req = Tensor._grad_enabled and (self.requires_grad or other.requires_grad)
        out = Tensor(self.data + other.data, requires_grad=req)

        if req:
            out._prev = {self, other}
            out._op = "add"

            def _backward():
                if out.grad is None:
                    return
                
                if self.requires_grad:
                    self.__init_grad()
                    self.grad += _unbroadcast(out.grad, self.data.shape)

                if other.requires_grad:
                    other.__init_grad()
                    other.grad += _unbroadcast(out.grad, other.data.shape)

            out._backward = _backward
        else:
            out._op = "add"
        return out

    def __mul__(self, other):

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        req = Tensor._grad_enabled and (self.requires_grad or other.requires_grad)
        out = Tensor(self.data * other.data, requires_grad=req)
        
        if req:
            out._prev = {self, other}
            out._op = "mul"

            def _backward():
                if out.grad is None:
                    return

                if self.requires_grad:
                    self.__init_grad()
                    grad_self = out.grad * other.data
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                if other.requires_grad:
                    other.__init_grad()
                    grad_other = out.grad * self.data
                    other.grad += _unbroadcast(grad_other, other.data.shape)

            out._backward = _backward
        else:
            out._op = "mul"
        return out

    def __neg__(self):

        req = Tensor._grad_enabled and self.requires_grad
        out = Tensor((-1) * self.data, requires_grad=req)
            
        if req:    
            out._prev = {self}
            out._op = "neg"

            def _backward():
                if out.grad is None:
                    return
                
                if self.requires_grad:
                    self.__init_grad()
                    self.grad += _unbroadcast(-out.grad, self.data.shape)

            out._backward = _backward
        else:
            out._op = "neg"
        return out

    def __sub__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __pow__(self, p):

        req = Tensor._grad_enabled and self.requires_grad
        out = Tensor(self.data ** p, requires_grad=req)
        
        if req:
            out._prev = {self}
            out._op = "pow"

            def _backward():
                if out.grad is None:
                    return
                
                if self.requires_grad:
                    self.__init_grad()
                    grad_self = out.grad * (p * (self.data ** (p - 1)))
                    self.grad += _unbroadcast(grad_self, self.data.shape)

            out._backward = _backward
        else:
            out._op = "pow"
        return out
    
    def sum(self, axis=None, keepdims=False):
        
        req = Tensor._grad_enabled and (self.requires_grad)
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=req)
        if req:
            out._prev = {self}
            out._op = "sum"

            def _backward():
                if out.grad is None:
                    return
                if not self.requires_grad:
                    return

                self.__init_grad()  # or your __init_grad, use the correct name
                grad = out.grad

                # If summed over all elements, grad is scalar-like -> broadcast to input
                if axis is None:
                    self.grad += np.ones_like(self.data) * grad
                    return

                # Normalize axis to tuple
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)

                # If keepdims=False, reinsert reduced dims so broadcast works
                if not keepdims:
                    for a in sorted(axes):
                        grad = np.expand_dims(grad, axis=a)

                # Broadcast once to input shape
                self.grad += np.broadcast_to(grad, self.data.shape)

            out._backward = _backward
        else:
            out._op = "sum"
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            axes = tuple([a if a >= 0 else a + self.data.ndim for a in axes])
            denom = 1
            for a in axes:
                denom *= self.data.shape[a]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return self * (other ** -1)

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other * (self ** -1)

    def zero_grad(self):
        self.grad = None

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other - self
    
    def relu(self):
        
        req = Tensor._grad_enabled and self.requires_grad
        out = Tensor(np.maximum(0, self.data), requires_grad=req)
        
        if req:
            out._prev = {self}
            out._op = "relu"

            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    self.__init_grad()
                    mask = (self.data > 0).astype(self.data.dtype)
                    grad_self = out.grad * mask
                    self.grad += _unbroadcast(grad_self, self.data.shape)

            out._backward = _backward
        else:
            out._op = "relu"
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, requires_grad={self.requires_grad})"

    def __matmul__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        req = Tensor._grad_enabled and (self.requires_grad or other.requires_grad)

        out = Tensor(self.data @ other.data, requires_grad=req)
        
        if req:
            out._prev = {self, other}
            out._op = "matmul"

            def _backward():
                if out.grad is None:
                    return
                
                if self.requires_grad:
                    self.__init_grad()
                    grad_self = out.grad @ other.data.T
                    self.grad += _unbroadcast(grad_self, self.data.shape)

                if other.requires_grad:
                    other.__init_grad()
                    grad_other = self.data.T @ out.grad
                    other.grad += _unbroadcast(grad_other, other.data.shape)

            out._backward = _backward
        else:
            out._op = "matmul"
        return out
    
    def sigmoid(self):
        out_data = 1 / (1 + np.exp(-self.data))
        out = Tensor(out_data, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "sigmoid"

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.__init_grad()
                grad_self = out.grad * out.data * (1 - out.data)
                self.grad += _unbroadcast(grad_self, self.data.shape)

        out._backward = _backward
        return out
    
    def log(self):
        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "log"

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.__init_grad()
                self.grad += _unbroadcast(out.grad / self.data, self.data.shape)
        
        out._backward = _backward
        return out
    
    def exp(self):
        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "exp"

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.__init_grad()
                self.grad += _unbroadcast(out.grad * out.data, self.data.shape)

        out._backward = _backward
        return out
    
    def abs(self):
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "abs"

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.__init_grad()
                sign = np.sign(self.data)
                self.grad += _unbroadcast(out.grad * sign, self.data.shape)

        out._backward = _backward
        return out
    
    def max(self, axis=None, keepdims=False):
        # used for numerical stability; treat as constant (no grad)
        return Tensor(np.max(self.data, axis=axis, keepdims=keepdims), requires_grad=False)
    
    def mean(self, axis=None, keepdims=False):
        if axis is None:
            denom = self.data.size
        else:
            # number of elements reduced
            if isinstance(axis, int): axis_tuple = (axis,)
            else: axis_tuple = tuple(axis)
            denom = 1
            for ax in axis_tuple:
                denom *= self.data.shape[ax]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / denom)

    def logsumexp(self, axis=None, keepdims=False):
        m = self.max(axis=axis, keepdims=True)               # constant Tensor
        shifted = self - m
        s = shifted.exp().sum(axis=axis, keepdims=True).log()
        out = s + m
        if not keepdims and axis is not None:
            # optional squeeze behavior if you implemented it; otherwise keepdims=True everywhere
            pass
        return out






