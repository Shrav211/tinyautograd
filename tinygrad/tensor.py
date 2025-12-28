import numpy as np

def _unbroadcast(grad, target_shape):
    g = np.array(grad)

    # If target is scalar, everything was broadcast to something bigger → sum all
    if target_shape == ():
        return np.array(g.sum())

    # If grad has extra leading dims, sum them out
    while g.ndim > len(target_shape):
        g = g.sum(axis=0)

    # Now same ndim; for broadcasted dims (target=1), sum over that axis
    # Iterate from last axis backward to avoid axis index shifting issues
    for axis in range(len(target_shape) - 1, -1, -1):
        ts = target_shape[axis]
        gs = g.shape[axis]
        if ts == 1 and gs != 1:
            g = g.sum(axis=axis, keepdims=True)

    return g

#tensor should contain value, gradient, who created it and the how to push the gradients backwards
class Tensor:
    #basically a tensor should represent a value or node in a graph
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data) if not isinstance(data, np.ndarray) else data
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = set()
        self._backward = lambda: None  

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

        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

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
        return out

    def __mul__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                self.__init_grad()
                grad_self = out.grad * other.data
                self.grad += _unbroadcast(grad_self, self.data.shape)

            if other.requires_grad:
                other.__init_grad()
                grad_self = out.grad * self.data
                other.grad += _unbroadcast(grad_self, other.data.shape)

        out._backward = _backward
        return out

    def __neg__(self):

        out = Tensor((-1) * self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                self.__init_grad()
                self.grad += _unbroadcast(-out.grad, self.data.shape)

        out._backward = _backward
        return out

    def __sub__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)

    def __pow__(self, p):

        out = Tensor(self.data ** p, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                self.__init_grad()
                grad_self = out.grad * (p * (self.data ** (p - 1)))
                self.grad += _unbroadcast(grad_self, self.data.shape)

        out._backward = _backward
        return out
    
    def sum(self):

        out = Tensor(self.data.sum(), requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            
            if self.requires_grad:
                self.__init_grad()
                self.grad += np.ones_like(self.data) * out.grad

        out._backward = _backward
        return out

    def zero_grad(self):
        self.grad = None

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other):
        
        other = other if isinstance(other, Tensor) else Tensor(other)

        return other - self
    
x = Tensor(3, True)
z = 2 * x + 1
z.backward()
print(x.grad)  # expect 2





