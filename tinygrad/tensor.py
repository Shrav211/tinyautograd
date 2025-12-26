#tensor should contain value, gradient, who created it and the how to push the gradients backwards

class Tensor:
    #basically a tensor should represent a value or node in a graph
    def __init__(self, data, requires_grad=False):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._prev = set()
        self._backward = lambda: None  

    def backward(self):
        # Implement the backward pass to compute gradients
        self.grad = 1
        
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

    def __add__(self, other):

        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if not out.requires_grad:
                return
            
            if self.requires_grad:
                if self.grad is None: self.grad = 0
                self.grad += out.grad

            if other.requires_grad:
                if other.grad is None: other.grad = 0
                other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if not out.requires_grad:
                return 
            
            if self.requires_grad:
                if self.grad is None: self.grad = 0
                self.grad += out.grad * other.data

            if other.requires_grad:
                if other.grad is None: other.grad = 0
                other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __neg__(self):

        out = Tensor((-1) * self.data, requires_grad=self.requires_grad)
        out._prev = {self}

        def _backward():
            if not out.requires_grad:
                return
            
            if self.requires_grad:
                if self.grad is None: self.grad = 0
                self.grad += (-1) * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):

        out = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}

        def _backward():
            if not out.requires_grad:
                return
            
            if self.requires_grad:
                if self.grad is None: self.grad = 0
                self.grad += out.grad

            if other.requires_grad:
                if other.grad is None: other.grad = 0
                other.grad += (-1) * out.grad

        out._backward = _backward
        return out

    def __pow__(self, p):

        out = Tensor(self.data ** p, requires_grad=self.requires_grad)
        out._prev = {self.other}

        def _backward():
            if not out.requires_grad:
                return
            
            if self.requires_grad:
                if self.grad is None: self.grad = 0
                self.grad += out.grad * p * (self.data ** (p-1))

        out._backward = _backward
        return out

x = Tensor(2, True)
y = Tensor(3, True)
z = x - y
z.backward()
print(x.grad, y.grad)

