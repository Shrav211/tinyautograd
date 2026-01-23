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

#Conv2d
def _pair(x):
    return (x, x) if isinstance(x, int) else x

def im2col(x, kH, kW, stride=1, padding=0):
    # x: (N, C, H, W)
    sH, sW = _pair(stride)
    pH, pW = _pair(padding)

    N, C, H, W = x.shape
    H_p, W_p = H + 2 * pH, W + 2 * pW
    x_pad = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)), mode="constant")

    out_h = (H_p - kH) // sH + 1
    out_w = (W_p - kW) // sW + 1

    # building columns: (N, out_h, out_w, C, kH, kW)
    cols = np.empty((N, out_h, out_w, C, kH, kW), dtype=x.dtype)
    for i in range(kH):
        i_end = i + sH * out_h
        for j in range(kW):
            j_end = j + sW * out_w
            cols[..., i, j] = x_pad[:, :, i:i_end:sH, j:j_end:sW].transpose(0, 2, 3, 1)

    # Flatten to (N * out_h * out_w, C * kH * kW)
    x_col = cols.reshape(N * out_h * out_w, C * kH * kW)
    return x_col, out_h, out_w, x_pad.shape

def col2im(dX_col, x_pad_shape, kH, kW, stride=1, padding=0):
    sH, sW = _pair(stride)
    pH, pW = _pair(padding)

    N, C, H_p, W_p = x_pad_shape
    out_h = (H_p - kH) // sH + 1
    out_w = (W_p - kW) // sW + 1

    cols = dX_col.reshape(N, out_h, out_w, C, kH, kW)
    dx_pad = np.zeros((N, C, H_p, W_p), dtype=dX_col.dtype)

    for i in range(kH):
        i_end = i + sH*out_h
        for j in range(kW):
            j_end = j + sW*out_w
            dx_pad[:, :, i:i_end:sH, j:j_end:sW] += cols[..., i, j].transpose(0, 3, 1, 2)

    # remove padding
    if pH == 0 and pW == 0:
        return dx_pad
    return dx_pad[:, :, pH:-pH, pW:-pW]

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
        self._hooks = []

    @property
    def _track_grad(self):
        # i want to track this only if this tensor wants grad AND grad tracking is enabled globally
        return self.requires_grad and Tensor._grad_enabled

    def detach(self):
        # Will return a new tensor that shares the same data, but no autograd history
        out = Tensor(self.data, requires_grad=False)
        out._op = "detach" #useful for viz
        return out

    def backward(self, retain_graph=False):
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

        # add graph freeing, after all grads are computed, free the graph
        if not retain_graph:
            for v in topo:
                v._prev = set()
                v._backward = lambda: None
                v._op = ""

    def register_hook(self, fn):
        #fn: callable that takes (grad_np) and return a (grad_np)
        self._hooks.append(fn)
        return fn
    
    def _apply_hooks(self, grad):
        for h in self._hooks:
            grad = h(grad)
        return grad 

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
                    #self.grad += _unbroadcast(out.grad, self.data.shape)
                    grad_contrib = _unbroadcast(out.grad, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

                if other.requires_grad:
                    other.__init_grad()
                    # other.grad += _unbroadcast(out.grad, other.data.shape)
                    grad_contrib = _unbroadcast(out.grad, other.data.shape)
                    grad_contrib = other._apply_hooks(grad_contrib)
                    other.grad += grad_contrib

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
                    # self.grad += _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

                if other.requires_grad:
                    other.__init_grad()
                    grad_other = out.grad * self.data
                    # other.grad += _unbroadcast(grad_other, other.data.shape)
                    grad_contrib = _unbroadcast(grad_other, other.data.shape)
                    grad_contrib = other._apply_hooks(grad_contrib)
                    other.grad += grad_contrib

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
                    # self.grad += _unbroadcast(-out.grad, self.data.shape)
                    grad_contrib = _unbroadcast(-out.grad, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

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
                    # self.grad += _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

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
                # self.grad += np.broadcast_to(grad, self.data.shape)
                grad_contrib = np.broadcast_to(grad, self.data.shape)
                grad_contrib = self._apply_hooks(grad_contrib)
                self.grad += grad_contrib

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
                    # self.grad += _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

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
                    # self.grad += _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

                if other.requires_grad:
                    other.__init_grad()
                    grad_other = self.data.T @ out.grad
                    other.grad += _unbroadcast(grad_other, other.data.shape)
                    grad_contrib = _unbroadcast(grad_other, other.data.shape)
                    grad_contrib = other._apply_hooks(grad_contrib)
                    other.grad += grad_contrib

            out._backward = _backward
        else:
            out._op = "matmul"
        return out
    
    def sigmoid(self):
        out_data = 1 / (1 + np.exp(-self.data))
        req = Tensor._grad_enabled and self.requires_grad

        out = Tensor(out_data, requires_grad=req)
        
        if req:
            out._prev = {self}
            out._op = "sigmoid"

            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    self.__init_grad()
                    grad_self = out.grad * out.data * (1 - out.data)
                    # self.grad += _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = _unbroadcast(grad_self, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

            out._backward = _backward
        else:
            out._op = "relu"
        return out
    
    def log(self):
        
        req = Tensor._grad_enabled and self.requires_grad

        out = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        
        if req:
            out._prev = {self}
            out._op = "log"

            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    self.__init_grad()
                    # self.grad += _unbroadcast(out.grad / self.data, self.data.shape)
                    grad_contrib = _unbroadcast(out.grad / self.data, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib
            
            out._backward = _backward
        else:
            out._op = "log"
        return out
    
    def exp(self):
        
        req = Tensor._grad_enabled and self.requires_grad

        out = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        
        if req:
            out._prev = {self}
            out._op = "exp"

            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    self.__init_grad()
                    # self.grad += _unbroadcast(out.grad * out.data, self.data.shape)
                    grad_contrib = _unbroadcast(out.grad * out.data, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

            out._backward = _backward
        else:
            out._op = "exp"
        return out
    
    def abs(self):
        
        req = Tensor._grad_enabled and self.requires_grad
        out = Tensor(np.abs(self.data), requires_grad=self.requires_grad)
            
        if req:    
            out._prev = {self}
            out._op = "abs"

            def _backward():
                if out.grad is None:
                    return
                if self.requires_grad:
                    self.__init_grad()
                    sign = np.sign(self.data)
                    # self.grad += _unbroadcast(out.grad * sign, self.data.shape)
                    grad_contrib = _unbroadcast(out.grad * sign, self.data.shape)
                    grad_contrib = self._apply_hooks(grad_contrib)
                    self.grad += grad_contrib

            out._backward = _backward
        else:
            out._op = "abs"
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
            pass
        return out
    
    def reshape(self, *shape):
        # allow reshape((a,b)) or reshape(a,b)
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        req = getattr(Tensor, "_grad_enabled", True) and self.requires_grad
        out = Tensor(self.data.reshape(shape), requires_grad=req)
        out._op = "reshape"

        if not req:
            return out

        out._prev = {self}

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.__init_grad()
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out
    
    def transpose(self, *axes):
        """
        transpose() with either:
        - no args -> reverse axes
        - a tuple/list -> that permutation
        - separate ints -> permutation
        """
        if len(axes) == 0:
            perm = tuple(reversed(range(self.data.ndim)))
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        else:
            perm = tuple(axes)

        req = getattr(Tensor, "_grad_enabled", True) and self.requires_grad
        out = Tensor(self.data.transpose(perm), requires_grad=req)
        out._op = "transpose"

        if not req:
            return out

        out._prev = {self}

        # inverse permutation: inv[perm[i]] = i
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        inv = tuple(inv)

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                self.__init_grad()
                self.grad += out.grad.transpose(inv)

        out._backward = _backward
        return out

    def conv2d(self, weight, bias=None, stride=1, padding=0):
        """
        self:   x (N, C, H, W)
        weight: w (F, C, kH, kW)
        bias:   b (F,) or None
        returns out: (N, F, out_h, out_w)
        """
        x = self
        w = weight if isinstance(weight, Tensor) else Tensor(weight, requires_grad=False)
        b = None if bias is None else (bias if isinstance(bias, Tensor) else Tensor(bias, requires_grad=False))

        # respect no_grad
        req = getattr(Tensor, "_grad_enabled", True) and (x.requires_grad or w.requires_grad or (b is not None and b.requires_grad))

        X = x.data
        W = w.data
        if X.ndim != 4 or W.ndim != 4:
            raise ValueError(f"conv2d expects x (N,C,H,W) and w (F,C,kH,kW); got {X.shape}, {W.shape}")

        F, Cw, kH, kW = W.shape
        N, Cx, H, W_in = X.shape
        if Cw != Cx:
            raise ValueError(f"conv2d channel mismatch: x has C={Cx}, w has C={Cw}")

        X_col, out_h, out_w, x_pad_shape = im2col(X, kH, kW, stride=stride, padding=padding)  # (N*out_h*out_w, C*kH*kW)
        W_col = W.reshape(F, -1)  # (F, C*kH*kW)

        # Y_col: (N*out_h*out_w, F)
        Y_col = X_col @ W_col.T
        if b is not None:
            Y_col = Y_col + b.data.reshape(1, F)

        out_data = Y_col.reshape(N, out_h, out_w, F).transpose(0, 3, 1, 2)  # (N,F,out_h,out_w)
        out = Tensor(out_data, requires_grad=req)
        out._op = "conv2d"

        if not req:
            return out

        out._prev = {x, w} if b is None else {x, w, b}

        def _backward():
            if out.grad is None:
                return

            dY = out.grad  # (N,F,out_h,out_w)
            dY_col = dY.transpose(0, 2, 3, 1).reshape(N*out_h*out_w, F)  # (N*out_h*out_w, F)

            # grads w.r.t weight
            if w.requires_grad:
                w._Tensor__init_grad() if hasattr(w, "_Tensor__init_grad") else w.__init_grad()
                dW_col = dY_col.T @ X_col  # (F, C*kH*kW)
                w.grad += dW_col.reshape(w.data.shape)

            # grads w.r.t bias
            if b is not None and b.requires_grad:
                b._Tensor__init_grad() if hasattr(b, "_Tensor__init_grad") else b.__init_grad()
                db = dY_col.sum(axis=0)  # (F,)
                b.grad += db.reshape(b.data.shape)

            # grads w.r.t input
            if x.requires_grad:
                x._Tensor__init_grad() if hasattr(x, "_Tensor__init_grad") else x.__init_grad()
                dX_col = dY_col @ W_col  # (N*out_h*out_w, C*kH*kW)
                dX = col2im(dX_col, x_pad_shape, kH, kW, stride=stride, padding=padding)  # (N,C,H,W)
                x.grad += dX

        out._backward = _backward
        return out








