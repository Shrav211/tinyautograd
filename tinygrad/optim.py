import numpy as np

class SGD:
    def __init__(self, params, lr=1e-2):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for p in self.params:
            if p.requires_grad:
                p.data -= self.lr * p.grad

class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        #state
        self.t = 0
        self.m = {} # first moment
        self.v = {} # second moment

        for p in self.params:
            self.m[id(p)] = np.zeros_like(p.data, dtype=float)
            self.v[id(p)] = np.zeros_like(p.data, dtype=float)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        for p in self.params:
            if p.grad is None:
                continue

            g = p.grad
            key = id(p)

            #update biased moments
            self.m[key] = b1 * self.m[key] + (1 - b1) * g
            self.v[key] = b2 * self.v[key] + (1 - b2) * (g * g)

            #bias correction
            m_hat = self.m[key] / (1 - (b1 ** self.t))
            v_hat = self.v[key] / (1 - (b2 ** self.t))

            #parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            