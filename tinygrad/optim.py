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

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-12):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        #state
        self.t = 0
        self.m = {} # first moment
        self.v = {} # second moment

        for p in self.params:
            self.m[id(p)] = np.zeros_like(p.data)
            self.v[id(p)] = np.zeros_like(p.data)

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
            pid = id(p)

            #update biased moments
            self.m[pid] = b1 * self.m[pid] + (1 - b1) * g
            self.v[pid] = b2 * self.v[pid] + (1 - b2) * (g * g)

            #bias correction
            m_hat = self.m[pid] / (1 - (b1 ** self.t))
            v_hat = self.v[pid] / (1 - (b2 ** self.t))

            #parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            #decoupled weight decay
            p.data -= self.lr * self.weight_decay * p.data
            