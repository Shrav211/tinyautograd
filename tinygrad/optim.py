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

