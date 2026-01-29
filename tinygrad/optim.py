import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = list(params)

    def clip_grad_norm_(self, max_norm):
        total_sq = 0.0
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            total_sq += float(np.sum(g * g))
        total_norm = float(np.sqrt(total_sq))

        if total_norm > max_norm:
            scale = max_norm / total_norm
            for p in self.params:
                if p.grad is None:
                    continue
                p.grad = p.grad * scale

        return total_norm

class SGD(Optimizer):
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

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-12):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # state
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def state_dict(self):
        return {
            "t": self.t,
            "m": [mm.copy() for mm in self.m],
            "v": [vv.copy() for vv in self.v],
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, sd):
        self.t = int(sd["t"])
        self.lr = float(sd.get("lr", self.lr))
        b1, b2 = sd.get("betas", (self.beta1, self.beta2))
        self.beta1, self.beta2 = float(b1), float(b2)
        self.eps = float(sd.get("eps", self.eps))
        self.weight_decay = float(sd.get("weight_decay", self.weight_decay))

        m = sd["m"]
        v = sd["v"]
        if len(m) != len(self.params) or len(v) != len(self.params):
            raise ValueError(f"AdamW state mismatch: got {len(m)} slots, expected {len(self.params)}")

        # shape checks
        for i, p in enumerate(self.params):
            if m[i].shape != p.data.shape or v[i].shape != p.data.shape:
                raise ValueError(f"AdamW slot shape mismatch at idx {i}: "
                                 f"m {m[i].shape} v {v[i].shape} vs param {p.data.shape}")

        self.m = [mm.copy() for mm in m]
        self.v = [vv.copy() for vv in v]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self, clip_norm=None):
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        if clip_norm is not None:
            self.clip_grad_norm_(clip_norm)

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # update biased moments
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * (g * g)

            # bias correction
            m_hat = self.m[i] / (1 - (b1 ** self.t))
            v_hat = self.v[i] / (1 - (b2 ** self.t))

            # parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # decoupled weight decay
            p.data -= self.lr * self.weight_decay * p.data

    

            