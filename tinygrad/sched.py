import math

class LRScheduler:
    def __init__(self, optimizer):
        self.opt = optimizer
        self.last_step = -1

    def step(self):
        self.last_step += 1
        lr = self.get_lr(self.last_step)
        self.opt.lr = lr
        return lr
    
    def get_lr(self, step: int) -> float:
        raise NotImplementedError
    
class StepLR(LRScheduler):
    # multiply lr by gamma every 'step_size' steps
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1):
        super().__init__(optimizer) 
        self.step_size = step_size
        self.gamma = gamma
        self.base_lr = optimizer.lr

    def get_lr(self, step: int) -> float:
        k = step // self.step_size
        return self.base_lr * (self.gamma ** k)
    
class CosineAnnealingLR(LRScheduler):
    # Cosine decay from base_lr -> eta_min over T_max steps.
    def __init__(self, optimizer, T_max: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.T_max = max(1, T_max)
        self.eta_min = eta_min
        self.base_lr = optimizer.lr

    def get_lr(self, step: int) -> float:
        t = min(step, self.T_max)
        cos = (1 + math.cos(math.pi * t / self.T_max)) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * cos
    
class WarmupCosineLR(LRScheduler):
    #Linear warmup for warmup_steps, then cosine decay to eta_min over total_steps.
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, eta_min: float = 0.0):
        super().__init__(optimizer)
        self.warmup_steps = max(0, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.eta_min = eta_min
        self.base_lr = optimizer.lr

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # linear warmup from 0 -> base_lr
            return self.base_lr * (step + 1) / self.warmup_steps

        # cosine part
        t = step - self.warmup_steps
        T = max(1, self.total_steps - self.warmup_steps)
        t = min(t, T)
        cos = (1 + math.cos(math.pi * t / T)) / 2
        return self.eta_min + (self.base_lr - self.eta_min) * cos