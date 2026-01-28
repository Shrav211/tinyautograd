import numpy as np

class GradScaler:
    def __init__(self, init_scale=2.0**16, growth_factor=2.0, backoff_factor=0.5,
                 growth_interval=2000, min_scale=1.0):
        self.scale = float(init_scale)
        self.growth_factor = float(growth_factor)
        self.backoff_factor = float(backoff_factor)
        self.growth_interval = int(growth_interval)
        self.min_scale = float(min_scale)
        self._good_steps = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def unscale_(self, params):
        inv = 1.0 / self.scale
        for p in params:
            if p.grad is None:
                continue
            p.grad = p.grad * inv

    def found_inf(self, params):
        for p in params:
            if p.grad is None:
                continue
            g = p.grad
            if not np.all(np.isfinite(g)):
                return True
        return False

    def update(self, found_inf):
        if found_inf:
            self.scale = max(self.min_scale, self.scale * self.backoff_factor)
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._good_steps = 0
