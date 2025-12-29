from tinygrad.nn import Module, Linear

class TwoLayer(Module):
    def __init__(self):
        self.l1 = Linear()
        self.l2 = Linear()

    def __call__(self, x):
        return self.l2(self.l1(x))
    
m = TwoLayer()
ps = m.parameters()

print(len(ps))
for p in ps:
    print(p.data, p.requires_grad)