import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Module, Conv2d, Linear
from tinygrad.optim import AdamW

def softmax_cross_entropy_logits(logits: Tensor, y: np.ndarray):
    """
    logits: (N, C)
    y: (N,) int labels
    returns scalar Tensor
    """
    # use your existing cross_entropy_logits if you already have it
    from tinygrad.nn import cross_entropy_logits
    return cross_entropy_logits(logits, y)

class TinyCNN(Module):
    def __init__(self):
        super().__init__()
        self.c1 = Conv2d(1, 4, 3, stride=1, padding=1)   # (N,4,H,W)
        self.c2 = Conv2d(4, 8, 3, stride=2, padding=1)   # (N,8,H/2,W/2)
        # if input is 8x8 => after stride2 conv => 4x4
        self.fc = Linear(8*4*4, 3, init="xavier")

    def __call__(self, x: Tensor) -> Tensor:
        h = self.c1(x).relu()
        h = self.c2(h).relu()
        # flatten (N, 8, 4, 4) -> (N, 128)
        N = h.data.shape[0]
        # h = Tensor(h.data.reshape(N, -1), requires_grad=h.requires_grad)
        # h._prev = {h}  
        h = h.reshape(N, -1) # reshape as a tensor op
        return self.fc(h)

def make_tiny_images(seed=0):
    rng = np.random.RandomState(seed)
    N = 30
    # 8x8 "images", 3 classes
    X = rng.randn(N, 1, 8, 8).astype(float)
    y = rng.randint(0, 3, size=(N,))
    return X, y

def accuracy(logits, y):
    pred = np.argmax(logits, axis=1)
    return np.mean(pred == y)

def main():
    X, y = make_tiny_images(seed=0)

    model = TinyCNN()
    opt = AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    for step in range(1000):
        opt.zero_grad()

        logits = model(Tensor(X, requires_grad=False))
        loss = softmax_cross_entropy_logits(logits, y)
        loss.backward()

        opt.step()

        if step % 100 == 0:
            acc = accuracy(logits.data, y)
            print(f"step {step:4d}  loss {float(loss.data):.4f}  acc {acc:.3f}")

    # final
    logits = model(Tensor(X, requires_grad=False)).data
    print("FINAL acc:", accuracy(logits, y))

if __name__ == "__main__":
    main()
