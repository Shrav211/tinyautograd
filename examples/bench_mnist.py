import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP_BN, cross_entropy_logits  # or your MNIST model
from tinygrad.optim import AdamW
from tinygrad.amp import GradScaler
from tinygrad.datasets.mnist import MNIST
from tinygrad.data import DataLoader  # whatever you named it

def run(mode="fp32", steps=200, batch_size=128):
    train_ds = MNIST(root="data/mnist", train=True, normalize=True, flatten=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = MLP_BN(784, 256, 10, dropout_p=0.0)
    model.train()
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    scaler = GradScaler() if mode == "amp" else None

    t0 = time.perf_counter()
    n_seen = 0

    it = iter(train_loader)
    for s in range(steps):
        xb, yb = next(it)

        opt.zero_grad()

        if mode == "amp":
            x = Tensor(xb.astype(np.float16), requires_grad=False)
        else:
            x = Tensor(xb.astype(np.float32), requires_grad=False)

        logits = model(x)
        loss = cross_entropy_logits(logits, yb)

        if mode == "amp":
            scaled = scaler.scale_loss(loss)
            scaled.backward()
            scaler.unscale_(model.parameters())
            if scaler.found_inf(model.parameters()):
                opt.zero_grad()
                scaler.update(True)
                continue
            opt.step()
            scaler.update(False)
        else:
            loss.backward()
            opt.step()

        n_seen += xb.shape[0]

    t1 = time.perf_counter()
    dt = t1 - t0
    print(f"mode={mode} steps={steps} batch={batch_size}")
    print(f"  steps/sec:   {steps/dt:.2f}")
    print(f"  samples/sec: {n_seen/dt:.2f}")

def main():
    run("fp32", steps=200, batch_size=128)
    run("amp",  steps=200, batch_size=128)

if __name__ == "__main__":
    main()
