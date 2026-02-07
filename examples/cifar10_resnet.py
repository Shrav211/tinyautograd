# examples/cifar10_resnet.py
import numpy as np
try:
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()  # Clear cache
except:
    cp = None

from tinygrad.tensor import Tensor, no_grad
from tinygrad.data import DataLoader, cifar10_collate
from tinygrad.datasets.cifar10 import CIFAR10
from tinygrad.nn import ResNetCIFAR, cross_entropy_logits
from tinygrad.optim import AdamW
from tinygrad.amp import GradScaler
from tinygrad.device import to_numpy
from tinygrad.sched import WarmupCosineLR

def acc_from_logits(logits_np, y_np):
    return float((np.argmax(logits_np, axis=1) == y_np).mean())

def evaluate(model, loader, device="cpu"):
    model.eval()
    tot_loss = 0.0
    tot_acc  = 0.0
    n = 0
    
    with no_grad():
        for xb, yb in loader:
            x = Tensor(xb, requires_grad=False).to(device)
            logits = model(x)
            loss = cross_entropy_logits(logits, yb)
            
            bs = xb.shape[0]
            tot_loss += float(loss.data) * bs
            tot_acc  += acc_from_logits(to_numpy(logits.data), yb) * bs
            n += bs
            
            # Free memory
            del x, logits, loss
    
    # Clear GPU cache
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    
    model.train()
    return tot_loss / n, tot_acc / n

def main(device="cuda", mode="fp32"):
    train_ds = CIFAR10(root="data/cifar10", train=True, normalize=True)
    test_ds  = CIFAR10(root="data/cifar10", train=False, normalize=True)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=64,
        shuffle=True, 
        collate_fn=cifar10_collate, 
        seed=0
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=128,
        shuffle=False, 
        collate_fn=cifar10_collate
    )
    
    model = ResNetCIFAR(num_classes=10, n=3, use_cudnn=True)
    model.to(device)
    model.train()
    
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    scheduler = WarmupCosineLR(opt, warmup_steps=500, total_steps=2000)
    scaler = GradScaler() if (mode == "amp") else None
    
    steps = 8000
    eval_every = 500
    
    # CREATE INFINITE ITERATOR
    def infinite_loader():
        while True:
            for batch in train_loader:
                yield batch
    
    it = infinite_loader()  # ← Changed this!
    
    for step in range(steps):
        xb, yb = next(it)
        x = Tensor(xb, requires_grad=False).to(device)
        
        opt.zero_grad()
        
        if mode == "amp":
            x = Tensor(x.data.astype(x.xp.float16), requires_grad=False)
        
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
            scheduler.step()
            scaler.update(False)
        else:
            loss.backward()
            opt.step()
            scheduler.step()
        
        # Free memory after each step
        if step % 10 == 0 and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        
        if step % eval_every == 0:
            te_loss, te_acc = evaluate(model, test_loader, device=device)
            print(f"step {step:4d} loss {float(loss.data):.4f} te_acc {te_acc:.3f}")
    
    te_loss, te_acc = evaluate(model, test_loader, device=device)
    print(f"\nFINAL: loss {te_loss:.4f} acc {te_acc:.4f}")

if __name__ == "__main__":
    main(device="cuda", mode="fp32")  # Try "amp" if still OOM