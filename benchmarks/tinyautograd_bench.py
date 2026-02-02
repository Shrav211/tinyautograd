import time
import json
import numpy as np
try:
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
except:
    cp = None

from tinygrad.tensor import Tensor, no_grad
from tinygrad.data import DataLoader, cifar10_collate
from tinygrad.datasets.cifar10 import CIFAR10
from tinygrad.nn import ResNetCIFAR, cross_entropy_logits
from tinygrad.optim import AdamW
from tinygrad.device import to_numpy


def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    
    with no_grad():
        for xb, yb in loader:
            x = Tensor(xb, requires_grad=False).to(device)
            logits = model(x)
            loss = cross_entropy_logits(logits, yb)
            
            pred = np.argmax(to_numpy(logits.data), axis=1)
            correct += (pred == yb).sum()
            total += yb.shape[0]
            test_loss += float(loss.data) * yb.shape[0]
            
            del x, logits, loss
    
    if cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    
    model.train()
    return test_loss / total, correct / total


def benchmark_tinygrad(device='cuda', steps=2000, batch_size=64):
    """Benchmark TinyGrad ResNet-20 on CIFAR-10"""
    
    print(f"TinyAutoGrad Benchmark - Device: {device}, Steps: {steps}, Batch: {batch_size}")
    
    # Data
    train_ds = CIFAR10(root="data/cifar10", train=True, normalize=True)
    test_ds = CIFAR10(root="data/cifar10", train=False, normalize=True)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              collate_fn=cifar10_collate, seed=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, 
                             collate_fn=cifar10_collate)
    
    # Model
    model = ResNetCIFAR(num_classes=10, n=3)
    model.to(device)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    # Warmup
    print("Warming up...")
    warmup_iter = iter(train_loader)
    for _ in range(20):
        try:
            xb, yb = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            xb, yb = next(warmup_iter)
        
        x = Tensor(xb, requires_grad=False).to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_logits(logits, yb)
        loss.backward()
        optimizer.step()
    
    if device == 'cuda' and cp is not None:
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    
    # Benchmark
    print("Running benchmark...")
    
    losses = []
    
    def infinite_loader():
        while True:
            for batch in train_loader:
                yield batch
    
    train_iter = infinite_loader()
    
    start_time = time.perf_counter()
    
    for step in range(steps):
        xb, yb = next(train_iter)
        x = Tensor(xb, requires_grad=False).to(device)
        
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy_logits(logits, yb)
        loss.backward()
        optimizer.step()
        
        losses.append(float(loss.data))
        
        if step % 10 == 0 and cp is not None:
            cp.get_default_memory_pool().free_all_blocks()
        
        if (step + 1) % 500 == 0:
            te_loss, te_acc = evaluate(model, test_loader, device=device)
            print(f"Step {step+1:4d} | Loss: {float(loss.data):.4f} | Acc: {te_acc:.4f}")
    
    if device == 'cuda' and cp is not None:
        cp.cuda.Stream.null.synchronize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Final evaluation
    final_loss, final_acc = evaluate(model, test_loader, device=device)
    
    results = {
        'framework': 'tinygrad',
        'device': device,
        'steps': steps,
        'batch_size': batch_size,
        'total_time_seconds': total_time,
        'steps_per_second': steps / total_time,
        'samples_per_second': (steps * batch_size) / total_time,
        'final_accuracy': final_acc,
        'final_loss': final_loss,
        'loss_history': losses[::100],
    }
    
    print(f"\nTinyAutoGrad Results:")
    print(f"Total time: {total_time:.2f}s")
    print(f"Steps/sec: {results['steps_per_second']:.2f}")
    print(f"Samples/sec: {results['samples_per_second']:.2f}")
    print(f"Final accuracy: {final_acc:.4f}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--output', type=str, default='benchmarks/results/tinyautograd_results.json')
    args = parser.parse_args()
    
    results = benchmark_tinygrad(args.device, args.steps, args.batch_size)
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")