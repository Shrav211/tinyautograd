# benchmarks/pytorch_baseline.py
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = torch.relu(out)
        return out


class ResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(16, 16, 3, 1)
        self.layer2 = self._make_layer(16, 32, 3, 2)
        self.layer3 = self._make_layer(32, 64, 3, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
    
    def _make_layer(self, in_ch, out_ch, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def benchmark_pytorch(device='cuda', steps=2000, batch_size=64):
    """Benchmark PyTorch ResNet-20 on CIFAR-10"""
    
    print(f"PyTorch Benchmark - Device: {device}, Steps: {steps}, Batch: {batch_size}")
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    
    # Model
    model = ResNet20().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    # Warmup
    print("Warming up...")
    model.train()
    warmup_iter = iter(train_loader)
    for _ in range(20):
        try:
            x, y = next(warmup_iter)
        except StopIteration:
            warmup_iter = iter(train_loader)
            x, y = next(warmup_iter)
        
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark
    print("Running benchmark...")
    model.train()
    
    losses = []
    start_time = time.perf_counter()
    
    train_iter = iter(train_loader)
    for step in range(steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 500 == 0:
            # Evaluate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test, y_test = x_test.to(device), y_test.to(device)
                    outputs = model(x_test)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y_test.size(0)
                    correct += (predicted == y_test).sum().item()
            
            acc = correct / total
            print(f"Step {step+1:4d} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
            model.train()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    total_time = end_time - start_time
    
    # Final evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            outputs = model(x_test)
            loss = criterion(outputs, y_test)
            test_loss += loss.item() * y_test.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
    
    final_acc = correct / total
    final_loss = test_loss / total
    
    results = {
        'framework': 'pytorch',
        'device': device,
        'steps': steps,
        'batch_size': batch_size,
        'total_time_seconds': total_time,
        'steps_per_second': steps / total_time,
        'samples_per_second': (steps * batch_size) / total_time,
        'final_accuracy': final_acc,
        'final_loss': final_loss,
        'loss_history': losses[::100],  # Sample every 100 steps
    }
    
    print(f"\nPyTorch Results:")
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
    parser.add_argument('--output', type=str, default='benchmarks/results/pytorch_results.json')
    args = parser.parse_args()
    
    results = benchmark_pytorch(args.device, args.steps, args.batch_size)
    
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")