# examples/mnist_cnn.py
import numpy as np
from tinygrad.tensor import Tensor, no_grad
from tinygrad.optim import AdamW
from tinygrad.nn import MNIST_CNN, cross_entropy_logits
from tinygrad.datasets.mnist import MNIST
from tinygrad.data import DataLoader, mnist_cnn_collate

def accuracy_from_logits(logits_np, y_np):
    pred = np.argmax(logits_np, axis=1)
    return float(np.mean(pred == y_np))

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    
    with no_grad():
        for xb, yb in loader:
            x = Tensor(xb, requires_grad=False)
            logits = model(x)
            loss = cross_entropy_logits(logits, yb)
            
            bs = xb.shape[0]
            total_loss += float(loss.data) * bs
            total_acc += accuracy_from_logits(logits.data, yb) * bs
            n += bs
    
    model.train()
    return total_loss / n, total_acc / n

def main():
    # Load data
    train_ds = MNIST(root="data/mnist", train=True, normalize=True, flatten=False)
    test_ds = MNIST(root="data/mnist", train=False, normalize=True, flatten=False)
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, 
                              collate_fn=mnist_cnn_collate, seed=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, 
                             collate_fn=mnist_cnn_collate)
    
    print("="*50)
    print("DATASET INFO")
    print("="*50)
    print(f"Train size: {len(train_ds)}")
    print(f"Test size: {len(test_ds)}")
    
    # Check first batch
    xb, yb = next(iter(train_loader))
    print(f"Batch X shape: {xb.shape}")  # Should be (128, 1, 28, 28)
    print(f"Batch Y shape: {yb.shape}")  # Should be (128,)
    print(f"X range: [{xb.min():.3f}, {xb.max():.3f}]")  # Should be [0, 1]
    print(f"Sample labels: {yb[:10]}")
    print("="*50)
    
    # Model and optimizer
    model = MNIST_CNN()
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # Training
    epochs = 3
    step = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        for xb, yb in train_loader:
            model.train()
            x = Tensor(xb, requires_grad=False)
            
            opt.zero_grad()
            logits = model(x)
            loss = cross_entropy_logits(logits, yb)
            loss.backward()
            opt.step()
            
            # Periodic evaluation
            if step % 100 == 0:
                tr_acc = accuracy_from_logits(logits.data, yb)
                te_loss, te_acc = evaluate(model, test_loader)
                print(f"Step {step:4d}  loss {float(loss.data):.4f}  "
                      f"tr_acc {tr_acc:.3f}  te_acc {te_acc:.3f}")
            
            step += 1
        
        # End of epoch evaluation
        te_loss, te_acc = evaluate(model, test_loader)
        print(f">>> Epoch {epoch + 1} complete: test_acc {te_acc:.4f}")
    
    # Final evaluation
    final_loss, final_acc = evaluate(model, test_loader)
    print("\n" + "="*50)
    print(f"FINAL TEST ACCURACY: {final_acc:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()