# examples/autograd_viz_sanity.py
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import MLP_LN, cross_entropy_logits
from tinygrad.viz import save_dot

def main():
    X = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]], dtype=float)
    y = np.array([0,1,1,0], dtype=int)  # for binary you'd use BCE; this is just graph demo

    model = MLP_LN(2, 8, 2, dropout_p=0.0)  # 2 classes for CE demo
    logits = model(Tensor(X, requires_grad=False))
    loss = cross_entropy_logits(logits, y)

    # optional: label
    loss._name = "loss"

    save_dot(loss, "graph.dot")
    print("Wrote graph.dot. Render with: dot -Tpng graph.dot -o graph.png")

if __name__ == "__main__":
    main()
