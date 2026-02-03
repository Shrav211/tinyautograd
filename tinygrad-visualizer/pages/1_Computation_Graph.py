import sys
import os

# Add parent directory to path
TINYGRAD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if TINYGRAD_ROOT not in sys.path:
    sys.path.insert(0, TINYGRAD_ROOT)

import streamlit as st
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

st.title("📊 Computation Graph Explorer")

# Initialize model in session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.current_step = 0
    st.session_state.activations = []

# Build simple network
if st.sidebar.button("Build XOR Network"):
    # Create a simple 2-layer network
    class XORNet:
        def __init__(self):
            self.fc1 = Linear(2, 4)
            self.fc2 = Linear(4, 1)
        
        def __call__(self, x):
            h = self.fc1(x).relu()
            out = self.fc2(h)
            return out
    
    st.session_state.model = XORNet()
    st.success("✅ Network built!")

# Forward pass with step-through
if st.session_state.model is not None:
    st.subheader("Input Data")
    
    # XOR inputs
    x_vals = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y_vals = [[0], [1], [1], [0]]
    
    sample_idx = st.slider("Select sample", 0, 3, 0)
    
    col1, col2 = st.columns(2)
    col1.metric("Input", str(x_vals[sample_idx]))
    col2.metric("Target", str(y_vals[sample_idx][0]))
    
    if st.button("▶️ Forward Pass"):
        x = Tensor([x_vals[sample_idx]])
        
        with st.expander("Layer 1: Linear (2→4)", expanded=True):
            h = st.session_state.model.fc1(x)
            st.write(f"Shape: `{h.data.shape}`")
            st.write(f"Values: `{h.data}`")
        
        with st.expander("Activation: ReLU", expanded=True):
            h_relu = h.relu()
            st.write(f"After ReLU: `{h_relu.data}`")
        
        with st.expander("Layer 2: Linear (4→1)", expanded=True):
            out = st.session_state.model.fc2(h_relu)
            st.write(f"Output: `{out.data}`")
            st.write(f"Target: `{y_vals[sample_idx][0]}`")
        
        # Show difference
        error = abs(float(out.data) - y_vals[sample_idx][0])
        st.metric("Prediction Error", f"{error:.4f}")
        
        if error < 0.1:
            st.success("🎯 Close to target!")
        else:
            st.info("🔄 Network needs training")
else:
    st.info("👈 Click 'Build XOR Network' in the sidebar to get started")