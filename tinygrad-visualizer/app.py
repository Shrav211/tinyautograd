import streamlit as st
import numpy as np
import sys
import os
import time

# Add tinygrad to path
sys.path.insert(0, os.path.abspath('..'))

from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.optim import SGD
from utils.network_drawer import NetworkDrawer

st.set_page_config(page_title="XOR Network Visualizer", layout="wide")

st.title("🧠 Neural Network Visualizer - XOR Problem")

# ============================================
# Sidebar Controls
# ============================================

st.sidebar.header("⚙️ Configuration")

# Dataset selection
dataset = st.sidebar.selectbox("Dataset", ["XOR", "AND", "OR"])

# Network architecture
st.sidebar.subheader("Network Architecture")
input_size = 2
hidden_size = st.sidebar.slider("Hidden neurons", 2, 10, 4)
output_size = 1

# Training config
st.sidebar.subheader("Training Parameters")
learning_rate = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.1, 0.001)
epochs = st.sidebar.number_input("Epochs", 1, 1000, 100)

# ============================================
# Initialize Session State
# ============================================

if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = {'loss': [], 'weights': [], 'activations': []}

# ============================================
# Data
# ============================================

def get_data(dataset_name):
    """Get training data for selected dataset"""
    if dataset_name == "XOR":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    elif dataset_name == "AND":
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [0], [0], [1]], dtype=np.float32)
    else:  # OR
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
        y = np.array([[0], [1], [1], [1]], dtype=np.float32)
    return X, y

X, y = get_data(dataset)

# Display data
st.sidebar.subheader("📊 Training Data")
st.sidebar.dataframe({
    'Input 1': X[:, 0],
    'Input 2': X[:, 1],
    'Output': y[:, 0]
})

# ============================================
# Model Definition
# ============================================

class XORNetwork:
    def __init__(self, hidden_size):
        self.fc1 = Linear(2, hidden_size)
        self.fc2 = Linear(hidden_size, 1)
        self.activations = []  # Store for visualization
        
    def __call__(self, x):
        self.activations = []
        
        # Input
        self.activations.append(x.data.copy())
        
        # Hidden layer
        h = self.fc1(x)
        self.activations.append(h.data.copy())
        h = h.relu()
        self.activations.append(h.data.copy())
        
        # Output layer
        out = self.fc2(h)
        self.activations.append(out.data.copy())
        
        return out
    
    def get_weights(self):
        """Return weights for visualization"""
        return [
            self.fc1.W.data.copy(),
            self.fc2.W.data.copy()
        ]
    
    def parameters(self):
        """Return all parameters for optimizer"""
        return self.fc1.parameters() + self.fc2.parameters()

# ============================================
# Training
# ============================================

def train_network(model, X, y, lr, epochs, progress_bar, status_text):
    """Train the network and store history"""
    optimizer = SGD(model.parameters(), lr=lr)
    history = {'loss': [], 'weights': [], 'activations': []}
    
    for epoch in range(epochs):
        # Forward pass
        x_tensor = Tensor(X, requires_grad=False)
        y_tensor = Tensor(y, requires_grad=False)
        
        predictions = model(x_tensor)
        
        # MSE Loss
        loss = ((predictions - y_tensor) ** 2).mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Store history (sample every 10 epochs to save memory)
        if epoch % max(1, epochs // 100) == 0:
            history['loss'].append(float(loss.data))
            history['weights'].append(model.get_weights())
            history['activations'].append([a.copy() for a in model.activations])
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {float(loss.data):.6f}")
    
    return history

# ============================================
# Main Interface
# ============================================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Visualization")
    network_placeholder = st.empty()

with col2:
    st.subheader("Training Progress")
    
    # Build/Reset button
    if st.button("🔄 Build Network"):
        st.session_state.model = XORNetwork(hidden_size)
        st.session_state.training_history = {'loss': [], 'weights': [], 'activations': []}
        st.success("Network initialized!")
    
    # Train button
    if st.session_state.model is not None:
        if st.button("▶️ Train Network"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Train
            history = train_network(
                st.session_state.model,
                X, y,
                learning_rate,
                epochs,
                progress_bar,
                status_text
            )
            
            st.session_state.training_history = history
            st.success(f"✅ Training complete! Final loss: {history['loss'][-1]:.6f}")
        
        # Test network
        st.subheader("🧪 Test Network")
        test_sample = st.selectbox("Select test input", 
                                   [f"[{X[i,0]:.0f}, {X[i,1]:.0f}]" for i in range(len(X))])
        test_idx = int(test_sample.split('[')[1].split(',')[0])
        
        if st.button("Test"):
            x_test = Tensor(X[test_idx:test_idx+1], requires_grad=False)
            pred = st.session_state.model(x_test)
            
            col_a, col_b = st.columns(2)
            col_a.metric("Prediction", f"{float(pred.data[0,0]):.4f}")
            col_b.metric("Target", f"{y[test_idx,0]:.0f}")
            
            error = abs(float(pred.data[0,0]) - y[test_idx,0])
            if error < 0.1:
                st.success("🎯 Correct!")
            else:
                st.warning("❌ Incorrect")

# ============================================
# Visualization
# ============================================

if st.session_state.model is not None:
    # Get current state
    weights = st.session_state.model.get_weights()
    
    # If we have training history, show evolution
    if st.session_state.training_history['weights']:
        st.subheader("Training Evolution")
        
        epoch_slider = st.slider(
            "View epoch",
            0,
            len(st.session_state.training_history['weights']) - 1,
            len(st.session_state.training_history['weights']) - 1
        )
        
        weights = st.session_state.training_history['weights'][epoch_slider]
        activations = st.session_state.training_history['activations'][epoch_slider]
        
        # Draw network
        drawer = NetworkDrawer([2, hidden_size, 1])
        fig = drawer.draw_network(
            weights=weights,
            activations=activations
        )
        network_placeholder.pyplot(fig)
        
        # Loss curve
        st.line_chart(st.session_state.training_history['loss'])
    else:
        # Just show initial network
        drawer = NetworkDrawer([2, hidden_size, 1])
        fig = drawer.draw_network(weights=weights)
        network_placeholder.pyplot(fig)
else:
    st.info("👈 Click 'Build Network' to get started")

# In app.py, add animation section

st.sidebar.subheader("🎬 Animation")
show_animation = st.sidebar.checkbox("Animate forward/backward pass")

if show_animation and st.session_state.model is not None:
    animation_speed = st.sidebar.slider("Animation speed", 0.1, 2.0, 1.0)
    
    if st.button("▶️ Play Forward Pass"):
        # Animate data flowing through network
        for layer_idx in range(len(st.session_state.model.activations)):
            # Highlight current layer
            drawer = NetworkDrawer([2, hidden_size, 1])
            fig = drawer.draw_network(
                weights=weights,
                activations=st.session_state.model.activations[:layer_idx+1],
                highlight_layer=layer_idx
            )
            network_placeholder.pyplot(fig)
            time.sleep(1.0 / animation_speed)
    
    if st.button("◀️ Play Backward Pass"):
        # Animate gradients flowing backward
        # (similar but in reverse, with red color)
        pass