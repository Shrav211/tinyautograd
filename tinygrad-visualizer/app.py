import streamlit as st
import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt  # ✅ FIXED: Was "import matplotlib as plt"
import matplotlib.patches as patches

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
        self.activations = []
        self.gradients = []
        
    def __call__(self, x):
        self.activations = []
        
        # Input
        self.activations.append(x.data.copy())
        
        # Hidden layer
        h = self.fc1(x)
        self.activations.append(h.data.copy())
        h_relu = h.relu()
        self.activations.append(h_relu.data.copy())
        
        # Output layer
        out = self.fc2(h_relu)
        self.activations.append(out.data.copy())
        
        return out
    
    def get_weights(self):
        """Return weights for visualization"""
        W1 = self.fc1.W.data.copy()
        W2 = self.fc2.W.data.copy()
        
        # Handle transposing if needed
        if W1.shape == (2, hidden_size):
            W1 = W1.T
        if W2.shape == (hidden_size, 1):
            W2 = W2.T
        
        return [W1, W2]
    
    def get_gradients(self):
        """Return weight gradients for visualization"""
        if self.fc1.W.grad is None or self.fc2.W.grad is None:
            return None
        
        G1 = self.fc1.W.grad.copy()
        G2 = self.fc2.W.grad.copy()
        
        # Handle transposing if needed
        if G1.shape == (2, hidden_size):
            G1 = G1.T
        if G2.shape == (hidden_size, 1):
            G2 = G2.T
        
        return [G1, G2]
    
    def parameters(self):
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

with col2:
    st.subheader("Controls")
    
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
# Visualization (in col1)
# ============================================

with col1:
    st.subheader("Network Visualization")
    
    if st.session_state.model is not None:
        # Get current state
        weights = st.session_state.model.get_weights()
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Static View", "▶️ Forward Animation", "◀️ Backward Animation"])
        
        # TAB 1: Static View
        with tab1:
            if st.session_state.training_history['weights']:
                # Show training evolution
                epoch_slider = st.slider(
                    "View epoch",
                    0,
                    len(st.session_state.training_history['weights']) - 1,
                    len(st.session_state.training_history['weights']) - 1,
                    key="epoch_slider"
                )
                
                weights = st.session_state.training_history['weights'][epoch_slider]
                activations = st.session_state.training_history['activations'][epoch_slider]
                
                drawer = NetworkDrawer([2, hidden_size, 1])
                fig = drawer.draw_network(
                    weights=weights,
                    activations=activations
                )
                st.pyplot(fig)
                plt.close(fig)
                
                # Loss curve
                st.subheader("Training Progress")
                st.line_chart(st.session_state.training_history['loss'])
            else:
                # Just show initial network
                drawer = NetworkDrawer([2, hidden_size, 1])
                fig = drawer.draw_network(weights=weights)
                st.pyplot(fig)
                plt.close(fig)
        
        # TAB 2: Forward Animation
        with tab2:
            animation_speed = st.slider("Speed (steps/sec)", 1, 10, 3, key="forward_speed")
            sample_idx = st.selectbox("Select input", range(len(X)), key="forward_sample",
                                     format_func=lambda i: f"[{X[i,0]:.0f}, {X[i,1]:.0f}] → {y[i,0]:.0f}")
            
            if st.button("▶️ Play Forward Pass", key="play_forward"):
                # Run forward pass
                x_tensor = Tensor(X[sample_idx:sample_idx+1], requires_grad=False)
                output = st.session_state.model(x_tensor)
                
                # Get activations
                activations = st.session_state.model.activations
                
                # Placeholders for animation
                progress_placeholder = st.empty()
                network_viz = st.empty()
                
                num_layers = len(activations)
                
                for layer_idx in range(num_layers):
                    # Highlight current layer and connections
                    highlight_connections = None
                    if layer_idx > 0:
                        highlight_connections = (layer_idx - 1, layer_idx)
                    
                    # Draw network with current highlight
                    drawer = NetworkDrawer([2, hidden_size, 1])
                    fig = drawer.draw_network(
                        weights=weights,
                        activations=activations[:layer_idx+1],
                        highlight_layer=layer_idx,
                        highlight_connections=highlight_connections,
                        flow_direction='forward'
                    )
                    
                    network_viz.pyplot(fig)
                    plt.close(fig)
                    
                    # Progress indicator
                    progress_placeholder.progress((layer_idx + 1) / num_layers)
                    
                    # Wait based on speed
                    time.sleep(1.0 / animation_speed)
                
                # Show final result
                st.success(f"✅ Output: {float(output.data[0,0]):.4f} | Target: {y[sample_idx,0]:.0f}")
        
        # TAB 3: Backward Animation
        with tab3:
            backward_speed = st.slider("Speed (steps/sec)", 1, 10, 3, key="backward_speed")
            sample_idx_back = st.selectbox("Select input", range(len(X)), key="backward_sample",
                                          format_func=lambda i: f"[{X[i,0]:.0f}, {X[i,1]:.0f}] → {y[i,0]:.0f}")
            
            if st.button("◀️ Play Backward Pass", key="play_backward"):
                # Run forward and backward pass
                x_tensor = Tensor(X[sample_idx_back:sample_idx_back+1], requires_grad=False)
                y_tensor = Tensor(y[sample_idx_back:sample_idx_back+1], requires_grad=False)
                
                output = st.session_state.model(x_tensor)
                loss = ((output - y_tensor) ** 2).mean()
                
                # Zero gradients first
                optimizer = SGD(st.session_state.model.parameters(), lr=0.1)
                optimizer.zero_grad()
                
                # Backward pass
                loss.backward()
                
                # Get gradients
                gradients = st.session_state.model.get_gradients()
                activations = st.session_state.model.activations
                
                # Placeholders for animation
                progress_placeholder = st.empty()
                network_viz = st.empty()
                
                num_layers = len(activations)
                
                for step in range(num_layers):
                    # Go backwards
                    layer_idx = num_layers - 1 - step
                    
                    # Highlight current layer and connections
                    highlight_connections = None
                    if layer_idx < num_layers - 1:
                        highlight_connections = (layer_idx, layer_idx + 1)
                    
                    # Draw network with gradient highlight
                    drawer = NetworkDrawer([2, hidden_size, 1])
                    fig = drawer.draw_network(
                        weights=weights,
                        activations=activations,
                        gradients=gradients,
                        highlight_layer=layer_idx,
                        highlight_connections=highlight_connections,
                        flow_direction='backward'
                    )
                    
                    network_viz.pyplot(fig)
                    plt.close(fig)
                    
                    # Progress indicator
                    progress_placeholder.progress((step + 1) / num_layers)
                    
                    # Wait
                    time.sleep(1.0 / backward_speed)
                
                st.success(f"✅ Gradients computed! Loss: {float(loss.data):.6f}")
                
                # Show gradient heatmap
                if gradients is not None:
                    st.subheader("Gradient Magnitudes")
                    
                    fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
                    
                    for i, (grad, name) in enumerate(zip(gradients, ['Layer 1', 'Layer 2'])):
                        im = axes[i].imshow(np.abs(grad), cmap='Reds', aspect='auto')
                        axes[i].set_title(f'{name} Gradients')
                        axes[i].set_xlabel('Input neurons')
                        axes[i].set_ylabel('Output neurons')
                        plt.colorbar(im, ax=axes[i])
                    
                    st.pyplot(fig2)
                    plt.close(fig2)
    
    else:
        st.info("👈 Click 'Build Network' in the right panel to get started")