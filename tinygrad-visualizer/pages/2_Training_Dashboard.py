import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("🎯 Training Dashboard")

# Training configuration
st.sidebar.header("Training Config")
lr = st.sidebar.slider("Learning Rate", 0.001, 0.1, 0.01, 0.001)
epochs = st.sidebar.number_input("Epochs", 1, 100, 10)

# Initialize history
if 'loss_history' not in st.session_state:
    st.session_state.loss_history = []
if 'acc_history' not in st.session_state:
    st.session_state.acc_history = []

# Start training button
if st.button("▶️ Start Training"):
    st.session_state.training = True

# Training loop (simulated)
if st.session_state.get('training', False):
    progress_bar = st.progress(0)
    status_text = st.empty()
    loss_chart = st.empty()
    
    for epoch in range(epochs):
        # Simulate training
        loss = 2.0 * np.exp(-epoch * 0.1) + np.random.randn() * 0.1
        acc = 0.5 + 0.4 * (1 - np.exp(-epoch * 0.1)) + np.random.randn() * 0.02
        
        st.session_state.loss_history.append(loss)
        st.session_state.acc_history.append(acc)
        
        # Update progress
        progress_bar.progress((epoch + 1) / epochs)
        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.3f}")
        
        # Update chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.plot(st.session_state.loss_history)
        ax1.set_title("Loss")
        ax1.set_xlabel("Epoch")
        ax2.plot(st.session_state.acc_history)
        ax2.set_title("Accuracy")
        ax2.set_xlabel("Epoch")
        loss_chart.pyplot(fig)
        plt.close()
        
        time.sleep(0.1)  # Simulate computation
    
    st.session_state.training = False
    st.success("Training complete!")

# Show history if exists
if st.session_state.loss_history:
    st.subheader("Training History")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
    ax1.plot(st.session_state.loss_history)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(st.session_state.acc_history)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    st.pyplot(fig)