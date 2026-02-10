# 🧠 TinyGrad

> A minimal deep learning framework built from scratch for educational purposes

**TinyGrad** is a clean-room implementation of a neural network framework in pure Python and NumPy/CuPy, demonstrating the core concepts behind modern deep learning libraries like PyTorch and TensorFlow.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ✨ Key Features

- 🔥 **Automatic Differentiation** - Full reverse-mode autograd engine with dynamic computation graphs
- 🚀 **GPU Acceleration** - CUDA support via CuPy with optimized convolution kernels
- 📊 **Common Layers** - Linear, Conv2d, BatchNorm, Dropout, Pooling, and more
- 🎯 **Modern Optimizers** - SGD, Adam, AdamW with learning rate scheduling
- 🧮 **Mixed Precision Training** - FP16 automatic mixed precision for faster training
- 📈 **Real Datasets** - MNIST and CIFAR-10 loaders with data augmentation
- 🏗️ **Modern Architectures** - MLP, CNN, ResNet implementations
- 🎨 **Interactive Visualizer** - Streamlit-based neural network visualization tool

## 🚀 Performance Benchmarks

### CIFAR-10 ResNet-20 Training Performance

Trained on **NVIDIA GTX 1050 Ti** (4GB VRAM) for 2000 steps (batch size 64).

## 🎯 Learning Objectives

This project demonstrates:

1. **Automatic Differentiation** - How autograd engines work under the hood
2. **GPU Programming** - CUDA acceleration with CuPy
3. **Performance Optimization** - Memory management, vectorization, kernel fusion
4. **Neural Network Architectures** - MLP, CNN, ResNet from scratch
5. **Training Dynamics** - Optimizers, learning rate schedules, regularization
6. **Software Engineering** - Clean API design, testing, documentation