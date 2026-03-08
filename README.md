# Neural Network From Scratch (NumPy)

A simple **feed-forward neural network implemented from scratch using NumPy**, designed for educational purposes.  
This project demonstrates how core deep learning components work internally, including:

- Forward propagation
- Backpropagation
- Activation functions
- Loss functions
- Mini-batch training
- Error tracking and visualization

The model is trained on the **scikit-learn Digits dataset** to classify handwritten digits.

---

### File Overview

| File | Description |
|-----|-------------|
| `functions.py` | Implements activation functions and loss functions |
| `neural_network.py` | Full neural network implementation (forward pass, backpropagation, training loop, visualization utilities) |
| `main.py` | Example script that trains the network on the sklearn digits dataset |

---

# Features

### Implemented from Scratch
No deep learning frameworks such as PyTorch or TensorFlow are used.

### Activation Functions
- Sigmoid
- ReLU
- Softmax

### Loss Function
- Log Loss (cross-entropy style)

### Neural Network Capabilities
- Fully connected feed-forward architecture
- Multiple hidden layers
- Custom activation functions
- Mini-batch gradient descent
- Backpropagation implementation
- Training cache for analysis
- Error visualization

### Visualization Tools
The network includes utilities to:

- Plot **training error evolution**
- Generate a **confusion matrix**

# Installation

Clone the repository:

```bash
git clone https://github.com/isaac-oldtown/neural-network-from-scratch.git
cd neural-network-from-scratch
```

Install dependencies:

```bash
pip install numpy matplotlib scikit-learn
```

---

# Running the Project

Run the training script:

```bash
cd src
python main.py
```

This will:

1. Load the digits dataset
2. Train the neural network
3. Plot the **training error curve**
4. Display a **confusion matrix**

---

# Learning Goals

This project was built to explore:

- How neural networks work internally
- The mathematics behind backpropagation
- How gradient descent updates weights
- How predictions and errors evolve during training

---

# Possible Improvements

Some ideas for extending the project:

- Add additional optimizers (Adam, RMSProp)
- Implement dropout
- Add weight initialization strategies
- Add support for different loss functions
- Implement dataset splitting (train/test)
- Vectorize more operations for speed
- Build a visualization of neuron activations
