#Phase 2: Manual Calculation (Build Intuition)
#Let's work through a concrete example by hand first:
#Example: h(x) = sigmoid(3x + 2)

#Outer function: f(u) = sigmoid(u) = 1/(1 + e^(-u))
#Inner function: g(x) = 3x + 2

#Step-by-step:

#g'(x) = 3
#f'(u) = sigmoid(u) · (1 - sigmoid(u))
#h'(x) = f'(g(x)) · g'(x) = sigmoid(3x + 2) · (1 - sigmoid(3x + 2)) · 3

#Multi-Layer Neural Network 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

# Simple 2-layer neural network
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(42)
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Store for visualization
        self.cache = {}
        
    def forward(self, X):
        """Forward pass: X -> h1 -> h2 -> output"""
        # Layer 1
        self.cache['X'] = X
        self.cache['z1'] = np.dot(X, self.W1) + self.b1
        self.cache['a1'] = sigmoid(self.cache['z1'])
        
        # Layer 2
        self.cache['z2'] = np.dot(self.cache['a1'], self.W2) + self.b2
        self.cache['a2'] = sigmoid(self.cache['z2'])
        
        return self.cache['a2']
    
    def backward(self, X, y, output):
        """Backward pass: Chain rule from output to input"""
        m = X.shape[0]
        
        # Output layer gradient (chain rule step 1)
        # dL/dz2 = dL/da2 * da2/dz2
        dz2 = output - y  # For binary cross-entropy + sigmoid
        dW2 = np.dot(self.cache['a1'].T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradient (chain rule step 2)
        # dL/dz1 = dL/dz2 * dz2/da1 * da1/dz1
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * sigmoid_derivative(self.cache['z1'])
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def train_step(self, X, y, learning_rate=0.1):
        """One training iteration"""
        output = self.forward(X)
        gradients = self.backward(X, y, output)
        
        # Update weights
        self.W1 -= learning_rate * gradients['dW1']
        self.b1 -= learning_rate * gradients['db1']
        self.W2 -= learning_rate * gradients['dW2']
        self.b2 -= learning_rate * gradients['db2']
        
        # Calculate loss
        loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
        return loss

# Generate XOR dataset (classic non-linear problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

print("=" * 70)
print("BACKPROPAGATION: CHAIN RULE IN A NEURAL NETWORK")
print("=" * 70)
print("\nSolving XOR Problem (Non-linear Classification)")
print("-" * 70)
print("Training Data:")
print(pd.DataFrame(np.hstack([X, y]), columns=['x1', 'x2', 'y']))

# Create and train network
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Training
epochs = 2000
losses = []
print(f"\nTraining for {epochs} epochs...")

for epoch in range(epochs):
    loss = nn.train_step(X, y, learning_rate=0.5)
    losses.append(loss)
    
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d}: Loss = {loss:.6f}")

print("\n" + "=" * 70)
print("FINAL PREDICTIONS")
print("=" * 70)
final_output = nn.forward(X)
predictions_df = pd.DataFrame({
    'x1': X[:, 0],
    'x2': X[:, 1],
    'Target': y.flatten(),
    'Prediction': final_output.flatten(),
    'Rounded': np.round(final_output.flatten())
})
print(predictions_df)

# Visualization
fig = plt.figure(figsize=(16, 10))

# Plot 1: Loss curve
ax1 = plt.subplot(2, 3, 1)
ax1.plot(losses, 'b-', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Decision boundary
ax2 = plt.subplot(2, 3, 2)
h = 0.01
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
ax2.scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], c='blue', s=200, 
            edgecolors='k', marker='o', label='Class 0', linewidth=2)
ax2.scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], c='red', s=200, 
            edgecolors='k', marker='s', label='Class 1', linewidth=2)
ax2.set_xlabel('x₁', fontsize=12)
ax2.set_ylabel('x₂', fontsize=12)
ax2.set_title('Decision Boundary', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Weight magnitudes
ax3 = plt.subplot(2, 3, 3)
w1_flat = nn.W1.flatten()
w2_flat = nn.W2.flatten()
ax3.hist(w1_flat, bins=15, alpha=0.6, label='Layer 1 Weights', color='blue')
ax3.hist(w2_flat, bins=15, alpha=0.6, label='Layer 2 Weights', color='red')
ax3.set_xlabel('Weight Value', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Weight Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Activation patterns (Layer 1)
ax4 = plt.subplot(2, 3, 4)
activations = nn.cache['a1']
im = ax4.imshow(activations.T, aspect='auto', cmap='viridis', interpolation='nearest')
ax4.set_xlabel('Training Sample', fontsize=12)
ax4.set_ylabel('Hidden Unit', fontsize=12)
ax4.set_title('Hidden Layer Activations', fontsize=14, fontweight='bold')
ax4.set_xticks(range(4))
ax4.set_xticklabels(['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
plt.colorbar(im, ax=ax4)

# Plot 5: Prediction vs Target
ax5 = plt.subplot(2, 3, 5)
x_pos = np.arange(len(X))
width = 0.35
ax5.bar(x_pos - width/2, y.flatten(), width, label='Target', alpha=0.8, color='green')
ax5.bar(x_pos + width/2, final_output.flatten(), width, label='Prediction', alpha=0.8, color='orange')
ax5.set_xlabel('Sample', fontsize=12)
ax5.set_ylabel('Value', fontsize=12)
ax5.set_title('Predictions vs Targets', fontsize=14, fontweight='bold')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['[0,0]', '[0,1]', '[1,0]', '[1,1]'])
ax5.legend()
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Chain rule visualization
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
ax6.text(0.5, 0.9, 'Chain Rule Breakdown', ha='center', fontsize=14, fontweight='bold', 
         transform=ax6.transAxes)

chain_text = """
Forward Pass:
  x → z₁ = W₁x + b₁
    → a₁ = σ(z₁)
      → z₂ = W₂a₁ + b₂
        → a₂ = σ(z₂) → Loss

Backward Pass (Chain Rule):
  ∂L/∂W₂ = ∂L/∂z₂ · ∂z₂/∂W₂
  ∂L/∂W₁ = ∂L/∂z₂ · ∂z₂/∂a₁ · ∂a₁/∂z₁ · ∂z₁/∂W₁

Each gradient flows backward
through the chain of derivatives!
"""

ax6.text(0.5, 0.5, chain_text, ha='center', va='center', fontsize=10, 
         family='monospace', transform=ax6.transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()


