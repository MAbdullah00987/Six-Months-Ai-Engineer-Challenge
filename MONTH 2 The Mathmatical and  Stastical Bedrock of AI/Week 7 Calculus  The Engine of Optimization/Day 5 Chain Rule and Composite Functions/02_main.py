
#NEURAL NETWORK BACKPROPAGATION
#Full implementation showing chain rule in action

#Topic 3: Multi-layer Neural Network with Manual Backpropagation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
np.random.seed(42)


# PART 1: NEURAL NETWORK CLASS WITH DETAILED BACKPROP
print("NEURAL NETWORK WITH CHAIN RULE BACKPROPAGATION")

class NeuralNetwork:
    """
    Two-layer neural network with detailed backpropagation
    Architecture: Input -> Hidden Layer -> Output
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        
        # Initialize weights with small random values
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Storage for gradients and activations
        self.cache = {}
        self.gradients = {}
        
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        """Derivative of sigmoid: σ'(z) = σ(z)(1 - σ(z))"""
        return a * (1 - a)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU"""
        return (z > 0).astype(float)
    
    def forward(self, X, verbose=False):
        """
        Forward pass through the network
        
        Layer 1: Z1 = X @ W1 + b1
                 A1 = relu(Z1)
        Layer 2: Z2 = A1 @ W2 + b2
                 A2 = sigmoid(Z2)
        """
        # Input layer to hidden layer
        self.cache['X'] = X
        self.cache['Z1'] = X @ self.W1 + self.b1
        self.cache['A1'] = self.relu(self.cache['Z1'])
        
        # Hidden layer to output layer
        self.cache['Z2'] = self.cache['A1'] @ self.W2 + self.b2
        self.cache['A2'] = self.sigmoid(self.cache['Z2'])
        
        if verbose:
            print("\nForward Pass:")
            print(f"  X shape: {X.shape}")
            print(f"  Z1 = X @ W1 + b1, shape: {self.cache['Z1'].shape}")
            print(f"  A1 = relu(Z1), shape: {self.cache['A1'].shape}")
            print(f"  Z2 = A1 @ W2 + b2, shape: {self.cache['Z2'].shape}")
            print(f"  A2 = sigmoid(Z2), shape: {self.cache['A2'].shape}")
        
        return self.cache['A2']
    
    def backward(self, X, y, verbose=False):
        """
        Backward pass using chain rule
        
        Chain rule applied layer by layer:
        1. dL/dZ2 = dL/dA2 * dA2/dZ2
        2. dL/dW2 = (dL/dZ2).T @ A1
        3. dL/dA1 = dL/dZ2 @ W2.T
        4. dL/dZ1 = dL/dA1 * dA1/dZ1
        5. dL/dW1 = X.T @ dL/dZ1
        """
        m = X.shape[0]  # number of samples
        
        # Output layer gradients
        # dL/dA2 for MSE loss: A2 - y
        dL_dA2 = self.cache['A2'] - y
        
        # Chain rule: dL/dZ2 = dL/dA2 * dA2/dZ2
        dA2_dZ2 = self.sigmoid_derivative(self.cache['A2'])
        dL_dZ2 = dL_dA2 * dA2_dZ2
        
        # Gradients for W2 and b2
        self.gradients['dW2'] = (self.cache['A1'].T @ dL_dZ2) / m
        self.gradients['db2'] = np.sum(dL_dZ2, axis=0, keepdims=True) / m
        
        # Backpropagate to hidden layer
        # Chain rule: dL/dA1 = dL/dZ2 @ W2.T
        dL_dA1 = dL_dZ2 @ self.W2.T
        
        # Chain rule: dL/dZ1 = dL/dA1 * dA1/dZ1
        dA1_dZ1 = self.relu_derivative(self.cache['Z1'])
        dL_dZ1 = dL_dA1 * dA1_dZ1
        
        # Gradients for W1 and b1
        self.gradients['dW1'] = (X.T @ dL_dZ1) / m
        self.gradients['db1'] = np.sum(dL_dZ1, axis=0, keepdims=True) / m
        
        if verbose:
            print("\nBackward Pass (Chain Rule):")
            print(f"  dL/dA2 shape: {dL_dA2.shape}")
            print(f"  dL/dZ2 = dL/dA2 * σ'(Z2), shape: {dL_dZ2.shape}")
            print(f"  dL/dW2 shape: {self.gradients['dW2'].shape}")
            print(f"  dL/dA1 = dL/dZ2 @ W2^T, shape: {dL_dA1.shape}")
            print(f"  dL/dZ1 = dL/dA1 * relu'(Z1), shape: {dL_dZ1.shape}")
            print(f"  dL/dW1 shape: {self.gradients['dW1'].shape}")
    
    def update_parameters(self):
        """Update parameters using gradient descent"""
        self.W1 -= self.lr * self.gradients['dW1']
        self.b1 -= self.lr * self.gradients['db1']
        self.W2 -= self.lr * self.gradients['dW2']
        self.b2 -= self.lr * self.gradients['db2']
    
    def compute_loss(self, y_true, y_pred):
        """Mean squared error loss"""
        return np.mean((y_true - y_pred)**2)
    
    def train(self, X, y, epochs, verbose_freq=100):
        """Train the network"""
        losses = []
        
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            losses.append(loss)
            
            # Backward pass
            self.backward(X, y)
            
            # Update parameters
            self.update_parameters()
            
            # Print progress
            if epoch % verbose_freq == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")
        
        return losses



# PART 2: CREATE SYNTHETIC DATASET
print("CREATING SYNTHETIC DATASET")


# Generate XOR-like problem (non-linearly separable)
def create_xor_dataset(n_samples=200):
    """Create XOR dataset"""
    np.random.seed(42)
    X = np.random.randn(n_samples, 2)
    y = ((X[:, 0] * X[:, 1]) > 0).astype(float).reshape(-1, 1)
    return X, y

X_train, y_train = create_xor_dataset(400)
print(f"\nDataset created:")
print(f"  Input shape: {X_train.shape}")
print(f"  Output shape: {y_train.shape}")
print(f"  Class distribution: {np.bincount(y_train.flatten().astype(int))}")


# PART 3: TRAIN THE NETWORK
print("TRAINING NEURAL NETWORK")

# Create and train network
nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1, learning_rate=0.5)

print("\nInitial forward and backward pass (verbose):")
y_pred_init = nn.forward(X_train[:5], verbose=True)
nn.backward(X_train[:5], y_train[:5], verbose=True)

print("\n" + "="*70)
print("\nTraining...")
losses = nn.train(X_train, y_train, epochs=1000, verbose_freq=200)

print("\nTraining complete!")


# PART 4: VISUALIZE RESULTS
print("CREATING VISUALIZATIONS")

fig = plt.figure(figsize=(16, 12))

# Plot 1: Loss curve
ax1 = plt.subplot(2, 3, 1)
ax1.plot(losses, linewidth=2, color='blue')
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontsize=11)
ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Original data
ax2 = plt.subplot(2, 3, 2)
scatter = ax2.scatter(X_train[:, 0], X_train[:, 1], 
                     c=y_train.flatten(), cmap='RdYlBu', 
                     s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
ax2.set_xlabel('X₁', fontsize=11)
ax2.set_ylabel('X₂', fontsize=11)
ax2.set_title('Original XOR Dataset', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Class')

# Plot 3: Decision boundary
ax3 = plt.subplot(2, 3, 3)
h = 0.02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

contour = ax3.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
ax3.scatter(X_train[:, 0], X_train[:, 1], 
           c=y_train.flatten(), cmap='RdYlBu',
           s=50, edgecolors='k', linewidth=1)
ax3.set_xlabel('X₁', fontsize=11)
ax3.set_ylabel('X₂', fontsize=11)
ax3.set_title('Learned Decision Boundary', fontsize=13, fontweight='bold')
plt.colorbar(contour, ax=ax3, label='Prediction')

# Plot 4: Weight distributions
ax4 = plt.subplot(2, 3, 4)
weights = [nn.W1.flatten(), nn.W2.flatten()]
ax4.hist(weights, bins=20, alpha=0.7, label=['W1', 'W2'])
ax4.set_xlabel('Weight Value', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Weight Distributions', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

# Plot 5: Gradient magnitudes
ax5 = plt.subplot(2, 3, 5)
grad_data = {
    'Layer': ['W1', 'b1', 'W2', 'b2'],
    'Magnitude': [
        np.linalg.norm(nn.gradients['dW1']),
        np.linalg.norm(nn.gradients['db1']),
        np.linalg.norm(nn.gradients['dW2']),
        np.linalg.norm(nn.gradients['db2'])
    ]
}
bars = ax5.bar(grad_data['Layer'], grad_data['Magnitude'], 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax5.set_ylabel('Gradient Magnitude', fontsize=11)
ax5.set_title('Final Gradient Magnitudes', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Prediction distribution
ax6 = plt.subplot(2, 3, 6)
y_pred_final = nn.forward(X_train)
ax6.hist(y_pred_final[y_train.flatten() == 0], bins=30, alpha=0.7, 
         label='Class 0', color='blue')
ax6.hist(y_pred_final[y_train.flatten() == 1], bins=30, alpha=0.7, 
         label='Class 1', color='red')
ax6.axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Threshold')
ax6.set_xlabel('Prediction Value', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Prediction Distribution', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neural_network_backprop.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'neural_network_backprop.png'")
plt.show()

# PART 5: ACCURACY METRICS
print("PERFORMANCE METRICS")

y_pred_final = nn.forward(X_train)
y_pred_class = (y_pred_final > 0.5).astype(float)
accuracy = np.mean(y_pred_class == y_train)

print(f"\nFinal Training Accuracy: {accuracy*100:.2f}%")
print(f"Final Loss: {losses[-1]:.6f}")


print("CHAIN RULE IN ACTION:")
print("Layer 2 (Output): dL/dW2 = (dL/dA2) * (dA2/dZ2) * (dZ2/dW2)")
print("Layer 1 (Hidden): dL/dW1 = (dL/dA2) * (dA2/dZ2) * (dZ2/dA1) * (dA1/dZ1) * (dZ1/dW1)")
print("\nEach gradient is computed by multiplying derivatives along the path!")
