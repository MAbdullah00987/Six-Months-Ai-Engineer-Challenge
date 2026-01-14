
#Part 3 - Backpropagation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("CHAIN RULE MASTERY - PART 3: BACKPROPAGATION IN ACTION")
# TOPIC 10: Simple Neural Network - Forward Pass

print("TOPIC 10: Building a 2-Layer Neural Network")


class SimpleNeuralNetwork:
    """
    A simple 2-layer neural network demonstrating chain rule in backpropagation
    Architecture: Input -> Hidden Layer -> Output Layer
    """
    
    def __init__(self, input_size=1, hidden_size=3, output_size=1, seed=42):
        np.random.seed(seed)
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        # Storage for intermediate values (needed for backprop)
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
        print(f"Network initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Output size: {output_size}")
        print(f"  W1 shape: {self.W1.shape}")
        print(f"  W2 shape: {self.W2.shape}")
    
    def sigmoid(self, x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid"""
        s = self.sigmoid(x)
        return s * (1 - s)
    
    def forward(self, x, verbose=False):
        """
        Forward pass through the network
        """
        self.x = x.reshape(-1, 1)  # Ensure column vector
        
        # Layer 1: Linear + Activation
        self.z1 = np.dot(self.x.T, self.W1) + self.b1  # z1 = x*W1 + b1
        self.a1 = self.sigmoid(self.z1)  # a1 = σ(z1)
        
        # Layer 2: Linear + Activation
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # z2 = a1*W2 + b2
        self.a2 = self.sigmoid(self.z2)  # a2 = σ(z2)
        
        if verbose:
            print(f"\nForward Pass:")
            print(f"  Input x: {self.x.flatten()}")
            print(f"  z1 = x*W1 + b1: {self.z1.flatten()}")
            print(f"  a1 = σ(z1): {self.a1.flatten()}")
            print(f"  z2 = a1*W2 + b2: {self.z2.flatten()}")
            print(f"  a2 = σ(z2) [OUTPUT]: {self.a2.flatten()}")
        
        return self.a2
    
    def backward(self, y_true, learning_rate=0.1, verbose=False):
        """
        Backward pass - Chain Rule in Action!
        This is where we apply the chain rule layer by layer
        """
        m = 1  # batch size
        
        # Compute loss derivative (MSE loss)
        # L = 1/2 * (y_pred - y_true)^2
        # dL/dy_pred = (y_pred - y_true)
        dL_da2 = self.a2 - y_true
        
        # ===== LAYER 2 BACKPROP (Chain Rule!) =====
        # We need: dL/dW2, dL/db2, dL/da1
        
        # 1. dL/dz2 = dL/da2 * da2/dz2
        #    da2/dz2 = sigmoid'(z2) = a2 * (1 - a2)
        dL_dz2 = dL_da2 * self.sigmoid_derivative(self.z2)
        
        # 2. dL/dW2 = a1.T * dL/dz2
        dL_dW2 = np.dot(self.a1.T, dL_dz2) / m
        
        # 3. dL/db2 = sum(dL/dz2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True) / m
        
        # 4. dL/da1 = dL/dz2 * W2.T (chain rule: propagate back)
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        
        # ===== LAYER 1 BACKPROP (Chain Rule!) =====
        # We need: dL/dW1, dL/db1
        
        # 1. dL/dz1 = dL/da1 * da1/dz1
        #    da1/dz1 = sigmoid'(z1) = a1 * (1 - a1)
        dL_dz1 = dL_da1 * self.sigmoid_derivative(self.z1)
        
        # 2. dL/dW1 = x.T * dL/dz1
        dL_dW1 = np.dot(self.x, dL_dz1) / m
        
        # 3. dL/db1 = sum(dL/dz1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True) / m
        
        if verbose:
            print(f"\nBackward Pass (Chain Rule):")
            print(f"  dL/da2: {dL_da2.flatten()}")
            print(f"  dL/dz2 = dL/da2 * σ'(z2): {dL_dz2.flatten()}")
            print(f"  dL/dW2: {dL_dW2.flatten()}")
            print(f"  dL/da1: {dL_da1.flatten()}")
            print(f"  dL/dz1 = dL/da1 * σ'(z1): {dL_dz1.flatten()}")
            print(f"  dL/dW1: {dL_dW1.flatten()}")
        
        # Update weights (Gradient Descent)
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        
        return {
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1,
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2
        }


# TOPIC 11: Training the Network - Chain Rule in Action!
print("TOPIC 11: Training with Backpropagation")

# Create training data: y = sin(x)
np.random.seed(42)
X_train = np.linspace(-np.pi, np.pi, 50)
y_train = np.sin(X_train)

# Initialize network
nn = SimpleNeuralNetwork(input_size=1, hidden_size=5, output_size=1)

# Demonstrate one forward and backward pass
print("\n" + "-" * 70)
print("DEMONSTRATION: Single Training Step")
print("-" * 70)

sample_x = 1.0
sample_y = np.sin(sample_x)

print(f"\nTraining sample: x={sample_x:.2f}, y_true={sample_y:.4f}")

# Forward pass
y_pred = nn.forward(np.array([sample_x]), verbose=True)

# Compute loss
loss = 0.5 * (y_pred - sample_y)**2
print(f"\nLoss: {loss[0][0]:.6f}")

# Backward pass
gradients = nn.backward(sample_y, learning_rate=0.1, verbose=True)

# Train the network
print("\n" + "-" * 70)
print("TRAINING THE NETWORK")
print("-" * 70)

epochs = 1000
learning_rate = 0.5
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for x, y in zip(X_train, y_train):
        # Forward pass
        y_pred = nn.forward(np.array([x]))
        
        # Compute loss
        loss = 0.5 * (y_pred - y)**2
        epoch_loss += loss[0][0]
        
        # Backward pass
        nn.backward(y, learning_rate)
    
    avg_loss = epoch_loss / len(X_train)
    losses.append(avg_loss)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# TOPIC 12: Visualizing Learning Progress

print("\n" + "=" * 70)
print("TOPIC 12: Visualizing the Learning Process")
print("=" * 70)

# Generate predictions
X_test = np.linspace(-np.pi, np.pi, 100)
y_test = np.sin(X_test)
y_pred_final = np.array([nn.forward(np.array([x]))[0][0] for x in X_test])

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Learning Curve
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (MSE)', fontsize=12)
ax1.set_title('Learning Curve: Loss vs Epochs', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_yscale('log')

# Plot 2: Predictions vs True Function
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(X_test, y_test, 'b-', linewidth=3, label='True: y=sin(x)', alpha=0.7)
ax2.plot(X_test, y_pred_final, 'r--', linewidth=2, label='Neural Network Prediction')
ax2.scatter(X_train, y_train, c='green', s=30, alpha=0.5, label='Training Data')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Neural Network Function Approximation', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# Plot 3: Error Distribution
ax3 = fig.add_subplot(gs[1, 1])
y_pred_train = np.array([nn.forward(np.array([x]))[0][0] for x in X_train])
errors = y_train - y_pred_train
ax3.hist(errors, bins=20, color='purple', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Prediction Error', fontsize=12)
ax3.set_ylabel('Frequency', fontsize=12)
ax3.set_title('Error Distribution', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Weight Visualization - Layer 1
ax4 = fig.add_subplot(gs[2, 0])
w1_df = pd.DataFrame(nn.W1, columns=[f'H{i+1}' for i in range(nn.W1.shape[1])])
w1_df.index = ['Input']
sns.heatmap(w1_df, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            ax=ax4, cbar_kws={'label': 'Weight Value'})
ax4.set_title('Layer 1 Weights (W1)', fontsize=14, fontweight='bold')

# Plot 5: Weight Visualization - Layer 2
ax5 = fig.add_subplot(gs[2, 1])
w2_df = pd.DataFrame(nn.W2, columns=['Output'])
w2_df.index = [f'H{i+1}' for i in range(nn.W2.shape[0])]
sns.heatmap(w2_df, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
            ax=ax5, cbar_kws={'label': 'Weight Value'})
ax5.set_title('Layer 2 Weights (W2)', fontsize=14, fontweight='bold')

plt.savefig('backpropagation_training.png', dpi=150, bbox_inches='tight')
print("\n✓ Training visualization saved as 'backpropagation_training.png'")
plt.show()


# TOPIC 13: Gradient Flow Visualization
print("TOPIC 13: Visualizing Gradient Flow (Chain Rule)")


# Compute gradients for a single sample
sample_idx = 25
x_sample = X_train[sample_idx]
y_sample = y_train[sample_idx]

# Forward pass
y_pred = nn.forward(np.array([x_sample]))
loss = 0.5 * (y_pred - y_sample)**2

# Backward pass to get gradients
gradients = nn.backward(y_sample, learning_rate=0)  # lr=0 to not update

# Create gradient flow visualization
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

layers = ['Input\nx', 'Hidden\nz1, a1', 'Output\nz2, a2', 'Loss\nL']
x_positions = np.array([0, 1, 2, 3])

# Draw nodes
for i, (x_pos, layer) in enumerate(zip(x_positions, layers)):
    circle = plt.Circle((x_pos, 0.5), 0.3, color='lightblue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(x_pos, 0.5, layer, ha='center', va='center', fontsize=11, fontweight='bold')

# Draw forward arrows
for i in range(len(x_positions) - 1):
    ax.annotate('', xy=(x_positions[i+1]-0.35, 0.5), xytext=(x_positions[i]+0.35, 0.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    ax.text((x_positions[i] + x_positions[i+1])/2, 0.7, 'forward', 
            ha='center', fontsize=10, color='blue')

# Draw backward arrows (gradients)
for i in range(len(x_positions) - 1, 0, -1):
    ax.annotate('', xy=(x_positions[i-1]+0.35, 0.3), xytext=(x_positions[i]-0.35, 0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text((x_positions[i-1] + x_positions[i])/2, 0.1, 'backprop', 
            ha='center', fontsize=10, color='red')

# Add gradient values
ax.text(2.5, -0.2, f'dL/dW2:\n{gradients["dL_dW2"][0,0]:.4f}', 
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
ax.text(0.5, -0.2, f'dL/dW1:\n{gradients["dL_dW1"][0,0]:.4f}', 
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlim(-0.5, 3.5)
ax.set_ylim(-0.5, 1.2)
ax.axis('off')
ax.set_title('Gradient Flow Through Network (Chain Rule)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('gradient_flow.png', dpi=150, bbox_inches='tight')
print("\n✓ Gradient flow visualization saved")
plt.show()


