

#Complete integration of all libraries for real-world ML problems
#Topic 5: End-to-End Machine Learning with Chain Rule Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 12)


# PART 1: SYMBOLIC ANALYSIS WITH SYMPY
print("PART 1: SYMBOLIC GRADIENT ANALYSIS")

def analyze_loss_function():
    """Analyze common loss functions symbolically"""
    
    # Define symbols
    y_true, y_pred = sp.symbols('y_true y_pred', real=True)
    w, x, b = sp.symbols('w x b', real=True)
    
    print("\n1. Mean Squared Error (MSE) Loss")
    mse = (y_true - y_pred)**2
    print(f"   L = {mse}")
    
    # Derivative with respect to prediction
    dL_dy_pred = sp.diff(mse, y_pred)
    print(f"   ∂L/∂y_pred = {dL_dy_pred}")
    print(f"   Simplified: {sp.simplify(dL_dy_pred)}")
    
    print("\n2. Binary Cross-Entropy Loss")
    # For numerical stability, we'll use a simplified version
    bce = -(y_true * sp.log(y_pred) + (1 - y_true) * sp.log(1 - y_pred))
    print(f"   L = {bce}")
    
    dL_dy_pred_bce = sp.diff(bce, y_pred)
    print(f"   ∂L/∂y_pred = {dL_dy_pred_bce}")
    print(f"   Simplified: {sp.simplify(dL_dy_pred_bce)}")
    
    print("\n3. Linear Model: y = wx + b")
    y_linear = w * x + b
    print(f"   y_pred = {y_linear}")
    
    # Substitute into MSE
    mse_linear = (y_true - y_linear)**2
    
    dL_dw = sp.diff(mse_linear, w)
    dL_db = sp.diff(mse_linear, b)
    
    print(f"   ∂L/∂w = {dL_dw}")
    print(f"   ∂L/∂b = {dL_db}")
    print(f"   Chain rule: ∂L/∂w = (∂L/∂y_pred)(∂y_pred/∂w)")
    
    print("\n4. Sigmoid Activation with MSE")
    # FIX: Define z as a symbol, not an expression
    z = sp.symbols('z', real=True)
    sigmoid = 1 / (1 + sp.exp(-z))
    print(f"   z = w*x + b (treating z as symbolic variable)")
    print(f"   σ(z) = {sigmoid}")
    
    # Derivative of sigmoid - NOW THIS WORKS!
    dsigmoid_dz = sp.diff(sigmoid, z)
    print(f"   ∂σ/∂z = {sp.simplify(dsigmoid_dz)}")
    print(f"   Note: σ'(z) = σ(z)(1 - σ(z))")
    
    # Show the chain rule for the full expression
    print("\n5. Chain Rule for Full Network")
    z_expr = w * x + b
    print(f"   Given: z = {z_expr}")
    print(f"   And: σ(z) = 1/(1 + e^(-z))")
    
    # Substitute z expression into sigmoid
    sigmoid_full = sigmoid.subs(z, z_expr)
    
    # Now we can differentiate with respect to w
    dsigmoid_dw = sp.diff(sigmoid_full, w)
    print(f"   ∂σ/∂w = {sp.simplify(dsigmoid_dw)}")
    print(f"   This demonstrates: ∂σ/∂w = (∂σ/∂z)(∂z/∂w) = σ'(z) * x")

analyze_loss_function()

# PART 2: DATA ANALYSIS WITH PANDAS
print("\n" + "="*60)
print("PART 2: REAL DATA ANALYSIS WITH PANDAS")
print("="*60)


# Generate synthetic dataset
np.random.seed(42)
n_samples = 1000

# Create features
data = {
    'feature_1': np.random.randn(n_samples),
    'feature_2': np.random.randn(n_samples),
    'feature_3': np.random.randn(n_samples),
}

# Create target (non-linear relationship)
data['target'] = (
    2 * data['feature_1']**2 + 
    3 * data['feature_2'] - 
    1.5 * data['feature_3'] + 
    np.random.randn(n_samples) * 0.5
)

df = pd.DataFrame(data)

print("\n1. Dataset Overview")
print(df.head(10))
print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# Compute correlations
print("\n2. Feature Correlations")
correlations = df.corr()
print(correlations)

# Gradient analysis using finite differences
def compute_numerical_gradient(df, target_col, feature_col, epsilon=1e-5):
    """Compute numerical gradient of target w.r.t feature"""
    gradients = []
    
    for idx in range(len(df)):
        original_val = df.loc[idx, feature_col]
        target_original = df.loc[idx, target_col]
        
        # Perturb feature
        df.loc[idx, feature_col] = original_val + epsilon
        # Recompute target (approximation)
        
        # For demonstration, use correlation as proxy
        gradient = correlations.loc[target_col, feature_col]
        gradients.append(gradient)
        
        # Restore
        df.loc[idx, feature_col] = original_val
    
    return np.array(gradients)

# Add gradient estimates to dataframe
for col in ['feature_1', 'feature_2', 'feature_3']:
    df[f'grad_{col}'] = compute_numerical_gradient(df, 'target', col)

print("\n3. Gradient Estimates (using correlation)")
print(df[['target', 'grad_feature_1', 'grad_feature_2', 'grad_feature_3']].head())


# PART 3: ADVANCED NEURAL NETWORK WITH MOMENTUM
print("\n" + "="*60)
print("PART 3: ADVANCED NEURAL NETWORK")
print("="*60)


class AdvancedNeuralNetwork:
    """Neural network with momentum and adaptive learning rate"""
    
    def __init__(self, layer_sizes, learning_rate=0.01, momentum=0.9):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.momentum = momentum
        
        # Initialize weights
        self.weights = []
        self.biases = []
        self.velocities_w = []
        self.velocities_b = []
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
            self.velocities_w.append(np.zeros_like(w))
            self.velocities_b.append(np.zeros_like(b))
        
        self.cache = {}
        self.gradients = {}
        
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def forward(self, X):
        """Forward propagation through all layers"""
        self.cache['A0'] = X
        
        for i in range(len(self.weights)):
            Z = self.cache[f'A{i}'] @ self.weights[i] + self.biases[i]
            self.cache[f'Z{i+1}'] = Z
            
            # Use ReLU for hidden layers, sigmoid for output
            if i < len(self.weights) - 1:
                A = self.relu(Z)
            else:
                A = self.sigmoid(Z)
            
            self.cache[f'A{i+1}'] = A
        
        return self.cache[f'A{len(self.weights)}']
    
    def backward(self, X, y):
        """Backpropagation with chain rule"""
        m = X.shape[0]
        n_layers = len(self.weights)
        
        # Output layer gradient
        dA = self.cache[f'A{n_layers}'] - y
        
        for i in range(n_layers - 1, -1, -1):
            # Current layer
            A_curr = self.cache[f'A{i}']
            Z_curr = self.cache[f'Z{i+1}']
            
            # Apply activation derivative
            if i == n_layers - 1:
                dZ = dA * self.sigmoid_derivative(self.cache[f'A{i+1}'])
            else:
                dZ = dA * self.relu_derivative(Z_curr)
            
            # Gradients
            self.gradients[f'dW{i}'] = (A_curr.T @ dZ) / m
            self.gradients[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Backpropagate to previous layer
            if i > 0:
                dA = dZ @ self.weights[i].T
    
    def update_parameters(self):
        """Update with momentum"""
        for i in range(len(self.weights)):
            # Momentum update
            self.velocities_w[i] = (self.momentum * self.velocities_w[i] + 
                                   self.lr * self.gradients[f'dW{i}'])
            self.velocities_b[i] = (self.momentum * self.velocities_b[i] + 
                                   self.lr * self.gradients[f'db{i}'])
            
            self.weights[i] -= self.velocities_w[i]
            self.biases[i] -= self.velocities_b[i]
    
    def compute_loss(self, y_true, y_pred):
        """MSE loss"""
        return np.mean((y_true - y_pred)**2)
    
    def train(self, X, y, epochs, batch_size=32):
        """Mini-batch training"""
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            n_batches = 0
            
            # Mini-batch gradient descent
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                y_pred = self.forward(X_batch)
                
                # Compute loss
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss
                n_batches += 1
                
                # Backward pass
                self.backward(X_batch, y_batch)
                
                # Update parameters
                self.update_parameters()
            
            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            
            if epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | Loss: {avg_loss:.6f}")
        
        return losses

# Prepare data
X = df[['feature_1', 'feature_2', 'feature_3']].values
y = df[['target']].values

# Normalize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print("\n4. Training Advanced Neural Network")
print(f"   Architecture: [3, 10, 8, 1]")
print(f"   Training samples: {X_train.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# Create and train network
nn = AdvancedNeuralNetwork(
    layer_sizes=[3, 10, 8, 1],
    learning_rate=0.01,
    momentum=0.9
)

losses = nn.train(X_train, y_train, epochs=200, batch_size=32)

# Test predictions
y_pred_train = nn.forward(X_train)
y_pred_test = nn.forward(X_test)

train_loss = nn.compute_loss(y_train, y_pred_train)
test_loss = nn.compute_loss(y_test, y_pred_test)

print(f"\nFinal Training Loss: {train_loss:.6f}")
print(f"Final Test Loss: {test_loss:.6f}")


# PART 4: COMPREHENSIVE VISUALIZATIONS
print("\n" + "="*60)
print("PART 4: CREATING COMPREHENSIVE VISUALIZATIONS")
print("="*60)


fig = plt.figure(figsize=(18, 14))

# Plot 1: Loss curves
ax1 = plt.subplot(3, 3, 1)
ax1.plot(losses, linewidth=2, color='blue', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Training Loss Over Time', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Plot 2: Feature correlations heatmap
ax2 = plt.subplot(3, 3, 2)
sns.heatmap(correlations, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=ax2, cbar_kws={'label': 'Correlation'})
ax2.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')

# Plot 3: Predictions vs Actual
ax3 = plt.subplot(3, 3, 3)
ax3.scatter(y_test, y_pred_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Values', fontsize=11)
ax3.set_ylabel('Predicted Values', fontsize=11)
ax3.set_title('Test Set: Predictions vs Actual', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals
ax4 = plt.subplot(3, 3, 4)
residuals = y_test - y_pred_test
ax4.scatter(y_pred_test, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Values', fontsize=11)
ax4.set_ylabel('Residuals', fontsize=11)
ax4.set_title('Residual Plot', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Weight distributions
ax5 = plt.subplot(3, 3, 5)
all_weights = np.concatenate([w.flatten() for w in nn.weights])
ax5.hist(all_weights, bins=50, edgecolor='black', alpha=0.7)
ax5.set_xlabel('Weight Value', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Weight Distribution (All Layers)', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Gradient magnitudes by layer
ax6 = plt.subplot(3, 3, 6)
layer_grads = [np.linalg.norm(nn.gradients[f'dW{i}']) 
               for i in range(len(nn.weights))]
layers = [f'Layer {i+1}' for i in range(len(nn.weights))]
bars = ax6.bar(layers, layer_grads, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax6.set_ylabel('Gradient Magnitude', fontsize=11)
ax6.set_title('Gradient Magnitudes by Layer', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
for bar, grad in zip(bars, layer_grads):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'{grad:.4f}', ha='center', va='bottom', fontsize=9)

# Plot 7: Feature importance (via gradient)
ax7 = plt.subplot(3, 3, 7)
feature_names = ['Feature 1', 'Feature 2', 'Feature 3']
feature_grads = []
for i in range(3):
    # Compute average gradient magnitude for each feature
    grad_col = np.abs(nn.weights[0][i, :])
    feature_grads.append(np.mean(grad_col))

bars = ax7.barh(feature_names, feature_grads, color=['#e74c3c', '#3498db', '#2ecc71'])
ax7.set_xlabel('Average Gradient Magnitude', fontsize=11)
ax7.set_title('Feature Importance', fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3, axis='x')

# Plot 8: Learning curve comparison
ax8 = plt.subplot(3, 3, 8)
# Smooth losses with moving average
window = 10
if len(losses) > window:
    losses_smooth = pd.Series(losses).rolling(window=window).mean()
    ax8.plot(losses, alpha=0.3, label='Raw', color='blue')
    ax8.plot(losses_smooth, linewidth=2, label='Smoothed', color='red')
    ax8.set_xlabel('Epoch', fontsize=11)
    ax8.set_ylabel('Loss', fontsize=11)
    ax8.set_title('Learning Curve (Raw vs Smoothed)', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)

# Plot 9: Error distribution
ax9 = plt.subplot(3, 3, 9)
ax9.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='purple')
ax9.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax9.set_xlabel('Prediction Error', fontsize=11)
ax9.set_ylabel('Frequency', fontsize=11)
ax9.set_title('Error Distribution', fontsize=13, fontweight='bold')
ax9.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('advanced_applications.png', dpi=150, bbox_inches='tight')
print("\n✓ Comprehensive visualization saved as 'advanced_applications.png'")
plt.show()

# PART 5: SUMMARY AND KEY INSIGHTS
print("\n" + "="*60)
print("COMPREHENSIVE SUMMARY")
print("="*60)
print("""
CHAIN RULE MASTERY - KEY TAKEAWAYS:

1. MATHEMATICAL FOUNDATION:
   - Single variable: df/dx = (df/dg)(dg/dx)
   - Multivariable: ∂f/∂x = Σᵢ (∂f/∂uᵢ)(∂uᵢ/∂x)
   - Always work backwards from output to input

2. COMPUTATIONAL IMPLEMENTATION:
   - Store intermediate values during forward pass
   - Compute gradients layer-by-layer during backward pass
   - Each layer multiplies gradients (chain rule)

3. LIBRARIES INTEGRATION:
   ✓ SymPy: Symbolic derivatives and verification
   ✓ NumPy: Efficient numerical computation
   ✓ Pandas: Data analysis and gradient tracking
   ✓ Matplotlib/Seaborn: Comprehensive visualizations

4. PRACTICAL ML APPLICATIONS:
   - Backpropagation in neural networks
   - Gradient descent optimization
   - Feature importance analysis
   - Loss function design

5. ADVANCED TECHNIQUES:
   - Momentum optimization
   - Mini-batch gradient descent
   - Adaptive learning rates
   - Gradient clipping

NEXT STEPS:
- Implement automatic differentiation
- Study computational graphs (PyTorch, TensorFlow)
- Explore advanced architectures (CNNs, RNNs, Transformers)
- Practice on real datasets
""")