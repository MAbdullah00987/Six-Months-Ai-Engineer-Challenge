import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)


# LINEAR REGRESSION WITH GRADIENT DESCENT


print("="*80)
print("LINEAR REGRESSION: GRADIENT DESCENT FROM SCRATCH")
print("="*80)

# Generate synthetic data
np.random.seed(42)
X_train = np.random.rand(100, 1) * 10
y_train = 3 * X_train.flatten() + 7 + np.random.randn(100) * 2

print(f"Dataset: {len(X_train)} samples")
print(f"True relationship: y = 3x + 7 (with noise)")

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.w = None  # weight
        self.b = None  # bias
        self.cost_history = []
        
    def compute_cost(self, X, y, w, b):
        """Mean Squared Error: J = (1/2m) Σ(ŷᵢ - yᵢ)²"""
        m = len(y)
        predictions = X @ w + b
        cost = (1/(2*m)) * np.sum((predictions - y)**2)
        return cost
    
    def compute_gradients(self, X, y, w, b):
        """
        Gradients:
        ∂J/∂w = (1/m) Σ(ŷᵢ - yᵢ) * xᵢ
        ∂J/∂b = (1/m) Σ(ŷᵢ - yᵢ)
        """
        m = len(y)
        predictions = X @ w + b
        
        dw = (1/m) * X.T @ (predictions - y)
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        """Train using gradient descent"""
        m, n = X.shape
        self.w = np.zeros((n, 1))
        self.b = 0
        
        self.weight_history = []
        self.bias_history = []
        
        for i in range(self.iterations):
            # Store history BEFORE update
            self.weight_history.append(self.w.copy())
            self.bias_history.append(self.b)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y.reshape(-1, 1), self.w, self.b)
            
            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            # Compute and store cost
            cost = self.compute_cost(X, y.reshape(-1, 1), self.w, self.b)
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}, w = {self.w[0,0]:.4f}, b = {self.b:.4f}")
        
        return self
    
    def predict(self, X):
        return X @ self.w + self.b

# Train model
model = LinearRegressionGD(learning_rate=0.01, iterations=500)
model.fit(X_train, y_train)

print(f"\nFinal parameters:")
print(f"  Learned weight: {model.w[0,0]:.4f} (true: 3.0)")
print(f"  Learned bias: {model.b:.4f} (true: 7.0)")

# ============================================================================
# VISUALIZATION 1: Training Progress
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Linear Regression: Gradient Descent Training', fontsize=16, fontweight='bold')

# Plot 1: Data and fitted line
axes[0, 0].scatter(X_train, y_train, alpha=0.6, s=50, label='Training data')
X_line = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X_line)
axes[0, 0].plot(X_line, y_pred, 'r-', linewidth=3, label=f'Fitted: y = {model.w[0,0]:.2f}x + {model.b:.2f}')
axes[0, 0].plot(X_line, 3*X_line + 7, 'g--', linewidth=2, label='True: y = 3x + 7', alpha=0.7)
axes[0, 0].set_xlabel('x', fontsize=12)
axes[0, 0].set_ylabel('y', fontsize=12)
axes[0, 0].set_title('Data and Fitted Line', fontsize=13)
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Cost function over iterations
axes[0, 1].plot(model.cost_history, 'b-', linewidth=2)
axes[0, 1].set_xlabel('Iteration', fontsize=12)
axes[0, 1].set_ylabel('Cost (MSE)', fontsize=12)
axes[0, 1].set_title('Cost Function Convergence', fontsize=13)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Parameter evolution
iterations = range(len(model.weight_history))
weights = [w[0,0] for w in model.weight_history]
biases = model.bias_history

ax3 = axes[1, 0]
ax3.plot(iterations, weights, 'b-', linewidth=2, label='Weight (w)')
ax3.axhline(y=3, color='b', linestyle='--', alpha=0.5, label='True weight = 3')
ax3.set_xlabel('Iteration', fontsize=12)
ax3.set_ylabel('Weight', fontsize=12, color='b')
ax3.tick_params(axis='y', labelcolor='b')

ax3_twin = ax3.twinx()
ax3_twin.plot(iterations, biases, 'r-', linewidth=2, label='Bias (b)')
ax3_twin.axhline(y=7, color='r', linestyle='--', alpha=0.5, label='True bias = 7')
ax3_twin.set_ylabel('Bias', fontsize=12, color='r')
ax3_twin.tick_params(axis='y', labelcolor='r')

ax3.set_title('Parameter Evolution During Training', fontsize=13)
ax3.legend(loc='upper left', fontsize=10)
ax3_twin.legend(loc='upper right', fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Cost surface (3D visualization)
from mpl_toolkits.mplot3d import Axes3D
ax4 = fig.add_subplot(2, 2, 4, projection='3d')

# Create mesh for cost surface
w_range = np.linspace(0, 6, 50)
b_range = np.linspace(0, 14, 50)
W, B = np.meshgrid(w_range, b_range)

# Compute cost for each (w, b) pair
Z = np.zeros_like(W)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        Z[j, i] = model.compute_cost(X_train, y_train.reshape(-1, 1), 
                                     np.array([[w_range[i]]]), b_range[j])

# Plot surface
surf = ax4.plot_surface(W, B, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Plot gradient descent path - FIXED: ensure all arrays have same length
sample_indices = range(0, len(model.cost_history), 10)
weights_array = np.array([model.weight_history[i][0,0] for i in sample_indices])
biases_array = np.array([model.bias_history[i] for i in sample_indices])
costs_array = np.array([model.cost_history[i] for i in sample_indices])

ax4.plot(weights_array, biases_array, costs_array, 'r-', linewidth=3, alpha=0.9)
ax4.plot([weights_array[0]], [biases_array[0]], [costs_array[0]], 
        'go', markersize=10, label='Start')
ax4.plot([weights_array[-1]], [biases_array[-1]], [costs_array[-1]], 
        'ro', markersize=10, label='End')

ax4.set_xlabel('Weight (w)', fontsize=11)
ax4.set_ylabel('Bias (b)', fontsize=11)
ax4.set_zlabel('Cost', fontsize=11)
ax4.set_title('Cost Surface and Gradient Descent Path', fontsize=13)
ax4.legend(fontsize=10)

plt.tight_layout()
plt.savefig('linear_regression_gd.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# LOGISTIC REGRESSION WITH GRADIENT DESCENT
# ============================================================================

print("\n" + "="*80)
print("LOGISTIC REGRESSION: BINARY CLASSIFICATION")
print("="*80)

# Generate binary classification data
from sklearn.datasets import make_classification
X_clf, y_clf = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                   n_informative=2, n_clusters_per_class=1,
                                   random_state=42)

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.w = None
        self.b = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid function: σ(z) = 1 / (1 + e^(-z))"""
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y, w, b):
        """
        Binary cross-entropy loss:
        J = -(1/m) Σ[yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
        """
        m = len(y)
        z = X @ w + b
        predictions = self.sigmoid(z)
        
        # Avoid log(0)
        predictions = np.clip(predictions, 1e-10, 1 - 1e-10)
        
        cost = -(1/m) * np.sum(y * np.log(predictions) + (1-y) * np.log(1-predictions))
        return cost
    
    def compute_gradients(self, X, y, w, b):
        """
        Gradients:
        ∂J/∂w = (1/m) X^T (σ(Xw+b) - y)
        ∂J/∂b = (1/m) Σ(σ(Xw+b) - y)
        """
        m = len(y)
        z = X @ w + b
        predictions = self.sigmoid(z)
        
        dw = (1/m) * X.T @ (predictions - y)
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros((n, 1))
        self.b = 0
        
        for i in range(self.iterations):
            dw, db = self.compute_gradients(X, y.reshape(-1, 1), self.w, self.b)
            
            self.w -= self.lr * dw
            self.b -= self.lr * db
            
            cost = self.compute_cost(X, y.reshape(-1, 1), self.w, self.b)
            self.cost_history.append(cost)
            
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        return self
    
    def predict_proba(self, X):
        z = X @ self.w + self.b
        return self.sigmoid(z)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

# Train logistic regression
log_model = LogisticRegressionGD(learning_rate=0.1, iterations=1000)
log_model.fit(X_clf, y_clf)

# ============================================================================
# VISUALIZATION 2: Logistic Regression
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Logistic Regression: Binary Classification', fontsize=16, fontweight='bold')

# Plot 1: Decision boundary
h = 0.02  # mesh step size
x_min, x_max = X_clf[:, 0].min() - 1, X_clf[:, 0].max() + 1
y_min, y_max = X_clf[:, 1].min() - 1, X_clf[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = log_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter = axes[0].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, 
                         cmap='RdYlBu', edgecolors='k', s=80, alpha=0.8)
axes[0].set_xlabel('Feature 1', fontsize=12)
axes[0].set_ylabel('Feature 2', fontsize=12)
axes[0].set_title('Decision Boundary', fontsize=13)
plt.colorbar(scatter, ax=axes[0], label='Class')

# Plot 2: Cost over iterations
axes[1].plot(log_model.cost_history, 'b-', linewidth=2)
axes[1].set_xlabel('Iteration', fontsize=12)
axes[1].set_ylabel('Cost (Binary Cross-Entropy)', fontsize=12)
axes[1].set_title('Training Loss Convergence', fontsize=13)
axes[1].grid(True, alpha=0.3)

# Plot 3: Probability contours
Z_proba = log_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_proba = Z_proba.reshape(xx.shape)

contour = axes[2].contourf(xx, yy, Z_proba, levels=20, cmap='RdYlBu', alpha=0.7)
axes[2].scatter(X_clf[:, 0], X_clf[:, 1], c=y_clf, 
               cmap='RdYlBu', edgecolors='k', s=80, alpha=0.8)
axes[2].contour(xx, yy, Z_proba, levels=[0.5], colors='black', linewidths=3)
axes[2].set_xlabel('Feature 1', fontsize=12)
axes[2].set_ylabel('Feature 2', fontsize=12)
axes[2].set_title('Probability Contours (Decision boundary at 0.5)', fontsize=13)
plt.colorbar(contour, ax=axes[2], label='P(y=1)')

plt.tight_layout()
plt.savefig('logistic_regression_gd.png', dpi=300, bbox_inches='tight')
plt.show()

# Calculate accuracy
predictions = log_model.predict(X_clf)
accuracy = np.mean(predictions.flatten() == y_clf) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")

# ============================================================================
# NEURAL NETWORK GRADIENT VISUALIZATION
# ============================================================================

print("\n" + "="*80)
print("NEURAL NETWORK: BACKPROPAGATION (Chain Rule of Derivatives)")
print("="*80)

class SimpleNeuralNetwork:
    """Simple 2-layer neural network for visualization"""
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate
        self.loss_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def forward(self, X):
        """Forward propagation"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y):
        """Backpropagation: compute gradients using chain rule"""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients (chain rule!)
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.z1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = np.mean((output - y)**2)
            self.loss_history.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y)
            
            # Update weights
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

# Train neural network
X_nn = X_clf
y_nn = y_clf.reshape(-1, 1)

nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
nn.train(X_nn, y_nn, epochs=1000)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Neural Network Training with Backpropagation', fontsize=16, fontweight='bold')

# Plot loss
axes[0].plot(nn.loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Training Loss (Backpropagation via Chain Rule)', fontsize=13)
axes[0].grid(True, alpha=0.3)

# Plot decision boundary
h = 0.02
x_min, x_max = X_nn[:, 0].min() - 1, X_nn[:, 0].max() + 1
y_min, y_max = X_nn[:, 1].min() - 1, X_nn[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z = (Z >= 0.5).astype(int).reshape(xx.shape)

axes[1].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
axes[1].scatter(X_nn[:, 0], X_nn[:, 1], c=y_nn.flatten(), 
               cmap='RdYlBu', edgecolors='k', s=80, alpha=0.8)
axes[1].set_xlabel('Feature 1', fontsize=12)
axes[1].set_ylabel('Feature 2', fontsize=12)
axes[1].set_title('Neural Network Decision Boundary', fontsize=13)

plt.tight_layout()
plt.savefig('neural_network_gd.png', dpi=300, bbox_inches='tight')
plt.show()