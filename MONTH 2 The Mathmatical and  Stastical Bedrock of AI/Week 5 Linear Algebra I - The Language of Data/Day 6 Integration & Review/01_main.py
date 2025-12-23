#Day 6: Integration & Review

#Read: Mathematics for ML Chapter 2-3 summary sections
#Review all Coursera Week 1-4 key takeaways

#Practice (3-4 hours):

#Project 1 (Complete): Data Representation
#Load full dataset (100+ samples, 4-5 features)
#Represent as feature matrix X and target vector y
#Compute X^T X (Gram matrix)
#Calculate column-wise means using matrix operations
#Standardize features using broadcasting
#Compute correlation matrix
#Visualize data matrix as heatmap


#Consolidation:
#Revisit all 10 projects - ensure they run without errors
#Write a 1-page summary connecting projects to ML concepts:

#How matrix multiplication relates to neural network forward pass
#How linear systems appear in regression
#How transformations relate to data preprocessing

#Create a GitHub repository with all projects documented
#Exercise Set:
#Complete any remaining Chapter 2-3 exercises
#Take Coursera Week 1-4 quizzes/assignments

#Topics Of the day:
#1. Load Full Dataset
#2. Standardize Features Using Broadcasting
#3. Compute Correlation Matrix
#4. Visualize Data Matrix as Heatmap
#5. Matrix Multiplication in Neural Networks
#6. Linear Systems in Regression
#7. Transformations in Data Preprocessing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*70)
print("SECTION 1: LOAD FULL DATASET")
print("="*70)

# Generate synthetic dataset for demonstration
np.random.seed(42)
n_samples = 1000
n_features = 5

# Create feature matrix X with some patterns
X = np.random.randn(n_samples, n_features)
X[:, 1] = X[:, 0] * 0.8 + np.random.randn(n_samples) * 0.3  # Correlated feature

# Create target with real relationship
y = 3*X[:, 0] + 2*X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

# Create DataFrame
feature_names = [f'feature_{i}' for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

print(f"\nDataset Shape: {df.shape}")
print(f"Memory Usage: {df.memory_usage().sum() / 1024:.2f} KB")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe()}")

# KEY LEARNING: Understanding data shape is crucial for matrix operations
print(f"\n[INSIGHT] X is a {X.shape} matrix: {n_samples} samples × {n_features} features")
print(f"[INSIGHT] Each row is a data point, each column is a feature")

print("\n" + "="*70)
print("SECTION 2: STANDARDIZE FEATURES USING BROADCASTING")
print("="*70)

def standardize_with_broadcasting(X):
    """
    Standardize: z = (x - μ) / σ
    This demonstrates NumPy broadcasting!
    """
    # Calculate mean and std along axis 0 (columns)
    mean = np.mean(X, axis=0)  # Shape: (n_features,)
    std = np.std(X, axis=0)    # Shape: (n_features,)
    
    # Broadcasting magic happens here!
    # X shape: (1000, 5)
    # mean shape: (5,) → broadcasts to (1, 5) → broadcasts to (1000, 5)
    X_standardized = (X - mean) / std
    
    return X_standardized, mean, std

# Apply standardization
X_std, mean, std = standardize_with_broadcasting(X)

print("\n[BROADCASTING EXPLAINED]")
print(f"X shape: {X.shape}")
print(f"mean shape: {mean.shape}")
print(f"When we compute X - mean:")
print(f"  NumPy automatically expands mean from (5,) to (1,5) to (1000,5)")
print(f"  This is called BROADCASTING")

print(f"\nOriginal mean per feature: {np.round(mean, 2)}")
print(f"Original std per feature: {np.round(std, 2)}")
print(f"\nStandardized mean: {np.round(X_std.mean(axis=0), 10)}")  # ~0
print(f"Standardized std: {np.round(X_std.std(axis=0), 2)}")      # ~1

# Visualize the effect
fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)

ax1 = fig.add_subplot(gs[0, 0])
ax1.boxplot(X)
ax1.set_title('Original Features', fontsize=14, fontweight='bold')
ax1.set_xlabel('Feature Index')
ax1.set_ylabel('Value')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.boxplot(X_std)
ax2.set_title('Standardized Features', fontsize=14, fontweight='bold')
ax2.set_xlabel('Feature Index')
ax2.set_ylabel('Standardized Value')
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(X[:, 0], bins=30, alpha=0.5, label='Original')
ax3.hist(X_std[:, 0], bins=30, alpha=0.5, label='Standardized')
ax3.set_title('Feature 0 Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Value')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_standardization.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 01_standardization.png")

# Broadcasting example breakdown
print("\n[BROADCASTING RULES]")
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([10, 20, 30])             # (3,)
result = A - b

print(f"A (2×3):\n{A}")
print(f"\nb (3,):\n{b}")
print(f"\nA - b:\n{result}")
print("\nBroadcasting rule: b (3,) → (1,3) → (2,3)")

print("\n" + "="*70)
print("SECTION 3: COMPUTE CORRELATION MATRIX")
print("="*70)

def compute_correlation_from_scratch(X):
    """
    Correlation matrix using matrix multiplication
    Formula: R = (1/(n-1)) * X_std^T @ X_std
    """
    # Standardize first
    X_centered = X - X.mean(axis=0)
    X_std = X_centered / X.std(axis=0)
    
    n = X_std.shape[0]
    
    # Key insight: X^T @ X gives sum of element-wise products
    # For standardized data, this is correlation!
    corr_matrix = (X_std.T @ X_std) / (n - 1)
    
    return corr_matrix

# Compute correlation
corr_manual = compute_correlation_from_scratch(X)
corr_numpy = np.corrcoef(X.T)

print("\n[CORRELATION MATRIX COMPUTATION]")
print(f"X_std shape: {X_std.shape}")
print(f"X_std.T shape: {X_std.T.shape}")
print(f"Correlation matrix shape: {corr_manual.shape}")
print(f"\nCorrelation Matrix:\n{np.round(corr_manual, 3)}")
print(f"\nDifference from np.corrcoef: {np.max(np.abs(corr_manual - corr_numpy)):.10f}")

# Visualize correlations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Correlation heatmap
sns.heatmap(corr_manual, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            xticklabels=feature_names, yticklabels=feature_names, ax=axes[0])
axes[0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

# Scatter plot of most correlated features
axes[1].scatter(X[:, 0], X[:, 1], alpha=0.5, s=20)
axes[1].set_xlabel(f'{feature_names[0]}', fontsize=12)
axes[1].set_ylabel(f'{feature_names[1]}', fontsize=12)
axes[1].set_title(f'Most Correlated Features (r={corr_manual[0,1]:.3f})', 
                  fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_correlation.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 02_correlation.png")

# Understanding the math
print("\n[MATRIX MULTIPLICATION FOR CORRELATION]")
X_small = np.array([[1, 2], [2, 4], [3, 5]])
X_small_std = (X_small - X_small.mean(axis=0)) / X_small.std(axis=0)

print(f"Small example (3 samples, 2 features):")
print(f"Standardized data:\n{X_small_std}")
print(f"\nX^T @ X:\n{X_small_std.T @ X_small_std}")
print(f"\nCorrelation:\n{(X_small_std.T @ X_small_std) / 2}")

print("\n" + "="*70)
print("SECTION 4: VISUALIZE DATA MATRIX AS HEATMAP")
print("="*70)

# Create dataset with clear patterns
np.random.seed(123)
n_samp, n_feat = 50, 20
X_pattern = np.random.randn(n_samp, n_feat)

# Add structure: first group likes first features, second group likes last features
X_pattern[:25, :10] += 2.5  # Group 1
X_pattern[25:, 10:] += 2.5  # Group 2

# Standardize
X_pattern_std = (X_pattern - X_pattern.mean(axis=0)) / X_pattern.std(axis=0)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 1. Raw data heatmap
ax1 = fig.add_subplot(gs[0, 0])
im1 = ax1.imshow(X_pattern, aspect='auto', cmap='viridis', interpolation='nearest')
ax1.set_title('Raw Data Matrix', fontsize=14, fontweight='bold')
ax1.set_xlabel('Features')
ax1.set_ylabel('Samples')
ax1.axhline(y=24.5, color='red', linestyle='--', linewidth=2, label='Group boundary')
ax1.axvline(x=9.5, color='red', linestyle='--', linewidth=2)
plt.colorbar(im1, ax=ax1, label='Value')

# 2. Standardized data
ax2 = fig.add_subplot(gs[0, 1])
im2 = ax2.imshow(X_pattern_std, aspect='auto', cmap='RdBu_r', 
                 vmin=-3, vmax=3, interpolation='nearest')
ax2.set_title('Standardized Data Matrix', fontsize=14, fontweight='bold')
ax2.set_xlabel('Features')
ax2.set_ylabel('Samples')
plt.colorbar(im2, ax=ax2, label='Standardized Value')

# 3. Feature correlation
corr_feat = np.corrcoef(X_pattern.T)
ax3 = fig.add_subplot(gs[1, 0])
im3 = ax3.imshow(corr_feat, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
ax3.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
ax3.set_xlabel('Feature')
ax3.set_ylabel('Feature')
plt.colorbar(im3, ax=ax3, label='Correlation')

# 4. Sample similarity
sample_sim = np.corrcoef(X_pattern)
ax4 = fig.add_subplot(gs[1, 1])
im4 = ax4.imshow(sample_sim, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
ax4.set_title('Sample Similarity Matrix', fontsize=14, fontweight='bold')
ax4.set_xlabel('Sample')
ax4.set_ylabel('Sample')
ax4.axhline(y=24.5, color='white', linestyle='--', linewidth=2)
ax4.axvline(x=24.5, color='white', linestyle='--', linewidth=2)
plt.colorbar(im4, ax=ax4, label='Similarity')

plt.savefig('03_heatmaps.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 03_heatmaps.png")

print("\n[HEATMAP INSIGHTS]")
print("✓ Two distinct groups visible in the data")
print("✓ Group 1 (samples 0-24): High values in features 0-9")
print("✓ Group 2 (samples 25-49): High values in features 10-19")
print("✓ Sample similarity shows block structure (groups cluster together)")

print("\n" + "="*70)
print("SECTION 5: MATRIX MULTIPLICATION → NEURAL NETWORK FORWARD PASS")
print("="*70)

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize with small random weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
        print(f"\n[NETWORK ARCHITECTURE]")
        print(f"Input layer: {input_size} neurons")
        print(f"Hidden layer: {hidden_size} neurons")
        print(f"Output layer: {output_size} neurons")
        print(f"\nW1 shape: {self.W1.shape} - connects input to hidden")
        print(f"W2 shape: {self.W2.shape} - connects hidden to output")
    
    def relu(self, Z):
        """ReLU: max(0, z)"""
        return np.maximum(0, Z)
    
    def forward(self, X):
        """
        Forward propagation through the network
        """
        # Layer 1: input → hidden
        # Matrix multiplication: (batch_size, input) @ (input, hidden)
        self.Z1 = X @ self.W1 + self.b1  # Broadcasting adds bias
        self.A1 = self.relu(self.Z1)
        
        # Layer 2: hidden → output
        # Matrix multiplication: (batch_size, hidden) @ (hidden, output)
        self.Z2 = self.A1 @ self.W2 + self.b2
        
        return self.Z2
    
    def visualize_forward_pass(self, X):
        """Visualize what happens during forward pass"""
        output = self.forward(X)
        
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Input
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(X[:20].T, aspect='auto', cmap='viridis')
        ax1.set_title(f'Input X\nShape: {X.shape}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Features')
        ax1.set_xlabel('Samples (first 20)')
        plt.colorbar(im1, ax=ax1)
        
        # Weights 1
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(self.W1, aspect='auto', cmap='RdBu_r')
        ax2.set_title(f'Weights W1\nShape: {self.W1.shape}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hidden Units')
        ax2.set_ylabel('Input Features')
        plt.colorbar(im2, ax=ax2)
        
        # Hidden activation
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(self.A1[:20].T, aspect='auto', cmap='viridis')
        ax3.set_title(f'Hidden Layer A1\nShape: {self.A1.shape}', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Hidden Units')
        ax3.set_xlabel('Samples (first 20)')
        plt.colorbar(im3, ax=ax3)
        
        # Weights 2
        ax4 = fig.add_subplot(gs[1, 0])
        im4 = ax4.imshow(self.W2, aspect='auto', cmap='RdBu_r')
        ax4.set_title(f'Weights W2\nShape: {self.W2.shape}', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Output Units')
        ax4.set_ylabel('Hidden Units')
        plt.colorbar(im4, ax=ax4)
        
        # Output
        ax5 = fig.add_subplot(gs[1, 1])
        im5 = ax5.imshow(output[:20].T, aspect='auto', cmap='viridis')
        ax5.set_title(f'Output\nShape: {output.shape}', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Output Units')
        ax5.set_xlabel('Samples (first 20)')
        plt.colorbar(im5, ax=ax5)
        
        # Flow diagram
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        ax6.text(0.5, 0.9, 'FORWARD PASS FLOW', ha='center', 
                fontsize=14, weight='bold')
        ax6.text(0.5, 0.75, 'X @ W1 + b1 → Z1', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax6.text(0.5, 0.6, '↓', ha='center', fontsize=16)
        ax6.text(0.5, 0.50, 'ReLU(Z1) → A1', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax6.text(0.5, 0.35, '↓', ha='center', fontsize=16)
        ax6.text(0.5, 0.25, 'A1 @ W2 + b2 → Output', ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        ax6.text(0.5, 0.08, f'Total Parameters: {self.W1.size + self.W2.size:,}',
                ha='center', fontsize=10)
        
        plt.savefig('04_neural_network.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved: 04_neural_network.png")

# Create and test network
nn = SimpleNeuralNetwork(input_size=5, hidden_size=10, output_size=3)

# Forward pass with batch
batch = X_std[:32]
output = nn.forward(batch)

print(f"\n[FORWARD PASS SHAPES]")
print(f"Input batch: {batch.shape}")
print(f"After layer 1 (Z1): {nn.Z1.shape}")
print(f"After ReLU (A1): {nn.A1.shape}")
print(f"Final output: {output.shape}")

nn.visualize_forward_pass(batch)

# Demonstrate matrix multiplication step by step
print("\n[MATRIX MULTIPLICATION BREAKDOWN]")
X_demo = batch[:3]  # 3 samples
print(f"3 samples × 5 features:")
print(X_demo)
print(f"\nW1 (5 × 10):")
print(np.round(nn.W1[:, :3], 2), "...")
Z1_demo = X_demo @ nn.W1
print(f"\nResult Z1 (3 × 10):")
print(np.round(Z1_demo[:, :3], 2), "...")

print("\n" + "="*70)
print("SECTION 6: LINEAR SYSTEMS IN REGRESSION")
print("="*70)

class LinearRegression:
    def __init__(self):
        self.weights = None
        self.bias = None
    
    def fit_normal_equation(self, X, y):
        """
        Normal Equation: θ = (X^T X)^(-1) X^T y
        Directly solves the linear system!
        """
        # Add intercept term
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # Solve: X^T X θ = X^T y
        XtX = X_with_intercept.T @ X_with_intercept
        Xty = X_with_intercept.T @ y
        
        # θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(XtX) @ Xty
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def fit_gradient_descent(self, X, y, lr=0.01, epochs=1000):
        """
        Iterative optimization using gradient descent
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        losses = []
        
        for epoch in range(epochs):
            # Prediction: ŷ = Xw + b
            y_pred = X @ self.weights + self.bias
            
            # Loss: MSE = (1/n) Σ(y - ŷ)²
            loss = np.mean((y_pred - y)**2)
            losses.append(loss)
            
            # Gradients (using matrix operations!)
            error = y_pred - y
            dw = (2/n_samples) * (X.T @ error)  # ∂L/∂w
            db = (2/n_samples) * np.sum(error)   # ∂L/∂b
            
            # Update
            self.weights -= lr * dw
            self.bias -= lr * db
        
        return self, losses
    
    def predict(self, X):
        return X @ self.weights + self.bias

# Fit both models
print("\n[COMPARING REGRESSION METHODS]")

lr_normal = LinearRegression()
lr_normal.fit_normal_equation(X_std, y)
y_pred_normal = lr_normal.predict(X_std)

lr_gd = LinearRegression()
lr_gd, losses = lr_gd.fit_gradient_descent(X_std, y, lr=0.1, epochs=1000)
y_pred_gd = lr_gd.predict(X_std)

print("\nNormal Equation weights:")
print(np.round(lr_normal.weights, 4))
print(f"Bias: {lr_normal.bias:.4f}")

print("\nGradient Descent weights:")
print(np.round(lr_gd.weights, 4))
print(f"Bias: {lr_gd.bias:.4f}")

print(f"\nMax weight difference: {np.max(np.abs(lr_normal.weights - lr_gd.weights)):.6f}")

# Visualize
fig = plt.figure(figsize=(15, 5))
gs = gridspec.GridSpec(1, 3, figure=fig)

# Loss curve
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(losses, linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('MSE Loss', fontsize=12)
ax1.set_title('Gradient Descent Convergence', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Predictions vs Actual
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y, y_pred_normal, alpha=0.5, s=20, label='Normal Eq', color='blue')
ax2.scatter(y, y_pred_gd, alpha=0.3, s=20, label='Grad Descent', color='red')
ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', label='Perfect', linewidth=2)
ax2.set_xlabel('Actual', fontsize=12)
ax2.set_ylabel('Predicted', fontsize=12)
ax2.set_title('Predictions vs Actual', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Residuals
residuals = y - y_pred_normal
ax3 = fig.add_subplot(gs[0, 2])
ax3.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax3.set_xlabel('Residual', fontsize=12)
ax3.set_ylabel('Count', fontsize=12)
ax3.set_title('Residual Distribution', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('05_regression.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 05_regression.png")

print("\n[LINEAR SYSTEM EXPLANATION]")
print("We're solving: X @ w = y")
print("More precisely: X^T @ X @ w = X^T @ y")
print(f"\nX^T @ X shape: {(X_std.T @ X_std).shape}")
print("This is a square matrix that we can invert!")
print(f"Condition number: {np.linalg.cond(X_std.T @ X_std):.2f}")
print("(Lower is better; high values indicate numerical issues)")

print("\n" + "="*70)
print("SECTION 7: TRANSFORMATIONS IN DATA PREPROCESSING")
print("="*70)

class DataTransformations:
    
    @staticmethod
    def pca_transform(X, n_components=2):
        """
        Principal Component Analysis
        Finds directions of maximum variance
        """
        # Center data
        X_centered = X - X.mean(axis=0)
        
        # Covariance matrix
        cov_matrix = (X_centered.T @ X_centered) / (len(X) - 1)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort descending
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        components = eigenvectors[:, :n_components]
        
        # Transform: This is matrix multiplication!
        X_transformed = X_centered @ components
        
        return X_transformed, components, eigenvalues
    
    @staticmethod
    def rotation_matrix_2d(angle_degrees):
        """2D rotation matrix"""
        theta = np.radians(angle_degrees)
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

# PCA demonstration
print("\n[PCA TRANSFORMATION]")
X_pca, components, eigenvalues = DataTransformations.pca_transform(X_std, n_components=2)

print(f"Original shape: {X_std.shape}")
print(f"PCA shape: {X_pca.shape}")
print(f"\nVariance explained by each component:")
var_explained = eigenvalues / eigenvalues.sum() * 100
for i, var in enumerate(var_explained[:2]):
    print(f"  PC{i+1}: {var:.2f}%")
print(f"  Total (2 PCs): {var_explained[:2].sum():.2f}%")

# Rotation demonstration
print("\n[ROTATION TRANSFORMATION]")
angle = 45
R = DataTransformations.rotation_matrix_2d(angle)
X_2d = X_std[:100, :2]
X_rotated = X_2d @ R.T

print(f"Rotation matrix (45°):\n{np.round(R, 3)}")
print(f"\nOriginal data shape: {X_2d.shape}")
print(f"Rotated data shape: {X_rotated.shape}")

# Check distance preservation
dist_original = np.linalg.norm(X_2d, axis=1)
dist_rotated = np.linalg.norm(X_rotated, axis=1)
print(f"Distances preserved? {np.allclose(dist_original, dist_rotated)}")

# Visualize transformations
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Original 2D data
ax1 = fig.add_subplot(gs[0, 0])
scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=y[:100], 
                       cmap='viridis', alpha=0.6, s=30)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_title('Original 2D Data', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
plt.colorbar(scatter1, ax=ax1, label='Target')

# PCA transformed
ax2 = fig.add_subplot(gs[0, 1])
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                       cmap='viridis', alpha=0.6, s=30)
ax2.set_xlabel('PC 1')
ax2.set_ylabel('PC 2')
ax2.set_title('PCA Transformed', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Target')

# Rotated data
ax3 = fig.add_subplot(gs[0, 2])
scatter3 = ax3.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y[:100],
                       cmap='viridis', alpha=0.6, s=30)
ax3.set_xlabel('Rotated Feature 1')
ax3.set_ylabel('Rotated Feature 2')
ax3.set_title(f'45° Rotation', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axis('equal')
plt.colorbar(scatter3, ax=ax3, label='Target')

# Variance explained
ax4 = fig.add_subplot(gs[1, 0])
ax4.bar(range(len(eigenvalues)), eigenvalues, color='steelblue', edgecolor='black')
ax4.set_xlabel('Component', fontsize=12)
ax4.set_ylabel('Eigenvalue', fontsize=12)
ax4.set_title('Variance by Component', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Cumulative variance
ax5 = fig.add_subplot(gs[1, 1])
cumsum_var = np.cumsum(var_explained)
ax5.plot(range(1, len(cumsum_var)+1), cumsum_var, 'o-', linewidth=2, markersize=8)
ax5.axhline(y=90, color='r', linestyle='--', label='90% threshold')
ax5.set_xlabel('Number of Components', fontsize=12)
ax5.set_ylabel('Cumulative Variance (%)', fontsize=12)
ax5.set_title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Transformation comparison
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
ax6.text(0.5, 0.9, 'TRANSFORMATION SUMMARY', ha='center', 
        fontsize=14, weight='bold')
ax6.text(0.1, 0.7, 'PCA:', fontsize=12, weight='bold')
ax6.text(0.1, 0.6, f'  • Reduces dimensions', fontsize=10)
ax6.text(0.1, 0.5, f'  • Captures {var_explained[:2].sum():.1f}% variance', fontsize=10)
ax6.text(0.1, 0.4, f'  • Removes correlation', fontsize=10)

ax6.text(0.1, 0.25, 'Rotation:', fontsize=12, weight='bold')
ax6.text(0.1, 0.15, f'  • Preserves distances', fontsize=10)
ax6.text(0.1, 0.05, f'  • Changes orientation only', fontsize=10)

plt.savefig('06_transformations.png', dpi=150, bbox_inches='tight')
print("\nSaved: 06_transformations.png")



"""
  BROADCASTING: NumPy automatically expands arrays for element-wise operations
  Example: (1000, 5) - (5,) → automatically broadcasts to (1000, 5)

 CORRELATION: X^T @ X computes all pairwise dot products
  For standardized data, this gives correlation matrix

 MATRIX MULTIPLICATION: Core of neural networks
  Each layer: output = input @ weights + bias

 LINEAR SYSTEMS: Regression solves X @ w = y
  Normal equation: w = (X^T X)^(-1) X^T y
  Gradient descent: iteratively approximates solution

 TRANSFORMATIONS: All are matrix operations
  • Standardization: element-wise operations with broadcasting
  • PCA: eigendecomposition + matrix multiplication
  • Rotation: matrix multiplication with rotation matrix

PRACTICE TIPS:
1. Always check shapes before operations
2. Understand what each axis represents
3. Visualize intermediate results
4. Start with small examples to verify logic
5. Use broadcasting to avoid loops


"""

