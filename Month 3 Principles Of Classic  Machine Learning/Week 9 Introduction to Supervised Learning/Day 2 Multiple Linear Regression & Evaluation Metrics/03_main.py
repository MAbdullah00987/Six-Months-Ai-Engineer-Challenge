
#Part 3. Visualizing Linear Regression with Matplotlib & Seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100

# Single feature for 2D visualization
X1 = np.random.randn(n_samples, 1) * 2
y1 = 4 + 3 * X1 + np.random.randn(n_samples, 1) * 1.5

# Two features for 3D visualization
X2 = np.random.randn(n_samples, 2)
y2 = 4 + 3 * X2[:, 0] + 2 * X2[:, 1] + np.random.randn(n_samples) * 0.8

# Train simple models
X1_b = np.c_[np.ones((n_samples, 1)), X1]
theta1 = np.linalg.inv(X1_b.T @ X1_b) @ X1_b.T @ y1

X2_b = np.c_[np.ones((n_samples, 1)), X2]
theta2 = np.linalg.inv(X2_b.T @ X2_b) @ X2_b.T @ y2.reshape(-1, 1)

# Predictions
y1_pred = X1_b @ theta1
y2_pred = X2_b @ theta2

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# 1. SIMPLE LINEAR REGRESSION (2D)
# ============================================================================
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(X1, y1, alpha=0.6, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
ax1.plot(X1, y1_pred, 'r-', linewidth=2, label=f'y = {theta1[0,0]:.2f} + {theta1[1,0]:.2f}x')
ax1.set_xlabel('X (Feature)', fontsize=11, fontweight='bold')
ax1.set_ylabel('y (Target)', fontsize=11, fontweight='bold')
ax1.set_title('Simple Linear Regression Fit', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# ============================================================================
# 2. RESIDUALS PLOT
# ============================================================================
ax2 = plt.subplot(3, 3, 2)
residuals1 = y1 - y1_pred
ax2.scatter(y1_pred, residuals1, alpha=0.6, s=50, color='coral', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax2.set_ylabel('Residuals (ε = y - ŷ)', fontsize=11, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# ============================================================================
# 3. RESIDUALS HISTOGRAM
# ============================================================================
ax3 = plt.subplot(3, 3, 3)
ax3.hist(residuals1, bins=20, alpha=0.7, color='mediumpurple', edgecolor='black')
ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Residuals', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# ============================================================================
# 4. PREDICTED vs ACTUAL
# ============================================================================
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y1, y1_pred, alpha=0.6, s=50, color='seagreen', edgecolors='black', linewidth=0.5)
min_val, max_val = y1.min(), y1.max()
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Values', fontsize=11, fontweight='bold')
ax4.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
ax4.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)

# ============================================================================
# 5. Q-Q PLOT (Normality of Residuals)
# ============================================================================
ax5 = plt.subplot(3, 3, 5)
from scipy import stats
stats.probplot(residuals1.flatten(), dist="norm", plot=ax5)
ax5.set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold')
ax5.grid(alpha=0.3)

# ============================================================================
# 6. 3D REGRESSION PLANE (Multiple Features)
# ============================================================================
ax6 = plt.subplot(3, 3, 6, projection='3d')
ax6.scatter(X2[:, 0], X2[:, 1], y2, c='steelblue', marker='o', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)

# Create mesh for regression plane
x1_range = np.linspace(X2[:, 0].min(), X2[:, 0].max(), 20)
x2_range = np.linspace(X2[:, 1].min(), X2[:, 1].max(), 20)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)
y_mesh = theta2[0, 0] + theta2[1, 0] * x1_mesh + theta2[2, 0] * x2_mesh

ax6.plot_surface(x1_mesh, x2_mesh, y_mesh, alpha=0.3, cmap='coolwarm')
ax6.set_xlabel('X₁', fontsize=10, fontweight='bold')
ax6.set_ylabel('X₂', fontsize=10, fontweight='bold')
ax6.set_zlabel('y', fontsize=10, fontweight='bold')
ax6.set_title('3D Regression Plane', fontsize=12, fontweight='bold')

# ============================================================================
# 7. GRADIENT DESCENT CONVERGENCE
# ============================================================================
ax7 = plt.subplot(3, 3, 7)

# Simulate gradient descent
def gradient_descent_track(X, y, learning_rate=0.1, n_iter=100):
    m, n = X.shape
    theta = np.zeros((n, 1))
    cost_history = []
    
    for i in range(n_iter):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * X.T @ errors
        theta = theta - learning_rate * gradient
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)
    
    return cost_history

costs = gradient_descent_track(X1_b, y1, learning_rate=0.1, n_iter=100)
ax7.plot(costs, linewidth=2, color='darkred')
ax7.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax7.set_ylabel('Cost J(θ)', fontsize=11, fontweight='bold')
ax7.set_title('Gradient Descent Convergence', fontsize=12, fontweight='bold')
ax7.grid(alpha=0.3)

# ============================================================================
# 8. LEARNING RATE COMPARISON
# ============================================================================
ax8 = plt.subplot(3, 3, 8)
learning_rates = [0.01, 0.05, 0.1, 0.5]
colors = ['blue', 'green', 'orange', 'red']

for lr, color in zip(learning_rates, colors):
    costs = gradient_descent_track(X1_b, y1, learning_rate=lr, n_iter=50)
    ax8.plot(costs, linewidth=2, label=f'α = {lr}', color=color)

ax8.set_xlabel('Iteration', fontsize=11, fontweight='bold')
ax8.set_ylabel('Cost J(θ)', fontsize=11, fontweight='bold')
ax8.set_title('Impact of Learning Rate', fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(alpha=0.3)

# ============================================================================
# 9. FEATURE CORRELATION HEATMAP
# ============================================================================
ax9 = plt.subplot(3, 3, 9)
df = pd.DataFrame(np.c_[X2, y2], columns=['X₁', 'X₂', 'y'])
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax9)
ax9.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('linear_regression_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()


print("VISUALIZATION COMPLETE")
print("\nKey Insights from Visualizations:")
print("1. Scatter + Regression Line: Shows the linear relationship")
print("2. Residuals Plot: Should show random scatter (no patterns)")
print("3. Residuals Histogram: Should be approximately normal")
print("4. Predicted vs Actual: Points should lie on diagonal line")
print("5. Q-Q Plot: Tests normality assumption of residuals")
print("6. 3D Plane: Visualizes regression in higher dimensions")
print("7. Convergence: Shows gradient descent optimization")
print("8. Learning Rate: Demonstrates impact on convergence speed")
print("9. Correlation: Shows relationships between variables")
