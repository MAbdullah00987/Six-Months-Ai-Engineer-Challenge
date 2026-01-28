#Day 4: Gradient Boosting & XGBoost (The King)

#Objective: Master the algorithm that wins 80% of Tabular Kaggle competitions.
#Concept: Gradient Boosting Machines (GBM), Learning Rate, Residuals (predicting the error of the previous tree).
#Task: Project - Customer Churn Prediction.
#Use the Telco Customer Churn dataset.
#Train an XGBClassifier.
#Crucial: Learn to use the early_stopping_rounds parameter to prevent overfitting automatically.


"""
DAY 4: GRADIENT BOOSTING & XGBOOST - COMPLETE MASTERY
=====================================================
Fixed version for Windows - saves files to current directory

Topics Covered:
1. Gradient Boosting Machines (GBM) - Core Concepts
2. Learning Rate & Its Impact
3. Residuals (Predicting Errors)
4. XGBoost Implementation
5. Feature Importance Analysis
6. Hyperparameter Tuning
7. Advanced Visualizations with Matplotlib, Seaborn
8. Statistical Analysis with SciPy, StatsModels
9. Mathematical Foundations with SymPy
10. Animations with Manim (preparation)

Libraries: numpy, pandas, matplotlib, seaborn, scipy, statsmodels, sklearn, sympy
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize_scalar
import sympy as sp
from sympy import symbols, diff, simplify, lambdify
import warnings
import os
warnings.filterwarnings('ignore')

# sklearn imports
from sklearn.datasets import make_regression, make_classification, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Create output directory in current working directory
output_dir = os.path.join(os.getcwd(), 'gradient_boosting_outputs')
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("DAY 4: GRADIENT BOOSTING & XGBOOST MASTERY")
print("="*80)
print()

# ============================================================================
# SECTION 1: MATHEMATICAL FOUNDATION - RESIDUALS & GRADIENT DESCENT
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: MATHEMATICAL FOUNDATION - UNDERSTANDING RESIDUALS")
print("="*80)

# Using SymPy to understand the mathematical foundation
print("\n1.1 Loss Function Mathematics (SymPy)")
print("-" * 40)

# Define symbolic variables
y_true, y_pred = symbols('y_true y_pred')

# Mean Squared Error Loss
mse_loss = (y_true - y_pred)**2
print(f"MSE Loss Function: L = {mse_loss}")

# Gradient of loss w.r.t prediction
gradient = diff(mse_loss, y_pred)
print(f"Gradient ∂L/∂ŷ = {gradient}")
print(f"Simplified: {simplify(gradient)}")

# This shows us that residual = -gradient/2 for MSE
print("\nKey Insight: Residual = y_true - y_pred = -gradient/2")
print("Gradient Boosting predicts the NEGATIVE GRADIENT of the loss function")

# ============================================================================
# SECTION 2: VISUAL INTUITION - HOW GRADIENT BOOSTING WORKS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: VISUAL INTUITION - STEP-BY-STEP BOOSTING")
print("="*80)

# Generate simple dataset
np.random.seed(42)
X_simple = np.linspace(0, 10, 100).reshape(-1, 1)
y_simple = 2 * X_simple.ravel() + np.sin(X_simple.ravel() * 2) + np.random.normal(0, 0.5, 100)

print("\n2.1 Gradient Boosting: Building Trees Step by Step")
print("-" * 40)

# Manual implementation to show the process
class SimpleGradientBoosting:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X, y):
        # Start with mean prediction
        self.initial_prediction = np.mean(y)
        predictions = np.full(len(y), self.initial_prediction)
        
        # Store history for visualization
        self.prediction_history = [predictions.copy()]
        self.residual_history = []
        
        for i in range(self.n_estimators):
            # Calculate residuals (negative gradient for MSE)
            residuals = y - predictions
            self.residual_history.append(residuals.copy())
            
            # Fit tree to residuals
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions
            update = tree.predict(X)
            predictions += self.learning_rate * update
            self.prediction_history.append(predictions.copy())
            
        return self
    
    def predict(self, X):
        predictions = np.full(len(X), self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * tree.predict(X)
        return predictions

# Fit the model
gb_simple = SimpleGradientBoosting(n_estimators=5, learning_rate=0.1, max_depth=3)
gb_simple.fit(X_simple, y_simple)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Gradient Boosting: Sequential Tree Building Process', fontsize=16, fontweight='bold')

for i in range(6):
    row = i // 3
    col = i % 3
    ax = axes[row, col]
    
    # Plot data
    ax.scatter(X_simple, y_simple, alpha=0.5, s=20, label='True Data', color='blue')
    
    if i == 0:
        # Initial prediction (mean)
        ax.axhline(y=gb_simple.initial_prediction, color='red', linestyle='--', 
                   linewidth=2, label=f'Initial (Mean): {gb_simple.initial_prediction:.2f}')
        ax.set_title('Step 0: Initial Prediction (Mean)', fontweight='bold')
        mse = mean_squared_error(y_simple, gb_simple.prediction_history[0])
        ax.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    else:
        # Show cumulative prediction
        predictions = gb_simple.prediction_history[i]
        ax.plot(X_simple, predictions, color='red', linewidth=2, 
                label=f'Prediction after {i} tree(s)')
        ax.set_title(f'Step {i}: After Adding Tree {i}', fontweight='bold')
        mse = mean_squared_error(y_simple, predictions)
        ax.text(0.05, 0.95, f'MSE: {mse:.3f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
    
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, '01_gradient_boosting_process.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# SECTION 3: RESIDUAL ANALYSIS - THE HEART OF BOOSTING
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: RESIDUAL ANALYSIS - PREDICTING ERRORS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Residual Analysis: How Boosting Corrects Errors', fontsize=16, fontweight='bold')

# Plot 1: Residuals at each step
ax = axes[0, 0]
for i in range(min(4, len(gb_simple.residual_history))):
    residuals = gb_simple.residual_history[i]
    ax.scatter(X_simple, residuals, alpha=0.6, s=30, label=f'After Tree {i}')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('X', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residuals Decrease with Each Tree', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Residual distribution
ax = axes[0, 1]
for i in [0, len(gb_simple.residual_history)-1]:
    residuals = gb_simple.residual_history[i]
    ax.hist(residuals, bins=20, alpha=0.6, label=f'After Tree {i}')
ax.set_xlabel('Residual Value', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Residual Distribution: Initial vs Final', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: MSE reduction
ax = axes[1, 0]
mse_values = [mean_squared_error(y_simple, pred) for pred in gb_simple.prediction_history]
ax.plot(range(len(mse_values)), mse_values, marker='o', linewidth=2, markersize=8, color='darkred')
ax.set_xlabel('Number of Trees', fontsize=12)
ax.set_ylabel('Mean Squared Error', fontsize=12)
ax.set_title('MSE Reduction with Each Tree', fontweight='bold')
ax.grid(True, alpha=0.3)

# Add percentage improvement
for i, mse in enumerate(mse_values):
    if i > 0:
        improvement = ((mse_values[0] - mse) / mse_values[0]) * 100
        ax.text(i, mse, f'{improvement:.1f}%', ha='center', va='bottom', fontsize=9)

# Plot 4: Q-Q plot for residuals
ax = axes[1, 1]
final_residuals = gb_simple.residual_history[-1]
stats.probplot(final_residuals, dist="norm", plot=ax)
ax.set_title('Q-Q Plot: Final Residuals vs Normal Distribution', fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, '02_residual_analysis.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Statistical tests on residuals
print("\n3.1 Statistical Analysis of Residuals (SciPy)")
print("-" * 40)

final_residuals = gb_simple.residual_history[-1]

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(final_residuals)
print(f"Shapiro-Wilk Test for Normality:")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  P-value: {shapiro_p:.4f}")
print(f"  Interpretation: {'Residuals are normally distributed' if shapiro_p > 0.05 else 'Residuals are NOT normally distributed'}")

# Mean and standard deviation
print(f"\nResidual Statistics:")
print(f"  Mean: {np.mean(final_residuals):.6f} (should be ≈ 0)")
print(f"  Std Dev: {np.std(final_residuals):.4f}")
print(f"  Min: {np.min(final_residuals):.4f}")
print(f"  Max: {np.max(final_residuals):.4f}")

# ============================================================================
# SECTION 4: LEARNING RATE IMPACT - BIAS-VARIANCE TRADEOFF
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: LEARNING RATE - THE MOST IMPORTANT HYPERPARAMETER")
print("="*80)

print("\n4.1 Comparing Different Learning Rates")
print("-" * 40)

learning_rates = [0.01, 0.1, 0.3, 1.0]
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Impact of Learning Rate on Gradient Boosting', fontsize=16, fontweight='bold')

for idx, lr in enumerate(learning_rates):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    # Train model with specific learning rate
    gb_lr = SimpleGradientBoosting(n_estimators=20, learning_rate=lr, max_depth=3)
    gb_lr.fit(X_simple, y_simple)
    
    # Calculate MSE for each step
    mse_progression = [mean_squared_error(y_simple, pred) for pred in gb_lr.prediction_history]
    
    # Plot
    ax.scatter(X_simple, y_simple, alpha=0.3, s=20, label='Data', color='gray')
    final_pred = gb_lr.predict(X_simple)
    ax.plot(X_simple, final_pred, color=colors[idx], linewidth=2, label=f'LR={lr}')
    ax.set_title(f'Learning Rate = {lr}', fontweight='bold', fontsize=14)
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    
    # Add final MSE
    final_mse = mse_progression[-1]
    ax.text(0.05, 0.95, f'Final MSE: {final_mse:.3f}\nTrees: 20', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = os.path.join(output_dir, '03_learning_rate_impact.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# MSE progression comparison
fig, ax = plt.subplots(figsize=(12, 7))
for lr, color in zip(learning_rates, colors):
    gb_lr = SimpleGradientBoosting(n_estimators=50, learning_rate=lr, max_depth=3)
    gb_lr.fit(X_simple, y_simple)
    mse_progression = [mean_squared_error(y_simple, pred) for pred in gb_lr.prediction_history]
    ax.plot(range(len(mse_progression)), mse_progression, marker='o', 
            linewidth=2, label=f'LR = {lr}', color=color, markersize=4)

ax.set_xlabel('Number of Trees', fontsize=12)
ax.set_ylabel('Mean Squared Error', fontsize=12)
ax.set_title('MSE Convergence for Different Learning Rates', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 50)

# Add annotations
ax.text(0.6, 0.7, 'Lower LR:\n- Slower convergence\n- Better generalization\n- Needs more trees', 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
        fontsize=10)
ax.text(0.6, 0.3, 'Higher LR:\n- Faster convergence\n- Risk of overfitting\n- Fewer trees needed', 
        transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
        fontsize=10)

plt.tight_layout()
output_path = os.path.join(output_dir, '04_learning_rate_convergence.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# ============================================================================
# SECTION 5: REAL DATASET - DIABETES PREDICTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: REAL WORLD APPLICATION - DIABETES DATASET")
print("="*80)

# Load diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Create DataFrame for analysis
feature_names = diabetes.feature_names
df_diabetes = pd.DataFrame(X, columns=feature_names)
df_diabetes['target'] = y

print("\n5.1 Dataset Overview")
print("-" * 40)
print(f"Shape: {df_diabetes.shape}")
print(f"\nFeatures: {list(feature_names)}")
print(f"\nFirst few rows:")
print(df_diabetes.head())

print(f"\nStatistical Summary:")
print(df_diabetes.describe())

# Correlation analysis
print("\n5.2 Correlation Analysis (Pandas & Seaborn)")
print("-" * 40)

# Calculate correlations
correlations = df_diabetes.corr()['target'].drop('target').sort_values(ascending=False)
print("Feature correlations with target:")
print(correlations)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Correlation heatmap
ax = axes[0]
sns.heatmap(df_diabetes.corr(), annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=ax, square=True, linewidths=1)
ax.set_title('Feature Correlation Heatmap', fontweight='bold', fontsize=14)

# Feature importance from correlations
ax = axes[1]
colors_corr = ['green' if x > 0 else 'red' for x in correlations.values]
ax.barh(range(len(correlations)), correlations.values, color=colors_corr, alpha=0.7)
ax.set_yticks(range(len(correlations)))
ax.set_yticklabels(correlations.index)
ax.set_xlabel('Correlation with Target', fontsize=12)
ax.set_title('Feature Correlations with Disease Progression', fontweight='bold', fontsize=14)
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output_path = os.path.join(output_dir, '05_diabetes_correlation.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

# Split data for the remaining sections
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================================
# SECTION 6: GRADIENT BOOSTING vs SINGLE TREE
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: GRADIENT BOOSTING vs SINGLE DECISION TREE")
print("="*80)

print("\n6.1 Training Models")
print("-" * 40)

# Single Decision Tree
single_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
single_tree.fit(X_train, y_train)
tree_pred = single_tree.predict(X_test)
tree_mse = mean_squared_error(y_test, tree_pred)
tree_r2 = r2_score(y_test, tree_pred)

print(f"Single Decision Tree:")
print(f"  MSE: {tree_mse:.2f}")
print(f"  R²: {tree_r2:.4f}")

# Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    subsample=0.8
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)
gb_r2 = r2_score(y_test, gb_pred)

print(f"\nGradient Boosting:")
print(f"  MSE: {gb_mse:.2f}")
print(f"  R²: {gb_r2:.4f}")

improvement = ((tree_mse - gb_mse) / tree_mse) * 100
print(f"\n✓ Improvement: {improvement:.2f}%")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Predictions comparison
ax = axes[0]
ax.scatter(y_test, tree_pred, alpha=0.6, s=50, label='Single Tree', color='blue')
ax.scatter(y_test, gb_pred, alpha=0.6, s=50, label='Gradient Boosting', color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'k--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('True Values', fontsize=12)
ax.set_ylabel('Predictions', fontsize=12)
ax.set_title('Predictions: Single Tree vs Gradient Boosting', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals comparison
ax = axes[1]
tree_residuals = y_test - tree_pred
gb_residuals = y_test - gb_pred
ax.scatter(tree_pred, tree_residuals, alpha=0.6, s=50, label='Single Tree', color='blue')
ax.scatter(gb_pred, gb_residuals, alpha=0.6, s=50, label='Gradient Boosting', color='red')
ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Values', fontsize=12)
ax.set_ylabel('Residuals', fontsize=12)
ax.set_title('Residual Analysis', fontweight='bold', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# Performance metrics
ax = axes[2]
metrics = ['MSE', 'R²']
tree_metrics = [tree_mse, tree_r2]
gb_metrics = [gb_mse, gb_r2]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax.bar(x - width/2, tree_metrics, width, label='Single Tree', alpha=0.8, color='blue')
bars2 = ax.bar(x + width/2, gb_metrics, width, label='Gradient Boosting', alpha=0.8, color='red')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
output_path = os.path.join(output_dir, '06_comparison_tree_vs_gb.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_path}")
plt.close()

print("\n" + "="*80)
print("SCRIPT COMPLETED SUCCESSFULLY!")
print(f"All visualizations saved to: {output_dir}")
print("="*80)