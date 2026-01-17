
#Task: Project - Feature Scaling Impact.
#Create a dataset with a curve. Fit a Linear model (Underfit).
#Fit a Polynomial model degree 2 (Good fit).
#Fit a Polynomial model degree 20 (Overfit).
#Crucial: Apply StandardScaler before training to see why scaling matters for convergence.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# ============================================================================
# STEP 1: Create Dataset with a Curve
# ============================================================================
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 3 * X.squeeze()**2 - 20 * X.squeeze() + 50 + np.random.randn(100) * 10

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("="*80)
print("FEATURE SCALING IMPACT ON POLYNOMIAL REGRESSION")
print("="*80)
print(f"\nDataset Shape: {X.shape}")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")

# ============================================================================
# STEP 2: Train Models WITHOUT Scaling
# ============================================================================
print("\n" + "="*80)
print("TRAINING WITHOUT FEATURE SCALING")
print("="*80)

# Model 1: Linear (Underfit)
lr_no_scale = LinearRegression()
lr_no_scale.fit(X_train, y_train)
y_pred_lr_no_scale = lr_no_scale.predict(X_test)

print("\n1. LINEAR MODEL (UNDERFIT) - No Scaling:")
print(f"   R² Score: {r2_score(y_test, y_pred_lr_no_scale):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr_no_scale)):.4f}")

# Model 2: Polynomial degree 2 (Good fit)
poly2_features_no_scale = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2_no_scale = poly2_features_no_scale.fit_transform(X_train)
X_test_poly2_no_scale = poly2_features_no_scale.transform(X_test)

poly2_no_scale = LinearRegression()
poly2_no_scale.fit(X_train_poly2_no_scale, y_train)
y_pred_poly2_no_scale = poly2_no_scale.predict(X_test_poly2_no_scale)

print("\n2. POLYNOMIAL DEGREE 2 (GOOD FIT) - No Scaling:")
print(f"   R² Score: {r2_score(y_test, y_pred_poly2_no_scale):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly2_no_scale)):.4f}")
print(f"   Feature range after transformation: [{X_train_poly2_no_scale.min():.2f}, {X_train_poly2_no_scale.max():.2f}]")

# Model 3: Polynomial degree 20 (Overfit)
poly20_features_no_scale = PolynomialFeatures(degree=20, include_bias=False)
X_train_poly20_no_scale = poly20_features_no_scale.fit_transform(X_train)
X_test_poly20_no_scale = poly20_features_no_scale.transform(X_test)

poly20_no_scale = LinearRegression()
poly20_no_scale.fit(X_train_poly20_no_scale, y_train)
y_pred_poly20_no_scale = poly20_no_scale.predict(X_test_poly20_no_scale)

print("\n3. POLYNOMIAL DEGREE 20 (OVERFIT) - No Scaling:")
print(f"   R² Score: {r2_score(y_test, y_pred_poly20_no_scale):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly20_no_scale)):.4f}")
print(f"   Feature range after transformation: [{X_train_poly20_no_scale.min():.2e}, {X_train_poly20_no_scale.max():.2e}]")
print(f"   ⚠️  Notice the EXTREME range! This causes numerical instability.")

# ============================================================================
# STEP 3: Train Models WITH Scaling
# ============================================================================
print("\n" + "="*80)
print("TRAINING WITH FEATURE SCALING (StandardScaler)")
print("="*80)

# Scale original features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nAfter StandardScaler:")
print(f"X_train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
print(f"X_train range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# Model 1: Linear (Underfit) - WITH SCALING
lr_scaled = LinearRegression()
lr_scaled.fit(X_train_scaled, y_train)
y_pred_lr_scaled = lr_scaled.predict(X_test_scaled)

print("\n1. LINEAR MODEL (UNDERFIT) - With Scaling:")
print(f"   R² Score: {r2_score(y_test, y_pred_lr_scaled):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr_scaled)):.4f}")

# Model 2: Polynomial degree 2 (Good fit) - WITH SCALING
poly2_features_scaled = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly2_scaled = poly2_features_scaled.fit_transform(X_train_scaled)
X_test_poly2_scaled = poly2_features_scaled.transform(X_test_scaled)

poly2_scaled = LinearRegression()
poly2_scaled.fit(X_train_poly2_scaled, y_train)
y_pred_poly2_scaled = poly2_scaled.predict(X_test_poly2_scaled)

print("\n2. POLYNOMIAL DEGREE 2 (GOOD FIT) - With Scaling:")
print(f"   R² Score: {r2_score(y_test, y_pred_poly2_scaled):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly2_scaled)):.4f}")
print(f"   Feature range after transformation: [{X_train_poly2_scaled.min():.2f}, {X_train_poly2_scaled.max():.2f}]")

# Model 3: Polynomial degree 20 (Overfit) - WITH SCALING
poly20_features_scaled = PolynomialFeatures(degree=20, include_bias=False)
X_train_poly20_scaled = poly20_features_scaled.fit_transform(X_train_scaled)
X_test_poly20_scaled = poly20_features_scaled.transform(X_test_scaled)

poly20_scaled = LinearRegression()
poly20_scaled.fit(X_train_poly20_scaled, y_train)
y_pred_poly20_scaled = poly20_scaled.predict(X_test_poly20_scaled)

print("\n3. POLYNOMIAL DEGREE 20 (OVERFIT) - With Scaling:")
print(f"   R² Score: {r2_score(y_test, y_pred_poly20_scaled):.4f}")
print(f"   RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_poly20_scaled)):.4f}")
print(f"   Feature range after transformation: [{X_train_poly20_scaled.min():.2f}, {X_train_poly20_scaled.max():.2f}]")
print(f"   ✓ Much better range! Scaling prevents numerical overflow.")

# ============================================================================
# STEP 4: Visualization
# ============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Feature Scaling Impact on Polynomial Regression', fontsize=16, fontweight='bold')

# Sort for smooth plotting
X_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
X_plot_scaled = scaler.transform(X_plot)

# Row 1: WITHOUT SCALING
# Linear
axes[0, 0].scatter(X_train, y_train, alpha=0.5, label='Train data')
axes[0, 0].scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
axes[0, 0].plot(X_plot, lr_no_scale.predict(X_plot), 'r-', linewidth=2, label='Linear fit')
axes[0, 0].set_title('Linear Model (UNDERFIT)\nNo Scaling', fontweight='bold')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('y')
axes[0, 0].legend()
axes[0, 0].text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred_lr_no_scale):.4f}', 
                transform=axes[0, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Polynomial degree 2
X_plot_poly2 = poly2_features_no_scale.transform(X_plot)
axes[0, 1].scatter(X_train, y_train, alpha=0.5, label='Train data')
axes[0, 1].scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
axes[0, 1].plot(X_plot, poly2_no_scale.predict(X_plot_poly2), 'g-', linewidth=2, label='Poly-2 fit')
axes[0, 1].set_title('Polynomial Degree 2 (GOOD FIT)\nNo Scaling', fontweight='bold')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('y')
axes[0, 1].legend()
axes[0, 1].text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred_poly2_no_scale):.4f}', 
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Polynomial degree 20
X_plot_poly20 = poly20_features_no_scale.transform(X_plot)
axes[0, 2].scatter(X_train, y_train, alpha=0.5, label='Train data')
axes[0, 2].scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
axes[0, 2].plot(X_plot, poly20_no_scale.predict(X_plot_poly20), 'm-', linewidth=2, label='Poly-20 fit')
axes[0, 2].set_title('Polynomial Degree 20 (OVERFIT)\nNo Scaling', fontweight='bold')
axes[0, 2].set_xlabel('X')
axes[0, 2].set_ylabel('y')
axes[0, 2].legend()
axes[0, 2].text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred_poly20_no_scale):.4f}', 
                transform=axes[0, 2].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Row 2: WITH SCALING
# Linear
X_plot_poly2_scaled = poly2_features_scaled.transform(X_plot_scaled)
X_plot_poly20_scaled = poly20_features_scaled.transform(X_plot_scaled)

axes[1, 0].scatter(X_train, y_train, alpha=0.5, label='Train data')
axes[1, 0].scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
axes[1, 0].plot(X_plot, lr_scaled.predict(X_plot_scaled), 'r-', linewidth=2, label='Linear fit')
axes[1, 0].set_title('Linear Model (UNDERFIT)\nWith Scaling', fontweight='bold')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].legend()
axes[1, 0].text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred_lr_scaled):.4f}', 
                transform=axes[1, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Polynomial degree 2
axes[1, 1].scatter(X_train, y_train, alpha=0.5, label='Train data')
axes[1, 1].scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
axes[1, 1].plot(X_plot, poly2_scaled.predict(X_plot_poly2_scaled), 'g-', linewidth=2, label='Poly-2 fit')
axes[1, 1].set_title('Polynomial Degree 2 (GOOD FIT)\nWith Scaling', fontweight='bold')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('y')
axes[1, 1].legend()
axes[1, 1].text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred_poly2_scaled):.4f}', 
                transform=axes[1, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

# Polynomial degree 20
axes[1, 2].scatter(X_train, y_train, alpha=0.5, label='Train data')
axes[1, 2].scatter(X_test, y_test, alpha=0.5, color='orange', label='Test data')
axes[1, 2].plot(X_plot, poly20_scaled.predict(X_plot_poly20_scaled), 'm-', linewidth=2, label='Poly-20 fit')
axes[1, 2].set_title('Polynomial Degree 20 (OVERFIT)\nWith Scaling', fontweight='bold')
axes[1, 2].set_xlabel('X')
axes[1, 2].set_ylabel('y')
axes[1, 2].legend()
axes[1, 2].text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred_poly20_scaled):.4f}', 
                transform=axes[1, 2].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 5: Summary Table
# ============================================================================
print("\n" + "="*80)
print("SUMMARY COMPARISON TABLE")
print("="*80)

results = pd.DataFrame({
    'Model': ['Linear', 'Poly-2', 'Poly-20'] * 2,
    'Scaling': ['No'] * 3 + ['Yes'] * 3,
    'R² Score': [
        r2_score(y_test, y_pred_lr_no_scale),
        r2_score(y_test, y_pred_poly2_no_scale),
        r2_score(y_test, y_pred_poly20_no_scale),
        r2_score(y_test, y_pred_lr_scaled),
        r2_score(y_test, y_pred_poly2_scaled),
        r2_score(y_test, y_pred_poly20_scaled)
    ],
    'RMSE': [
        np.sqrt(mean_squared_error(y_test, y_pred_lr_no_scale)),
        np.sqrt(mean_squared_error(y_test, y_pred_poly2_no_scale)),
        np.sqrt(mean_squared_error(y_test, y_pred_poly20_no_scale)),
        np.sqrt(mean_squared_error(y_test, y_pred_lr_scaled)),
        np.sqrt(mean_squared_error(y_test, y_pred_poly2_scaled)),
        np.sqrt(mean_squared_error(y_test, y_pred_poly20_scaled))
    ]
})

print("\n", results.to_string(index=False))

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("""
1. UNDERFITTING (Linear Model):
   - Too simple to capture the quadratic relationship
   - Low R² score regardless of scaling
   - Scaling doesn't help because the model is fundamentally too simple

2. GOOD FIT (Polynomial Degree 2):
   - Matches the true data generation process
   - High R² score with both scaled and unscaled
   - Scaling helps with numerical stability

3. OVERFITTING (Polynomial Degree 20):
   - Too complex, fits training noise
   - WITHOUT scaling: Numerical instability due to huge feature values (10^20)
   - WITH scaling: Stable convergence, but still overfits the training data
   
4. WHY SCALING MATTERS:
   - Prevents numerical overflow in high-degree polynomials
   - Helps gradient-based algorithms converge faster
   - Ensures all features contribute equally to the model
   - Critical for regularization (L1/L2) to work properly
   
5. BEST PRACTICE:
   - Always scale features before polynomial transformation
   - Use cross-validation to select the right polynomial degree
   - Consider regularization (Ridge/Lasso) for high-degree polynomials
""")