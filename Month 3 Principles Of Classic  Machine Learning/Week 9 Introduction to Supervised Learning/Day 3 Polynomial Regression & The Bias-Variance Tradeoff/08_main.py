
#Project 3: Fish Weight Prediction**
# Apply polynomial features
# Compare different degrees of polynomial
# Use Ridge regression to prevent overfitting

#Fish Weight Prediction using Polynomial Features and Ridge Regression
#=====================================================================
#This project demonstrates:
#1. Polynomial feature transformation
#2. Comparison of different polynomial degrees
#3. Ridge regression to prevent overfitting
#4. Model evaluation and visualization


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. GENERATE OR LOAD FISH DATASET
# ============================================================================

def create_fish_dataset(n_samples=200):
    """
    Create a synthetic fish dataset with realistic relationships
    Features: Length1, Length2, Length3, Height, Width
    Target: Weight
    """
    np.random.seed(42)
    
    # Generate fish measurements with correlations
    length1 = np.random.uniform(7, 60, n_samples)  # Length from head to tail
    length2 = length1 * 0.95 + np.random.normal(0, 2, n_samples)  # Slightly shorter
    length3 = length1 * 0.90 + np.random.normal(0, 2, n_samples)  # Even shorter
    height = length1 * 0.35 + np.random.normal(0, 1.5, n_samples)  # Height proportional
    width = length1 * 0.25 + np.random.normal(0, 1, n_samples)  # Width proportional
    
    # Weight is non-linear function of dimensions (volume relationship)
    weight = (0.01 * length1**2 * height * width + 
              np.random.normal(0, 50, n_samples))
    weight = np.maximum(weight, 0)  # No negative weights
    
    # Create species categorical variable
    species = np.random.choice(['Bream', 'Roach', 'Whitefish', 'Perch', 'Pike'], n_samples)
    
    df = pd.DataFrame({
        'Species': species,
        'Length1': length1,
        'Length2': length2,
        'Length3': length3,
        'Height': height,
        'Width': width,
        'Weight': weight
    })
    
    return df

# Create dataset
print("=" * 80)
print("FISH WEIGHT PREDICTION PROJECT")
print("=" * 80)
print("\n1. Creating Fish Dataset...")
df = create_fish_dataset(200)

print(f"\nDataset Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())

print("\nDataset Statistics:")
print(df.describe())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("2. Exploratory Data Analysis")
print("=" * 80)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Correlation analysis
print("\nCorrelation with Weight:")
correlations = df.drop('Species', axis=1).corr()['Weight'].sort_values(ascending=False)
print(correlations)

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Fish Weight Prediction - Exploratory Data Analysis', fontsize=16, y=1.02)

# Feature distributions
features = ['Length1', 'Length2', 'Length3', 'Height', 'Width']
for idx, feature in enumerate(features):
    row = idx // 3
    col = idx % 3
    axes[row, col].scatter(df[feature], df['Weight'], alpha=0.5)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Weight')
    axes[row, col].set_title(f'Weight vs {feature}')
    
    # Add correlation coefficient
    corr = df[[feature, 'Weight']].corr().iloc[0, 1]
    axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                        transform=axes[row, col].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Weight distribution
axes[1, 2].hist(df['Weight'], bins=30, edgecolor='black', alpha=0.7)
axes[1, 2].set_xlabel('Weight')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Weight Distribution')

plt.tight_layout()
plt.savefig('fish_eda.png', dpi=300, bbox_inches='tight')
print("\n✓ EDA plot saved as 'fish_eda.png'")

# ============================================================================
# 3. DATA PREPARATION
# ============================================================================

print("\n" + "=" * 80)
print("3. Data Preparation")
print("=" * 80)

# Select features (dropping species for simplicity)
X = df[features].values
y = df['Weight'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# ============================================================================
# 4. POLYNOMIAL FEATURE TRANSFORMATION & MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("4. Comparing Polynomial Degrees")
print("=" * 80)

def evaluate_polynomial_model(degree, alpha=1.0, use_ridge=True):
    """
    Create and evaluate a polynomial regression model
    """
    # Create pipeline
    if use_ridge:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('ridge', Ridge(alpha=alpha))
        ])
        model_name = f"Ridge (α={alpha})"
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('linear', LinearRegression())
        ])
        model_name = "Linear Regression"
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                 scoring='r2')
    
    return {
        'degree': degree,
        'model_name': model_name,
        'model': model,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred
    }

# Compare different polynomial degrees
degrees = [1, 2, 3, 4, 5]
results = []

print("\nEvaluating Linear Regression with different polynomial degrees:")
print("-" * 80)
for degree in degrees:
    result = evaluate_polynomial_model(degree, use_ridge=False)
    results.append(result)
    print(f"Degree {degree}: Train R² = {result['train_r2']:.4f}, "
          f"Test R² = {result['test_r2']:.4f}, "
          f"Test RMSE = {result['test_rmse']:.2f}")

# ============================================================================
# 5. RIDGE REGRESSION TO PREVENT OVERFITTING
# ============================================================================

print("\n" + "=" * 80)
print("5. Ridge Regression for Overfitting Prevention")
print("=" * 80)

# Test different alpha values with degree 3 polynomial
alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
ridge_results = []

print("\nEvaluating Ridge Regression (Degree 3) with different alpha values:")
print("-" * 80)
for alpha in alphas:
    result = evaluate_polynomial_model(degree=3, alpha=alpha, use_ridge=True)
    ridge_results.append(result)
    print(f"Alpha {alpha:7.3f}: Train R² = {result['train_r2']:.4f}, "
          f"Test R² = {result['test_r2']:.4f}, "
          f"Test RMSE = {result['test_rmse']:.2f}")

# ============================================================================
# 6. VISUALIZATION OF RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("6. Visualizing Results")
print("=" * 80)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. R² Score vs Polynomial Degree
ax1 = fig.add_subplot(gs[0, 0])
degrees_list = [r['degree'] for r in results]
train_r2_list = [r['train_r2'] for r in results]
test_r2_list = [r['test_r2'] for r in results]
ax1.plot(degrees_list, train_r2_list, 'o-', label='Train R²', linewidth=2)
ax1.plot(degrees_list, test_r2_list, 's-', label='Test R²', linewidth=2)
ax1.set_xlabel('Polynomial Degree')
ax1.set_ylabel('R² Score')
ax1.set_title('Model Performance vs Polynomial Degree')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. RMSE vs Polynomial Degree
ax2 = fig.add_subplot(gs[0, 1])
train_rmse_list = [r['train_rmse'] for r in results]
test_rmse_list = [r['test_rmse'] for r in results]
ax2.plot(degrees_list, train_rmse_list, 'o-', label='Train RMSE', linewidth=2)
ax2.plot(degrees_list, test_rmse_list, 's-', label='Test RMSE', linewidth=2)
ax2.set_xlabel('Polynomial Degree')
ax2.set_ylabel('RMSE')
ax2.set_title('RMSE vs Polynomial Degree')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Cross-validation scores
ax3 = fig.add_subplot(gs[0, 2])
cv_means = [r['cv_mean'] for r in results]
cv_stds = [r['cv_std'] for r in results]
ax3.errorbar(degrees_list, cv_means, yerr=cv_stds, fmt='o-', linewidth=2, capsize=5)
ax3.set_xlabel('Polynomial Degree')
ax3.set_ylabel('Cross-Validation R² Score')
ax3.set_title('Cross-Validation Performance')
ax3.grid(True, alpha=0.3)

# 4. Ridge Regression: R² vs Alpha
ax4 = fig.add_subplot(gs[1, 0])
alpha_list = [r['degree'] for r in ridge_results]  # Using alpha values
train_r2_ridge = [r['train_r2'] for r in ridge_results]
test_r2_ridge = [r['test_r2'] for r in ridge_results]
ax4.semilogx(alphas, train_r2_ridge, 'o-', label='Train R²', linewidth=2)
ax4.semilogx(alphas, test_r2_ridge, 's-', label='Test R²', linewidth=2)
ax4.set_xlabel('Alpha (Regularization Strength)')
ax4.set_ylabel('R² Score')
ax4.set_title('Ridge Regression: R² vs Alpha (Degree 3)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Ridge Regression: RMSE vs Alpha
ax5 = fig.add_subplot(gs[1, 1])
train_rmse_ridge = [r['train_rmse'] for r in ridge_results]
test_rmse_ridge = [r['test_rmse'] for r in ridge_results]
ax5.semilogx(alphas, train_rmse_ridge, 'o-', label='Train RMSE', linewidth=2)
ax5.semilogx(alphas, test_rmse_ridge, 's-', label='Test RMSE', linewidth=2)
ax5.set_xlabel('Alpha (Regularization Strength)')
ax5.set_ylabel('RMSE')
ax5.set_title('Ridge Regression: RMSE vs Alpha (Degree 3)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Best model predictions vs actual (test set)
ax6 = fig.add_subplot(gs[1, 2])
best_idx = np.argmax([r['test_r2'] for r in ridge_results])
best_result = ridge_results[best_idx]
ax6.scatter(y_test, best_result['y_test_pred'], alpha=0.6)
ax6.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax6.set_xlabel('Actual Weight')
ax6.set_ylabel('Predicted Weight')
ax6.set_title(f'Best Model: α={alphas[best_idx]} (Test Set)')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Residuals plot for best model
ax7 = fig.add_subplot(gs[2, 0])
residuals = y_test - best_result['y_test_pred']
ax7.scatter(best_result['y_test_pred'], residuals, alpha=0.6)
ax7.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax7.set_xlabel('Predicted Weight')
ax7.set_ylabel('Residuals')
ax7.set_title('Residual Plot (Best Model)')
ax7.grid(True, alpha=0.3)

# 8. Residuals distribution
ax8 = fig.add_subplot(gs[2, 1])
ax8.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
ax8.set_xlabel('Residuals')
ax8.set_ylabel('Frequency')
ax8.set_title('Residuals Distribution')
ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax8.grid(True, alpha=0.3)

# 9. Feature importance (for degree 2 polynomial)
ax9 = fig.add_subplot(gs[2, 2])
degree_2_model = results[1]['model']  # Degree 2
poly_features = degree_2_model.named_steps['poly']
feature_names = poly_features.get_feature_names_out(features)
if hasattr(degree_2_model.named_steps['linear'], 'coef_'):
    coefs = degree_2_model.named_steps['linear'].coef_
else:
    coefs = degree_2_model.named_steps['ridge'].coef_

# Get top 10 features by absolute coefficient
top_indices = np.argsort(np.abs(coefs))[-10:]
ax9.barh(range(len(top_indices)), coefs[top_indices])
ax9.set_yticks(range(len(top_indices)))
ax9.set_yticklabels([feature_names[i] for i in top_indices], fontsize=8)
ax9.set_xlabel('Coefficient Value')
ax9.set_title('Top 10 Feature Coefficients (Degree 2)')
ax9.grid(True, alpha=0.3, axis='x')

plt.suptitle('Fish Weight Prediction: Comprehensive Analysis', 
             fontsize=18, y=0.995)
plt.savefig('fish_polynomial_ridge_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Analysis plot saved as 'fish_polynomial_ridge_analysis.png'")

# ============================================================================
# 7. FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("7. FINAL SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print("\n📊 Performance Summary:")
print("-" * 80)

# Best linear model
best_linear_idx = np.argmax([r['test_r2'] for r in results])
best_linear = results[best_linear_idx]
print(f"\nBest Linear Model (Degree {best_linear['degree']}):")
print(f"  Train R²: {best_linear['train_r2']:.4f}")
print(f"  Test R²:  {best_linear['test_r2']:.4f}")
print(f"  Test RMSE: {best_linear['test_rmse']:.2f}")
print(f"  CV R²:    {best_linear['cv_mean']:.4f} ± {best_linear['cv_std']:.4f}")

# Best ridge model
best_ridge_idx = np.argmax([r['test_r2'] for r in ridge_results])
best_ridge = ridge_results[best_ridge_idx]
print(f"\nBest Ridge Model (α={alphas[best_ridge_idx]}):")
print(f"  Train R²: {best_ridge['train_r2']:.4f}")
print(f"  Test R²:  {best_ridge['test_r2']:.4f}")
print(f"  Test RMSE: {best_ridge['test_rmse']:.2f}")
print(f"  CV R²:    {best_ridge['cv_mean']:.4f} ± {best_ridge['cv_std']:.4f}")

print("\n🎯 Key Insights:")
print("-" * 80)
print("1. Higher polynomial degrees capture more complex relationships")
print("2. But they can lead to overfitting (high train R², lower test R²)")
print("3. Ridge regression helps prevent overfitting through regularization")
print("4. The alpha parameter controls the strength of regularization")
print("5. Cross-validation helps select the optimal hyperparameters")

print("\n💡 Recommendations:")
print("-" * 80)
print(f"✓ Use polynomial degree 3 with Ridge regression")
print(f"✓ Set alpha = {alphas[best_ridge_idx]} for best generalization")
print(f"✓ Expected test R² ≈ {best_ridge['test_r2']:.4f}")
print(f"✓ Expected RMSE ≈ {best_ridge['test_rmse']:.2f}")

print("\n" + "=" * 80)
print("PROJECT COMPLETE!")
print("=" * 80)
print("\nFiles generated:")
print("  • fish_eda.png")
print("  • fish_polynomial_ridge_analysis.png")
print("\nYou've successfully:")
print("  ✓ Applied polynomial feature transformation")
print("  ✓ Compared different polynomial degrees")
print("  ✓ Used Ridge regression to prevent overfitting")
print("  ✓ Evaluated models using multiple metrics")
print("  ✓ Visualized the results comprehensively")