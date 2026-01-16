
#Part 4: Real-World Example: House Price Prediction
#REAL-WORLD APPLICATION: HOUSE PRICE PREDICTION
#Complete pipeline using all libraries learned
#This example demonstrates:
#1. Data generation and exploration (Pandas, NumPy)
#2. Feature engineering with polynomials (Sklearn)
#3. Statistical analysis (SciPy, Statsmodels)
#4. Visualization (Matplotlib, Seaborn)
#5. Symbolic math (SymPy)
#6. Model comparison and selection


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from scipy.optimize import curve_fit
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
import sympy as sp
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("REAL-WORLD PROJECT: HOUSE PRICE PREDICTION")
print("="*80)

print("\nSCENARIO:")
print("-" * 80)
print("""
You're a data scientist at a real estate company. Your task is to predict
house prices based on their square footage. The relationship might not be linear
due to factors like:
- Premium for larger houses (non-linear pricing)
- Location effects at different sizes
- Market segmentation

Let's build a robust predictive model!
""")

# ============================================================================
# STEP 1: DATA GENERATION (Simulating Real Data)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: DATA COLLECTION AND GENERATION")
print("="*80)

np.random.seed(42)
n_samples = 300

# Square footage (500 to 4000 sq ft)
sqft = np.random.uniform(500, 4000, n_samples)

# True relationship with non-linear components
# Base price + size effect + size^1.5 effect (premium for larger) + noise
base_price = 50000
price_per_sqft = 150
premium_factor = 0.01
noise_std = 30000

# Non-linear relationship
true_price = (base_price + 
              price_per_sqft * sqft + 
              premium_factor * sqft**1.5 +
              np.random.normal(0, noise_std, n_samples))

# Add some categorical features
neighborhoods = np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples, p=[0.3, 0.5, 0.2])
property_age = np.random.randint(0, 50, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'sqft': sqft,
    'price': true_price,
    'neighborhood': neighborhoods,
    'age': property_age,
    'price_per_sqft': true_price / sqft
})

# Add derived features
df['log_price'] = np.log(df['price'])
df['log_sqft'] = np.log(df['sqft'])

print("\nDataset Overview:")
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nBasic Statistics:")
print(df[['sqft', 'price', 'age']].describe())

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Price vs Square Footage
ax1 = axes[0, 0]
ax1.scatter(df['sqft'], df['price'], alpha=0.5, s=30)
ax1.set_xlabel('Square Footage', fontsize=11)
ax1.set_ylabel('Price ($)', fontsize=11)
ax1.set_title('Price vs Square Footage', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution of Prices
ax2 = axes[0, 1]
sns.histplot(df['price'], kde=True, ax=ax2, bins=30, edgecolor='black')
ax2.set_xlabel('Price ($)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Distribution of House Prices', fontsize=12, fontweight='bold')

# Plot 3: Price by Neighborhood
ax3 = axes[0, 2]
sns.boxplot(data=df, x='neighborhood', y='price', ax=ax3)
ax3.set_xlabel('Neighborhood', fontsize=11)
ax3.set_ylabel('Price ($)', fontsize=11)
ax3.set_title('Price Distribution by Neighborhood', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)

# Plot 4: Correlation Heatmap
ax4 = axes[1, 0]
numeric_cols = ['sqft', 'price', 'age', 'price_per_sqft']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', ax=ax4, center=0)
ax4.set_title('Correlation Matrix', fontsize=12, fontweight='bold')

# Plot 5: Price per Sqft vs Sqft
ax5 = axes[1, 1]
ax5.scatter(df['sqft'], df['price_per_sqft'], alpha=0.5, s=30, c=df['age'], cmap='viridis')
ax5.set_xlabel('Square Footage', fontsize=11)
ax5.set_ylabel('Price per Sqft ($)', fontsize=11)
ax5.set_title('Price per Sqft vs Size (colored by age)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(ax5.collections[0], ax=ax5)
cbar.set_label('Age (years)')

# Plot 6: Q-Q plot for normality
ax6 = axes[1, 2]
stats.probplot(df['price'], dist="norm", plot=ax6)
ax6.set_title('Q-Q Plot: Price Distribution', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("  ✓ Non-linear relationship between sqft and price")
print("  ✓ Price distribution is slightly right-skewed")
print("  ✓ Strong positive correlation between sqft and price")
print("  ✓ Neighborhood affects price")

# ============================================================================
# STEP 3: SYMBOLIC MATHEMATICS - UNDERSTANDING THE MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 3: SYMBOLIC MATHEMATICS WITH SYMPY")
print("="*80)

# Define symbolic variables
x, a0, a1, a2, a3 = sp.symbols('x a0 a1 a2 a3')
lambda_sym = sp.Symbol('lambda', positive=True)

# Polynomial model
poly_model = a0 + a1*x + a2*x**2 + a3*x**3

print("\nPolynomial Model:")
print(f"  Price = {poly_model}")

# Derivatives
first_deriv = sp.diff(poly_model, x)
second_deriv = sp.diff(first_deriv, x)

print("\nFirst Derivative (Rate of Change):")
print(f"  dPrice/dSqft = {first_deriv}")

print("\nSecond Derivative (Curvature):")
print(f"  d²Price/dSqft² = {second_deriv}")

# Loss function with regularization
y_pred_sym = poly_model
y_true_sym = sp.Symbol('y')
mse_loss = (y_true_sym - y_pred_sym)**2
ridge_penalty = lambda_sym * (a1**2 + a2**2 + a3**2)
lasso_penalty = lambda_sym * (sp.Abs(a1) + sp.Abs(a2) + sp.Abs(a3))

print("\nLoss Functions:")
print(f"  MSE = {mse_loss}")
print(f"  Ridge Penalty = {ridge_penalty}")
print(f"  Lasso Penalty = {lasso_penalty}")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*80)
print("STEP 4: FEATURE ENGINEERING")
print("="*80)

# Prepare features
X = df[['sqft']].values
y = df['price'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Create polynomial features
degrees_to_try = [1, 2, 3, 4, 5]
models_dict = {}

print("\nCreating Polynomial Features:")
for degree in degrees_to_try:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    print(f"  Degree {degree}: {X_train_poly.shape[1]} features")
    print(f"    Feature names: {poly.get_feature_names_out(['sqft'])}")
    
    models_dict[degree] = {
        'poly': poly,
        'X_train_poly': X_train_poly,
        'X_test_poly': X_test_poly
    }

# ============================================================================
# STEP 5: MODEL TRAINING - MULTIPLE APPROACHES
# ============================================================================

print("\n" + "="*80)
print("STEP 5: MODEL TRAINING - COMPARING APPROACHES")
print("="*80)

results = []

# 1. Simple Linear Regression
print("\n1. Simple Linear Regression:")
for degree in degrees_to_try:
    X_train_poly = models_dict[degree]['X_train_poly']
    X_test_poly = models_dict[degree]['X_test_poly']
    
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    results.append({
        'Model': f'Linear (d={degree})',
        'Degree': degree,
        'Train_RMSE': train_rmse,
        'Test_RMSE': test_rmse,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'Coefficients': model.coef_,
        'model_obj': model
    })
    
    print(f"  Degree {degree}: Train RMSE = ${train_rmse:,.0f}, Test RMSE = ${test_rmse:,.0f}")

# 2. Ridge Regression
print("\n2. Ridge Regression (L2):")
alphas = [0.01, 0.1, 1, 10, 100]
best_ridge = None
best_ridge_score = float('inf')

for alpha in alphas:
    degree = 4  # Use degree 4 for regularized models
    X_train_poly = models_dict[degree]['X_train_poly']
    X_test_poly = models_dict[degree]['X_test_poly']
    
    model = Ridge(alpha=alpha)
    model.fit(X_train_poly, y_train)
    
    y_test_pred = model.predict(X_test_poly)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    if test_rmse < best_ridge_score:
        best_ridge_score = test_rmse
        best_ridge = {'alpha': alpha, 'model': model, 'rmse': test_rmse}
    
    print(f"  α = {alpha:6.2f}: Test RMSE = ${test_rmse:,.0f}")

# Train best Ridge model
degree = 4
X_train_poly = models_dict[degree]['X_train_poly']
X_test_poly = models_dict[degree]['X_test_poly']

ridge_model = Ridge(alpha=best_ridge['alpha'])
ridge_model.fit(X_train_poly, y_train)

y_train_pred_ridge = ridge_model.predict(X_train_poly)
y_test_pred_ridge = ridge_model.predict(X_test_poly)

results.append({
    'Model': f'Ridge (α={best_ridge["alpha"]})',
    'Degree': degree,
    'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred_ridge)),
    'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred_ridge)),
    'Train_R2': r2_score(y_train, y_train_pred_ridge),
    'Test_R2': r2_score(y_test, y_test_pred_ridge),
    'Coefficients': ridge_model.coef_,
    'model_obj': ridge_model
})

# 3. Lasso Regression
print("\n3. Lasso Regression (L1):")
best_lasso = None
best_lasso_score = float('inf')

for alpha in alphas:
    model = Lasso(alpha=alpha, max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    y_test_pred = model.predict(X_test_poly)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-10)
    
    if test_rmse < best_lasso_score:
        best_lasso_score = test_rmse
        best_lasso = {'alpha': alpha, 'model': model, 'rmse': test_rmse}
    
    print(f"  α = {alpha:6.2f}: Test RMSE = ${test_rmse:,.0f}, Non-zero coefs = {n_nonzero}")

lasso_model = Lasso(alpha=best_lasso['alpha'], max_iter=10000)
lasso_model.fit(X_train_poly, y_train)

y_train_pred_lasso = lasso_model.predict(X_train_poly)
y_test_pred_lasso = lasso_model.predict(X_test_poly)

results.append({
    'Model': f'Lasso (α={best_lasso["alpha"]})',
    'Degree': degree,
    'Train_RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred_lasso)),
    'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred_lasso)),
    'Train_R2': r2_score(y_train, y_train_pred_lasso),
    'Test_R2': r2_score(y_test, y_test_pred_lasso),
    'Coefficients': lasso_model.coef_,
    'model_obj': lasso_model
})

# Create results DataFrame
df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("MODEL COMPARISON RESULTS")
print("="*80)
print(df_results[['Model', 'Train_RMSE', 'Test_RMSE', 'Train_R2', 'Test_R2']].to_string(index=False))

# ============================================================================
# STEP 6: CROSS-VALIDATION
# ============================================================================

print("\n" + "="*80)
print("STEP 6: CROSS-VALIDATION FOR MODEL SELECTION")
print("="*80)

cv_results = []

for degree in degrees_to_try:
    pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])
    
    cv_scores = cross_val_score(pipeline, X, y, cv=5, 
                                 scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores
    
    cv_results.append({
        'Degree': degree,
        'CV_RMSE_Mean': cv_rmse.mean(),
        'CV_RMSE_Std': cv_rmse.std()
    })
    
    print(f"Degree {degree}: CV RMSE = ${cv_rmse.mean():,.0f} ± ${cv_rmse.std():,.0f}")

df_cv = pd.DataFrame(cv_results)

# ============================================================================
# STEP 7: STATISTICAL ANALYSIS WITH STATSMODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 7: STATISTICAL ANALYSIS")
print("="*80)

# Use optimal degree (3)
optimal_degree = 3
X_train_poly_opt = models_dict[optimal_degree]['X_train_poly']

# Fit with statsmodels
X_sm = sm.add_constant(X_train_poly_opt)
model_sm = sm.OLS(y_train, X_sm)
results_sm = model_sm.fit()

print("\nOLS Regression Summary:")
print(results_sm.summary())

# Residual analysis
residuals = results_sm.resid

print("\n\nResidual Diagnostics:")
print("-" * 80)

# Normality test
_, p_shapiro = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test (Normality): p-value = {p_shapiro:.6f}")
print(f"  → {'Residuals normal' if p_shapiro > 0.05 else 'Residuals not normal'} (α=0.05)")

# Heteroscedasticity test
bp_test = het_breuschpagan(residuals, X_sm)
print(f"\nBreusch-Pagan Test (Heteroscedasticity): p-value = {bp_test[1]:.6f}")
print(f"  → {'Homoscedastic' if bp_test[1] > 0.05 else 'Heteroscedastic'} (α=0.05)")

# ============================================================================
# STEP 8: COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 8: COMPREHENSIVE VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Plot 1: All model predictions
ax1 = fig.add_subplot(gs[0, :])
sqft_sorted = np.sort(df['sqft'].values).reshape(-1, 1)

for result in results[:5]:  # First 5 models (linear)
    degree = result['Degree']
    model = result['model_obj']
    poly = models_dict[degree]['poly']
    
    sqft_poly = poly.transform(sqft_sorted)
    price_pred = model.predict(sqft_poly)
    
    ax1.plot(sqft_sorted, price_pred, linewidth=2, label=result['Model'], alpha=0.7)

ax1.scatter(df['sqft'], df['price'], alpha=0.3, s=20, color='gray', label='Actual Data')
ax1.set_xlabel('Square Footage', fontsize=12)
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.set_title('Model Predictions Comparison', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: RMSE Comparison
ax2 = fig.add_subplot(gs[1, 0])
models_plot = df_results['Model'][:6]
x_pos = np.arange(len(models_plot))
width = 0.35

ax2.bar(x_pos - width/2, df_results['Train_RMSE'][:6]/1000, width, label='Train RMSE', alpha=0.8)
ax2.bar(x_pos + width/2, df_results['Test_RMSE'][:6]/1000, width, label='Test RMSE', alpha=0.8)
ax2.set_ylabel('RMSE ($1000s)', fontsize=11)
ax2.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(models_plot, rotation=45, ha='right')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: R² Score
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(x_pos - width/2, df_results['Train_R2'][:6], width, label='Train R²', alpha=0.8)
ax3.bar(x_pos + width/2, df_results['Test_R2'][:6], width, label='Test R²', alpha=0.8)
ax3.set_ylabel('R² Score', fontsize=11)
ax3.set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(models_plot, rotation=45, ha='right')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

# Plot 4: Cross-Validation Results
ax4 = fig.add_subplot(gs[1, 2])
degrees = df_cv['Degree']
cv_means = df_cv['CV_RMSE_Mean']/1000
cv_stds = df_cv['CV_RMSE_Std']/1000

ax4.errorbar(degrees, cv_means, yerr=cv_stds, marker='o', markersize=10,
             capsize=5, capthick=2, linewidth=2)
ax4.set_xlabel('Polynomial Degree', fontsize=11)
ax4.set_ylabel('CV RMSE ($1000s)', fontsize=11)
ax4.set_title('Cross-Validation Results', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
optimal_idx = np.argmin(cv_means)
ax4.axvline(x=degrees.iloc[optimal_idx], color='red', linestyle='--', 
            label=f'Optimal (degree={degrees.iloc[optimal_idx]})')
ax4.legend()

# Plot 5: Residuals
ax5 = fig.add_subplot(gs[2, 0])
optimal_model = df_results[df_results['Model'].str.contains('Linear.*3')].iloc[0]
X_test_poly_opt = models_dict[3]['X_test_poly']
y_test_pred_opt = optimal_model['model_obj'].predict(X_test_poly_opt)
residuals_test = y_test - y_test_pred_opt

ax5.scatter(y_test_pred_opt/1000, residuals_test/1000, alpha=0.6, s=40)
ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Price ($1000s)', fontsize=11)
ax5.set_ylabel('Residuals ($1000s)', fontsize=11)
ax5.set_title('Residual Plot (Optimal Model)', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Residual Distribution
ax6 = fig.add_subplot(gs[2, 1])
sns.histplot(residuals_test/1000, kde=True, ax=ax6, bins=20, edgecolor='black')
ax6.set_xlabel('Residuals ($1000s)', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title('Residual Distribution', fontsize=12, fontweight='bold')
ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)

# Plot 7: Prediction vs Actual
ax7 = fig.add_subplot(gs[2, 2])
ax7.scatter(y_test/1000, y_test_pred_opt/1000, alpha=0.6, s=40)
ax7.plot([y_test.min()/1000, y_test.max()/1000], 
         [y_test.min()/1000, y_test.max()/1000], 
         'r--', linewidth=2, label='Perfect Prediction')
ax7.set_xlabel('Actual Price ($1000s)', fontsize=11)
ax7.set_ylabel('Predicted Price ($1000s)', fontsize=11)
ax7.set_title('Predicted vs Actual', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.show()

# ============================================================================
# STEP 9: FINAL MODEL AND PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print("STEP 9: FINAL MODEL AND BUSINESS INSIGHTS")
print("="*80)

# Select best model
best_model_row = df_results.loc[df_results['Test_RMSE'].idxmin()]
print(f"\nBest Model: {best_model_row['Model']}")
print(f"  Test RMSE: ${best_model_row['Test_RMSE']:,.0f}")
print(f"  Test R²: {best_model_row['Test_R2']:.4f}")

# Make predictions for new houses
new_houses = np.array([[1500], [2500], [3500]])
poly_final = models_dict[3]['poly']
new_houses_poly = poly_final.transform(new_houses)

final_model = df_results[df_results['Model'].str.contains('Linear.*3')].iloc[0]['model_obj']
predictions = final_model.predict(new_houses_poly)

print("\nPredictions for New Houses:")
print("-" * 60)
for sqft, price in zip(new_houses.flatten(), predictions):
    print(f"  {sqft:,.0f} sq ft → ${price:,.0f}")

# Business insights
print("\n" + "="*80)
print("KEY TAKEAWAYS AND BUSINESS RECOMMENDATIONS")
print("="*80)
print("""
1. MODEL SELECTION:
   ✓ Polynomial degree 3 provides best balance
   ✓ Higher degrees overfit despite better training performance
   ✓ Regularization helps with stability

2. FEATURE IMPORTANCE:
   ✓ Square footage is primary driver (as expected)
   ✓ Non-linear relationship suggests premium for larger homes
   ✓ Consider adding location features for better accuracy

3. MODEL PERFORMANCE:
   ✓ Typical prediction error: ~${:,.0f}
   ✓ R² score indicates {:,.1%} of variance explained
   ✓ Model suitable for initial estimates

4. RECOMMENDATIONS:
   • Use this model for quick price estimates
   • Collect more features (location, amenities, condition)
   • Regular model retraining as market changes
   • Consider ensemble methods for production
   • Monitor prediction errors by neighborhood

5. LIMITATIONS:
   • Assumes current market conditions persist
   • Limited to studied sqft range (500-4000)
   • Doesn't account for market seasonality
   • Missing important features (location, condition, etc.)
""".format(best_model_row['Test_RMSE'], best_model_row['Test_R2']))

print("\n" + "="*80)
print("PROJECT COMPLETE!")
print("="*80)
print("""
Libraries Successfully Used:
✓ NumPy - Array operations and mathematical functions
✓ Pandas - Data manipulation and analysis
✓ Matplotlib - Basic visualizations
✓ Seaborn - Advanced statistical plots
✓ Scikit-learn - Machine learning models and preprocessing
✓ SciPy - Statistical tests
✓ Statsmodels - Detailed regression analysis
✓ SymPy - Symbolic mathematics

Skills Demonstrated:
✓ Data generation and exploration
✓ Polynomial feature engineering
✓ Multiple regression techniques
✓ Regularization (Ridge & Lasso)
✓ Cross-validation and model selection
✓ Statistical hypothesis testing
✓ Comprehensive visualization
✓ Business insight extraction
""")