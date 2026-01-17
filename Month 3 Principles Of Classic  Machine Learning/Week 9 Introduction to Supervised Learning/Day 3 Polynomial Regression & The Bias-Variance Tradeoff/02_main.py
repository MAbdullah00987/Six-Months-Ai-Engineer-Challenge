
#Part 2: Advanced Seaborn Visualizations for Regression
#ADVANCED SEABORN VISUALIZATIONS FOR POLYNOMIAL REGRESSION
#Beautiful statistical visualizations using Seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set beautiful Seaborn style
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)
sns.set_palette("husl")

print("="*70)
print("SEABORN ADVANCED VISUALIZATIONS")
print("="*70)

# Generate data
np.random.seed(42)
n_samples = 150
X = np.linspace(-3, 3, n_samples)
y_true = 0.5 * X**3 - 2 * X**2 + X + 1
y = y_true + np.random.normal(0, 2, n_samples)

# Create comprehensive DataFrame
X_reshaped = X.reshape(-1, 1)
degrees = [1, 3, 10]
predictions = {}

for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_reshaped)
    model = LinearRegression().fit(X_poly, y)
    predictions[f'Degree_{degree}'] = model.predict(X_poly)

df = pd.DataFrame({
    'X': X,
    'y': y,
    'y_true': y_true,
    **predictions
})

# Add residuals
for degree in degrees:
    df[f'Residual_{degree}'] = df['y'] - df[f'Degree_{degree}']

print("\n1. SEABORN REGRESSION PLOT WITH POLYNOMIAL FITS")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Scatter with regression line (degree 1)
ax1 = axes[0, 0]
sns.regplot(data=df, x='X', y='y', order=1, ax=ax1, 
            scatter_kws={'alpha': 0.5, 's': 50},
            line_kws={'color': 'red', 'linewidth': 2.5})
ax1.set_title('Seaborn regplot - Linear Fit (Underfitting)', fontsize=13, fontweight='bold')
ax1.set_xlabel('X', fontsize=11)
ax1.set_ylabel('y', fontsize=11)

# Plot 2: Polynomial degree 2
ax2 = axes[0, 1]
sns.regplot(data=df, x='X', y='y', order=2, ax=ax2,
            scatter_kws={'alpha': 0.5, 's': 50, 'color': 'blue'},
            line_kws={'color': 'darkred', 'linewidth': 2.5})
ax2.set_title('Seaborn regplot - Quadratic Fit', fontsize=13, fontweight='bold')
ax2.set_xlabel('X', fontsize=11)
ax2.set_ylabel('y', fontsize=11)

# Plot 3: Polynomial degree 3 (optimal)
ax3 = axes[1, 0]
sns.regplot(data=df, x='X', y='y', order=3, ax=ax3,
            scatter_kws={'alpha': 0.5, 's': 50, 'color': 'green'},
            line_kws={'color': 'darkgreen', 'linewidth': 2.5})
ax3.set_title('Seaborn regplot - Cubic Fit (Optimal)', fontsize=13, fontweight='bold')
ax3.set_xlabel('X', fontsize=11)
ax3.set_ylabel('y', fontsize=11)

# Plot 4: High degree (overfitting)
ax4 = axes[1, 1]
sns.regplot(data=df, x='X', y='y', order=10, ax=ax4,
            scatter_kws={'alpha': 0.5, 's': 50, 'color': 'orange'},
            line_kws={'color': 'darkorange', 'linewidth': 2.5})
ax4.set_title('Seaborn regplot - Degree 10 (Overfitting)', fontsize=13, fontweight='bold')
ax4.set_xlabel('X', fontsize=11)
ax4.set_ylabel('y', fontsize=11)

plt.tight_layout()
plt.show()

print("\n2. RESIDUAL PLOTS USING SEABORN")
print("-" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    residuals = df[f'Residual_{degree}']
    fitted = df[f'Degree_{degree}']
    
    # Residual plot
    sns.residplot(data=df, x=fitted, y=residuals, ax=ax,
                  scatter_kws={'alpha': 0.6, 's': 50},
                  line_kws={'color': 'red', 'linewidth': 2})
    ax.set_title(f'Residuals - Degree {degree}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Fitted Values', fontsize=11)
    ax.set_ylabel('Residuals', fontsize=11)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

plt.tight_layout()
plt.show()

print("\n3. DISTRIBUTION PLOTS FOR RESIDUALS")
print("-" * 70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    residuals = df[f'Residual_{degree}']
    
    # Distribution plot with KDE and histogram
    sns.histplot(residuals, kde=True, ax=ax, stat='density',
                 bins=25, edgecolor='black', alpha=0.7)
    
    # Add normal distribution overlay
    mu, sigma = residuals.mean(), residuals.std()
    x_norm = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy import stats
    ax.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 
            'r--', linewidth=2.5, label='Normal Distribution')
    
    ax.set_title(f'Residual Distribution - Degree {degree}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend()

plt.tight_layout()
plt.show()

print("\n4. JOINT PLOT: X vs Y WITH MARGINAL DISTRIBUTIONS")
print("-" * 70)

# Create joint plot
g = sns.jointplot(data=df, x='X', y='y', kind='reg',
                  height=8, ratio=4,
                  marginal_kws=dict(bins=20, fill=True, alpha=0.7),
                  joint_kws=dict(scatter_kws={'alpha': 0.5, 's': 40}))
g.fig.suptitle('Joint Distribution: X vs Y with Linear Regression', 
               fontsize=14, fontweight='bold', y=1.02)
plt.show()

print("\n5. PAIR PLOT: MULTIPLE POLYNOMIAL PREDICTIONS")
print("-" * 70)

# Prepare data for pairplot
df_pair = df[['y', 'Degree_1', 'Degree_3', 'Degree_10']].copy()
df_pair.columns = ['Observed', 'Linear', 'Cubic', 'Degree 10']

# Create pairplot
g = sns.pairplot(df_pair, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30})
g.fig.suptitle('Pair Plot: Observed vs Predictions', fontsize=14, fontweight='bold', y=1.02)
plt.show()

print("\n6. HEATMAP: CORRELATION BETWEEN PREDICTIONS")
print("-" * 70)

# Correlation matrix
correlation_matrix = df_pair.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Observed vs Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nCorrelation with Observed Values:")
print(correlation_matrix['Observed'].sort_values(ascending=False))

print("\n7. VIOLIN PLOTS: RESIDUAL DISTRIBUTIONS BY DEGREE")
print("-" * 70)

# Reshape data for violin plot
residual_data = []
for degree in degrees:
    residuals = df[f'Residual_{degree}']
    for res in residuals:
        residual_data.append({'Degree': f'Degree {degree}', 'Residual': res})

df_residuals = pd.DataFrame(residual_data)

plt.figure(figsize=(12, 6))
sns.violinplot(data=df_residuals, x='Degree', y='Residual',
               palette='Set2', inner='box')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.title('Residual Distribution by Polynomial Degree', fontsize=14, fontweight='bold')
plt.ylabel('Residuals', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.tight_layout()
plt.show()

print("\n8. BOX PLOTS: RESIDUAL SPREAD COMPARISON")
print("-" * 70)

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_residuals, x='Degree', y='Residual',
            palette='pastel', width=0.6)
sns.swarmplot(data=df_residuals, x='Degree', y='Residual',
              color='black', alpha=0.3, size=3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
plt.title('Box Plot: Residual Spread with Individual Points', fontsize=14, fontweight='bold')
plt.ylabel('Residuals', fontsize=12)
plt.xlabel('Model', fontsize=12)
plt.tight_layout()
plt.show()

print("\n9. RIDGE & LASSO: COEFFICIENT PATHS WITH SEABORN")
print("-" * 70)

# Generate high-degree polynomial
degree_reg = 10
poly_reg = PolynomialFeatures(degree=degree_reg, include_bias=False)
X_poly_reg = poly_reg.fit_transform(X_reshaped)
X_train, X_test, y_train, y_test = train_test_split(X_poly_reg, y, test_size=0.2, random_state=42)

# Try different alphas
alphas = np.logspace(-3, 2, 30)
ridge_coefs = []
lasso_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_train, y_train)
    lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train, y_train)
    
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

# Plot coefficient paths
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ridge coefficients
for i in range(min(5, degree_reg)):
    ax1.semilogx(alphas, ridge_coefs[:, i], linewidth=2.5, label=f'Coef {i+1}')
ax1.set_xlabel('Regularization Parameter (α)', fontsize=12)
ax1.set_ylabel('Coefficient Value', fontsize=12)
ax1.set_title('Ridge Regression: Coefficient Paths', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Lasso coefficients
for i in range(min(5, degree_reg)):
    ax2.semilogx(alphas, lasso_coefs[:, i], linewidth=2.5, label=f'Coef {i+1}')
ax2.set_xlabel('Regularization Parameter (α)', fontsize=12)
ax2.set_ylabel('Coefficient Value', fontsize=12)
ax2.set_title('Lasso Regression: Coefficient Paths (Sparsity)', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.show()

print("\n10. STRIP PLOT: ACTUAL VS PREDICTED VALUES")
print("-" * 70)

# Create categories for better visualization
df_strip = df.copy()
df_strip['X_category'] = pd.cut(df_strip['X'], bins=5, 
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, degree in enumerate(degrees):
    ax = axes[idx]
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame({
        'X_category': df_strip['X_category'],
        'Actual': df_strip['y'],
        'Predicted': df_strip[f'Degree_{degree}']
    })
    
    # Melt for seaborn
    plot_data_melt = pd.melt(plot_data, id_vars=['X_category'], 
                              value_vars=['Actual', 'Predicted'],
                              var_name='Type', value_name='Value')
    
    sns.stripplot(data=plot_data_melt, x='X_category', y='Value', 
                  hue='Type', ax=ax, alpha=0.6, dodge=True, size=5)
    ax.set_title(f'Actual vs Predicted - Degree {degree}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X Range', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.legend(title='Type')

plt.tight_layout()
plt.show()

print("\n11. FINAL METRICS COMPARISON WITH SEABORN")
print("-" * 70)

# Calculate comprehensive metrics
metrics_data = []
for degree in degrees:
    pred = df[f'Degree_{degree}']
    residual = df[f'Residual_{degree}']
    
    mse = mean_squared_error(df['y'], pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residual))
    r2 = r2_score(df['y'], pred)
    
    metrics_data.append({
        'Degree': f'Degree {degree}',
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })

df_metrics = pd.DataFrame(metrics_data)

# Create subplots for metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# MSE
ax1 = axes[0, 0]
sns.barplot(data=df_metrics, x='Degree', y='MSE', ax=ax1, palette='viridis')
ax1.set_title('Mean Squared Error', fontsize=12, fontweight='bold')
ax1.set_ylabel('MSE', fontsize=11)

# RMSE
ax2 = axes[0, 1]
sns.barplot(data=df_metrics, x='Degree', y='RMSE', ax=ax2, palette='plasma')
ax2.set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=11)

# MAE
ax3 = axes[1, 0]
sns.barplot(data=df_metrics, x='Degree', y='MAE', ax=ax3, palette='magma')
ax3.set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
ax3.set_ylabel('MAE', fontsize=11)

# R²
ax4 = axes[1, 1]
sns.barplot(data=df_metrics, x='Degree', y='R²', ax=ax4, palette='cividis')
ax4.set_title('R² Score (Coefficient of Determination)', fontsize=12, fontweight='bold')
ax4.set_ylabel('R²', fontsize=11)
ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Perfect Fit')
ax4.legend()

plt.tight_layout()
plt.show()

print("\nDetailed Metrics:")
print(df_metrics.to_string(index=False))

print("\n" + "="*70)
print("SEABORN VISUALIZATION SUMMARY")
print("="*70)
print("""
Seaborn Features Demonstrated:
================================
1. regplot() - Regression plots with polynomial fits
2. residplot() - Residual visualization
3. histplot() - Histograms with KDE
4. jointplot() - Bivariate distributions with marginals
5. pairplot() - Multiple pairwise relationships
6. heatmap() - Correlation matrices
7. violinplot() - Distribution shapes
8. boxplot() - Statistical summaries
9. swarmplot() - Individual data points
10. stripplot() - Categorical scatter plots
11. barplot() - Metric comparisons

Key Advantages of Seaborn:
===========================
✓ Beautiful default styling
✓ Statistical visualizations made easy
✓ Integrated with pandas DataFrames
✓ Automatic color palettes
✓ Built-in statistical computations
✓ High-level interface for complex plots

Best Practices:
===============
• Use appropriate plot types for your data
• Choose color palettes that enhance understanding
• Add clear titles and labels
• Include statistical overlays (KDE, confidence intervals)
• Combine multiple plot types for comprehensive analysis
""")