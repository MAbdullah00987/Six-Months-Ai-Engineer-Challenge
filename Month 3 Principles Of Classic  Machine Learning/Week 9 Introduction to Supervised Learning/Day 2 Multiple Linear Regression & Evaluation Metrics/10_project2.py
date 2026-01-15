

#Project 2: Salary Prediction
# Dataset: Create or find a salary vs experience dataset
# Build your first linear regression model
# Visualize the regression line
# Calculate R² score and MSE


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# 1. CREATE DATASET
# ============================================================================
print("=" * 70)
print("SALARY PREDICTION MODEL - LINEAR REGRESSION")
print("=" * 70)

# Create a realistic salary vs experience dataset
np.random.seed(42)
n_samples = 100

# Generate experience data (0-15 years)
experience = np.random.uniform(0, 15, n_samples)

# Generate salary with realistic relationship: base salary + experience factor + noise
base_salary = 30000
salary_per_year = 5000
noise = np.random.normal(0, 5000, n_samples)
salary = base_salary + salary_per_year * experience + noise

# Create DataFrame
df = pd.DataFrame({
    'YearsExperience': experience,
    'Salary': salary
})

# Sort by experience for better visualization
df = df.sort_values('YearsExperience').reset_index(drop=True)

print("\n1. DATASET OVERVIEW")
print("-" * 70)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nDataset Statistics:")
print(df.describe())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 70)

# Check for missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Correlation analysis
correlation = df['YearsExperience'].corr(df['Salary'])
print(f"\nPearson Correlation: {correlation:.4f}")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot
axes[0, 0].scatter(df['YearsExperience'], df['Salary'], alpha=0.6, edgecolors='k')
axes[0, 0].set_xlabel('Years of Experience', fontsize=12)
axes[0, 0].set_ylabel('Salary ($)', fontsize=12)
axes[0, 0].set_title('Salary vs Experience - Raw Data', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Distribution of Experience
axes[0, 1].hist(df['YearsExperience'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
axes[0, 1].set_xlabel('Years of Experience', fontsize=12)
axes[0, 1].set_ylabel('Frequency', fontsize=12)
axes[0, 1].set_title('Distribution of Experience', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Distribution of Salary
axes[1, 0].hist(df['Salary'], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
axes[1, 0].set_xlabel('Salary ($)', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Distribution of Salary', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Box plots
bp = axes[1, 1].boxplot([df['YearsExperience'], df['Salary']/1000], 
                         labels=['Experience (years)', 'Salary ($1000s)'],
                         patch_artist=True)
for patch, color in zip(bp['boxes'], ['skyblue', 'lightcoral']):
    patch.set_facecolor(color)
axes[1, 1].set_ylabel('Value', fontsize=12)
axes[1, 1].set_title('Box Plots - Outlier Detection', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: 01_exploratory_analysis.png")
plt.show()

# ============================================================================
# 3. PREPARE DATA FOR MODELING
# ============================================================================
print("\n3. DATA PREPARATION")
print("-" * 70)

# Separate features and target
X = df[['YearsExperience']].values
y = df['Salary'].values

print(f"Feature shape (X): {X.shape}")
print(f"Target shape (y): {y.shape}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# ============================================================================
# 4. BUILD LINEAR REGRESSION MODEL
# ============================================================================
print("\n4. MODEL TRAINING")
print("-" * 70)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get model parameters
slope = model.coef_[0]
intercept = model.intercept_

print(f"Model trained successfully!")
print(f"\nModel Parameters:")
print(f"  • Slope (coefficient): ${slope:,.2f} per year")
print(f"  • Intercept: ${intercept:,.2f}")
print(f"\nModel Equation:")
print(f"  Salary = {intercept:,.2f} + {slope:,.2f} × Experience")

# ============================================================================
# 5. MAKE PREDICTIONS
# ============================================================================
print("\n5. PREDICTIONS")
print("-" * 70)

# Predict on training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Show some sample predictions
print("\nSample Predictions (Test Set):")
print("-" * 70)
sample_df = pd.DataFrame({
    'Experience': X_test[:10].flatten(),
    'Actual Salary': y_test[:10],
    'Predicted Salary': y_test_pred[:10],
    'Difference': y_test[:10] - y_test_pred[:10]
})
print(sample_df.to_string(index=False))

# ============================================================================
# 6. MODEL EVALUATION
# ============================================================================
print("\n6. MODEL EVALUATION")
print("=" * 70)

# Training metrics
train_r2 = r2_score(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_train_pred)

# Testing metrics
test_r2 = r2_score(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTRAINING SET METRICS:")
print("-" * 70)
print(f"  R² Score:              {train_r2:.4f}")
print(f"  Mean Squared Error:    ${train_mse:,.2f}")
print(f"  Root Mean Squared Error: ${train_rmse:,.2f}")
print(f"  Mean Absolute Error:   ${train_mae:,.2f}")

print("\nTEST SET METRICS:")
print("-" * 70)
print(f"  R² Score:              {test_r2:.4f}")
print(f"  Mean Squared Error:    ${test_mse:,.2f}")
print(f"  Root Mean Squared Error: ${test_rmse:,.2f}")
print(f"  Mean Absolute Error:   ${test_mae:,.2f}")

print("\n" + "=" * 70)
print("INTERPRETATION:")
print("=" * 70)
print(f"• R² Score of {test_r2:.4f} means the model explains {test_r2*100:.2f}% of variance")
print(f"• On average, predictions are off by ${test_mae:,.2f} (MAE)")
print(f"• RMSE of ${test_rmse:,.2f} shows typical prediction error")

# ============================================================================
# 7. VISUALIZE RESULTS
# ============================================================================
print("\n7. CREATING VISUALIZATIONS...")
print("-" * 70)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Regression Line with Training Data
axes[0, 0].scatter(X_train, y_train, alpha=0.6, label='Training Data', 
                   color='blue', edgecolors='k', s=60)
axes[0, 0].plot(X_train, y_train_pred, color='red', linewidth=2.5, 
                label=f'Regression Line\ny = {intercept:.0f} + {slope:.0f}x')
axes[0, 0].set_xlabel('Years of Experience', fontsize=12)
axes[0, 0].set_ylabel('Salary ($)', fontsize=12)
axes[0, 0].set_title('Training Set - Linear Regression Fit', 
                     fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Test Set Predictions
axes[0, 1].scatter(X_test, y_test, alpha=0.6, label='Actual Test Data', 
                   color='green', edgecolors='k', s=60)
axes[0, 1].scatter(X_test, y_test_pred, alpha=0.6, label='Predicted Values', 
                   color='red', marker='s', edgecolors='k', s=60)
axes[0, 1].plot(X_test, y_test_pred, color='red', linewidth=2, 
                linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Years of Experience', fontsize=12)
axes[0, 1].set_ylabel('Salary ($)', fontsize=12)
axes[0, 1].set_title(f'Test Set Predictions (R² = {test_r2:.4f})', 
                     fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residual Plot
residuals = y_test - y_test_pred
axes[1, 0].scatter(y_test_pred, residuals, alpha=0.6, 
                   color='purple', edgecolors='k', s=60)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Salary ($)', fontsize=12)
axes[1, 0].set_ylabel('Residuals ($)', fontsize=12)
axes[1, 0].set_title('Residual Plot - Error Analysis', 
                     fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Predicted vs Actual
axes[1, 1].scatter(y_test, y_test_pred, alpha=0.6, 
                   color='orange', edgecolors='k', s=60)
# Perfect prediction line
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect Prediction')
axes[1, 1].set_xlabel('Actual Salary ($)', fontsize=12)
axes[1, 1].set_ylabel('Predicted Salary ($)', fontsize=12)
axes[1, 1].set_title(f'Predicted vs Actual (RMSE = ${test_rmse:,.0f})', 
                     fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_regression_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_regression_analysis.png")
plt.show()

# ============================================================================
# 8. STATISTICAL ANALYSIS
# ============================================================================
print("\n8. STATISTICAL ANALYSIS")
print("=" * 70)

# Perform statistical tests on residuals
stat, p_value = stats.shapiro(residuals)
print(f"\nNormality Test (Shapiro-Wilk):")
print(f"  Test Statistic: {stat:.4f}")
print(f"  P-value: {p_value:.4f}")
if p_value > 0.05:
    print("  ✓ Residuals appear normally distributed (p > 0.05)")
else:
    print("  ✗ Residuals may not be normally distributed (p ≤ 0.05)")

# Durbin-Watson test for autocorrelation
from scipy.stats import jarque_bera
jb_stat, jb_pval = jarque_bera(residuals)
print(f"\nJarque-Bera Test:")
print(f"  Test Statistic: {jb_stat:.4f}")
print(f"  P-value: {jb_pval:.4f}")

# ============================================================================
# 9. SAVE MODEL AND RESULTS
# ============================================================================
print("\n9. SAVING RESULTS")
print("-" * 70)

# Save the dataset
df.to_csv('salary_dataset.csv', index=False)
print("✓ Saved: salary_dataset.csv")

# Save predictions
results_df = pd.DataFrame({
    'Experience': X_test.flatten(),
    'Actual_Salary': y_test,
    'Predicted_Salary': y_test_pred,
    'Residual': residuals
})
results_df.to_csv('predictions.csv', index=False)
print("✓ Saved: predictions.csv")

# Save model summary
with open('model_summary.txt', 'w') as f:
    f.write("LINEAR REGRESSION MODEL SUMMARY\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Model Equation: Salary = {intercept:.2f} + {slope:.2f} × Experience\n\n")
    f.write(f"Training Set Metrics:\n")
    f.write(f"  R² Score: {train_r2:.4f}\n")
    f.write(f"  RMSE: ${train_rmse:,.2f}\n")
    f.write(f"  MAE: ${train_mae:,.2f}\n\n")
    f.write(f"Test Set Metrics:\n")
    f.write(f"  R² Score: {test_r2:.4f}\n")
    f.write(f"  MSE: ${test_mse:,.2f}\n")
    f.write(f"  RMSE: ${test_rmse:,.2f}\n")
    f.write(f"  MAE: ${test_mae:,.2f}\n")
print("✓ Saved: model_summary.txt")

print("\n" + "=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nGenerated Files:")
print("  1. 01_exploratory_analysis.png")
print("  2. 02_regression_analysis.png")
print("  3. salary_dataset.csv")
print("  4. predictions.csv")
print("  5. model_summary.txt")
print("\n" + "=" * 70)