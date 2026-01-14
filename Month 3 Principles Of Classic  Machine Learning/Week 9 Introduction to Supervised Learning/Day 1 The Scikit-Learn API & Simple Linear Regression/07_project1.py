

#Task: Project - Salary Prediction.
#Use a simple dataset (YearsExperience vs. Salary).
#Split data using train_test_split.
#Train a LinearRegression model.
#Visualize the "Best Fit Line" using Matplotlib.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import statsmodels.api as sm

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. CREATE SAMPLE DATASET
print("=" * 60)
print("SALARY PREDICTION PROJECT")
print("=" * 60)

# Generate sample data
np.random.seed(42)
years_experience = np.array([1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
                             4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8,
                             7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5])

# Salary with some realistic variation (base: 30000 + 5000*years + noise)
salary = 30000 + 5000 * years_experience + np.random.normal(0, 3000, len(years_experience))

# Create DataFrame
data = pd.DataFrame({
    'YearsExperience': years_experience,
    'Salary': salary
})

print("\n1. DATASET OVERVIEW")
print("-" * 60)
print(data.head(10))
print(f"\nDataset Shape: {data.shape}")
print(f"\nDataset Statistics:")
print(data.describe())

# 2. DATA EXPLORATION
print("\n\n2. DATA EXPLORATION")
print("-" * 60)
print(f"Correlation between Years and Salary: {data.corr().iloc[0, 1]:.4f}")
print(f"Mean Salary: ${data['Salary'].mean():.2f}")
print(f"Median Salary: ${data['Salary'].median():.2f}")

# 3. SPLIT DATA
print("\n\n3. SPLITTING DATA (80% Train, 20% Test)")
print("-" * 60)

X = data[['YearsExperience']]
y = data['Salary']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set size: {len(X_train)} samples")
print(f"Testing set size: {len(X_test)} samples")

# 4. TRAIN LINEAR REGRESSION MODEL (scikit-learn)
print("\n\n4. TRAINING LINEAR REGRESSION MODEL")
print("-" * 60)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model Coefficient (Slope): {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")
print(f"\nEquation: Salary = {model.intercept_:.2f} + {model.coef_[0]:.2f} × Years")

# 5. MAKE PREDICTIONS
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 6. MODEL EVALUATION
print("\n\n5. MODEL EVALUATION")
print("-" * 60)

# Training metrics
train_r2 = r2_score(y_train, y_pred_train)
train_mse = mean_squared_error(y_train, y_pred_train)
train_rmse = np.sqrt(train_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)

print("Training Set Metrics:")
print(f"  R² Score: {train_r2:.4f}")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  MAE: ${train_mae:.2f}")

# Testing metrics
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\nTesting Set Metrics:")
print(f"  R² Score: {test_r2:.4f}")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  MAE: ${test_mae:.2f}")

# 7. STATSMODELS ANALYSIS (for statistical details)
print("\n\n6. STATISTICAL ANALYSIS (using Statsmodels)")
print("-" * 60)

X_train_sm = sm.add_constant(X_train)
stats_model = sm.OLS(y_train, X_train_sm).fit()
print(stats_model.summary())

# 8. VISUALIZATION
print("\n\n7. CREATING VISUALIZATIONS")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training Data with Best Fit Line
axes[0, 0].scatter(X_train, y_train, color='blue', alpha=0.6, s=100, label='Training Data')
axes[0, 0].plot(X_train, y_pred_train, color='red', linewidth=2, label='Best Fit Line')
axes[0, 0].set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Training Data - Linear Regression Model', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Test Data with Predictions
axes[0, 1].scatter(X_test, y_test, color='green', alpha=0.6, s=100, label='Test Data')
axes[0, 1].scatter(X_test, y_pred_test, color='red', alpha=0.6, s=100, marker='x', 
                   label='Predictions', linewidths=3)
# Add best fit line through entire range
x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_range_pred = model.predict(x_range)
axes[0, 1].plot(x_range, y_range_pred, color='red', linewidth=2, linestyle='--', 
                label='Best Fit Line')
axes[0, 1].set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Test Data - Model Predictions', fontsize=14, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Residuals Plot
residuals = y_test - y_pred_test
axes[1, 0].scatter(y_pred_test, residuals, color='purple', alpha=0.6, s=100)
axes[1, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1, 0].set_xlabel('Predicted Salary ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Residuals ($)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Residual Plot (Test Set)', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: All Data with Best Fit Line
axes[1, 1].scatter(X, y, color='darkblue', alpha=0.5, s=100, label='All Data Points')
axes[1, 1].plot(x_range, y_range_pred, color='red', linewidth=3, label='Best Fit Line')
axes[1, 1].set_xlabel('Years of Experience', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Salary ($)', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Complete Dataset with Linear Regression', fontsize=14, fontweight='bold')
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

# Add equation text
equation_text = f'y = {model.intercept_:.2f} + {model.coef_[0]:.2f}x\nR² = {test_r2:.4f}'
axes[1, 1].text(0.05, 0.95, equation_text, transform=axes[1, 1].transAxes,
                fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('salary_prediction_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualization saved as 'salary_prediction_analysis.png'")
plt.show()

# 9. PREDICTION EXAMPLE
print("\n\n8. SAMPLE PREDICTIONS")
print("-" * 60)

sample_years = np.array([[3.5], [5.0], [7.5]])
sample_predictions = model.predict(sample_years)

for years, salary in zip(sample_years.flatten(), sample_predictions):
    print(f"Experience: {years} years → Predicted Salary: ${salary:,.2f}")

