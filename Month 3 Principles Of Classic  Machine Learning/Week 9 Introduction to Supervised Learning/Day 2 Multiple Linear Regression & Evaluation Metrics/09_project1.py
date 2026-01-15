
#Project - California Housing Prediction.
#Note: The "Boston Housing" dataset is deprecated in Scikit-Learn due to ethical issues. Use sklearn.datasets.
#fetch_california_housing instead.
#Predict median house values based on 8 different features (income, age, rooms, etc.).
#Calculate RMSE to see how far off your predictions are in dollars.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ==================== 1. LOAD DATA ====================
print("=" * 60)
print("CALIFORNIA HOUSING PRICE PREDICTION PROJECT")
print("=" * 60)

# Load the California housing dataset
california = fetch_california_housing()
print("\n1. Loading California Housing Dataset...")
print(f"   Dataset shape: {california.data.shape}")
print(f"   Target shape: {california.target.shape}")

# Create DataFrame for better visualization
df = pd.DataFrame(california.data, columns=california.feature_names)
df['MedHouseVal'] = california.target

print("\n2. Dataset Information:")
print(df.info())
print("\n3. First few rows:")
print(df.head())

# ==================== 2. EXPLORATORY DATA ANALYSIS ====================
print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# Statistical summary
print("\n4. Statistical Summary:")
print(df.describe())

# Check for missing values
print("\n5. Missing Values:")
print(df.isnull().sum())

# Target variable analysis
print("\n6. Target Variable (Median House Value) Analysis:")
print(f"   Mean: ${df['MedHouseVal'].mean() * 100000:.2f}")
print(f"   Median: ${df['MedHouseVal'].median() * 100000:.2f}")
print(f"   Std Dev: ${df['MedHouseVal'].std() * 100000:.2f}")
print(f"   Min: ${df['MedHouseVal'].min() * 100000:.2f}")
print(f"   Max: ${df['MedHouseVal'].max() * 100000:.2f}")

# ==================== 3. VISUALIZATIONS ====================
print("\n7. Creating visualizations...")

# Distribution of target variable
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Median House Value (in $100,000s)')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Median House Values')
axes[0, 0].axvline(df['MedHouseVal'].mean(), color='red', linestyle='--', label='Mean')
axes[0, 0].legend()

# Correlation heatmap
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[0, 1], cbar_kws={'label': 'Correlation'})
axes[0, 1].set_title('Feature Correlation Heatmap')

# Feature importance (correlation with target)
correlations = df.corr()['MedHouseVal'].drop('MedHouseVal').sort_values()
correlations.plot(kind='barh', ax=axes[1, 0], color='steelblue')
axes[1, 0].set_xlabel('Correlation with Median House Value')
axes[1, 0].set_title('Feature Correlations with Target')
axes[1, 0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)

# Scatter plot: Most correlated feature vs target
most_correlated = correlations.abs().idxmax()
axes[1, 1].scatter(df[most_correlated], df['MedHouseVal'], alpha=0.3, s=10)
axes[1, 1].set_xlabel(most_correlated)
axes[1, 1].set_ylabel('Median House Value')
axes[1, 1].set_title(f'Strongest Predictor: {most_correlated}')

plt.tight_layout()
plt.savefig('california_housing_eda.png', dpi=300, bbox_inches='tight')
print("   Saved: california_housing_eda.png")
plt.show()

# ==================== 4. DATA PREPARATION ====================
print("\n" + "=" * 60)
print("DATA PREPARATION")
print("=" * 60)

# Separate features and target
X = california.data
y = california.target

print("\n8. Splitting data into train and test sets (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"   Training set size: {X_train.shape[0]} samples")
print(f"   Test set size: {X_test.shape[0]} samples")

# Feature scaling
print("\n9. Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   Features scaled successfully!")

# ==================== 5. MODEL TRAINING ====================
print("\n" + "=" * 60)
print("MODEL TRAINING")
print("=" * 60)

# Dictionary to store models and results
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

print("\n10. Training and evaluating models...\n")

for name, model in models.items():
    print(f"Training {name}...")
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results.append({
        'Model': name,
        'Train RMSE': train_rmse,
        'Test RMSE': test_rmse,
        'Test RMSE ($)': test_rmse * 100000,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'CV RMSE': cv_rmse
    })
    
    print(f"   Test RMSE: ${test_rmse * 100000:,.2f}")
    print(f"   Test R² Score: {test_r2:.4f}")
    print(f"   Test MAE: ${test_mae * 100000:,.2f}\n")

# ==================== 6. RESULTS COMPARISON ====================
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results)
print("\n11. Complete Results Table:")
print(results_df.to_string(index=False))

# Find best model
best_model_idx = results_df['Test RMSE'].idxmin()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_rmse = results_df.loc[best_model_idx, 'Test RMSE ($)']

print(f"\n12. Best Model: {best_model_name}")
print(f"    Test RMSE: ${best_rmse:,.2f}")
print(f"    This means predictions are off by approximately ${best_rmse:,.2f} on average")

# ==================== 7. VISUALIZATION OF RESULTS ====================
print("\n13. Creating model comparison visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# RMSE Comparison
axes[0, 0].barh(results_df['Model'], results_df['Test RMSE ($)'], color='coral')
axes[0, 0].set_xlabel('Test RMSE (in dollars)')
axes[0, 0].set_title('Model Performance - RMSE Comparison')
axes[0, 0].invert_yaxis()

# R² Score Comparison
axes[0, 1].barh(results_df['Model'], results_df['Test R²'], color='skyblue')
axes[0, 1].set_xlabel('R² Score')
axes[0, 1].set_title('Model Performance - R² Score (Higher is Better)')
axes[0, 1].set_xlim(0, 1)
axes[0, 1].invert_yaxis()

# Actual vs Predicted for best model
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)
axes[1, 0].scatter(y_test, y_pred_best, alpha=0.3, s=10)
axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[1, 0].set_xlabel('Actual Values')
axes[1, 0].set_ylabel('Predicted Values')
axes[1, 0].set_title(f'Actual vs Predicted - {best_model_name}')
axes[1, 0].legend()

# Residuals plot
residuals = y_test - y_pred_best
axes[1, 1].scatter(y_pred_best, residuals, alpha=0.3, s=10)
axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Predicted Values')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title(f'Residual Plot - {best_model_name}')

plt.tight_layout()
plt.savefig('california_housing_results.png', dpi=300, bbox_inches='tight')
print("   Saved: california_housing_results.png")
plt.show()

# ==================== 8. FEATURE IMPORTANCE ====================
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    print("\n14. Feature Importance Analysis:")
    feature_importance = pd.DataFrame({
        'Feature': california.feature_names,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("   Saved: feature_importance.png")
    plt.show()

# ==================== 9. SAMPLE PREDICTIONS ====================
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print("\n15. Example predictions on test set:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)

for idx in sample_indices:
    actual = y_test[idx] * 100000
    predicted = y_pred_best[idx] * 100000
    error = abs(actual - predicted)
    
    print(f"\nSample {idx}:")
    print(f"   Actual Price: ${actual:,.2f}")
    print(f"   Predicted Price: ${predicted:,.2f}")
    print(f"   Error: ${error:,.2f}")
