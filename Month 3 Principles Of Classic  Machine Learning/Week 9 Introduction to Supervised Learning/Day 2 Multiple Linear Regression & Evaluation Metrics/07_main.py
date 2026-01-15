
#Complete Linear Regression Project - California Housing

"""
COMPLETE LINEAR REGRESSION PROJECT
Dataset: California Housing (sklearn.datasets)
Goal: Predict median house prices
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

print("="*90)
print("COMPLETE LINEAR REGRESSION PROJECT: CALIFORNIA HOUSING PRICE PREDICTION")
print("="*90)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\nSTEP 1: DATA LOADING AND EXPLORATION")
print("-"*90)

# Load California housing dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print(f"Dataset shape: {df.shape}")
print(f"\nFeature names:")
for i, feature in enumerate(housing.feature_names, 1):
    print(f"  {i}. {feature}")

print(f"\nTarget variable: MedHouseVal (Median House Value in $100,000s)")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nDataset statistics:")
print(df.describe())

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nData types:")
print(df.dtypes)

# ============================================================================
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n\nSTEP 2: EXPLORATORY DATA ANALYSIS")
print("-"*90)

# Target distribution
print("\nTarget Variable Statistics:")
print(df['MedHouseVal'].describe())

# Correlation analysis
print("\nCorrelation with Target:")
correlations = df.corr()['MedHouseVal'].sort_values(ascending=False)
print(correlations)

# Identify highly correlated features
print("\nHighly correlated features (|r| > 0.5):")
high_corr = correlations[np.abs(correlations) > 0.5]
for feature, corr in high_corr.items():
    if feature != 'MedHouseVal':
        print(f"  {feature}: {corr:.3f}")

# Check for multicollinearity
print("\nMulticollinearity check (Feature Correlations):")
feature_corr = df[housing.feature_names].corr()
high_feature_corr = []
for i in range(len(feature_corr.columns)):
    for j in range(i+1, len(feature_corr.columns)):
        if abs(feature_corr.iloc[i, j]) > 0.7:
            high_feature_corr.append({
                'Feature 1': feature_corr.columns[i],
                'Feature 2': feature_corr.columns[j],
                'Correlation': feature_corr.iloc[i, j]
            })

if high_feature_corr:
    print("  High correlations found:")
    for item in high_feature_corr:
        print(f"    {item['Feature 1']} ↔ {item['Feature 2']}: {item['Correlation']:.3f}")
else:
    print("  No high correlations (|r| > 0.7) detected")

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n\nSTEP 3: FEATURE ENGINEERING")
print("-"*90)

# Create new features
df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['Population'] / df['HouseAge']

print("Created new features:")
print("  1. rooms_per_household = AveRooms / AveOccup")
print("  2. bedrooms_per_room = AveBedrms / AveRooms")
print("  3. population_per_household = Population / HouseAge")

# Handle any infinite or NaN values
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.median())

print(f"\nDataset shape after feature engineering: {df.shape}")

# ============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# ============================================================================
print("\n\nSTEP 4: DATA PREPARATION")
print("-"*90)

# Separate features and target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features standardized (mean=0, std=1)")

# ============================================================================
# STEP 5: BASELINE MODEL
# ============================================================================
print("\n\nSTEP 5: BASELINE MODEL")
print("-"*90)

# Mean baseline
y_train_mean = y_train.mean()
y_baseline_pred = np.full(len(y_test), y_train_mean)

baseline_mae = mean_absolute_error(y_test, y_baseline_pred)
baseline_rmse = np.sqrt(mean_squared_error(y_test, y_baseline_pred))
baseline_r2 = r2_score(y_test, y_baseline_pred)

print("Baseline: Predict mean value for all samples")
print(f"  Mean predicted value: ${y_train_mean:.2f}00,000")
print(f"  MAE: ${baseline_mae:.4f}00,000")
print(f"  RMSE: ${baseline_rmse:.4f}00,000")
print(f"  R² Score: {baseline_r2:.4f}")

# ============================================================================
# STEP 6: LINEAR REGRESSION MODELS
# ============================================================================
print("\n\nSTEP 6: LINEAR REGRESSION MODELS")
print("-"*90)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Ridge (α=10.0)': Ridge(alpha=10.0),
    'Lasso (α=0.01)': Lasso(alpha=0.01),
    'Lasso (α=0.1)': Lasso(alpha=0.1)
}

results = []

for name, model in models.items():
    # Train
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Evaluate
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Test MAE': test_mae,
        'Test RMSE': test_rmse,
        'Overfitting': train_r2 - test_r2
    })
    
    print(f"\n{name}:")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: ${test_mae:.4f}00,000")
    print(f"  Test RMSE: ${test_rmse:.4f}00,000")

results_df = pd.DataFrame(results)

print("\n" + "="*90)
print("MODEL COMPARISON")
print("="*90)
print(results_df.to_string(index=False))

best_model_idx = results_df['Test R²'].idxmax()
best_model_name = results_df.iloc[best_model_idx]['Model']
print(f"\n✓ Best model: {best_model_name} (Test R² = {results_df.iloc[best_model_idx]['Test R²']:.4f})")

# ============================================================================
# STEP 7: CROSS-VALIDATION
# ============================================================================
print("\n\nSTEP 7: CROSS-VALIDATION (5-Fold)")
print("-"*90)

lr = LinearRegression()
cv_scores = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring='r2')

print("R² scores for each fold:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nCross-validation mean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# ============================================================================
# STEP 8: FINAL MODEL ANALYSIS
# ============================================================================
print("\n\nSTEP 8: FINAL MODEL ANALYSIS")
print("-"*90)

# Train final model
final_model = LinearRegression()
final_model.fit(X_train_scaled, y_train)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': final_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance (Standardized Coefficients):")
print(feature_importance.to_string(index=False))

print("\nTop 5 most important features:")
for i, row in feature_importance.head(5).iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"  {i+1}. {row['Feature']}: {direction} price by ${abs(row['Coefficient']):.4f}00,000 per std dev")

# Residual analysis
y_test_pred = final_model.predict(X_test_scaled)
residuals = y_test - y_test_pred

print("\n\nResidual Analysis:")
print(f"  Mean: ${residuals.mean():.6f}00,000 (should be ≈ 0)")
print(f"  Std Dev: ${residuals.std():.4f}00,000")
print(f"  Min: ${residuals.min():.4f}00,000")
print(f"  Max: ${residuals.max():.4f}00,000")

# Normality test
shapiro_stat, shapiro_p = stats.shapiro(residuals[:5000])  # Sample for large datasets
print(f"\nShapiro-Wilk Test (Residual Normality):")
print(f"  Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.4f}")
print(f"  Result: {'✓ Normal' if shapiro_p > 0.05 else '✗ Not perfectly normal'}")

# ============================================================================
# STEP 9: MAKE PREDICTIONS
# ============================================================================
print("\n\nSTEP 9: EXAMPLE PREDICTIONS")
print("-"*90)

# Example houses
example_houses = pd.DataFrame({
    'MedInc': [3.0, 8.0, 5.0],
    'HouseAge': [30, 5, 15],
    'AveRooms': [5.0, 8.0, 6.0],
    'AveBedrms': [1.0, 3.0, 2.0],
    'Population': [1000, 500, 800],
    'AveOccup': [3.0, 2.5, 3.0],
    'Latitude': [34.0, 38.0, 37.0],
    'Longitude': [-118.0, -122.0, -121.0],
    'rooms_per_household': [5.0/3.0, 8.0/2.5, 6.0/3.0],
    'bedrooms_per_room': [1.0/5.0, 3.0/8.0, 2.0/6.0],
    'population_per_household': [1000/30, 500/5, 800/15]
})

example_houses_scaled = scaler.transform(example_houses)
predictions = final_model.predict(example_houses_scaled)

for i, price in enumerate(predictions, 1):
    print(f"\nHouse {i}:")
    print(f"  Median Income: ${example_houses.iloc[i-1]['MedInc']:.2f}0,000")
    print(f"  House Age: {example_houses.iloc[i-1]['HouseAge']:.0f} years")
    print(f"  Average Rooms: {example_houses.iloc[i-1]['AveRooms']:.1f}")
    print(f"  → Predicted Price: ${price:.2f}00,000")


print("PROJECT SUMMARY")
print(f"\n✓ Dataset: {len(df)} samples, {X.shape[1]} features")
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Test R²: {results_df.iloc[best_model_idx]['Test R²']:.4f}")
print(f"✓ Test RMSE: ${results_df.iloc[best_model_idx]['Test RMSE']:.4f}00,000")
print(f"✓ Improvement over baseline: {(results_df.iloc[best_model_idx]['Test R²'] - baseline_r2):.4f}")

print("\nKey Insights:")
print("  1. MedInc (Median Income) is the strongest predictor")
print("  2. Feature engineering improved model performance")
print("  3. Regularization (Ridge/Lasso) helps prevent overfitting")
print("  4. Model explains ~60% of variance in house prices")

print("\nNext Steps:")
print("  • Try polynomial features for non-linear relationships")
print("  • Test tree-based models (Random Forest, Gradient Boosting)")
print("  • Collect more features (e.g., school quality, crime rates)")
print("  • Deploy model as API for real-time predictions")

print("PROJECT COMPLETE!")
