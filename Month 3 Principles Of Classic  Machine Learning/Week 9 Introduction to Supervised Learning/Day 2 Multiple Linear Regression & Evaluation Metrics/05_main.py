
#Part 5. Scikit-Learn Implementation

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("SCIKIT-LEARN LINEAR REGRESSION - COMPLETE GUIDE")
print("="*80)

# ============================================================================
# 1. DATASET PREPARATION
# ============================================================================
print("\n1. DATASET PREPARATION")
print("-"*80)

# Generate realistic dataset
np.random.seed(42)
n_samples = 200

# Create features
data = pd.DataFrame({
    'square_feet': np.random.randint(600, 4000, n_samples),
    'bedrooms': np.random.randint(1, 6, n_samples),
    'bathrooms': np.random.uniform(1, 4, n_samples),
    'age': np.random.randint(0, 100, n_samples),
    'distance_to_city': np.random.uniform(0, 50, n_samples)
})

# True relationship with interactions and noise
y = (100000 + 
     200 * data['square_feet'] + 
     25000 * data['bedrooms'] + 
     15000 * data['bathrooms'] - 
     1000 * data['age'] - 
     500 * data['distance_to_city'] +
     0.05 * data['square_feet'] * data['bedrooms'] +  # Interaction
     np.random.randn(n_samples) * 40000)

print("Dataset Information:")
print(f"  Samples: {n_samples}")
print(f"  Features: {data.shape[1]}")
print(f"\nFeature names: {list(data.columns)}")
print(f"\nFirst 3 samples:")
print(data.head(3))

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================
print("\n2. TRAIN-TEST SPLIT")
print("-"*80)

X_train, X_test, y_train, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 3. BASIC LINEAR REGRESSION
# ============================================================================
print("\n3. BASIC LINEAR REGRESSION")
print("-"*80)

# Create and train model
lr = LinearRegression()
lr.fit(X_train, y_train)

print("Model trained successfully!")
print(f"\nIntercept (θ₀): ${lr.intercept_:,.2f}")
print("\nCoefficients:")
for feature, coef in zip(data.columns, lr.coef_):
    print(f"  {feature:20s}: {coef:10.2f}")

# Make predictions
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Evaluate
print("\n" + "-"*80)
print("EVALUATION METRICS")
print("-"*80)

def evaluate_model(y_true, y_pred, dataset_name=""):
    """Compute and display all evaluation metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{dataset_name}:")
    print(f"  MAE (Mean Absolute Error):       ${mae:,.2f}")
    print(f"  MSE (Mean Squared Error):        {mse:,.2f}")
    print(f"  RMSE (Root Mean Squared Error):  ${rmse:,.2f}")
    print(f"  R² Score:                         {r2:.4f}")
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

metrics_train = evaluate_model(y_train, y_train_pred, "Training Set")
metrics_test = evaluate_model(y_test, y_test_pred, "Test Set")

# ============================================================================
# 4. FEATURE SCALING
# ============================================================================
print("\n\n4. FEATURE SCALING (STANDARDIZATION)")
print("-"*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr_scaled = LinearRegression()
lr_scaled.fit(X_train_scaled, y_train)

print("Scaled coefficients:")
for feature, coef in zip(data.columns, lr_scaled.coef_):
    print(f"  {feature:20s}: {coef:10.2f}")

print("\nNote: Coefficients are on standardized scale")
print("→ Larger absolute values indicate more important features")

# ============================================================================
# 5. REGULARIZATION (Ridge, Lasso, ElasticNet)
# ============================================================================
print("\n\n5. REGULARIZATION TECHNIQUES")
print("-"*80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (L2)': Ridge(alpha=100.0),
    'Lasso (L1)': Lasso(alpha=100.0),
    'ElasticNet (L1+L2)': ElasticNet(alpha=100.0, l1_ratio=0.5)
}

results = []

for name, model in models.items():
    # Train on scaled data
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        'Model': name,
        'R² Score': r2,
        'RMSE': rmse,
        'Non-zero Coefs': np.sum(np.abs(model.coef_) > 1e-10)
    })
    
    print(f"\n{name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  Non-zero coefficients: {np.sum(np.abs(model.coef_) > 1e-10)}/{len(model.coef_)}")

results_df = pd.DataFrame(results)
print("\n" + "-"*80)
print("MODEL COMPARISON:")
print(results_df.to_string(index=False))

# ============================================================================
# 6. POLYNOMIAL FEATURES
# ============================================================================
print("\n\n6. POLYNOMIAL FEATURES")
print("-"*80)

poly_degrees = [1, 2, 3]
poly_results = []

for degree in poly_degrees:
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Train model
    lr_poly = LinearRegression()
    lr_poly.fit(X_train_poly, y_train)
    
    # Evaluate
    y_pred = lr_poly.predict(X_test_poly)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    poly_results.append({
        'Degree': degree,
        'Features': X_train_poly.shape[1],
        'R² Score': r2,
        'RMSE': rmse
    })
    
    print(f"\nDegree {degree}:")
    print(f"  Number of features: {X_train_poly.shape[1]}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: ${rmse:,.2f}")

poly_results_df = pd.DataFrame(poly_results)
print("\n" + "-"*80)
print("POLYNOMIAL REGRESSION COMPARISON:")
print(poly_results_df.to_string(index=False))

# ============================================================================
# 7. CROSS-VALIDATION
# ============================================================================
print("\n\n7. CROSS-VALIDATION")
print("-"*80)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

lr_cv = LinearRegression()
cv_scores = cross_val_score(lr_cv, data, y, cv=kf, 
                             scoring='r2')

print(f"5-Fold Cross-Validation R² Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Also evaluate RMSE
cv_rmse = -cross_val_score(lr_cv, data, y, cv=kf, 
                            scoring='neg_root_mean_squared_error')
print(f"\nMean RMSE: ${cv_rmse.mean():,.2f} (+/- ${cv_rmse.std() * 2:,.2f})")

# ============================================================================
# 8. PIPELINE FOR PRODUCTION
# ============================================================================
print("\n\n8. SKLEARN PIPELINE (Production-Ready)")
print("-"*80)

# Create comprehensive pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('regressor', Ridge(alpha=100.0))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Evaluate
y_pred_pipeline = pipeline.predict(X_test)
r2_pipeline = r2_score(y_test, y_pred_pipeline)
rmse_pipeline = np.sqrt(mean_squared_error(y_test, y_pred_pipeline))

print("Pipeline: Scaler → Polynomial(2) → Ridge(100)")
print(f"  R² Score: {r2_pipeline:.4f}")
print(f"  RMSE: ${rmse_pipeline:,.2f}")

# Make prediction on new data
new_house = pd.DataFrame({
    'square_feet': [2500],
    'bedrooms': [4],
    'bathrooms': [2.5],
    'age': [5],
    'distance_to_city': [10.0]
})

predicted_price = pipeline.predict(new_house)
print(f"\nNew house prediction:")
print(f"  Features: {new_house.to_dict('records')[0]}")
print(f"  Predicted Price: ${predicted_price[0]:,.2f}")

# ============================================================================
# 9. RESIDUAL ANALYSIS
# ============================================================================
print("\n\n9. RESIDUAL ANALYSIS")
print("-"*80)

residuals = y_test - y_test_pred

print(f"Residual Statistics:")
print(f"  Mean: ${residuals.mean():,.2f} (should be ≈ 0)")
print(f"  Std Dev: ${residuals.std():,.2f}")
print(f"  Min: ${residuals.min():,.2f}")
print(f"  Max: ${residuals.max():,.2f}")

# ============================================================================
# 10. FEATURE IMPORTANCE
# ============================================================================
print("\n\n10. FEATURE IMPORTANCE ANALYSIS")
print("-"*80)

# Standardize to compare coefficients
scaler_importance = StandardScaler()
X_scaled_all = scaler_importance.fit_transform(data)
lr_importance = LinearRegression()
lr_importance.fit(X_scaled_all, y)

feature_importance = pd.DataFrame({
    'Feature': data.columns,
    'Coefficient': lr_importance.coef_,
    'Abs_Coefficient': np.abs(lr_importance.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("Feature Importance (by absolute coefficient on standardized data):")
print(feature_importance[['Feature', 'Coefficient']].to_string(index=False))

print("\nMost important features:")
for i, row in feature_importance.head(3).iterrows():
    print(f"  {i+1}. {row['Feature']}: {row['Coefficient']:.2f}")

# ============================================================================
# SUMMARY
# ============================================================================

print("SUMMARY - KEY TAKEAWAYS")
print("\n1. Linear Regression: Simple, interpretable baseline")
print("2. Regularization: Ridge/Lasso prevent overfitting")
print("3. Polynomial Features: Capture non-linear relationships")
print("4. Cross-Validation: Robust performance estimation")
print("5. Pipelines: Production-ready ML workflows")
print("\nBest practices:")
print("  ✓ Always split train/test")
print("  ✓ Scale features for regularization")
print("  ✓ Use cross-validation")
print("  ✓ Check residuals")
print("  ✓ Validate assumptions")
