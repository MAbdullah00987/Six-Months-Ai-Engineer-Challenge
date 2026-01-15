
#Part 5: Scikit-Learn Complete ML Pipeline


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("SECTION 1: The Scikit-Learn Estimator API")
print("="*70)

# 1.1 Generate simple dataset
X_simple = np.random.randn(100, 1) * 10
y_simple = 3 + 2 * X_simple.ravel() + np.random.randn(100) * 2

print("\n1.1 THE THREE-STEP WORKFLOW:")
print("Step 1: CREATE estimator")
model = LinearRegression()
print(f"   model = LinearRegression()")
print(f"   Type: {type(model)}")

print("\nStep 2: FIT (train) on data")
model.fit(X_simple, y_simple)
print(f"   model.fit(X, y)")
print(f"   Learned parameters: w0={model.intercept_:.3f}, w1={model.coef_[0]:.3f}")

print("\nStep 3: PREDICT on new data")
X_new = np.array([[5], [10], [15]])
predictions = model.predict(X_new)
print(f"   predictions = model.predict(X_new)")
print(f"   X_new = {X_new.ravel()}")
print(f"   Predictions = {predictions}")

print("\nBonus: SCORE (evaluate) on test data")
score = model.score(X_simple, y_simple)
print(f"   R² = model.score(X, y) = {score:.4f}")

print("\n" + "="*70)
print("SECTION 2: Train/Test Split - The Right Way")
print("="*70)

# 2.1 Load real dataset
print("\n2.1 Loading California Housing dataset...")
housing = fetch_california_housing()
X_full = pd.DataFrame(housing.data, columns=housing.feature_names)
y_full = housing.target

print(f"Dataset shape: {X_full.shape}")
print(f"Features: {list(X_full.columns)}")
print(f"\nFirst 3 samples:")
print(X_full.head(3))

# 2.2 Proper train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

print(f"\n2.2 TRAIN/TEST SPLIT:")
print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X_full)*100:.0f}%)")
print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X_full)*100:.0f}%)")
print(f"Feature dimensions: {X_train.shape[1]}")

print("\n" + "="*70)
print("SECTION 3: Feature Scaling - Critical for ML")
print("="*70)

# 3.1 Why scaling matters
print("\n3.1 FEATURE SCALES (before scaling):")
print(X_train.describe().loc[['mean', 'std']])

# 3.2 Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # IMPORTANT: only fit on train!

print("\n3.2 AFTER STANDARDIZATION:")
print(f"Mean: {X_train_scaled.mean(axis=0)}")  # should be ~0
print(f"Std: {X_train_scaled.std(axis=0)}")    # should be ~1

print("\n⚠️ CRITICAL RULE: Only fit() scaler on training data!")
print("   - scaler.fit_transform(X_train)  ✓")
print("   - scaler.transform(X_test)       ✓")
print("   - scaler.fit_transform(X_test)   ✗ DATA LEAKAGE!")

print("\n" + "="*70)
print("SECTION 4: Training & Evaluating Models")
print("="*70)

# 4.1 Train model
print("\n4.1 TRAINING LINEAR REGRESSION:")
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

print(f"✓ Model trained")
print(f"Intercept: {lr.intercept_:.4f}")
print(f"Coefficients (first 3): {lr.coef_[:3]}")

# 4.2 Predictions
y_train_pred = lr.predict(X_train_scaled)
y_test_pred = lr.predict(X_test_scaled)

print("\n4.2 PREDICTIONS:")
print(f"First 5 predictions: {y_test_pred[:5]}")
print(f"First 5 actuals: {y_test[:5]}")

# 4.3 Evaluation metrics
print("\n4.3 EVALUATION METRICS:")
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE: {test_mae:.4f}")

if train_r2 - test_r2 > 0.1:
    print("⚠️ Warning: Large gap suggests overfitting!")
else:
    print("✓ Model generalizes well")

print("\n" + "="*70)
print("SECTION 5: Cross-Validation - Better Evaluation")
print("="*70)

# 5.1 K-Fold CV
print("\n5.1 5-FOLD CROSS-VALIDATION:")
cv_scores = cross_val_score(lr, X_train_scaled, y_train, 
                             cv=5, scoring='r2')

print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
print(f"Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Compare with test set
print(f"\nTest set R²: {test_r2:.4f}")
print("✓ Cross-validation gives more robust estimate")

print("\n" + "="*70)
print("SECTION 6: Regularization - Ridge & Lasso")
print("="*70)

# 6.1 Ridge Regression (L2)
print("\n6.1 RIDGE REGRESSION (L2):")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)

ridge_train_r2 = ridge.score(X_train_scaled, y_train)
ridge_test_r2 = ridge.score(X_test_scaled, y_test)

print(f"Alpha (λ): {ridge.alpha}")
print(f"Train R²: {ridge_train_r2:.4f}")
print(f"Test R²: {ridge_test_r2:.4f}")
print(f"Coefficients (first 3): {ridge.coef_[:3]}")

# 6.2 Lasso Regression (L1)
print("\n6.2 LASSO REGRESSION (L1):")
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)

lasso_train_r2 = lasso.score(X_train_scaled, y_train)
lasso_test_r2 = lasso.score(X_test_scaled, y_test)

print(f"Alpha (λ): {lasso.alpha}")
print(f"Train R²: {lasso_train_r2:.4f}")
print(f"Test R²: {lasso_test_r2:.4f}")
print(f"Coefficients (first 3): {lasso.coef_[:3]}")
print(f"Features with zero coeff: {np.sum(lasso.coef_ == 0)}/{len(lasso.coef_)}")
print("✓ Lasso performs feature selection!")

# 6.3 Model comparison
print("\n6.3 MODEL COMPARISON:")
comparison = pd.DataFrame({
    'Model': ['Linear', 'Ridge', 'Lasso'],
    'Train R²': [train_r2, ridge_train_r2, lasso_train_r2],
    'Test R²': [test_r2, ridge_test_r2, lasso_test_r2],
    'Test RMSE': [
        np.sqrt(mean_squared_error(y_test, lr.predict(X_test_scaled))),
        np.sqrt(mean_squared_error(y_test, ridge.predict(X_test_scaled))),
        np.sqrt(mean_squared_error(y_test, lasso.predict(X_test_scaled)))
    ]
})
print(comparison.to_string(index=False))

print("\n" + "="*70)
print("SECTION 7: Hyperparameter Tuning with GridSearchCV")
print("="*70)

# 7.1 Grid search for best alpha
print("\n7.1 SEARCHING FOR OPTIMAL ALPHA:")
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

ridge_grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring='r2')
ridge_grid.fit(X_train_scaled, y_train)

print(f"Best alpha: {ridge_grid.best_params_['alpha']}")
print(f"Best CV score (R²): {ridge_grid.best_score_:.4f}")
print(f"Test R² with best params: {ridge_grid.score(X_test_scaled, y_test):.4f}")

# Show all results
results_df = pd.DataFrame(ridge_grid.cv_results_)
print("\nAll alpha values tested:")
print(results_df[['param_alpha', 'mean_test_score', 'std_test_score']])

print("\n" + "="*70)
print("SECTION 8: Scikit-Learn Pipeline - Production Ready")
print("="*70)

# 8.1 Create pipeline
print("\n8.1 BUILDING PIPELINE:")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

print("Pipeline steps:")
for name, step in pipeline.steps:
    print(f"  {name}: {step}")

# 8.2 Fit pipeline (scales + trains in one call!)
pipeline.fit(X_train, y_train)  # Note: unscaled X_train!
pipeline_test_r2 = pipeline.score(X_test, y_test)

print(f"\n8.2 PIPELINE RESULTS:")
print(f"Test R²: {pipeline_test_r2:.4f}")
print("✓ Pipeline handles scaling automatically!")

# 8.3 Pipeline with GridSearch
pipe_params = {
    'regressor__alpha': [0.1, 1, 10]
}

pipe_grid = GridSearchCV(pipeline, pipe_params, cv=5)
pipe_grid.fit(X_train, y_train)

print(f"\n8.3 PIPELINE + GRIDSEARCH:")
print(f"Best alpha: {pipe_grid.best_params_['regressor__alpha']}")
print(f"Best CV score: {pipe_grid.best_score_:.4f}")

print("\n" + "="*70)
print("SECTION 9: Learning Curves - Diagnose Performance")
print("="*70)

# 9.1 Generate learning curve
train_sizes, train_scores, val_scores = learning_curve(
    LinearRegression(), X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='r2'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

print("\n9.1 LEARNING CURVE DATA:")
print(f"Training sizes: {train_sizes}")
print(f"Train scores: {train_mean}")
print(f"Val scores: {val_mean}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation score')
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
ax.set_xlabel('Training Set Size', fontsize=12)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_title('Learning Curves: Diagnose Bias/Variance', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.savefig('sklearn_learning_curves.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved: sklearn_learning_curves.png")
plt.close()

print("\n" + "="*70)
print("SECTION 10: Feature Importance Analysis")
print("="*70)

# 10.1 Coefficients as importance
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': lr.coef_,
    'Abs_Coefficient': np.abs(lr.coef_)
}).sort_values('Abs_Coefficient', ascending=False)

print("\n10.1 FEATURE IMPORTANCE (by coefficient magnitude):")
print(coefficients)

# 10.2 Visualize
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(coefficients['Feature'], coefficients['Coefficient'])
ax.set_xlabel('Coefficient Value', fontsize=12)
ax.set_title('Feature Importance in Linear Regression', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='red', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100, bbox_inches='tight')
print("\n✓ Saved: feature_importance.png")
plt.close()

print("\n" + "="*70)
print("KEY SKLEARN CONCEPTS MASTERED")
print("="*70)
"""
✓ SCIKIT-LEARN API PATTERN:
1. model = Estimator()         # Create
2. model.fit(X_train, y_train) # Train
3. model.predict(X_test)       # Predict
4. model.score(X_test, y_test) # Evaluate

✓ ESSENTIAL WORKFLOWS:
- Train/test split: Always split before any processing!
- Feature scaling: Fit on train, transform on test
- Cross-validation: More robust than single split
- Pipelines: Automate preprocessing + modeling
- GridSearchCV: Find optimal hyperparameters

✓ MODELS COVERED:
- LinearRegression: No regularization
- Ridge (L2): Shrinks all coefficients
- Lasso (L1): Can zero out coefficients
- ElasticNet: Combines L1 + L2

✓ EVALUATION METRICS:
- R²: Proportion of variance explained (0 to 1)
- RMSE: Prediction error in original units
- MAE: Mean absolute error (robust to outliers)
- Cross-validation score: More reliable estimate

✓ DIAGNOSTICS:
- Learning curves: Detect over/underfitting
- Feature importance: Understand model decisions
- Residual analysis: Check assumptions

YOU NOW UNDERSTAND THE COMPLETE ML PIPELINE! 🎉
"""