
#Part 4. Statistical Analysis with SciPy & Statsmodels

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("LINEAR REGRESSION - COMPREHENSIVE STATISTICAL ANALYSIS")
print("="*80)

# Generate realistic dataset
np.random.seed(42)
n = 100

# Multiple features
X = pd.DataFrame({
    'square_feet': np.random.randint(800, 3000, n),
    'bedrooms': np.random.randint(1, 5, n),
    'age': np.random.randint(0, 50, n)
})

# True relationship with noise
y = (50000 + 
     150 * X['square_feet'] + 
     20000 * X['bedrooms'] - 
     2000 * X['age'] + 
     np.random.randn(n) * 30000)

print("\nDataset: House Price Prediction")
print(f"Samples: {n}")
print(f"Features: {list(X.columns)}")
print(f"\nFirst 5 rows:")
print(X.head())
print(f"\nTarget (Price) - First 5 values:")
print(y.head())

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print("\n" + "="*80)
print("1. DESCRIPTIVE STATISTICS")
print("="*80)

print("\nFeature Statistics:")
print(X.describe())

print("\nTarget Statistics:")
print(pd.Series(y).describe())

print("\nCorrelation Matrix:")
df_full = X.copy()
df_full['price'] = y
print(df_full.corr())

# ============================================================================
# 2. STATSMODELS REGRESSION (Full Statistical Output)
# ============================================================================
print("\n" + "="*80)
print("2. ORDINARY LEAST SQUARES (OLS) REGRESSION - STATSMODELS")
print("="*80)

# Add constant for intercept
X_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_const)
results = model.fit()

# Print comprehensive summary
print(results.summary())

# Extract key statistics
print("\n" + "-"*80)
print("KEY STATISTICS EXPLAINED:")
print("-"*80)
print(f"R-squared: {results.rsquared:.4f}")
print("  → Proportion of variance explained by the model")
print("  → Range: [0, 1], higher is better")

print(f"\nAdjusted R-squared: {results.rsquared_adj:.4f}")
print("  → R² adjusted for number of predictors")
print("  → Penalizes adding irrelevant features")

print(f"\nF-statistic: {results.fvalue:.4f}, p-value: {results.f_pvalue:.6f}")
print("  → Tests if model is better than intercept-only model")
print("  → p < 0.05 suggests model is statistically significant")

print(f"\nAIC: {results.aic:.2f}, BIC: {results.bic:.2f}")
print("  → Model selection criteria (lower is better)")
print("  → BIC penalizes complexity more than AIC")

# ============================================================================
# 3. COEFFICIENT ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. COEFFICIENT ANALYSIS")
print("="*80)

coef_df = pd.DataFrame({
    'Feature': X_const.columns,
    'Coefficient': results.params,
    'Std Error': results.bse,
    't-statistic': results.tvalues,
    'p-value': results.pvalues,
    'CI Lower': results.conf_int()[0],
    'CI Upper': results.conf_int()[1]
})

print(coef_df.to_string(index=False))

print("\nInterpretation:")
for idx, row in coef_df.iterrows():
    if row['Feature'] != 'const':
        print(f"\n{row['Feature']}:")
        print(f"  Coefficient: {row['Coefficient']:.2f}")
        print(f"  → Holding other features constant, 1 unit increase in {row['Feature']}")
        print(f"    leads to ${row['Coefficient']:.2f} change in price")
        print(f"  p-value: {row['p-value']:.4f} {'✓ Significant' if row['p-value'] < 0.05 else '✗ Not significant'}")

# ============================================================================
# 4. RESIDUAL DIAGNOSTICS
# ============================================================================
print("\n" + "="*80)
print("4. RESIDUAL DIAGNOSTICS")
print("="*80)

residuals = results.resid
fitted = results.fittedvalues

# Normality tests
print("\n4.1 NORMALITY OF RESIDUALS")
print("-"*80)

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test: W={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
print(f"  H0: Residuals are normally distributed")
print(f"  Result: {'✓ Normal' if shapiro_p > 0.05 else '✗ Not normal'} (α=0.05)")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(residuals, 'norm')
print(f"\nKolmogorov-Smirnov Test: D={ks_stat:.4f}, p-value={ks_p:.4f}")

# Jarque-Bera test (from statsmodels)
jb_stat = results.jarque_bera[0]
jb_p = results.jarque_bera[1]
print(f"\nJarque-Bera Test: JB={jb_stat:.4f}, p-value={jb_p:.4f}")
print(f"  → Tests normality using skewness and kurtosis")

# ============================================================================
# 4.2 HOMOSCEDASTICITY (Constant Variance)
# ============================================================================
print("\n4.2 HOMOSCEDASTICITY TEST")
print("-"*80)

# Breusch-Pagan test
bp_test = het_breuschpagan(residuals, X_const)
bp_lm, bp_lm_p, bp_f, bp_f_p = bp_test
print(f"Breusch-Pagan Test: LM={bp_lm:.4f}, p-value={bp_lm_p:.4f}")
print(f"  H0: Homoscedasticity (constant variance)")
print(f"  Result: {'✓ Homoscedastic' if bp_lm_p > 0.05 else '✗ Heteroscedastic'} (α=0.05)")

# White test
white_test = het_white(residuals, X_const)
white_lm, white_p, white_f, white_f_p = white_test
print(f"\nWhite Test: LM={white_lm:.4f}, p-value={white_p:.4f}")

# ============================================================================
# 4.3 AUTOCORRELATION
# ============================================================================
print("\n4.3 AUTOCORRELATION TEST")
print("-"*80)

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_stat:.4f}")
print(f"  Range: [0, 4]")
print(f"  → DW ≈ 2: No autocorrelation")
print(f"  → DW < 2: Positive autocorrelation")
print(f"  → DW > 2: Negative autocorrelation")
if 1.5 < dw_stat < 2.5:
    print(f"  Result: ✓ No significant autocorrelation")
else:
    print(f"  Result: ⚠ Possible autocorrelation detected")

# ============================================================================
# 5. MULTICOLLINEARITY CHECK
# ============================================================================
print("\n" + "="*80)
print("5. MULTICOLLINEARITY DIAGNOSTICS")
print("="*80)

# Variance Inflation Factor (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF):")
print(vif_data.to_string(index=False))
print("\nInterpretation:")
print("  VIF = 1: No correlation with other features")
print("  VIF < 5: Low multicollinearity (acceptable)")
print("  VIF > 10: High multicollinearity (problematic)")

for idx, row in vif_data.iterrows():
    if row['VIF'] < 5:
        status = "✓ Good"
    elif row['VIF'] < 10:
        status = "⚠ Moderate"
    else:
        status = "✗ High"
    print(f"  {row['Feature']}: VIF={row['VIF']:.2f} {status}")

# ============================================================================
# 6. OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("6. OUTLIER DETECTION")
print("="*80)

# Standardized residuals
standardized_residuals = residuals / residuals.std()
outliers = np.abs(standardized_residuals) > 3

print(f"Number of outliers (|z| > 3): {outliers.sum()}")
if outliers.sum() > 0:
    print(f"Outlier indices: {np.where(outliers)[0]}")

# Cook's Distance
influence = results.get_influence()
cooks_d = influence.cooks_distance[0]
threshold = 4 / len(residuals)

print(f"\nCook's Distance threshold: {threshold:.4f}")
print(f"Influential points (D > threshold): {(cooks_d > threshold).sum()}")

# ============================================================================
# 7. CONFIDENCE AND PREDICTION INTERVALS
# ============================================================================
print("\n" + "="*80)
print("7. CONFIDENCE AND PREDICTION INTERVALS")
print("="*80)

# Example prediction
new_house = pd.DataFrame({
    'square_feet': [2000],
    'bedrooms': [3],
    'age': [10]
})
new_house_const = sm.add_constant(new_house)

# Point prediction
prediction = results.predict(new_house_const)
print(f"\nExample House: {new_house.to_dict('records')[0]}")
print(f"Predicted Price: ${prediction.values[0]:,.2f}")

# Confidence interval for the mean
pred_ci = results.get_prediction(new_house_const).summary_frame(alpha=0.05)
print(f"\n95% Confidence Interval for MEAN price:")
print(f"  [{pred_ci['mean_ci_lower'].values[0]:,.2f}, {pred_ci['mean_ci_upper'].values[0]:,.2f}]")
print(f"  → We're 95% confident the average price is in this range")

print(f"\n95% Prediction Interval for THIS house:")
print(f"  [{pred_ci['obs_ci_lower'].values[0]:,.2f}, {pred_ci['obs_ci_upper'].values[0]:,.2f}]")
print(f"  → We're 95% confident this specific house's price is in this range")

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("8. MODEL COMPARISON")
print("="*80)

# Fit reduced models
models = {}

# Full model
models['Full'] = results

# Individual feature models
for col in X.columns:
    X_single = sm.add_constant(X[[col]])
    models[col] = sm.OLS(y, X_single).fit()

# Compare models
comparison = pd.DataFrame({
    'Model': list(models.keys()),
    'R²': [m.rsquared for m in models.values()],
    'Adj R²': [m.rsquared_adj for m in models.values()],
    'AIC': [m.aic for m in models.values()],
    'BIC': [m.bic for m in models.values()]
})

print(comparison.to_string(index=False))
print("\nBest Model by AIC:", comparison.loc[comparison['AIC'].idxmin(), 'Model'])
print("Best Model by BIC:", comparison.loc[comparison['BIC'].idxmin(), 'Model'])


print("STATISTICAL ANALYSIS COMPLETE")
