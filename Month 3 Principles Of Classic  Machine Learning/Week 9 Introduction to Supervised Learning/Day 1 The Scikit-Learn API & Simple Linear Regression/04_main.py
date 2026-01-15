
#Part 4: SciPy & Statsmodels - Statistical Analysis


import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson

np.random.seed(42)

print("="*70)
print("SECTION 1: SciPy Statistical Distributions")
print("="*70)

# 1.1 Normal distribution (foundation of linear regression assumptions)
print("\n1.1 NORMAL DISTRIBUTION:")
mu, sigma = 0, 1
x = np.linspace(-4, 4, 100)
pdf = stats.norm.pdf(x, mu, sigma)
cdf = stats.norm.cdf(x, mu, sigma)

print(f"Mean: {mu}, Std: {sigma}")
print(f"P(X < 0) = {stats.norm.cdf(0, mu, sigma):.3f}")
print(f"P(-1 < X < 1) = {stats.norm.cdf(1, mu, sigma) - stats.norm.cdf(-1, mu, sigma):.3f}")
print(f"95% confidence interval: [{stats.norm.ppf(0.025):.3f}, {stats.norm.ppf(0.975):.3f}]")

# 1.2 Generate data with normal errors
n_samples = 100
X = np.linspace(0, 10, n_samples)
true_slope = 2.5
true_intercept = 5
errors = np.random.normal(0, 2, n_samples)  # N(0, σ²)
y = true_intercept + true_slope * X + errors

print(f"\n1.2 GENERATED DATA:")
print(f"True model: y = {true_intercept} + {true_slope}·x + ε, where ε ~ N(0, 4)")
print(f"Samples: {n_samples}")

# 1.3 Test for normality of errors
print("\n1.3 NORMALITY TESTS:")
shapiro_stat, shapiro_p = stats.shapiro(errors)
ks_stat, ks_p = stats.kstest(errors, 'norm', args=(0, 2))

print(f"Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p-value={shapiro_p:.4f}")
print(f"Kolmogorov-Smirnov test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
print(f"Interpretation: p > 0.05 → errors are normally distributed ✓")

print("\n" + "="*70)
print("SECTION 2: Hypothesis Testing")
print("="*70)

# 2.1 T-test for slope significance
print("\n2.1 IS THE SLOPE SIGNIFICANTLY DIFFERENT FROM ZERO?")

# Fit simple linear regression manually
X_with_bias = np.column_stack([np.ones(n_samples), X])
coeffs = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
y_pred = X_with_bias @ coeffs
residuals = y - y_pred

# Standard error calculation
n, k = X_with_bias.shape
df = n - k  # degrees of freedom
mse = np.sum(residuals**2) / df
var_coeff = mse * np.linalg.inv(X_with_bias.T @ X_with_bias)
se_coeffs = np.sqrt(np.diag(var_coeff))

# T-statistics
t_stats = coeffs / se_coeffs
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df))

print(f"Estimated slope: {coeffs[1]:.4f}")
print(f"Standard error: {se_coeffs[1]:.4f}")
print(f"t-statistic: {t_stats[1]:.4f}")
print(f"p-value: {p_values[1]:.6f}")
print(f"95% CI: [{coeffs[1] - 1.96*se_coeffs[1]:.3f}, {coeffs[1] + 1.96*se_coeffs[1]:.3f}]")
print(f"Conclusion: {'Reject' if p_values[1] < 0.05 else 'Fail to reject'} H₀ (slope = 0)")

# 2.2 F-test for overall model significance
print("\n2.2 F-TEST (Overall model significance):")
ss_total = np.sum((y - y.mean())**2)
ss_residual = np.sum(residuals**2)
ss_regression = ss_total - ss_residual

ms_regression = ss_regression / (k - 1)
ms_residual = ss_residual / df

f_statistic = ms_regression / ms_residual
f_p_value = 1 - stats.f.cdf(f_statistic, k-1, df)

print(f"F-statistic: {f_statistic:.4f}")
print(f"p-value: {f_p_value:.6f}")
print(f"Conclusion: Model is {'significant' if f_p_value < 0.05 else 'not significant'}")

print("\n" + "="*70)
print("SECTION 3: Statsmodels - Comprehensive Regression Analysis")
print("="*70)

# 3.1 Fit model with statsmodels
X_sm = sm.add_constant(X)  # adds intercept
model = sm.OLS(y, X_sm)
results = model.fit()

print("\n3.1 FULL REGRESSION SUMMARY:")
print(results.summary())

# 3.2 Extract key statistics
print("\n3.2 KEY STATISTICS:")
print(f"R-squared: {results.rsquared:.4f}")
print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")
print(f"F-statistic: {results.fvalue:.4f} (p={results.f_pvalue:.6f})")
print(f"AIC: {results.aic:.2f}")
print(f"BIC: {results.bic:.2f}")
print(f"Condition Number: {results.condition_number:.2f}")

print("\n" + "="*70)
print("SECTION 4: Regression Diagnostics")
print("="*70)

# 4.1 Residual analysis
print("\n4.1 RESIDUAL DIAGNOSTICS:")
residuals_std = results.resid_pearson  # standardized residuals

# Test for autocorrelation
dw_stat = durbin_watson(results.resid)
print(f"Durbin-Watson statistic: {dw_stat:.4f}")
print(f"Interpretation: ~2 is ideal (no autocorrelation)")

# 4.2 Heteroscedasticity tests
print("\n4.2 HETEROSCEDASTICITY TESTS:")
bp_test = het_breuschpagan(results.resid, results.model.exog)
print(f"Breusch-Pagan test: LM={bp_test[0]:.4f}, p={bp_test[1]:.4f}")

white_test = het_white(results.resid, results.model.exog)
print(f"White test: LM={white_test[0]:.4f}, p={white_test[1]:.4f}")
print(f"Interpretation: p > 0.05 → homoscedasticity (constant variance) ✓")

print("\n" + "="*70)
print("SECTION 5: Multiple Regression with Multicollinearity")
print("="*70)

# 5.1 Create multiple features
X1 = np.linspace(0, 10, n_samples)
X2 = X1 + np.random.normal(0, 1, n_samples)  # correlated with X1
X3 = np.random.normal(0, 3, n_samples)  # independent

y_multi = 5 + 2*X1 + 3*X2 + 1.5*X3 + np.random.normal(0, 2, n_samples)

df_multi = pd.DataFrame({
    'X1': X1,
    'X2': X2,
    'X3': X3,
    'y': y_multi
})

# 5.2 Correlation matrix
print("\n5.1 CORRELATION MATRIX:")
corr_matrix = df_multi.corr()
print(corr_matrix)

# 5.3 Variance Inflation Factor (VIF)
print("\n5.2 MULTICOLLINEARITY CHECK (VIF):")
X_multi = df_multi[['X1', 'X2', 'X3']].values
X_multi_with_const = sm.add_constant(X_multi)

vif_data = pd.DataFrame()
vif_data["Feature"] = ['const', 'X1', 'X2', 'X3']
vif_data["VIF"] = [variance_inflation_factor(X_multi_with_const, i) 
                   for i in range(X_multi_with_const.shape[1])]
print(vif_data)
print("\nInterpretation:")
print("VIF > 10: Strong multicollinearity (problem)")
print("VIF > 5: Moderate multicollinearity (caution)")
print("VIF < 5: Low multicollinearity (acceptable)")

# 5.4 Fit multiple regression
model_multi = sm.OLS(y_multi, X_multi_with_const)
results_multi = model_multi.fit()

print("\n5.3 MULTIPLE REGRESSION RESULTS:")
print(results_multi.summary())

print("\n" + "="*70)
print("SECTION 6: Model Comparison & Selection")
print("="*70)

# Compare models with different feature sets
models_comparison = []

# Model 1: X1 only
X1_const = sm.add_constant(X1)
model1 = sm.OLS(y_multi, X1_const).fit()
models_comparison.append(('X1 only', model1))

# Model 2: X1 + X2
X12_const = sm.add_constant(df_multi[['X1', 'X2']])
model2 = sm.OLS(y_multi, X12_const).fit()
models_comparison.append(('X1 + X2', model2))

# Model 3: All features
models_comparison.append(('X1 + X2 + X3', results_multi))

print("\n6.1 MODEL COMPARISON:")
comparison_df = pd.DataFrame({
    'Model': [name for name, _ in models_comparison],
    'R²': [m.rsquared for _, m in models_comparison],
    'Adj. R²': [m.rsquared_adj for _, m in models_comparison],
    'AIC': [m.aic for _, m in models_comparison],
    'BIC': [m.bic for _, m in models_comparison],
    'RMSE': [np.sqrt(m.mse_resid) for _, m in models_comparison]
})
print(comparison_df.to_string(index=False))

print("\n6.2 INTERPRETATION:")
print("- Lower AIC/BIC = better model (penalizes complexity)")
print("- Higher Adj. R² = better (adjusts for # of features)")
print("- Lower RMSE = better predictions")
best_model_idx = comparison_df['AIC'].idxmin()
print(f"\nBest model by AIC: {comparison_df.loc[best_model_idx, 'Model']}")

print("\n" + "="*70)
print("SECTION 7: Cross-Validation Statistics")
print("="*70)

# 7.1 K-Fold cross-validation manually
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_idx, val_idx in kf.split(X_multi):
    X_train, X_val = X_multi[train_idx], X_multi[val_idx]
    y_train, y_val = y_multi[train_idx], y_multi[val_idx]
    
    X_train_const = sm.add_constant(X_train)
    X_val_const = sm.add_constant(X_val)
    
    model_cv = sm.OLS(y_train, X_train_const).fit()
    y_pred_val = model_cv.predict(X_val_const)
    
    mse = np.mean((y_val - y_pred_val)**2)
    cv_scores.append(mse)

print("\n7.1 5-FOLD CROSS-VALIDATION:")
print(f"Fold MSE scores: {[f'{s:.3f}' for s in cv_scores]}")
print(f"Mean MSE: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
print(f"Mean RMSE: {np.sqrt(np.mean(cv_scores)):.3f}")

print("\n" + "="*70)
print("KEY STATISTICAL INSIGHTS")
print("="*70)
"""
✓ TESTS PERFORMED:
1. Shapiro-Wilk & KS tests → Check normality of residuals
2. t-tests → Individual coefficient significance
3. F-test → Overall model significance
4. Durbin-Watson → Autocorrelation in residuals
5. Breusch-Pagan & White → Heteroscedasticity
6. VIF → Multicollinearity detection

✓ MODEL SELECTION CRITERIA:
- AIC/BIC: Lower is better (penalizes complexity)
- Adjusted R²: Higher is better (adjusts for features)
- Cross-validation: Estimates generalization error

✓ KEY ASSUMPTIONS CHECKED:
1. Linearity: Visual inspection of residual plots
2. Independence: Durbin-Watson statistic ≈ 2
3. Homoscedasticity: Breusch-Pagan p > 0.05
4. Normality: Shapiro-Wilk p > 0.05
5. No multicollinearity: VIF < 5

These statistical foundations are CRITICAL before using sklearn!
"""