
#Complete Statistics - Part 4: Advanced Topics

"""
PART 4: ADVANCED STATISTICAL CONCEPTS
Power Analysis, Multiple Testing, and Statsmodels Integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. STATISTICAL POWER ANALYSIS
# ============================================================================

print("=" * 80)
print("1. STATISTICAL POWER ANALYSIS")
print("=" * 80)

print("""
⚡ WHAT IS STATISTICAL POWER?
Power = Probability of correctly rejecting H₀ when it's false
Power = 1 - β (where β is Type II error rate)

🎯 FOUR CONNECTED COMPONENTS:
   1. Effect Size: How big is the difference?
   2. Sample Size: How many observations?
   3. Significance Level (α): Threshold for rejecting H₀
   4. Power (1-β): Probability of detecting true effect

   ⚠️ If you know 3, you can calculate the 4th!

📊 TYPICAL VALUES:
   • Power: 80% or 90% (higher = better, but needs more samples)
   • Alpha (α): 5% (0.05)
   • Effect Size: Depends on your domain
""")

# Example: Calculate required sample size for different effect sizes
effect_sizes = np.array([0.2, 0.5, 0.8])  # Small, Medium, Large (Cohen's d)
alpha = 0.05
power = 0.80

print(f"\n📊 SAMPLE SIZE REQUIREMENTS:")
print(f"   Alpha (α): {alpha}")
print(f"   Power: {power:.0%}")
print(f"\n   Effect Size (Cohen's d) | Sample Size per Group")
print(f"   {'-'*50}")

sample_sizes = []
for es in effect_sizes:
    n = tt_ind_solve_power(effect_size=es, alpha=alpha, power=power, 
                           alternative='two-sided')
    sample_sizes.append(int(np.ceil(n)))
    effect_name = {0.2: 'Small', 0.5: 'Medium', 0.8: 'Large'}[es]
    print(f"   {es:^6} ({effect_name:^6})        |   {int(np.ceil(n)):>6,}")

print(f"\n💡 INTERPRETATION:")
print(f"   • Smaller effects need MORE samples to detect")
print(f"   • Larger effects need FEWER samples to detect")
print(f"   • Power analysis helps plan experiments efficiently")

# Visualization: Power curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Power vs Sample Size for different effect sizes
sample_range = np.arange(10, 500, 10)
colors = ['red', 'orange', 'green']
effect_labels = ['Small (d=0.2)', 'Medium (d=0.5)', 'Large (d=0.8)']

for es, color, label in zip(effect_sizes, colors, effect_labels):
    powers = [tt_ind_solve_power(effect_size=es, alpha=alpha, nobs1=n, 
                                 alternative='two-sided') for n in sample_range]
    axes[0, 0].plot(sample_range, powers, linewidth=2, color=color, label=label)

axes[0, 0].axhline(0.8, color='black', linestyle='--', linewidth=1, label='80% Power')
axes[0, 0].set_xlabel('Sample Size per Group')
axes[0, 0].set_ylabel('Statistical Power')
axes[0, 0].set_title('Power vs Sample Size', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Power vs Effect Size for different sample sizes
effect_range = np.linspace(0.1, 1.2, 100)
sample_vals = [50, 100, 200, 400]
colors2 = ['red', 'orange', 'blue', 'green']

for n, color in zip(sample_vals, colors2):
    powers = [tt_ind_solve_power(effect_size=es, alpha=alpha, nobs1=n, 
                                 alternative='two-sided') for es in effect_range]
    axes[0, 1].plot(effect_range, powers, linewidth=2, color=color, label=f'n={n}')

axes[0, 1].axhline(0.8, color='black', linestyle='--', linewidth=1)
axes[0, 1].axvline(0.5, color='gray', linestyle=':', linewidth=1, label='Medium Effect')
axes[0, 1].set_xlabel('Effect Size (Cohen\'s d)')
axes[0, 1].set_ylabel('Statistical Power')
axes[0, 1].set_title('Power vs Effect Size', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Required sample size vs desired power
power_range = np.linspace(0.5, 0.99, 50)
for es, color, label in zip([0.2, 0.5, 0.8], colors, effect_labels):
    sample_reqs = [tt_ind_solve_power(effect_size=es, alpha=alpha, power=p, 
                                      alternative='two-sided') for p in power_range]
    axes[1, 0].plot(power_range, sample_reqs, linewidth=2, color=color, label=label)

axes[1, 0].axvline(0.8, color='black', linestyle='--', linewidth=1, label='Common Target')
axes[1, 0].set_xlabel('Desired Power')
axes[1, 0].set_ylabel('Required Sample Size per Group')
axes[1, 0].set_title('Sample Size Requirements', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Type I and Type II errors visualization
# Create confusion matrix style visualization
labels = ['H₀ True', 'H₀ False']
scenarios = ['Fail to Reject H₀', 'Reject H₀']

data = np.array([[1-alpha, 1-power],  # Beta (Type II error)
                 [alpha, power]])      # Alpha (Type I error), Power

im = axes[1, 1].imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Add text annotations
texts = [
    ['Correct\n(1-α)', 'Type II Error\n(β)'],
    ['Type I Error\n(α)', 'Correct\n(Power=1-β)']
]
for i in range(2):
    for j in range(2):
        text = axes[1, 1].text(j, i, f'{texts[i][j]}\n{data[i,j]:.2f}',
                               ha="center", va="center", color="black", 
                               fontsize=11, fontweight='bold')

axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_yticks([0, 1])
axes[1, 1].set_xticklabels(labels)
axes[1, 1].set_yticklabels(scenarios)
axes[1, 1].set_title('Decision Outcomes (α=0.05, Power=0.80)', 
                     fontsize=14, fontweight='bold')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.savefig('power_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. MULTIPLE TESTING PROBLEM
# ============================================================================

print("\n" + "=" * 80)
print("2. MULTIPLE TESTING PROBLEM")
print("=" * 80)

print("""
⚠️ THE PROBLEM:
When testing multiple hypotheses, the chance of at least one false positive
increases dramatically!

📊 EXAMPLE:
   • Test 1 hypothesis at α=0.05 → 5% chance of Type I error
   • Test 20 hypotheses at α=0.05 → 1 - (0.95)²⁰ = 64% chance of ≥1 false positive!

🛡️ SOLUTIONS:
   1. Bonferroni Correction: Divide α by number of tests
   2. Holm-Bonferroni: Sequential adjustment (less conservative)
   3. Benjamini-Hochberg (FDR): Control false discovery rate
""")

# Simulate multiple testing scenario
np.random.seed(42)
n_tests = 20
alpha = 0.05

# Generate p-values: 15 from null (uniform), 5 from alternative (low p-values)
p_values_null = np.random.uniform(0, 1, 15)
p_values_alternative = np.random.beta(0.5, 5, 5)  # Generate low p-values
p_values = np.concatenate([p_values_alternative, p_values_null])
np.random.shuffle(p_values)

print(f"\n📊 SIMULATED SCENARIO: Testing {n_tests} hypotheses")
print(f"   True Positives: 5 (real effects)")
print(f"   True Negatives: 15 (no effect)")
print(f"\n   P-values: {p_values[:5].round(4)}... (showing first 5)")

# Apply different correction methods
methods_results = {}

# 1. No correction
uncorrected = p_values < alpha
methods_results['Uncorrected'] = uncorrected

# 2. Bonferroni
bonferroni_alpha = alpha / n_tests
bonferroni = p_values < bonferroni_alpha
methods_results['Bonferroni'] = bonferroni

# 3. Holm-Bonferroni
reject_holm, pvals_holm, _, _ = multipletests(p_values, alpha=alpha, method='holm')
methods_results['Holm'] = reject_holm

# 4. Benjamini-Hochberg (FDR)
reject_fdr, pvals_fdr, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
methods_results['BH (FDR)'] = reject_fdr

print(f"\n🔍 RESULTS COMPARISON:")
print(f"   {'Method':<20} | Rejections | Adjusted α")
print(f"   {'-'*50}")
print(f"   {'Uncorrected':<20} | {sum(uncorrected):^10} | {alpha:.4f}")
print(f"   {'Bonferroni':<20} | {sum(bonferroni):^10} | {bonferroni_alpha:.4f}")
print(f"   {'Holm':<20} | {sum(reject_holm):^10} | Sequential")
print(f"   {'BH (FDR)':<20} | {sum(reject_fdr):^10} | Sequential")

print(f"\n💡 INTERPRETATION:")
print(f"   • Uncorrected: Most rejections, but highest false positive risk")
print(f"   • Bonferroni: Most conservative, may miss true effects")
print(f"   • Holm: Good balance, slightly less conservative")
print(f"   • BH (FDR): Best for exploratory research with many tests")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. P-value distribution
axes[0, 0].hist(p_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(alpha, color='red', linestyle='--', linewidth=2, 
                    label=f'α = {alpha}')
axes[0, 0].axvline(bonferroni_alpha, color='orange', linestyle='--', linewidth=2,
                    label=f'Bonferroni α = {bonferroni_alpha:.4f}')
axes[0, 0].set_xlabel('P-value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('P-value Distribution', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Sorted p-values with thresholds
sorted_indices = np.argsort(p_values)
sorted_pvals = p_values[sorted_indices]
ranks = np.arange(1, n_tests + 1)

axes[0, 1].scatter(ranks, sorted_pvals, s=80, alpha=0.7, color='blue', label='P-values')
axes[0, 1].axhline(alpha, color='red', linestyle='--', linewidth=2, label='Uncorrected')
axes[0, 1].axhline(bonferroni_alpha, color='orange', linestyle='--', linewidth=2, 
                    label='Bonferroni')

# BH threshold line
bh_threshold = (ranks / n_tests) * alpha
axes[0, 1].plot(ranks, bh_threshold, 'g--', linewidth=2, label='BH Threshold')

axes[0, 1].set_xlabel('Rank (sorted by p-value)')
axes[0, 1].set_ylabel('P-value')
axes[0, 1].set_title('Sorted P-values with Thresholds', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Comparison of rejections
methods = list(methods_results.keys())
rejections = [sum(methods_results[m]) for m in methods]

bars = axes[1, 0].bar(methods, rejections, color=['red', 'orange', 'blue', 'green'],
                       alpha=0.7, edgecolor='black', linewidth=2)
axes[1, 0].axhline(5, color='black', linestyle='--', linewidth=2, 
                    label='True Positives (5)')
axes[1, 0].set_ylabel('Number of Rejections')
axes[1, 0].set_title('Rejections by Method', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, rejections):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')

# 4. Family-wise error rate simulation
n_simulations = 1000
n_tests_range = np.arange(1, 51, 2)
fwer_results = []

for n_t in n_tests_range:
    false_positives = 0
    for _ in range(n_simulations):
        sim_pvals = np.random.uniform(0, 1, n_t)
        if any(sim_pvals < alpha):
            false_positives += 1
    fwer_results.append(false_positives / n_simulations)

axes[1, 1].plot(n_tests_range, fwer_results, linewidth=3, color='red',
                 label='Observed FWER')
axes[1, 1].axhline(alpha, color='green', linestyle='--', linewidth=2,
                    label=f'Target α = {alpha}')
axes[1, 1].fill_between(n_tests_range, alpha, fwer_results, 
                         where=(np.array(fwer_results) > alpha),
                         alpha=0.3, color='red')
axes[1, 1].set_xlabel('Number of Tests')
axes[1, 1].set_ylabel('Family-Wise Error Rate')
axes[1, 1].set_title('FWER Increases with Multiple Tests', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multiple_testing.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. USING STATSMODELS FOR REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("3. REGRESSION ANALYSIS WITH STATSMODELS")
print("=" * 80)

# Generate sample data: predicting house prices
np.random.seed(42)
n_samples = 100

square_feet = np.random.uniform(800, 3000, n_samples)
bedrooms = np.random.randint(1, 6, n_samples)
age = np.random.uniform(0, 50, n_samples)

# True relationship with some noise
price = (200 * square_feet + 50000 * bedrooms - 1000 * age + 
         np.random.normal(0, 50000, n_samples))

# Create DataFrame
df = pd.DataFrame({
    'price': price,
    'square_feet': square_feet,
    'bedrooms': bedrooms,
    'age': age
})

print(f"\n📊 DATA PREVIEW:")
print(df.head())
print(f"\n   Shape: {df.shape}")
print(f"\n   Summary Statistics:")
print(df.describe().round(2))

# Prepare data for regression
X = df[['square_feet', 'bedrooms', 'age']]
y = df['price']
X = sm.add_constant(X)  # Add intercept

# Fit OLS regression
model = sm.OLS(y, X)
results = model.fit()

print(f"\n📊 REGRESSION RESULTS:")
print(results.summary())

# Extract key metrics
print(f"\n🔑 KEY METRICS:")
print(f"   R-squared: {results.rsquared:.4f}")
print(f"   Adjusted R-squared: {results.rsquared_adj:.4f}")
print(f"   F-statistic: {results.fvalue:.2f}")
print(f"   Prob (F-statistic): {results.f_pvalue:.6f}")

print(f"\n📊 COEFFICIENTS:")
for i, name in enumerate(['Intercept', 'Square Feet', 'Bedrooms', 'Age']):
    coef = results.params[i]
    pval = results.pvalues[i]
    ci_low, ci_high = results.conf_int().iloc[i]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"   {name:15}: {coef:12,.2f}  (p={pval:.4f}) {sig}")
    print(f"                   95% CI: [{ci_low:,.2f}, {ci_high:,.2f}]")

# Predictions
y_pred = results.fittedvalues

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted
axes[0, 0].scatter(y, y_pred, alpha=0.6, s=50)
axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2)
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals vs Fitted
residuals = y - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6, s=50)
axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Fitted Values ($)')
axes[0, 1].set_ylabel('Residuals ($)')
axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 4. Coefficient plot
coef_df = pd.DataFrame({
    'Coefficient': results.params[1:],
    'Variable': ['Square Feet', 'Bedrooms', 'Age']
})
conf_int = results.conf_int().iloc[1:]
errors = [(results.params[i] - conf_int.iloc[i-1, 0]) for i in range(1, len(results.params))]

axes[1, 1].barh(coef_df['Variable'], coef_df['Coefficient'], 
                xerr=errors, capsize=5, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1, 1].set_xlabel('Coefficient Value')
axes[1, 1].set_title('Coefficients with 95% CI', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 80)
print("✅ PART 4 COMPLETE: Advanced Topics")
print("=" * 80)
print("\nKey Takeaways:")
print("1. Power Analysis: Plan sample sizes before experiments")
print("2. Multiple Testing: Adjust for multiple comparisons")
print("3. Statsmodels: Professional statistical modeling")
print("4. Always check assumptions and residuals")
print("\nNext: Run Part 5 for SymPy symbolic math!")