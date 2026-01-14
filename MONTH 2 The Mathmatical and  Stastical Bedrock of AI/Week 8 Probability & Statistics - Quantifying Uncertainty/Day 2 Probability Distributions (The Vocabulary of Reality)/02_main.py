
#Probability Distributions - Advanced Analysis

"""
PROBABILITY DISTRIBUTIONS - ADVANCED ANALYSIS
Central Limit Theorem, Real-world Applications, Statistical Testing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import lilliefors

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# CENTRAL LIMIT THEOREM DEMONSTRATION
# ============================================================================

print("="*80)
print("CENTRAL LIMIT THEOREM (CLT)")
print("="*80)
print("\nThe CLT states: Regardless of the population distribution,")
print("the sampling distribution of means approaches normal as n increases.\n")

# Start with a highly non-normal distribution (Exponential)
lambda_param = 1
sample_sizes = [1, 2, 5, 10, 30, 100]
n_experiments = 10000

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, n in enumerate(sample_sizes):
    # Draw samples and compute means
    sample_means = np.zeros(n_experiments)
    for i in range(n_experiments):
        sample = np.random.exponential(scale=1/lambda_param, size=n)
        sample_means[i] = np.mean(sample)
    
    # Plot histogram
    axes[idx].hist(sample_means, bins=50, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    
    # Overlay theoretical normal distribution
    mu = 1/lambda_param
    sigma = (1/lambda_param) / np.sqrt(n)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    theoretical_normal = stats.norm.pdf(x, loc=mu, scale=sigma)
    axes[idx].plot(x, theoretical_normal, 'r-', linewidth=2, label='Normal approximation')
    
    axes[idx].set_title(f'Sample Size n = {n}')
    axes[idx].set_xlabel('Sample Mean')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    
    # Add statistics
    text = f'Mean: {np.mean(sample_means):.3f}\nStd: {np.std(sample_means):.3f}'
    axes[idx].text(0.95, 0.95, text, transform=axes[idx].transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Central Limit Theorem: Exponential Distribution → Normal', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=150, bbox_inches='tight')
print("✓ Saved: central_limit_theorem.png\n")
plt.close()

# ============================================================================
# REAL-WORLD APPLICATION: A/B TESTING
# ============================================================================

print("\n" + "="*80)
print("REAL-WORLD APPLICATION: A/B TESTING")
print("="*80)
print("\nScenario: Testing two website designs for conversion rate\n")

# Simulate A/B test data
np.random.seed(42)
n_visitors_A = 1000
n_visitors_B = 1000
conversion_rate_A = 0.10  # 10% conversion
conversion_rate_B = 0.12  # 12% conversion (B is better)

conversions_A = np.random.binomial(1, conversion_rate_A, n_visitors_A)
conversions_B = np.random.binomial(1, conversion_rate_B, n_visitors_B)

# Calculate statistics
p_A = np.mean(conversions_A)
p_B = np.mean(conversions_B)
p_pooled = (np.sum(conversions_A) + np.sum(conversions_B)) / (n_visitors_A + n_visitors_B)

# Standard error
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_visitors_A + 1/n_visitors_B))

# Z-test
z_score = (p_B - p_A) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test

print(f"Group A: {n_visitors_A} visitors, {np.sum(conversions_A)} conversions ({p_A:.2%})")
print(f"Group B: {n_visitors_B} visitors, {np.sum(conversions_B)} conversions ({p_B:.2%})")
print(f"\nDifference: {(p_B - p_A):.2%}")
print(f"Z-score: {z_score:.4f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("\n✓ STATISTICALLY SIGNIFICANT: Reject null hypothesis")
    print("  Evidence suggests B performs better than A")
else:
    print("\n✗ NOT SIGNIFICANT: Fail to reject null hypothesis")
    print("  Insufficient evidence to conclude B is better")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Conversion rates
groups = ['A', 'B']
rates = [p_A, p_B]
colors = ['#FF6B6B', '#4ECDC4']
bars = axes[0].bar(groups, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Conversion Rate')
axes[0].set_title('A/B Test: Conversion Rates')
axes[0].set_ylim(0, max(rates) * 1.3)
for i, (bar, rate) in enumerate(zip(bars, rates)):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{rate:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Distribution of difference
n_simulations = 10000
diff_distribution = np.random.normal(loc=p_B - p_A, scale=se, size=n_simulations)
axes[1].hist(diff_distribution, bins=50, density=True, alpha=0.7, 
             color='purple', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
axes[1].axvline(p_B - p_A, color='green', linestyle='--', linewidth=2, label='Observed diff')
axes[1].set_xlabel('Difference in Conversion Rate (B - A)')
axes[1].set_ylabel('Density')
axes[1].set_title('Distribution of Difference')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Power analysis - sample size needed
effect_sizes = np.linspace(0.01, 0.05, 50)
sample_sizes_needed = []
for effect in effect_sizes:
    # Calculate required sample size for 80% power
    z_alpha = stats.norm.ppf(0.975)  # 95% confidence
    z_beta = stats.norm.ppf(0.80)    # 80% power
    p1, p2 = 0.10, 0.10 + effect
    p_avg = (p1 + p2) / 2
    n = ((z_alpha + z_beta)**2 * p_avg * (1 - p_avg)) / (effect**2)
    sample_sizes_needed.append(n * 2)  # Total for both groups

axes[2].plot(effect_sizes * 100, sample_sizes_needed, linewidth=2, color='darkblue')
axes[2].axhline(y=2000, color='red', linestyle='--', label='Current sample (2000)')
axes[2].set_xlabel('Effect Size (percentage points)')
axes[2].set_ylabel('Total Sample Size Needed')
axes[2].set_title('Sample Size Required for 80% Power')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ab_testing_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: ab_testing_analysis.png\n")
plt.close()

# ============================================================================
# DISTRIBUTION COMPARISON AND SELECTION
# ============================================================================

print("\n" + "="*80)
print("DISTRIBUTION FITTING: SELECTING THE RIGHT MODEL")
print("="*80)

# Generate mixed data
np.random.seed(123)
data = np.concatenate([
    np.random.normal(50, 10, 500),
    np.random.exponential(20, 300)
])

print("\nTesting different distributions on sample data...\n")

# Fit multiple distributions
distributions = {
    'Normal': stats.norm,
    'Exponential': stats.expon,
    'Gamma': stats.gamma,
    'Lognormal': stats.lognorm,
    'Weibull': stats.weibull_min
}

results = []
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (name, dist) in enumerate(distributions.items()):
    # Fit distribution
    params = dist.fit(data)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, lambda x: dist.cdf(x, *params))
    
    # AIC and BIC
    log_likelihood = np.sum(np.log(dist.pdf(data, *params) + 1e-10))
    k = len(params)
    n = len(data)
    aic = 2*k - 2*log_likelihood
    bic = k*np.log(n) - 2*log_likelihood
    
    results.append({
        'Distribution': name,
        'KS Statistic': ks_stat,
        'P-value': ks_p,
        'AIC': aic,
        'BIC': bic
    })
    
    # Plot
    axes[idx].hist(data, bins=40, density=True, alpha=0.6, 
                   color='lightblue', edgecolor='black', label='Data')
    x = np.linspace(data.min(), data.max(), 200)
    axes[idx].plot(x, dist.pdf(x, *params), 'r-', linewidth=2, label=f'{name} fit')
    axes[idx].set_title(f'{name}\nKS: {ks_stat:.4f}, p: {ks_p:.4f}')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Density')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

# Hide extra subplot
axes[-1].axis('off')

plt.suptitle('Distribution Fitting Comparison', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('distribution_fitting.png', dpi=150, bbox_inches='tight')
print("✓ Saved: distribution_fitting.png\n")
plt.close()

# Display results table
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AIC')
print("\nFitting Results (sorted by AIC - lower is better):")
print(results_df.to_string(index=False))
print(f"\nBest fit: {results_df.iloc[0]['Distribution']}")

# ============================================================================
# HYPOTHESIS TESTING WITH DISTRIBUTIONS
# ============================================================================

print("\n\n" + "="*80)
print("HYPOTHESIS TESTING: PRACTICAL EXAMPLES")
print("="*80)

# Example 1: One-sample t-test
print("\n--- Test 1: One-sample t-test ---")
print("Question: Is the mean IQ score significantly different from 100?\n")

iq_scores = np.random.normal(105, 15, 50)
t_stat, p_value_t = stats.ttest_1samp(iq_scores, 100)

print(f"Sample mean: {np.mean(iq_scores):.2f}")
print(f"Sample std: {np.std(iq_scores, ddof=1):.2f}")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value: {p_value_t:.4f}")
print(f"Conclusion: {'Reject' if p_value_t < 0.05 else 'Fail to reject'} null hypothesis")

# Example 2: Two-sample t-test
print("\n--- Test 2: Two-sample t-test ---")
print("Question: Do two teaching methods produce different test scores?\n")

method_A = np.random.normal(75, 10, 30)
method_B = np.random.normal(80, 10, 30)

t_stat2, p_value_t2 = stats.ttest_ind(method_A, method_B)

print(f"Method A - Mean: {np.mean(method_A):.2f}, Std: {np.std(method_A, ddof=1):.2f}")
print(f"Method B - Mean: {np.mean(method_B):.2f}, Std: {np.std(method_B, ddof=1):.2f}")
print(f"t-statistic: {t_stat2:.4f}")
print(f"p-value: {p_value_t2:.4f}")
print(f"Conclusion: {'Significant' if p_value_t2 < 0.05 else 'Not significant'} difference")

# Example 3: Chi-square test
print("\n--- Test 3: Chi-square goodness of fit ---")
print("Question: Does a die appear to be fair?\n")

die_rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=600, p=[0.15, 0.18, 0.17, 0.16, 0.17, 0.17])
observed = np.bincount(die_rolls, minlength=7)[1:]
expected = np.array([100, 100, 100, 100, 100, 100])

chi2_stat, p_value_chi2 = stats.chisquare(observed, expected)

print(f"Observed frequencies: {observed}")
print(f"Expected frequencies: {expected}")
print(f"Chi-square statistic: {chi2_stat:.4f}")
print(f"p-value: {p_value_chi2:.4f}")
print(f"Conclusion: Die is {'NOT fair' if p_value_chi2 < 0.05 else 'fair'} (at α=0.05)")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Test 1: One-sample t-test
axes[0].hist(iq_scores, bins=20, density=True, alpha=0.7, color='lightcoral', edgecolor='black')
x = np.linspace(iq_scores.min(), iq_scores.max(), 100)
axes[0].plot(x, stats.norm.pdf(x, 100, 15), 'b--', linewidth=2, label='Null (μ=100)')
axes[0].plot(x, stats.norm.pdf(x, np.mean(iq_scores), np.std(iq_scores, ddof=1)), 
             'r-', linewidth=2, label='Sample')
axes[0].axvline(100, color='blue', linestyle='--', alpha=0.5)
axes[0].axvline(np.mean(iq_scores), color='red', linestyle='--', alpha=0.5)
axes[0].set_xlabel('IQ Score')
axes[0].set_ylabel('Density')
axes[0].set_title(f'One-sample t-test\np-value: {p_value_t:.4f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Test 2: Two-sample t-test
axes[1].hist(method_A, bins=15, density=True, alpha=0.5, color='blue', 
             edgecolor='black', label='Method A')
axes[1].hist(method_B, bins=15, density=True, alpha=0.5, color='red', 
             edgecolor='black', label='Method B')
axes[1].axvline(np.mean(method_A), color='blue', linestyle='--', linewidth=2)
axes[1].axvline(np.mean(method_B), color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Test Score')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Two-sample t-test\np-value: {p_value_t2:.4f}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Test 3: Chi-square
x_chi = np.arange(1, 7)
width = 0.35
axes[2].bar(x_chi - width/2, observed, width, label='Observed', 
            color='skyblue', edgecolor='black')
axes[2].bar(x_chi + width/2, expected, width, label='Expected', 
            color='lightcoral', edgecolor='black')
axes[2].set_xlabel('Die Face')
axes[2].set_ylabel('Frequency')
axes[2].set_title(f'Chi-square Test\np-value: {p_value_chi2:.4f}')
axes[2].set_xticks(x_chi)
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('hypothesis_testing.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: hypothesis_testing.png\n")
plt.close()

# ============================================================================
# CONFIDENCE INTERVALS
# ============================================================================

print("\n" + "="*80)
print("CONFIDENCE INTERVALS")
print("="*80)

# Generate sample data
np.random.seed(42)
sample_data = np.random.normal(100, 15, 50)

# Calculate confidence intervals at different levels
confidence_levels = [0.90, 0.95, 0.99]
sample_mean = np.mean(sample_data)
sample_std = np.std(sample_data, ddof=1)
n = len(sample_data)

print(f"\nSample statistics:")
print(f"Mean: {sample_mean:.2f}")
print(f"Std Dev: {sample_std:.2f}")
print(f"Sample size: {n}")

fig, ax = plt.subplots(figsize=(12, 6))

colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
y_positions = [0, 1, 2]

for idx, conf_level in enumerate(confidence_levels):
    # Calculate confidence interval
    margin_of_error = stats.t.ppf((1 + conf_level) / 2, df=n-1) * (sample_std / np.sqrt(n))
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    print(f"\n{conf_level*100:.0f}% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"  Width: {ci_upper - ci_lower:.2f}")
    print(f"  Margin of error: ±{margin_of_error:.2f}")
    
    # Plot
    ax.plot([ci_lower, ci_upper], [y_positions[idx], y_positions[idx]], 
            'o-', linewidth=3, markersize=8, color=colors[idx], 
            label=f'{conf_level*100:.0f}% CI: [{ci_lower:.1f}, {ci_upper:.1f}]')
    ax.plot(sample_mean, y_positions[idx], 'ko', markersize=10)

# Add true population mean line
ax.axvline(100, color='red', linestyle='--', linewidth=2, label='True μ = 100', alpha=0.7)

ax.set_yticks(y_positions)
ax.set_yticklabels([f'{int(c*100)}%' for c in confidence_levels])
ax.set_xlabel('Value')
ax.set_ylabel('Confidence Level')
ax.set_title('Confidence Intervals at Different Levels', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('confidence_intervals.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: confidence_intervals.png\n")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("LEARNING SUMMARY: KEY TAKEAWAYS")
print("="*80)

summary = """
1. DISCRETE DISTRIBUTIONS
   - Bernoulli: Single trial (coin flip)
   - Binomial: n independent trials
   - Poisson: Count of rare events in fixed interval

2. CONTINUOUS DISTRIBUTIONS
   - Normal: Most important, describes many natural phenomena
   - Uniform: All values equally likely
   - Exponential: Time between events

3. CENTRAL LIMIT THEOREM
   - Sample means → Normal distribution (regardless of population)
   - Enables inference even when population isn't normal

4. HYPOTHESIS TESTING
   - t-tests: Compare means
   - Chi-square: Test categorical data
   - Always report p-values and effect sizes

5. PRACTICAL TOOLS
   - NumPy: Random sampling, array operations
   - SciPy: Statistical tests, distribution functions
   - Matplotlib/Seaborn: Visualization
   - StatsModels: Advanced statistical modeling
   - Pandas: Data manipulation

6. WHEN TO USE EACH DISTRIBUTION
   - Count data, fixed trials → Binomial
   - Count data, rate-based → Poisson
   - Continuous, symmetric → Normal
   - Time/lifetime → Exponential
   - Any bounded range → Uniform
"""

print(summary)
print("\n" + "="*80)
print("ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
print("="*80)
print("\nFiles created:")
print("  1. bernoulli_distribution.png")
print("  2. binomial_distribution.png")
print("  3. poisson_distribution.png")
print("  4. normal_distribution.png")
print("  5. uniform_distribution.png")
print("  6. exponential_distribution.png")
print("  7. central_limit_theorem.png")
print("  8. ab_testing_analysis.png")
print("  9. distribution_fitting.png")
print(" 10. hypothesis_testing.png")
print(" 11. confidence_intervals.png")
print("\nYou now have the foundation to:")
print("  ✓ Understand probability distributions deeply")
print("  ✓ Apply them to real-world problems")
print("  ✓ Use Python tools effectively")
print("  ✓ Make data-driven decisions with confidence")
print("\n" + "="*80)