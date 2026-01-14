

#Part 4. Z-Tests: Large Sample Testing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# ============================================================================
# Z-TESTS: When to Use vs T-Tests
# ============================================================================

print("=" * 80)
print("Z-TESTS: Testing with Known Population Parameters")
print("=" * 80)

print("""
Z-TEST vs T-TEST: When to Use Which?

USE Z-TEST WHEN:
✓ Population standard deviation (σ) is KNOWN
✓ Sample size is large (n ≥ 30) - Central Limit Theorem applies
✓ Population is normally distributed (or n is large enough)

USE T-TEST WHEN:
✓ Population standard deviation (σ) is UNKNOWN
✓ Using sample standard deviation (s) as estimate
✓ Smaller sample sizes (n < 30)

KEY DIFFERENCE:
• Z-test uses standard normal distribution (Z ~ N(0,1))
• T-test uses t-distribution (heavier tails for uncertainty)
• As n→∞, t-distribution → normal distribution

TYPES OF Z-TESTS:
1. One-Sample Z-Test: Compare sample mean to population mean
2. Two-Sample Z-Test: Compare two sample means
3. Z-Test for Proportions: Compare sample proportion to population
""")

# ============================================================================
# EXAMPLE 1: ONE-SAMPLE Z-TEST
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 1: ONE-SAMPLE Z-TEST")
print("=" * 80)

print("\nSCENARIO: IQ Testing")
print("We know population IQ has μ=100, σ=15 (standardized)")
print("A school of 50 students claims to have higher IQ.")

# Generate sample data
np.random.seed(42)
n = 50
population_mean = 100
population_std = 15
sample_iq = np.random.normal(loc=105, scale=population_std, size=n)

sample_mean = np.mean(sample_iq)
sample_size = len(sample_iq)

print(f"\nPopulation Parameters:")
print(f"  μ = {population_mean}")
print(f"  σ = {population_std} (KNOWN)")

print(f"\nSample Statistics:")
print(f"  n = {sample_size}")
print(f"  x̄ = {sample_mean:.2f}")

# Calculate Z-statistic
standard_error = population_std / np.sqrt(n)
z_statistic = (sample_mean - population_mean) / standard_error

print(f"\nZ-Test Calculation:")
print(f"  Standard Error (SE) = σ/√n = {population_std}/√{n} = {standard_error:.4f}")
print(f"  Z = (x̄ - μ) / SE")
print(f"  Z = ({sample_mean:.2f} - {population_mean}) / {standard_error:.4f}")
print(f"  Z = {z_statistic:.4f}")

# Calculate p-value (two-tailed)
p_value_two_tailed = 2 * (1 - norm.cdf(abs(z_statistic)))
p_value_one_tailed = 1 - norm.cdf(z_statistic)

print(f"\nHypotheses:")
print(f"  H₀: μ = 100 (school IQ same as population)")
print(f"  H₁: μ > 100 (school IQ higher than population)")

print(f"\nP-values:")
print(f"  Two-tailed: p = {p_value_two_tailed:.4f}")
print(f"  One-tailed: p = {p_value_one_tailed:.4f}")

alpha = 0.05
if p_value_one_tailed < alpha:
    print(f"\n✓ REJECT H₀: School has significantly higher IQ (p={p_value_one_tailed:.4f} < {alpha})")
else:
    print(f"\n✗ FAIL TO REJECT H₀: No evidence of higher IQ")

# Using statsmodels for verification
from statsmodels.stats import weightstats as stests
z_test_result = stests.ztest(sample_iq, value=population_mean)
print(f"\nStatsmodels Verification:")
print(f"  Z-statistic = {z_test_result[0]:.4f}")
print(f"  p-value (two-tailed) = {z_test_result[1]:.4f}")

# ============================================================================
# EXAMPLE 2: TWO-SAMPLE Z-TEST
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 2: TWO-SAMPLE Z-TEST")
print("=" * 80)

print("\nSCENARIO: Comparing Two Schools")
print("School A: 60 students, σ_A = 15 (known)")
print("School B: 70 students, σ_B = 15 (known)")

# Generate data
np.random.seed(123)
n_a = 60
n_b = 70
sigma_a = 15
sigma_b = 15

school_a = np.random.normal(loc=102, scale=sigma_a, size=n_a)
school_b = np.random.normal(loc=108, scale=sigma_b, size=n_b)

mean_a = np.mean(school_a)
mean_b = np.mean(school_b)

print(f"\nSample Statistics:")
print(f"  School A: n={n_a}, x̄={mean_a:.2f}, σ={sigma_a}")
print(f"  School B: n={n_b}, x̄={mean_b:.2f}, σ={sigma_b}")

# Two-sample Z-test
se_diff = np.sqrt((sigma_a**2 / n_a) + (sigma_b**2 / n_b))
z_two_sample = (mean_b - mean_a) / se_diff
p_two_sample = 2 * (1 - norm.cdf(abs(z_two_sample)))

print(f"\nTwo-Sample Z-Test:")
print(f"  SE_diff = √(σ²_A/n_A + σ²_B/n_B) = {se_diff:.4f}")
print(f"  Z = (x̄_B - x̄_A) / SE_diff = {z_two_sample:.4f}")
print(f"  p-value = {p_two_sample:.4f}")

if p_two_sample < alpha:
    print(f"\n✓ REJECT H₀: Schools have significantly different mean IQs")
    print(f"  Difference: {mean_b - mean_a:.2f} IQ points")
else:
    print(f"\n✗ FAIL TO REJECT H₀: No significant difference")

# ============================================================================
# EXAMPLE 3: Z-TEST FOR PROPORTIONS
# ============================================================================

print("\n" + "=" * 80)
print("EXAMPLE 3: Z-TEST FOR PROPORTIONS")
print("=" * 80)

print("\nSCENARIO: Click-Through Rate (CTR)")
print("National average CTR is 3% (p₀ = 0.03)")
print("Our ad campaign: 350 clicks out of 10,000 impressions")

# Data
n_impressions = 10000
n_clicks = 350
p_null = 0.03  # Null hypothesis proportion
p_sample = n_clicks / n_impressions

print(f"\nData:")
print(f"  Sample size: n = {n_impressions}")
print(f"  Successes: x = {n_clicks}")
print(f"  Sample proportion: p̂ = {p_sample:.4f}")
print(f"  Null proportion: p₀ = {p_null:.4f}")

# Z-test for proportion
se_prop = np.sqrt(p_null * (1 - p_null) / n_impressions)
z_prop = (p_sample - p_null) / se_prop
p_value_prop = 2 * (1 - norm.cdf(abs(z_prop)))

print(f"\nZ-Test for Proportion:")
print(f"  SE = √[p₀(1-p₀)/n] = {se_prop:.6f}")
print(f"  Z = (p̂ - p₀) / SE = {z_prop:.4f}")
print(f"  p-value = {p_value_prop:.4f}")

if p_value_prop < alpha:
    print(f"\n✓ REJECT H₀: Our CTR is significantly different from {p_null*100}%")
else:
    print(f"\n✗ FAIL TO REJECT H₀: CTR not significantly different")

# Using statsmodels
from statsmodels.stats.proportion import proportions_ztest
z_prop_sm, p_prop_sm = proportions_ztest(n_clicks, n_impressions, p_null)
print(f"\nStatsmodels Verification:")
print(f"  Z = {z_prop_sm:.4f}, p = {p_prop_sm:.4f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# Plot 1: One-Sample Z-Test Distribution
ax1 = fig.add_subplot(gs[0, :2])
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
ax1.plot(x, y, 'b-', linewidth=2, label='Standard Normal Distribution')

# Shade rejection regions (two-tailed)
critical_z = norm.ppf(1 - alpha/2)
x_left = x[x <= -critical_z]
x_right = x[x >= critical_z]
ax1.fill_between(x_left, norm.pdf(x_left), alpha=0.3, color='red', label=f'Rejection Region (α={alpha})')
ax1.fill_between(x_right, norm.pdf(x_right), alpha=0.3, color='red')

# Mark z-statistic
ax1.axvline(z_statistic, color='green', linestyle='--', linewidth=2, 
           label=f'Z-statistic = {z_statistic:.2f}')
ax1.axvline(-critical_z, color='red', linestyle=':', linewidth=1.5, 
           label=f'Critical values = ±{critical_z:.2f}')
ax1.axvline(critical_z, color='red', linestyle=':', linewidth=1.5)

ax1.set_xlabel('Z-value', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title(f'One-Sample Z-Test: IQ Scores\nZ={z_statistic:.2f}, p={p_value_two_tailed:.4f}', 
             fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Sample distribution
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(sample_iq, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
x_range = np.linspace(sample_iq.min(), sample_iq.max(), 100)
ax2.plot(x_range, norm.pdf(x_range, population_mean, population_std), 
        'r--', linewidth=2, label=f'Population N({population_mean},{population_std})')
ax2.axvline(sample_mean, color='green', linestyle='--', linewidth=2, 
           label=f'Sample Mean = {sample_mean:.1f}')
ax2.axvline(population_mean, color='red', linestyle='--', linewidth=2, 
           label=f'Pop Mean = {population_mean}')
ax2.set_xlabel('IQ Score', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('Sample Distribution', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plot 3: Two-Sample Comparison
ax3 = fig.add_subplot(gs[1, 0])
bp = ax3.boxplot([school_a, school_b], labels=['School A', 'School B'], 
                 patch_artist=True, widths=0.6)
colors = ['lightcoral', 'lightgreen']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('IQ Score', fontsize=11)
ax3.set_title(f'Two-Sample Z-Test\nZ={z_two_sample:.2f}, p={p_two_sample:.4f}', 
             fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Two distributions overlaid
ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(school_a, bins=20, alpha=0.5, color='coral', label='School A', edgecolor='black')
ax4.hist(school_b, bins=20, alpha=0.5, color='lightgreen', label='School B', edgecolor='black')
ax4.axvline(mean_a, color='darkred', linestyle='--', linewidth=2)
ax4.axvline(mean_b, color='darkgreen', linestyle='--', linewidth=2)
ax4.set_xlabel('IQ Score', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Means with error bars
ax5 = fig.add_subplot(gs[1, 2])
means = [mean_a, mean_b]
ses = [sigma_a/np.sqrt(n_a), sigma_b/np.sqrt(n_b)]
x_pos = [0, 1]
bars = ax5.bar(x_pos, means, yerr=ses, capsize=10, 
              color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['School A', 'School B'])
ax5.set_ylabel('Mean IQ ± SE', fontsize=11)
ax5.set_title('Mean Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Proportion Test
ax6 = fig.add_subplot(gs[2, 0])
categories = ['Clicks', 'No Clicks']
values = [n_clicks, n_impressions - n_clicks]
colors_prop = ['green', 'lightgray']
wedges, texts, autotexts = ax6.pie(values, labels=categories, autopct='%1.1f%%',
                                    colors=colors_prop, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax6.set_title(f'CTR: {p_sample*100:.2f}% (p={p_value_prop:.4f})', 
             fontsize=12, fontweight='bold')

# Plot 7: Proportion sampling distribution
ax7 = fig.add_subplot(gs[2, 1])
x_prop = np.linspace(p_null - 4*se_prop, p_null + 4*se_prop, 1000)
y_prop = norm.pdf(x_prop, p_null, se_prop)
ax7.plot(x_prop, y_prop, 'b-', linewidth=2, label='Sampling Distribution under H₀')

# Critical values
critical_prop = norm.ppf(1 - alpha/2) * se_prop + p_null
x_left_prop = x_prop[x_prop <= (p_null - (critical_prop - p_null))]
x_right_prop = x_prop[x_prop >= critical_prop]
ax7.fill_between(x_left_prop, norm.pdf(x_left_prop, p_null, se_prop), 
                alpha=0.3, color='red')
ax7.fill_between(x_right_prop, norm.pdf(x_right_prop, p_null, se_prop), 
                alpha=0.3, color='red')

ax7.axvline(p_sample, color='green', linestyle='--', linewidth=2, 
           label=f'Sample p̂ = {p_sample:.4f}')
ax7.axvline(p_null, color='red', linestyle='--', linewidth=2, 
           label=f'H₀: p₀ = {p_null:.4f}')
ax7.set_xlabel('Proportion', fontsize=11)
ax7.set_ylabel('Probability Density', fontsize=11)
ax7.set_title('Proportion Test Distribution', fontsize=12, fontweight='bold')
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

# Plot 8: Z vs T comparison
ax8 = fig.add_subplot(gs[2, 2])
x_comp = np.linspace(-4, 4, 1000)
y_z = norm.pdf(x_comp)
y_t5 = stats.t.pdf(x_comp, df=5)
y_t30 = stats.t.pdf(x_comp, df=30)

ax8.plot(x_comp, y_z, 'b-', linewidth=2, label='Z (Normal)')
ax8.plot(x_comp, y_t5, 'r--', linewidth=2, label='t (df=5)')
ax8.plot(x_comp, y_t30, 'g:', linewidth=2, label='t (df=30)')
ax8.set_xlabel('Value', fontsize=11)
ax8.set_ylabel('Probability Density', fontsize=11)
ax8.set_title('Z vs T Distributions\n(T → Z as df increases)', 
             fontsize=12, fontweight='bold')
ax8.legend()
ax8.grid(True, alpha=0.3)

plt.savefig('z_tests_complete.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: z_tests_complete.png")
plt.show()

# ============================================================================
# DECISION GUIDE TABLE
# ============================================================================

print("\n" + "=" * 80)
print("DECISION GUIDE: Z-TEST vs T-TEST")
print("=" * 80)

decision_table = pd.DataFrame({
    'Condition': [
        'σ known, any n',
        'σ unknown, n ≥ 30',
        'σ unknown, n < 30',
        'Comparing proportions',
        'Small sample, not normal'
    ],
    'Test to Use': [
        'Z-test',
        'Z-test or t-test (both OK)',
        't-test (required)',
        'Z-test for proportions',
        'Non-parametric test'
    ],
    'Distribution': [
        'Standard Normal Z(0,1)',
        'Normal (CLT applies)',
        't-distribution',
        'Normal (large n)',
        'None (rank-based)'
    ],
    'Standard Error': [
        'σ/√n',
        's/√n ≈ σ/√n',
        's/√n',
        '√[p(1-p)/n]',
        'Not applicable'
    ]
})

print(decision_table.to_string(index=False))

print("\n" + "=" * 80)
print("KEY FORMULAS SUMMARY")
print("=" * 80)

print("""
ONE-SAMPLE Z-TEST:
  Z = (x̄ - μ₀) / (σ/√n)
  where: x̄ = sample mean, μ₀ = hypothesized mean, σ = population SD

TWO-SAMPLE Z-TEST:
  Z = (x̄₁ - x̄₂) / √(σ₁²/n₁ + σ₂²/n₂)
  where: x̄₁, x̄₂ = sample means, σ₁, σ₂ = population SDs

Z-TEST FOR PROPORTION:
  Z = (p̂ - p₀) / √[p₀(1-p₀)/n]
  where: p̂ = sample proportion, p₀ = hypothesized proportion

CONFIDENCE INTERVAL (Z):
  x̄ ± Z_α/2 × (σ/√n)
  Common: 95% CI uses Z = 1.96
""")