
#Complete Statistics - Part 3: A/B Testing

"""
PART 3: A/B TESTING
Real-World Applications for Product Decisions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. A/B TESTING FUNDAMENTALS
# ============================================================================


print("1. A/B TESTING FUNDAMENTALS")

print("""
 WHAT IS A/B TESTING?
A method to compare two versions (A and B) to determine which performs better.

 COMMON USE CASES:
   • Website design changes
   • New UI features
   • Marketing campaigns
   • Product recommendations
   • Pricing strategies
   • Email subject lines

 THE PROCESS:
   1. Split users randomly into Group A (control) and Group B (variant)
   2. Collect data on key metric (clicks, conversions, time on page, etc.)
   3. Perform statistical test
   4. Make decision based on results

 KEY METRICS:
   • Conversion Rate: % of users who take desired action
   • Click-Through Rate (CTR): % of users who click
   • Revenue per User: Average money generated
   • Engagement Time: How long users interact
""")

# ============================================================================
# 2. EXAMPLE 1: WEBSITE BUTTON COLOR (PROPORTIONS TEST)
# ============================================================================

print("\n" + "=" * 80)
print("2. A/B TEST: BUTTON COLOR (Blue vs Green)")
print("=" * 80)

# Scenario: Testing if green button increases click rate
np.random.seed(42)

# Group A: Blue button (control)
n_a = 1000
clicks_a = 120  # 12% click rate
click_rate_a = clicks_a / n_a

# Group B: Green button (variant)
n_b = 1000
clicks_b = 145  # 14.5% click rate
click_rate_b = clicks_b / n_b

print(f"\n📊 DATA SUMMARY:")
print(f"   Group A (Blue Button):")
print(f"      Total Users: {n_a}")
print(f"      Clicks: {clicks_a}")
print(f"      Click Rate: {click_rate_a:.1%}")
print(f"\n   Group B (Green Button):")
print(f"      Total Users: {n_b}")
print(f"      Clicks: {clicks_b}")
print(f"      Click Rate: {click_rate_b:.1%}")
print(f"\n   Observed Difference: {(click_rate_b - click_rate_a):.1%}")

# Hypothesis setup
alpha = 0.05
print(f"\n HYPOTHESIS SETUP:")
print(f"   H₀: p_b = p_a (green button has same click rate)")
print(f"   H₁: p_b > p_a (green button has higher click rate)")
print(f"   Significance Level (α): {alpha}")

# Two-proportion z-test
pooled_proportion = (clicks_a + clicks_b) / (n_a + n_b)
se = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/n_a + 1/n_b))
z_statistic = (click_rate_b - click_rate_a) / se
p_value = 1 - stats.norm.cdf(z_statistic)  # One-tailed

print(f"\n TEST RESULTS:")
print(f"   Pooled Proportion: {pooled_proportion:.4f}")
print(f"   Standard Error: {se:.4f}")
print(f"   z-statistic: {z_statistic:.4f}")
print(f"   p-value: {p_value:.4f}")

print(f"\n DECISION:")
if p_value < alpha:
    print(f"    p-value ({p_value:.4f}) < α ({alpha})")
    print(f"   → REJECT H₀")
    print(f"   → Green button SIGNIFICANTLY INCREASES click rate")
    print(f"   → Recommendation: DEPLOY green button!")
else:
    print(f"    p-value ({p_value:.4f}) ≥ α ({alpha})")
    print(f"   → FAIL TO REJECT H₀")
    print(f"   → No significant difference")
    print(f"   → Recommendation: Keep blue button (no proven benefit)")

# Calculate confidence interval for difference
ci_lower = (click_rate_b - click_rate_a) - 1.96 * se
ci_upper = (click_rate_b - click_rate_a) + 1.96 * se

print(f"\n 95% CONFIDENCE INTERVAL FOR DIFFERENCE:")
print(f"   [{ci_lower:.1%}, {ci_upper:.1%}]")
print(f"   → We're 95% confident the true difference is in this range")

# Lift calculation
lift = ((click_rate_b - click_rate_a) / click_rate_a) * 100
print(f"\n LIFT: {lift:.1f}%")
print(f"   → Green button shows {lift:.1f}% relative improvement")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Bar chart of click rates
categories = ['Blue Button\n(Control)', 'Green Button\n(Variant)']
rates = [click_rate_a, click_rate_b]
colors_chart = ['blue', 'green']

bars = axes[0, 0].bar(categories, rates, color=colors_chart, alpha=0.7, edgecolor='black', linewidth=2)
axes[0, 0].set_ylabel('Click Rate')
axes[0, 0].set_title('Click Rates Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylim(0, max(rates) * 1.2)

for i, (bar, rate) in enumerate(zip(bars, rates)):
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{rate:.1%}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# 2. Conversion funnel
stages = ['Visitors', 'Clicks']
blue_funnel = [n_a, clicks_a]
green_funnel = [n_b, clicks_b]

x = np.arange(len(stages))
width = 0.35

axes[0, 1].bar(x - width/2, blue_funnel, width, label='Blue Button', color='blue', alpha=0.7)
axes[0, 1].bar(x + width/2, green_funnel, width, label='Green Button', color='green', alpha=0.7)
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Conversion Funnel', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(stages)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Confidence interval
diff = click_rate_b - click_rate_a
axes[1, 0].errorbar([1], [diff], yerr=[[diff - ci_lower], [ci_upper - diff]],
                     fmt='o', markersize=12, capsize=15, capthick=3, color='purple', linewidth=3)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=2, label='No Difference')
axes[1, 0].fill_between([0.5, 1.5], ci_lower, ci_upper, alpha=0.2, color='purple')
axes[1, 0].set_xlim(0.5, 1.5)
axes[1, 0].set_ylabel('Difference in Click Rate')
axes[1, 0].set_title('95% Confidence Interval for Difference', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks([1])
axes[1, 0].set_xticklabels(['Green - Blue'])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Statistical power visualization
sample_sizes = np.arange(100, 3000, 100)
powers = []
for n in sample_sizes:
    # Simulate power calculation
    se_temp = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (2/n))
    z_crit = stats.norm.ppf(1 - alpha)
    z_alt = (click_rate_b - click_rate_a) / se_temp
    power = 1 - stats.norm.cdf(z_crit - z_alt)
    powers.append(power)

axes[1, 1].plot(sample_sizes, powers, linewidth=3, color='blue')
axes[1, 1].axhline(0.8, color='red', linestyle='--', linewidth=2, label='80% Power (Target)')
axes[1, 1].axvline(n_a, color='green', linestyle='--', linewidth=2, label=f'Current n={n_a}')
axes[1, 1].fill_between(sample_sizes, 0.8, powers, where=(np.array(powers) >= 0.8),
                         alpha=0.2, color='green')
axes[1, 1].set_xlabel('Sample Size per Group')
axes[1, 1].set_ylabel('Statistical Power')
axes[1, 1].set_title('Statistical Power vs Sample Size', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ab_test_button.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. EXAMPLE 2: NEW FEATURE IMPACT (CONTINUOUS METRIC)
# ============================================================================

print("3. A/B TEST: NEW FEATURE IMPACT ON TIME SPENT")


# Scenario: Testing if new recommendation feature increases time on site
np.random.seed(42)

# Group A: Without new feature (control)
time_control = np.random.normal(8.5, 2.5, 500)  # Mean 8.5 minutes, std 2.5

# Group B: With new feature (variant)
time_variant = np.random.normal(9.2, 2.3, 500)  # Mean 9.2 minutes, std 2.3

print(f"\n DATA SUMMARY:")
print(f"   Control Group (No Feature):")
print(f"      Sample Size: {len(time_control)}")
print(f"      Mean Time: {np.mean(time_control):.2f} minutes")
print(f"      Std Dev: {np.std(time_control, ddof=1):.2f} minutes")
print(f"\n   Variant Group (With Feature):")
print(f"      Sample Size: {len(time_variant)}")
print(f"      Mean Time: {np.mean(time_variant):.2f} minutes")
print(f"      Std Dev: {np.std(time_variant, ddof=1):.2f} minutes")
print(f"\n   Observed Difference: {np.mean(time_variant) - np.mean(time_control):.2f} minutes")

print(f"\nHYPOTHESIS SETUP:")
print(f"   H₀: μ_variant = μ_control (no effect)")
print(f"   H₁: μ_variant > μ_control (feature increases time)")
print(f"   Significance Level (α): {alpha}")

# Two-sample t-test
t_stat, p_val = stats.ttest_ind(time_variant, time_control, alternative='greater')

print(f"\nTEST RESULTS:")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_val:.6f}")
print(f"   Degrees of freedom: {len(time_control) + len(time_variant) - 2}")

print(f"\nDECISION:")
if p_val < alpha:
    print(f"   p-value ({p_val:.6f}) < α ({alpha})")
    print(f"   → REJECT H₀")
    print(f"   → New feature SIGNIFICANTLY INCREASES time on site")
    print(f"   → Recommendation: LAUNCH the feature!")
else:
    print(f"    p-value ({p_val:.6f}) ≥ α ({alpha})")
    print(f"   → FAIL TO REJECT H₀")
    print(f"   → No significant evidence of improvement")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(time_control)-1)*np.var(time_control, ddof=1) + 
                      (len(time_variant)-1)*np.var(time_variant, ddof=1)) / 
                     (len(time_control) + len(time_variant) - 2))
cohens_d = (np.mean(time_variant) - np.mean(time_control)) / pooled_std

print(f"\n EFFECT SIZE (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("   → Small effect")
elif abs(cohens_d) < 0.5:
    print("   → Medium effect")
else:
    print("   → Large effect")

# Business impact
avg_increase = np.mean(time_variant) - np.mean(time_control)
percent_increase = (avg_increase / np.mean(time_control)) * 100
print(f"\n💼 BUSINESS IMPACT:")
print(f"   Average Increase: {avg_increase:.2f} minutes ({percent_increase:.1f}%)")
print(f"   If 10,000 users/day: +{avg_increase * 10000:.0f} total minutes/day")
print(f"   Equivalent to: {(avg_increase * 10000) / 60:.0f} extra hours of engagement/day")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distributions
axes[0, 0].hist(time_control, bins=30, alpha=0.5, label='Control', color='blue', density=True)
axes[0, 0].hist(time_variant, bins=30, alpha=0.5, label='Variant', color='orange', density=True)
axes[0, 0].axvline(np.mean(time_control), color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(np.mean(time_variant), color='orange', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Time on Site (minutes)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution Comparison', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Box plots
data_df = pd.DataFrame({
    'Control': time_control,
    'Variant': time_variant
})
axes[0, 1].boxplot([time_control, time_variant], labels=['Control', 'Variant'],
                    patch_artist=True,
                    boxprops=dict(alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
axes[0, 1].set_ylabel('Time on Site (minutes)')
axes[0, 1].set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Mean comparison with confidence intervals
means = [np.mean(time_control), np.mean(time_variant)]
sems = [stats.sem(time_control), stats.sem(time_variant)]
ci_95 = [1.96 * sem for sem in sems]

axes[1, 0].bar(['Control', 'Variant'], means, yerr=ci_95, capsize=10,
                alpha=0.7, color=['blue', 'orange'], edgecolor='black', linewidth=2)
axes[1, 0].set_ylabel('Mean Time (minutes)')
axes[1, 0].set_title('Mean Comparison with 95% CI', fontsize=14, fontweight='bold')
for i, (mean, ci) in enumerate(zip(means, ci_95)):
    axes[1, 0].text(i, mean + ci + 0.3, f'{mean:.2f}±{ci:.2f}',
                    ha='center', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Cumulative distribution
axes[1, 1].hist(time_control, bins=50, cumulative=True, density=True,
                alpha=0.6, label='Control', color='blue', histtype='step', linewidth=2)
axes[1, 1].hist(time_variant, bins=50, cumulative=True, density=True,
                alpha=0.6, label='Variant', color='orange', histtype='step', linewidth=2)
axes[1, 1].set_xlabel('Time on Site (minutes)')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ab_test_feature.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. SAMPLE SIZE CALCULATOR
# ============================================================================


print("4. SAMPLE SIZE CALCULATION")

def calculate_sample_size(p1, p2, alpha=0.05, power=0.8):
    """
    Calculate required sample size for proportion test
    
    Args:
        p1: Baseline conversion rate
        p2: Expected conversion rate with treatment
        alpha: Significance level
        power: Statistical power (1 - beta)
    """
    from statsmodels.stats.power import zt_ind_solve_power
    
    effect_size = (p2 - p1) / np.sqrt(p1 * (1 - p1))
    n = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, 
                            power=power, alternative='larger')
    return int(np.ceil(n))

# Example calculation
baseline_rate = 0.12
target_rate = 0.14
required_n = calculate_sample_size(baseline_rate, target_rate)

print(f"\nSAMPLE SIZE CALCULATOR:")
print(f"   Baseline Rate: {baseline_rate:.1%}")
print(f"   Target Rate: {target_rate:.1%}")
print(f"   Minimum Detectable Effect: {(target_rate - baseline_rate):.1%}")
print(f"   Significance Level (α): {alpha}")
print(f"   Statistical Power: 80%")
print(f"\n   Required Sample Size per Group: {required_n:,}")
print(f"   Total Required Sample Size: {required_n * 2:,}")


print(" PART 3 COMPLETE: A/B Testing")
print("\nKey Takeaways:")
print("1. A/B tests compare two versions to find the better one")
print("2. Use proportion tests for conversion/click rates")
print("3. Use t-tests for continuous metrics (time, revenue, etc.)")
print("4. Always calculate sample size before running test")
print("5. Consider practical significance, not just statistical")
print("6. Calculate business impact of observed differences")
print("\nNext: Run Part 4 for Advanced Topics!")