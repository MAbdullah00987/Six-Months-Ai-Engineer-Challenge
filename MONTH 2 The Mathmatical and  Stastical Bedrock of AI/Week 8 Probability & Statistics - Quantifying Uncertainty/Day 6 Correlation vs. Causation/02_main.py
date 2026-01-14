
#Part 2. Hypothesis Testing Framework

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t as t_dist
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# ============================================================================
# PART 1: HYPOTHESIS TESTING FRAMEWORK
# ============================================================================

print("=" * 80)
print("HYPOTHESIS TESTING FRAMEWORK: The Scientific Method in Statistics")
print("=" * 80)

print("""
THE 7-STEP HYPOTHESIS TESTING PROCESS:

1. STATE THE HYPOTHESES
   H₀ (Null Hypothesis):      No effect, no difference, status quo
   H₁ (Alternative Hypothesis): The claim you want to prove
   
2. CHOOSE SIGNIFICANCE LEVEL (α)
   Common: α = 0.05 (5% chance of Type I error)
   
3. SELECT THE TEST
   Based on: data type, sample size, assumptions
   
4. CALCULATE TEST STATISTIC
   Standardized measure of difference
   
5. FIND P-VALUE
   Probability of observing data this extreme if H₀ is true
   
6. MAKE DECISION
   If p-value < α: Reject H₀ (statistically significant)
   If p-value ≥ α: Fail to reject H₀ (not significant)
   
7. INTERPRET IN CONTEXT
   Statistical significance ≠ Practical significance
""")

# EXAMPLE 1: Drug Effectiveness Test

print("\n" + "=" * 80)
print("EXAMPLE 1: Testing New Drug Effectiveness")
print("=" * 80)

# Scenario
print("\nSCENARIO:")
print("A pharmaceutical company claims their new drug reduces")
print("blood pressure by more than 10 mmHg on average.")
print("We test 30 patients and measure the reduction.")

# Generate sample data
np.random.seed(42)
n = 30
sample_reductions = np.random.normal(loc=12, scale=5, size=n)  # True mean = 12
claimed_reduction = 10

print("\n1. HYPOTHESES:")
print(f"   H₀: μ ≤ 10 (drug is NOT better than claimed)")
print(f"   H₁: μ > 10 (drug IS better than claimed)")
print(f"   This is a ONE-TAILED test (we only care if it's better)")

print("\n2. SIGNIFICANCE LEVEL:")
alpha = 0.05
print(f"   α = {alpha} (5% risk of false positive)")

print("\n3. SAMPLE STATISTICS:")
sample_mean = np.mean(sample_reductions)
sample_std = np.std(sample_reductions, ddof=1)
sample_se = sample_std / np.sqrt(n)

print(f"   Sample mean (x̄) = {sample_mean:.2f} mmHg")
print(f"   Sample std (s) = {sample_std:.2f} mmHg")
print(f"   Standard error (SE) = {sample_se:.2f} mmHg")

print("\n4. TEST STATISTIC (t-statistic):")
# t = (x̄ - μ₀) / (s / √n)
t_statistic = (sample_mean - claimed_reduction) / sample_se
df = n - 1  # degrees of freedom

print(f"   t = (x̄ - μ₀) / SE")
print(f"   t = ({sample_mean:.2f} - {claimed_reduction}) / {sample_se:.2f}")
print(f"   t = {t_statistic:.4f}")
print(f"   Degrees of freedom = {df}")

print("\n5. P-VALUE:")
# For one-tailed test
p_value = 1 - t_dist.cdf(t_statistic, df)
print(f"   p-value = {p_value:.4f}")
print(f"   Interpretation: If H₀ is true, probability of seeing")
print(f"   data this extreme = {p_value*100:.2f}%")

print("\n6. DECISION:")
if p_value < alpha:
    print(f"   ✓ REJECT H₀ (p = {p_value:.4f} < α = {alpha})")
    print(f"   The drug IS significantly better than claimed!")
else:
    print(f"   ✗ FAIL TO REJECT H₀ (p = {p_value:.4f} ≥ α = {alpha})")
    print(f"   Insufficient evidence that drug is better.")

# Verification with scipy
t_stat_scipy, p_value_scipy = stats.ttest_1samp(sample_reductions, claimed_reduction)
p_value_one_tail = p_value_scipy / 2  # Convert to one-tailed

print("\n7. VERIFICATION (SciPy):")
print(f"   t-statistic = {t_stat_scipy:.4f}")
print(f"   p-value (one-tailed) = {p_value_one_tail:.4f}")

# ============================================================================
# VISUALIZATION: Understanding P-Values
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Sample distribution
axes[0, 0].hist(sample_reductions, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0, 0].axvline(sample_mean, color='red', linestyle='--', linewidth=2, label=f'Sample Mean = {sample_mean:.2f}')
axes[0, 0].axvline(claimed_reduction, color='green', linestyle='--', linewidth=2, label=f'H₀ Mean = {claimed_reduction}')
axes[0, 0].set_xlabel('Blood Pressure Reduction (mmHg)', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('Sample Data Distribution', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: t-distribution with critical region
x = np.linspace(-4, 4, 1000)
y = t_dist.pdf(x, df)

axes[0, 1].plot(x, y, 'b-', linewidth=2, label='t-distribution')
critical_value = t_dist.ppf(1 - alpha, df)
x_critical = x[x >= critical_value]
y_critical = t_dist.pdf(x_critical, df)
axes[0, 1].fill_between(x_critical, y_critical, alpha=0.3, color='red', label=f'Rejection Region (α={alpha})')
axes[0, 1].axvline(t_statistic, color='green', linestyle='--', linewidth=2, label=f't-stat = {t_statistic:.2f}')
axes[0, 1].axvline(critical_value, color='red', linestyle=':', linewidth=2, label=f'Critical value = {critical_value:.2f}')
axes[0, 1].set_xlabel('t-value', fontsize=11)
axes[0, 1].set_ylabel('Probability Density', fontsize=11)
axes[0, 1].set_title(f't-Distribution (df={df})\nOne-Tailed Test', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)


# PART 2: P-VALUES AND SIGNIFICANCE LEVELS


print("\n" + "=" * 80)
print("UNDERSTANDING P-VALUES")
print("=" * 80)

print("""
WHAT IS A P-VALUE?

Definition: The probability of observing data as extreme as (or more 
extreme than) what we observed, ASSUMING the null hypothesis is true.

Common Misconceptions:
✗ p-value is NOT the probability that H₀ is true
✗ p-value is NOT the probability we made a mistake
✓ p-value IS how surprising our data is under H₀

Interpretation:
• p < 0.001  →  Very strong evidence against H₀
• p < 0.01   →  Strong evidence against H₀
• p < 0.05   →  Moderate evidence against H₀
• p < 0.10   →  Weak evidence against H₀
• p ≥ 0.10   →  Little to no evidence against H₀

SIGNIFICANCE LEVELS (α):
α = 0.05 (most common): 5% false positive rate
α = 0.01 (stringent): 1% false positive rate
α = 0.10 (lenient): 10% false positive rate
""")

# Demonstrate different p-values
scenarios = {
    'Very Strong (p<0.001)': 4.0,
    'Strong (p<0.01)': 3.0,
    'Moderate (p<0.05)': 2.0,
    'Weak (p<0.10)': 1.7,
    'Not Significant': 1.0
}

print("\nP-VALUE SCENARIOS:")
for scenario, t_val in scenarios.items():
    p = 1 - t_dist.cdf(t_val, df)
    significance = "✓ Significant" if p < alpha else "✗ Not Significant"
    print(f"{scenario:25s}: t={t_val:.1f}, p={p:.4f} {significance}")

# Plot 3: P-value visualization
x = np.linspace(-4, 4, 1000)
y = t_dist.pdf(x, df)

axes[1, 0].plot(x, y, 'b-', linewidth=2)

# Shade different p-value regions
for t_val, color in [(4.0, 'darkred'), (3.0, 'red'), (2.0, 'orange'), (1.7, 'yellow')]:
    x_shade = x[x >= t_val]
    y_shade = t_dist.pdf(x_shade, df)
    axes[1, 0].fill_between(x_shade, y_shade, alpha=0.2, color=color)

axes[1, 0].set_xlabel('t-value', fontsize=11)
axes[1, 0].set_ylabel('Probability Density', fontsize=11)
axes[1, 0].set_title('P-Value Regions\n(Smaller area = Smaller p-value = Stronger evidence)', 
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)


# PART 3: TYPE I AND TYPE II ERRORS


print("\n" + "=" * 80)
print("TYPE I AND TYPE II ERRORS")
print("=" * 80)

print("""
THE TRUTH TABLE OF HYPOTHESIS TESTING:

                    │  H₀ is TRUE    │  H₀ is FALSE
────────────────────┼────────────────┼─────────────────
Reject H₀           │  TYPE I ERROR  │  ✓ Correct
(Say there IS       │  (False Pos)   │  (True Positive)
an effect)          │  Prob = α      │  Prob = Power
────────────────────┼────────────────┼─────────────────
Fail to Reject H₀   │  ✓ Correct     │  TYPE II ERROR
(Say there is NO    │  (True Neg)    │  (False Negative)
effect)             │  Prob = 1-α    │  Prob = β
────────────────────┴────────────────┴─────────────────

TYPE I ERROR (α):
• Rejecting H₀ when it's actually true
• "False Alarm" / "False Positive"
• Example: Concluding drug works when it doesn't
• Controlled by significance level α

TYPE II ERROR (β):
• Failing to reject H₀ when it's actually false
• "Missed Detection" / "False Negative"
• Example: Concluding drug doesn't work when it does
• Related to statistical power (1 - β)

STATISTICAL POWER (1 - β):
• Probability of correctly rejecting false H₀
• Influenced by: sample size, effect size, α
• Typical goal: Power ≥ 0.80 (80%)
""")

# Visualization of errors
truth_reality = ['H₀ True', 'H₀ False']
decisions = ['Fail to Reject H₀', 'Reject H₀']
outcomes = [
    ['Correct\n(True Negative)\nProb = 1-α', 'Type II Error\n(False Negative)\nProb = β'],
    ['Type I Error\n(False Positive)\nProb = α', 'Correct\n(True Positive)\nProb = 1-β']
]

colors = [
    ['lightgreen', 'lightcoral'],
    ['lightcoral', 'lightgreen']
]

ax = axes[1, 1]
ax.axis('off')
ax.set_xlim(0, 3)
ax.set_ylim(0, 3)

# Draw table
for i, decision in enumerate(decisions):
    for j, reality in enumerate(truth_reality):
        x, y = j + 0.5, 2 - i
        rect = plt.Rectangle((j, 1.5 - i), 1, 0.8, 
                             facecolor=colors[i][j], 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, outcomes[i][j], ha='center', va='center', 
               fontsize=9, fontweight='bold')

# Labels
ax.text(1.5, 2.8, 'Reality (Unknown Truth)', ha='center', fontsize=12, fontweight='bold')
ax.text(0.5, 2.5, 'H₀ True', ha='center', fontsize=10, fontweight='bold')
ax.text(1.5, 2.5, 'H₀ False', ha='center', fontsize=10, fontweight='bold')
ax.text(-0.3, 1.9, 'Reject H₀', rotation=90, va='center', fontsize=10, fontweight='bold')
ax.text(-0.3, 1.1, 'Fail to\nReject H₀', rotation=90, va='center', fontsize=10, fontweight='bold')
ax.text(-0.6, 1.5, 'Our Decision', rotation=90, va='center', fontsize=12, fontweight='bold')

ax.set_title('Decision Table: Errors in Hypothesis Testing', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('hypothesis_testing_framework.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: hypothesis_testing_framework.png")
plt.show()

print("KEY TAKEAWAYS FOR STRONGER LOGIC")


print("""
1. ALWAYS state hypotheses BEFORE seeing data
2. Choose α based on cost of Type I vs Type II error
3. Smaller p-value = Stronger evidence (but not proof!)
4. Statistical significance ≠ Practical importance
5. Never "accept" H₀, only "fail to reject" it
6. Report effect sizes, not just p-values
7. Consider statistical power when planning studies
""")