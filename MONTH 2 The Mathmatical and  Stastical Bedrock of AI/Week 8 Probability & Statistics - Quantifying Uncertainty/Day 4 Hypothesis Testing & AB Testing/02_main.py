
#omplete Statistics - Part 2: Hypothesis Testing

"""
PART 2: HYPOTHESIS TESTING
Understanding Statistical Significance and Decision Making
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
# 1. HYPOTHESIS TESTING FUNDAMENTALS
# ============================================================================


print("1. HYPOTHESIS TESTING FUNDAMENTALS")


print("""
 CORE CONCEPTS:

1. NULL HYPOTHESIS (H₀):
   → The "status quo" or "no effect" hypothesis
   → What we assume is true until proven otherwise
   → Example: "The new drug has NO effect"

2. ALTERNATIVE HYPOTHESIS (H₁ or Hₐ):
   → What we're trying to prove
   → The "there IS an effect" hypothesis
   → Example: "The new drug DOES have an effect"

3. P-VALUE:
   → Probability of observing our data (or more extreme) if H₀ is true
   → Low p-value = Strong evidence against H₀
   → High p-value = Weak evidence against H₀

4. SIGNIFICANCE LEVEL (α):
   → Threshold for rejecting H₀
   → Commonly α = 0.05 (5%)
   → If p-value < α, we reject H₀

5. TEST STATISTIC:
   → A number calculated from sample data
   → Helps us decide between H₀ and H₁
   → Examples: t-statistic, z-statistic, F-statistic
""")

# ============================================================================
# 2. ONE-SAMPLE T-TEST
# ============================================================================

print("\n" + "=" * 80)
print("2. ONE-SAMPLE T-TEST")
print("=" * 80)
print("Question: Is the average score different from 80?")
print("=" * 80)

# Sample data: Student test scores
np.random.seed(42)
scores = np.array([78, 85, 82, 90, 88, 75, 92, 87, 83, 89, 91, 86, 84, 88, 90])

print(f"\nSample Scores: {scores}")
print(f"Sample Size (n): {len(scores)}")
print(f"Sample Mean: {np.mean(scores):.2f}")
print(f"Sample Std Dev: {np.std(scores, ddof=1):.2f}")

# Set up hypotheses
population_mean = 80
alpha = 0.05

print(f"\n HYPOTHESIS SETUP:")
print(f"   H₀ (Null): μ = {population_mean} (average score is 80)")
print(f"   H₁ (Alternative): μ ≠ {population_mean} (average score is NOT 80)")
print(f"   Significance Level (α): {alpha}")

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(scores, population_mean)

print(f"\nTEST RESULTS:")
print(f"   t-statistic: {t_statistic:.4f}")
print(f"   p-value: {p_value:.4f}")
print(f"   Degrees of freedom: {len(scores) - 1}")

# Decision
print(f"\n DECISION:")
if p_value < alpha:
    print(f"    p-value ({p_value:.4f}) < α ({alpha})")
    print(f"   → REJECT H₀")
    print(f"   → The average score IS significantly different from {population_mean}")
else:
    print(f"    p-value ({p_value:.4f}) ≥ α ({alpha})")
    print(f"   → FAIL TO REJECT H₀")
    print(f"   → No significant evidence that average differs from {population_mean}")

# Calculate confidence interval
confidence_level = 0.95
mean_score = np.mean(scores)
sem = stats.sem(scores)
ci = stats.t.interval(confidence_level, len(scores)-1, mean_score, sem)

print(f"\n95% CONFIDENCE INTERVAL:")
print(f"   [{ci[0]:.2f}, {ci[1]:.2f}]")
print(f"   → We're 95% confident the true mean lies in this range")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Distribution with population mean
axes[0, 0].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black', density=True)
axes[0, 0].axvline(mean_score, color='red', linestyle='--', linewidth=2, 
                    label=f'Sample Mean: {mean_score:.2f}')
axes[0, 0].axvline(population_mean, color='green', linestyle='--', linewidth=2,
                    label=f'Hypothesized Mean: {population_mean}')
axes[0, 0].set_xlabel('Scores')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Sample Distribution vs Hypothesized Mean')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. T-distribution with critical regions
df = len(scores) - 1
x = np.linspace(-4, 4, 1000)
y = stats.t.pdf(x, df)
critical_value = stats.t.ppf(1 - alpha/2, df)

axes[0, 1].plot(x, y, 'b-', linewidth=2, label='t-distribution')
axes[0, 1].axvline(t_statistic, color='red', linestyle='--', linewidth=2,
                    label=f't-stat: {t_statistic:.2f}')
axes[0, 1].axvline(critical_value, color='orange', linestyle=':', linewidth=2,
                    label=f'Critical: ±{critical_value:.2f}')
axes[0, 1].axvline(-critical_value, color='orange', linestyle=':', linewidth=2)

# Shade rejection regions
x_left = x[x < -critical_value]
x_right = x[x > critical_value]
axes[0, 1].fill_between(x_left, stats.t.pdf(x_left, df), alpha=0.3, color='red',
                         label='Rejection Region')
axes[0, 1].fill_between(x_right, stats.t.pdf(x_right, df), alpha=0.3, color='red')

axes[0, 1].set_xlabel('t-value')
axes[0, 1].set_ylabel('Probability Density')
axes[0, 1].set_title(f't-Distribution (df={df}) with Critical Regions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Confidence interval visualization
axes[1, 0].errorbar([1], [mean_score], yerr=[[mean_score-ci[0]], [ci[1]-mean_score]],
                     fmt='o', markersize=10, capsize=10, capthick=2, color='blue',
                     label=f'95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]')
axes[1, 0].axhline(population_mean, color='red', linestyle='--', linewidth=2,
                    label=f'H₀: μ = {population_mean}')
axes[1, 0].set_xlim(0.5, 1.5)
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('95% Confidence Interval')
axes[1, 0].set_xticks([1])
axes[1, 0].set_xticklabels(['Sample'])
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. P-value interpretation
p_values = np.linspace(0, 0.1, 100)
colors = ['red' if p < alpha else 'green' for p in p_values]
axes[1, 1].scatter(p_values, [1]*len(p_values), c=colors, s=50, alpha=0.6)
axes[1, 1].axvline(alpha, color='black', linestyle='--', linewidth=2, label=f'α = {alpha}')
axes[1, 1].axvline(p_value, color='blue', linestyle='-', linewidth=3,
                    label=f'Our p-value: {p_value:.4f}')
axes[1, 1].set_xlabel('p-value')
axes[1, 1].set_title('P-value vs Significance Level')
axes[1, 1].set_yticks([])
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='x')
axes[1, 1].text(0.025, 0.5, 'Reject H₀', fontsize=12, color='red', ha='center')
axes[1, 1].text(0.075, 0.5, 'Fail to Reject H₀', fontsize=12, color='green', ha='center')

plt.tight_layout()
plt.savefig('one_sample_ttest.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. TWO-SAMPLE T-TEST (INDEPENDENT)
# ============================================================================


print("3. TWO-SAMPLE T-TEST (INDEPENDENT)")
print("Question: Do two groups have different average scores?")


# Two groups: Method A vs Method B
np.random.seed(42)
method_a = np.array([78, 82, 85, 88, 90, 92, 87, 85, 89, 91])
method_b = np.array([72, 75, 78, 80, 82, 85, 79, 81, 84, 86])

print(f"\nMethod A (n={len(method_a)}): {method_a}")
print(f"   Mean: {np.mean(method_a):.2f}, Std: {np.std(method_a, ddof=1):.2f}")
print(f"\nMethod B (n={len(method_b)}): {method_b}")
print(f"   Mean: {np.mean(method_b):.2f}, Std: {np.std(method_b, ddof=1):.2f}")

print(f"\n HYPOTHESIS SETUP:")
print(f"   H₀: μₐ = μᵦ (no difference between methods)")
print(f"   H₁: μₐ ≠ μᵦ (methods produce different results)")
print(f"   Significance Level (α): {alpha}")

# Perform independent two-sample t-test
t_stat, p_val = stats.ttest_ind(method_a, method_b)

print(f"\nTEST RESULTS:")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_val:.4f}")

print(f"\n DECISION:")
if p_val < alpha:
    print(f"   p-value ({p_val:.4f}) < α ({alpha})")
    print(f"   → REJECT H₀")
    print(f"   → Methods produce SIGNIFICANTLY DIFFERENT results")
else:
    print(f"   p-value ({p_val:.4f}) ≥ α ({alpha})")
    print(f"   → FAIL TO REJECT H₀")
    print(f"   → No significant difference between methods")

# Effect size (Cohen's d)
cohens_d = (np.mean(method_a) - np.mean(method_b)) / np.sqrt(
    ((len(method_a)-1)*np.var(method_a, ddof=1) + (len(method_b)-1)*np.var(method_b, ddof=1)) /
    (len(method_a) + len(method_b) - 2)
)
print(f"\nEFFECT SIZE (Cohen's d): {cohens_d:.4f}")
print(f"   → Small: 0.2, Medium: 0.5, Large: 0.8")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plots comparison
data_df = pd.DataFrame({
    'Method A': method_a,
    'Method B': method_b
})
axes[0, 0].boxplot([method_a, method_b], labels=['Method A', 'Method B'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[0, 0].set_ylabel('Scores')
axes[0, 0].set_title('Comparison of Two Methods')
axes[0, 0].grid(True, alpha=0.3)

# 2. Distributions
axes[0, 1].hist(method_a, bins=8, alpha=0.5, label='Method A', color='blue', density=True)
axes[0, 1].hist(method_b, bins=8, alpha=0.5, label='Method B', color='red', density=True)
axes[0, 1].axvline(np.mean(method_a), color='blue', linestyle='--', linewidth=2)
axes[0, 1].axvline(np.mean(method_b), color='red', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Scores')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Distribution Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Mean comparison with error bars
means = [np.mean(method_a), np.mean(method_b)]
sems = [stats.sem(method_a), stats.sem(method_b)]
axes[1, 0].bar(['Method A', 'Method B'], means, yerr=sems, capsize=10,
                alpha=0.7, color=['blue', 'red'])
axes[1, 0].set_ylabel('Mean Score')
axes[1, 0].set_title('Mean Comparison with Standard Error')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Individual data points
axes[1, 1].scatter([1]*len(method_a), method_a, alpha=0.6, s=100, label='Method A', color='blue')
axes[1, 1].scatter([2]*len(method_b), method_b, alpha=0.6, s=100, label='Method B', color='red')
axes[1, 1].hlines(np.mean(method_a), 0.8, 1.2, colors='blue', linewidth=3)
axes[1, 1].hlines(np.mean(method_b), 1.8, 2.2, colors='red', linewidth=3)
axes[1, 1].set_xlim(0.5, 2.5)
axes[1, 1].set_xticks([1, 2])
axes[1, 1].set_xticklabels(['Method A', 'Method B'])
axes[1, 1].set_ylabel('Scores')
axes[1, 1].set_title('Individual Data Points with Means')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('two_sample_ttest.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. PAIRED T-TEST
# ============================================================================


print("4. PAIRED T-TEST")
print("Question: Did scores improve after training?")


# Before and after training (same students)
np.random.seed(42)
before = np.array([70, 75, 68, 82, 78, 72, 85, 80, 76, 79])
after = np.array([75, 82, 73, 88, 85, 78, 90, 86, 82, 84])

print(f"\nBefore Training (n={len(before)}): {before}")
print(f"   Mean: {np.mean(before):.2f}")
print(f"\nAfter Training (n={len(after)}): {after}")
print(f"   Mean: {np.mean(after):.2f}")

differences = after - before
print(f"\nDifferences (After - Before): {differences}")
print(f"   Mean Difference: {np.mean(differences):.2f}")

print(f"\n🎯 HYPOTHESIS SETUP:")
print(f"   H₀: μdiff = 0 (no improvement)")
print(f"   H₁: μdiff > 0 (improvement occurred)")
print(f"   Significance Level (α): {alpha}")

# Perform paired t-test
t_stat_paired, p_val_paired = stats.ttest_rel(after, before, alternative='greater')

print(f"\n TEST RESULTS:")
print(f"   t-statistic: {t_stat_paired:.4f}")
print(f"   p-value: {p_val_paired:.4f}")

print(f"\n DECISION:")
if p_val_paired < alpha:
    print(f"    p-value ({p_val_paired:.4f}) < α ({alpha})")
    print(f"   → REJECT H₀")
    print(f"   → Training SIGNIFICANTLY IMPROVED scores")
else:
    print(f"    p-value ({p_val_paired:.4f}) ≥ α ({alpha})")
    print(f"   → FAIL TO REJECT H₀")
    print(f"   → No significant improvement")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Before vs After
x_pos = np.arange(len(before))
axes[0, 0].plot(x_pos, before, 'o-', label='Before', color='red', linewidth=2, markersize=8)
axes[0, 0].plot(x_pos, after, 's-', label='After', color='green', linewidth=2, markersize=8)
for i in range(len(before)):
    axes[0, 0].plot([i, i], [before[i], after[i]], 'k:', alpha=0.3)
axes[0, 0].set_xlabel('Student')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Before vs After Training')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Differences histogram
axes[0, 1].hist(differences, bins=8, alpha=0.7, color='purple', edgecolor='black')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
axes[0, 1].axvline(np.mean(differences), color='green', linestyle='--', linewidth=2,
                    label=f'Mean Diff: {np.mean(differences):.2f}')
axes[0, 1].set_xlabel('Score Difference (After - Before)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Improvements')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Box plot comparison
axes[1, 0].boxplot([before, after], labels=['Before', 'After'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[1, 0].set_ylabel('Scores')
axes[1, 0].set_title('Before vs After Distribution')
axes[1, 0].grid(True, alpha=0.3)

# 4. Difference scatter
axes[1, 1].scatter(range(len(differences)), differences, s=100, alpha=0.6, color='purple')
axes[1, 1].axhline(0, color='red', linestyle='--', linewidth=2, label='No Change')
axes[1, 1].axhline(np.mean(differences), color='green', linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(differences):.2f}')
axes[1, 1].set_xlabel('Student')
axes[1, 1].set_ylabel('Score Improvement')
axes[1, 1].set_title('Individual Improvements')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('paired_ttest.png', dpi=150, bbox_inches='tight')
plt.show()


print("PART 2 COMPLETE: Hypothesis Testing")
print("\nKey Takeaways:")
print("1. Always start with H₀ (null) and H₁ (alternative)")
print("2. Calculate test statistic and p-value")
print("3. Compare p-value to α (usually 0.05)")
print("4. Make decision: Reject H₀ or Fail to Reject H₀")
print("5. One-sample: Compare to known value")
print("6. Two-sample: Compare two independent groups")
print("7. Paired: Compare before/after on same subjects")
print("\nNext: Run Part 3 for A/B Testing!")