
#Part 3. T-Tests: Comparing Means

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 14)

# ============================================================================
# T-TESTS: WHEN AND HOW TO USE THEM
# ============================================================================

print("=" * 80)
print("T-TESTS: Comparing Means When Population Variance is Unknown")
print("=" * 80)

print("""
THREE TYPES OF T-TESTS:

1. ONE-SAMPLE T-TEST
   • Compare sample mean to known value
   • Example: Is average height different from 170cm?
   • H₀: μ = μ₀
   
2. INDEPENDENT TWO-SAMPLE T-TEST
   • Compare means of two independent groups
   • Example: Do men and women have different salaries?
   • H₀: μ₁ = μ₂
   
3. PAIRED T-TEST
   • Compare means of matched pairs
   • Example: Blood pressure before vs after treatment
   • H₀: μ_diff = 0

ASSUMPTIONS:
✓ Data is approximately normally distributed (robust to violations if n>30)
✓ For two-sample: equal variances (or use Welch's t-test)
✓ For paired: differences are normally distributed
✓ Observations are independent
""")

# ============================================================================
# TYPE 1: ONE-SAMPLE T-TEST
# ============================================================================

print("\n" + "=" * 80)
print("TYPE 1: ONE-SAMPLE T-TEST")
print("=" * 80)

print("\nSCENARIO: Quality Control")
print("A factory claims light bulbs last 1000 hours on average.")
print("We test 25 bulbs to verify this claim.")

# Generate data
np.random.seed(42)
n1 = 25
bulb_lifetimes = np.random.normal(loc=980, scale=50, size=n1)  # Actually less than claimed
claimed_mean = 1000

print(f"\nSample Statistics:")
print(f"  n = {n1}")
print(f"  x̄ = {np.mean(bulb_lifetimes):.2f} hours")
print(f"  s = {np.std(bulb_lifetimes, ddof=1):.2f} hours")

# Manual calculation
sample_mean = np.mean(bulb_lifetimes)
sample_std = np.std(bulb_lifetimes, ddof=1)
se = sample_std / np.sqrt(n1)
t_stat = (sample_mean - claimed_mean) / se
df = n1 - 1
p_value = 2 * t_dist.cdf(t_stat, df)  # Two-tailed

print(f"\nHypotheses:")
print(f"  H₀: μ = {claimed_mean} (claim is correct)")
print(f"  H₁: μ ≠ {claimed_mean} (claim is wrong)")

print(f"\nManual Calculation:")
print(f"  t = (x̄ - μ₀) / (s/√n)")
print(f"  t = ({sample_mean:.2f} - {claimed_mean}) / ({sample_std:.2f}/√{n1})")
print(f"  t = {t_stat:.4f}")
print(f"  df = {df}")
print(f"  p-value = {p_value:.4f}")

# Using scipy
t_scipy, p_scipy = stats.ttest_1samp(bulb_lifetimes, claimed_mean)

print(f"\nSciPy Verification:")
print(f"  t-statistic = {t_scipy:.4f}")
print(f"  p-value = {p_scipy:.4f}")

alpha = 0.05
if p_scipy < alpha:
    print(f"\n✓ REJECT H₀: Bulbs do NOT last {claimed_mean} hours on average")
else:
    print(f"\n✗ FAIL TO REJECT H₀: No evidence bulbs differ from {claimed_mean} hours")

# ============================================================================
# TYPE 2: INDEPENDENT TWO-SAMPLE T-TEST
# ============================================================================

print("\n" + "=" * 80)
print("TYPE 2: INDEPENDENT TWO-SAMPLE T-TEST")
print("=" * 80)

print("\nSCENARIO: A/B Testing")
print("Testing two website designs to see which has higher conversion rate.")

# Generate data for two groups
np.random.seed(123)
n_a = 50
n_b = 50
conversion_a = np.random.normal(loc=5.2, scale=1.5, size=n_a)  # Design A
conversion_b = np.random.normal(loc=6.0, scale=1.6, size=n_b)  # Design B (better)

print(f"\nGroup Statistics:")
print(f"  Design A: n={n_a}, x̄={np.mean(conversion_a):.2f}%, s={np.std(conversion_a, ddof=1):.2f}%")
print(f"  Design B: n={n_b}, x̄={np.mean(conversion_b):.2f}%, s={np.std(conversion_b, ddof=1):.2f}%")

# Test for equal variances (Levene's test)
stat_levene, p_levene = stats.levene(conversion_a, conversion_b)
print(f"\nLevene's Test for Equal Variances:")
print(f"  p-value = {p_levene:.4f}")
if p_levene > 0.05:
    print(f"  ✓ Variances are equal (use standard t-test)")
else:
    print(f"  ✗ Variances are unequal (use Welch's t-test)")

# Standard t-test (assumes equal variances)
t_standard, p_standard = stats.ttest_ind(conversion_a, conversion_b)

# Welch's t-test (does not assume equal variances)
t_welch, p_welch = stats.ttest_ind(conversion_a, conversion_b, equal_var=False)

print(f"\nHypotheses:")
print(f"  H₀: μ_A = μ_B (no difference in conversion rates)")
print(f"  H₁: μ_A ≠ μ_B (designs have different conversion rates)")

print(f"\nStandard t-test (pooled variance):")
print(f"  t = {t_standard:.4f}, p = {p_standard:.4f}")

print(f"\nWelch's t-test (unequal variance):")
print(f"  t = {t_welch:.4f}, p = {p_welch:.4f}")

if p_welch < alpha:
    print(f"\n✓ REJECT H₀: Design B has significantly different conversion rate")
    effect = np.mean(conversion_b) - np.mean(conversion_a)
    print(f"  Effect size: {effect:.2f}% difference")
else:
    print(f"\n✗ FAIL TO REJECT H₀: No significant difference between designs")

# ============================================================================
# TYPE 3: PAIRED T-TEST
# ============================================================================

print("\n" + "=" * 80)
print("TYPE 3: PAIRED T-TEST")
print("=" * 80)

print("\nSCENARIO: Before-After Study")
print("Testing if a training program improves test scores.")
print("Same students take test before and after training.")

# Generate paired data
np.random.seed(456)
n_students = 30
before_scores = np.random.normal(loc=70, scale=10, size=n_students)
improvement = np.random.normal(loc=5, scale=3, size=n_students)  # Average +5 points
after_scores = before_scores + improvement

print(f"\nSample Statistics:")
print(f"  n = {n_students} students")
print(f"  Before: x̄={np.mean(before_scores):.2f}, s={np.std(before_scores, ddof=1):.2f}")
print(f"  After:  x̄={np.mean(after_scores):.2f}, s={np.std(after_scores, ddof=1):.2f}")

# Calculate differences
differences = after_scores - before_scores
mean_diff = np.mean(differences)
std_diff = np.std(differences, ddof=1)
se_diff = std_diff / np.sqrt(n_students)

print(f"  Difference: x̄={mean_diff:.2f}, s={std_diff:.2f}")

# Paired t-test
t_paired, p_paired = stats.ttest_rel(before_scores, after_scores)

print(f"\nHypotheses:")
print(f"  H₀: μ_diff = 0 (no improvement)")
print(f"  H₁: μ_diff > 0 (scores improved)")

print(f"\nPaired t-test:")
print(f"  t = {t_paired:.4f}")
print(f"  df = {n_students - 1}")
print(f"  p-value (two-tailed) = {p_paired:.4f}")
print(f"  p-value (one-tailed) = {p_paired/2:.4f}")

if p_paired/2 < alpha:
    print(f"\n✓ REJECT H₀: Training significantly improved scores")
    print(f"  Average improvement: {mean_diff:.2f} points")
else:
    print(f"\n✗ FAIL TO REJECT H₀: No significant improvement")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# ===== ONE-SAMPLE T-TEST VISUALIZATIONS =====
ax1 = fig.add_subplot(gs[0, :2])
ax1.hist(bulb_lifetimes, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(sample_mean, color='red', linestyle='--', linewidth=2, label=f'Sample Mean = {sample_mean:.1f}h')
ax1.axvline(claimed_mean, color='green', linestyle='--', linewidth=2, label=f'Claimed = {claimed_mean}h')
ax1.set_xlabel('Bulb Lifetime (hours)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title(f'One-Sample t-test: Bulb Lifetimes\nt={t_stat:.2f}, p={p_value:.4f}', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# t-distribution for one-sample test
ax2 = fig.add_subplot(gs[0, 2])
x = np.linspace(-4, 4, 1000)
y = t_dist.pdf(x, df)
ax2.plot(x, y, 'b-', linewidth=2, label='t-distribution')
critical_val = t_dist.ppf(1 - alpha/2, df)
x_reject_left = x[x <= -critical_val]
x_reject_right = x[x >= critical_val]
ax2.fill_between(x_reject_left, t_dist.pdf(x_reject_left, df), alpha=0.3, color='red', label='Rejection Region')
ax2.fill_between(x_reject_right, t_dist.pdf(x_reject_right, df), alpha=0.3, color='red')
ax2.axvline(t_stat, color='green', linestyle='--', linewidth=2, label=f't = {t_stat:.2f}')
ax2.set_xlabel('t-value', fontsize=10)
ax2.set_ylabel('Density', fontsize=10)
ax2.set_title(f't-distribution (df={df})', fontsize=11, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ===== TWO-SAMPLE T-TEST VISUALIZATIONS =====
ax3 = fig.add_subplot(gs[1, 0])
bp_data = [conversion_a, conversion_b]
bp = ax3.boxplot(bp_data, labels=['Design A', 'Design B'], patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
    patch.set_facecolor(color)
ax3.set_ylabel('Conversion Rate (%)', fontsize=11)
ax3.set_title(f'Two-Sample t-test: A/B Testing\nt={t_welch:.2f}, p={p_welch:.4f}', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

ax4 = fig.add_subplot(gs[1, 1])
ax4.hist(conversion_a, bins=15, alpha=0.6, color='coral', label='Design A', edgecolor='black')
ax4.hist(conversion_b, bins=15, alpha=0.6, color='lightgreen', label='Design B', edgecolor='black')
ax4.set_xlabel('Conversion Rate (%)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[1, 2])
means = [np.mean(conversion_a), np.mean(conversion_b)]
stds = [np.std(conversion_a, ddof=1), np.std(conversion_b, ddof=1)]
x_pos = [0, 1]
ax5.bar(x_pos, means, yerr=stds, color=['coral', 'lightgreen'], 
        alpha=0.7, capsize=10, edgecolor='black', linewidth=2)
ax5.set_xticks(x_pos)
ax5.set_xticklabels(['Design A', 'Design B'])
ax5.set_ylabel('Mean Conversion (%)', fontsize=11)
ax5.set_title('Mean ± SD Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# ===== PAIRED T-TEST VISUALIZATIONS =====
ax6 = fig.add_subplot(gs[2, 0])
for i in range(n_students):
    ax6.plot([0, 1], [before_scores[i], after_scores[i]], 'o-', 
            color='gray', alpha=0.3, markersize=4)
ax6.plot([0, 1], [np.mean(before_scores), np.mean(after_scores)], 
        'ro-', linewidth=3, markersize=10, label='Mean')
ax6.set_xticks([0, 1])
ax6.set_xticklabels(['Before', 'After'])
ax6.set_ylabel('Test Score', fontsize=11)
ax6.set_title(f'Paired t-test: Training Effect\nt={t_paired:.2f}, p={p_paired:.4f}', fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

ax7 = fig.add_subplot(gs[2, 1])
ax7.hist(differences, bins=15, alpha=0.7, color='purple', edgecolor='black')
ax7.axvline(mean_diff, color='red', linestyle='--', linewidth=2, label=f'Mean Diff = {mean_diff:.2f}')
ax7.axvline(0, color='green', linestyle='--', linewidth=2, label='H₀: diff = 0')
ax7.set_xlabel('Score Difference (After - Before)', fontsize=11)
ax7.set_ylabel('Frequency', fontsize=11)
ax7.set_title('Distribution of Differences', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

ax8 = fig.add_subplot(gs[2, 2])
bp_paired = ax8.boxplot([before_scores, after_scores], labels=['Before', 'After'], patch_artist=True)
for patch, color in zip(bp_paired['boxes'], ['lightyellow', 'lightblue']):
    patch.set_facecolor(color)
ax8.set_ylabel('Test Score', fontsize=11)
ax8.set_title('Before vs After Comparison', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3, axis='y')

# ===== COMPARISON TABLE =====
ax9 = fig.add_subplot(gs[3, :])
ax9.axis('off')

comparison_data = [
    ['Test Type', 'Use Case', 'H₀', 'Example', 'Key Formula'],
    ['One-Sample', 'Compare to known value', 'μ = μ₀', 'Avg height = 170cm?', 't = (x̄-μ₀)/(s/√n)'],
    ['Independent\nTwo-Sample', 'Compare 2 groups', 'μ₁ = μ₂', 'Men vs Women salary', 't = (x̄₁-x̄₂)/SE_pooled'],
    ['Paired', 'Before-After comparison', 'μ_diff = 0', 'Before vs After treatment', 't = x̄_diff/(s_diff/√n)']
]

table = ax9.table(cellText=comparison_data, cellLoc='left', loc='center',
                 colWidths=[0.15, 0.25, 0.15, 0.25, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

for i in range(len(comparison_data)):
    for j in range(len(comparison_data[0])):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor(['#E3F2FD', '#FFF9C4', '#FFE0B2'][i-1])

ax9.set_title('T-Test Comparison Table', fontsize=14, fontweight='bold', pad=20)

plt.savefig('t_tests_complete.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: t_tests_complete.png")
plt.show()

print("\n" + "=" * 80)
print("KEY DECISION TREE FOR T-TESTS")
print("=" * 80)
print("""
        Are you comparing to a known value?
                    │
        ┌───────────┴───────────┐
       YES                      NO
        │                       │
   ONE-SAMPLE              Comparing 2 groups?
     t-test                     │
                    ┌───────────┴───────────┐
                Same subjects?         Different subjects?
                    │                       │
                PAIRED t-test      INDEPENDENT t-test
                                   (Check equal variances!)
""")