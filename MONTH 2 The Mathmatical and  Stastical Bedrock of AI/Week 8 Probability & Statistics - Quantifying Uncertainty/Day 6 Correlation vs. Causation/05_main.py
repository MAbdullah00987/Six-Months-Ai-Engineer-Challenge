
#Integrated Statistical Analysis Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, t as t_dist
import sympy as sp
from statsmodels.stats import weightstats
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

# ============================================================================
# REAL-WORLD INTEGRATED PROJECT
# ============================================================================

print("=" * 90)
print(" " * 20 + "INTEGRATED STATISTICAL ANALYSIS PROJECT")
print(" " * 15 + "E-Commerce A/B Testing with Complete Statistical Framework")
print("=" * 90)

# ============================================================================
# SCENARIO SETUP
# ============================================================================

print("\nSCENARIO:")
print("-" * 90)
print("""
An e-commerce company wants to test a new checkout page design.
They run an A/B test with:
  • Control Group (A): Current design
  • Treatment Group (B): New design

METRICS TO ANALYZE:
1. Conversion Rate (proportion who complete purchase)
2. Average Order Value (continuous variable)
3. Time on Page (continuous variable)

STATISTICAL QUESTIONS:
✓ Is there a significant difference in conversion rates?
✓ Does the new design increase average order value?
✓ Are conversion rate and time on page correlated?
✓ What's the confidence interval for the difference?
""")

# ============================================================================
# DATA GENERATION
# ============================================================================

np.random.seed(42)

# Group A: Control
n_a = 500
conversion_a = np.random.binomial(1, 0.12, n_a)  # 12% conversion rate
order_value_a = np.random.normal(85, 25, n_a)   # $85 ± $25
time_on_page_a = np.random.exponential(45, n_a) # 45s average

# Group B: Treatment (slightly better)
n_b = 500
conversion_b = np.random.binomial(1, 0.15, n_b)  # 15% conversion rate
order_value_b = np.random.normal(92, 28, n_b)    # $92 ± $28
time_on_page_b = np.random.exponential(50, n_b)  # 50s average

# Create DataFrames
df_a = pd.DataFrame({
    'group': 'A',
    'converted': conversion_a,
    'order_value': order_value_a,
    'time_on_page': time_on_page_a
})

df_b = pd.DataFrame({
    'group': 'B',
    'converted': conversion_b,
    'order_value': order_value_b,
    'time_on_page': time_on_page_b
})

df = pd.concat([df_a, df_b], ignore_index=True)

print("\n" + "=" * 90)
print("DATA SUMMARY")
print("=" * 90)

summary_stats = df.groupby('group').agg({
    'converted': ['sum', 'mean', 'count'],
    'order_value': ['mean', 'std'],
    'time_on_page': ['mean', 'std']
}).round(3)

print("\nGroup Statistics:")
print(summary_stats)

# ============================================================================
# ANALYSIS 1: CONVERSION RATE (Z-TEST FOR PROPORTIONS)
# ============================================================================

print("\n" + "=" * 90)
print("ANALYSIS 1: CONVERSION RATE COMPARISON (Z-Test for Proportions)")
print("=" * 90)

# Calculate proportions
p_a = df_a['converted'].mean()
p_b = df_b['converted'].mean()
n_converted_a = df_a['converted'].sum()
n_converted_b = df_b['converted'].sum()

print(f"\nConversion Rates:")
print(f"  Group A: {n_converted_a}/{n_a} = {p_a:.4f} ({p_a*100:.2f}%)")
print(f"  Group B: {n_converted_b}/{n_b} = {p_b:.4f} ({p_b*100:.2f}%)")
print(f"  Difference: {(p_b - p_a)*100:.2f} percentage points")

# Z-test for proportions (using pooled proportion)
p_pooled = (n_converted_a + n_converted_b) / (n_a + n_b)
se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))
z_conv = (p_b - p_a) / se_pooled
p_value_conv = 2 * (1 - norm.cdf(abs(z_conv)))

print(f"\nZ-Test for Proportions:")
print(f"  Pooled proportion: p̂ = {p_pooled:.4f}")
print(f"  Standard Error: SE = {se_pooled:.6f}")
print(f"  Z-statistic: Z = {z_conv:.4f}")
print(f"  p-value: p = {p_value_conv:.4f}")

# Verify with statsmodels
count = np.array([n_converted_b, n_converted_a])
nobs = np.array([n_b, n_a])
z_sm, p_sm = proportions_ztest(count, nobs)

print(f"\nStatsmodels Verification:")
print(f"  Z = {z_sm:.4f}, p = {p_sm:.4f}")

# 95% Confidence Interval for difference
z_critical = 1.96
se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
ci_lower = (p_b - p_a) - z_critical * se_diff
ci_upper = (p_b - p_a) + z_critical * se_diff

print(f"\n95% Confidence Interval for Difference:")
print(f"  ({ci_lower:.4f}, {ci_upper:.4f})")
print(f"  ({ci_lower*100:.2f}%, {ci_upper*100:.2f}%)")

if p_value_conv < 0.05:
    print(f"\n✓ CONCLUSION: Conversion rate is SIGNIFICANTLY HIGHER in Group B")
else:
    print(f"\n✗ CONCLUSION: No significant difference in conversion rates")

# ============================================================================
# ANALYSIS 2: ORDER VALUE (TWO-SAMPLE T-TEST)
# ============================================================================

print("\n" + "=" * 90)
print("ANALYSIS 2: AVERAGE ORDER VALUE (Independent Two-Sample t-Test)")
print("=" * 90)

# Extract order values for converters only
orders_a = df_a[df_a['converted'] == 1]['order_value']
orders_b = df_b[df_b['converted'] == 1]['order_value']

print(f"\nOrder Value Statistics (Converters Only):")
print(f"  Group A: n={len(orders_a)}, x̄=${orders_a.mean():.2f}, s=${orders_a.std():.2f}")
print(f"  Group B: n={len(orders_b)}, x̄=${orders_b.mean():.2f}, s=${orders_b.std():.2f}")

# Test for equal variances
stat_levene, p_levene = stats.levene(orders_a, orders_b)
print(f"\nLevene's Test for Equal Variances:")
print(f"  W = {stat_levene:.4f}, p = {p_levene:.4f}")

# Perform t-test (Welch's t-test - doesn't assume equal variances)
t_stat, p_value_order = stats.ttest_ind(orders_a, orders_b, equal_var=False)

print(f"\nWelch's t-Test:")
print(f"  t-statistic: t = {t_stat:.4f}")
print(f"  p-value: p = {p_value_order:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((orders_a.std()**2 + orders_b.std()**2) / 2)
cohens_d = (orders_b.mean() - orders_a.mean()) / pooled_std

print(f"\nEffect Size (Cohen's d): {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    effect_interp = "negligible"
elif abs(cohens_d) < 0.5:
    effect_interp = "small"
elif abs(cohens_d) < 0.8:
    effect_interp = "medium"
else:
    effect_interp = "large"
print(f"  Interpretation: {effect_interp} effect")

if p_value_order < 0.05:
    print(f"\n✓ CONCLUSION: Order value is SIGNIFICANTLY DIFFERENT between groups")
else:
    print(f"\n✗ CONCLUSION: No significant difference in order values")

# ============================================================================
# ANALYSIS 3: CORRELATION ANALYSIS
# ============================================================================

print("\n" + "=" * 90)
print("ANALYSIS 3: CORRELATION BETWEEN TIME ON PAGE AND CONVERSION")
print("=" * 90)

# Correlation for each group
for group_name, group_df in [('Group A', df_a), ('Group B', df_b)]:
    r, p_corr = stats.pearsonr(group_df['time_on_page'], group_df['converted'])
    
    print(f"\n{group_name}:")
    print(f"  Pearson r = {r:.4f}")
    print(f"  p-value = {p_corr:.4f}")
    
    if p_corr < 0.05:
        if r > 0:
            print(f"  ✓ SIGNIFICANT POSITIVE correlation")
        else:
            print(f"  ✓ SIGNIFICANT NEGATIVE correlation")
    else:
        print(f"  ✗ No significant correlation")

# Spearman correlation (non-parametric)
rho_a, p_rho_a = stats.spearmanr(df_a['time_on_page'], df_a['converted'])
rho_b, p_rho_b = stats.spearmanr(df_b['time_on_page'], df_b['converted'])

print(f"\nSpearman Rank Correlation:")
print(f"  Group A: ρ = {rho_a:.4f}, p = {p_rho_a:.4f}")
print(f"  Group B: ρ = {rho_b:.4f}, p = {p_rho_b:.4f}")

# ============================================================================
# SYMBOLIC MATHEMATICS WITH SYMPY
# ============================================================================

print("\n" + "=" * 90)
print("SYMBOLIC MATHEMATICS: Deriving Test Statistics with SymPy")
print("=" * 90)

# Define symbols
x_bar, mu_0, sigma, n = sp.symbols('bar{x} mu_0 sigma n', real=True, positive=True)
s = sp.symbols('s', real=True, positive=True)

# Z-statistic formula
z_formula = (x_bar - mu_0) / (sigma / sp.sqrt(n))
print(f"\nZ-Statistic Formula:")
print(f"  Z = {sp.latex(z_formula)}")
sp.pprint(z_formula)

# Standard Error
se_formula = sigma / sp.sqrt(n)
print(f"\nStandard Error:")
print(f"  SE = {sp.latex(se_formula)}")
sp.pprint(se_formula)

# T-statistic formula
t_formula = (x_bar - mu_0) / (s / sp.sqrt(n))
print(f"\nT-Statistic Formula:")
print(f"  t = {sp.latex(t_formula)}")
sp.pprint(t_formula)

# Derive confidence interval
alpha_sym = sp.symbols('alpha', real=True, positive=True)
z_alpha = sp.symbols('z_{alpha/2}', real=True, positive=True)

ci_lower_sym = x_bar - z_alpha * (sigma / sp.sqrt(n))
ci_upper_sym = x_bar + z_alpha * (sigma / sp.sqrt(n))

print(f"\n95% Confidence Interval:")
print(f"  Lower: {sp.latex(ci_lower_sym)}")
print(f"  Upper: {sp.latex(ci_upper_sym)}")

# Substitute actual values
actual_values = {
    x_bar: df_b['order_value'].mean(),
    sigma: df_b['order_value'].std(),
    n: len(df_b),
    z_alpha: 1.96
}

ci_lower_val = float(ci_lower_sym.subs(actual_values))
ci_upper_val = float(ci_upper_sym.subs(actual_values))

print(f"\nSubstituting Group B order values:")
print(f"  95% CI: (${ci_lower_val:.2f}, ${ci_upper_val:.2f})")

# ============================================================================
# COMPREHENSIVE VISUALIZATION
# ============================================================================

fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

# Plot 1: Conversion Rates
ax1 = fig.add_subplot(gs[0, 0])
groups = ['Group A\n(Control)', 'Group B\n(Treatment)']
conversions = [p_a * 100, p_b * 100]
colors = ['lightcoral', 'lightgreen']
bars = ax1.bar(groups, conversions, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

for i, (bar, val) in enumerate(zip(bars, conversions)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.3,
            f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')

ax1.set_ylabel('Conversion Rate (%)', fontsize=11)
ax1.set_title(f'Conversion Rate Comparison\nZ={z_conv:.2f}, p={p_value_conv:.4f}', 
             fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(conversions) * 1.2)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Conversion Rate Distribution
ax2 = fig.add_subplot(gs[0, 1])
x_prop = np.linspace(0, 0.25, 1000)
y_a = norm.pdf(x_prop, p_a, np.sqrt(p_a*(1-p_a)/n_a))
y_b = norm.pdf(x_prop, p_b, np.sqrt(p_b*(1-p_b)/n_b))

ax2.plot(x_prop, y_a, 'r-', linewidth=2, label='Group A')
ax2.plot(x_prop, y_b, 'g-', linewidth=2, label='Group B')
ax2.axvline(p_a, color='red', linestyle='--', alpha=0.7)
ax2.axvline(p_b, color='green', linestyle='--', alpha=0.7)
ax2.fill_between(x_prop, 0, y_a, alpha=0.2, color='red')
ax2.fill_between(x_prop, 0, y_b, alpha=0.2, color='green')

ax2.set_xlabel('Conversion Rate', fontsize=11)
ax2.set_ylabel('Probability Density', fontsize=11)
ax2.set_title('Sampling Distributions', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Statistical Power Visualization
ax3 = fig.add_subplot(gs[0, 2])
x_power = np.linspace(-4, 4, 1000)
y_null = norm.pdf(x_power, 0, 1)
y_alt = norm.pdf(x_power, 2, 1)

ax3.plot(x_power, y_null, 'b-', linewidth=2, label='H₀ Distribution')
ax3.plot(x_power, y_alt, 'r-', linewidth=2, label='H₁ Distribution')

critical = 1.96
x_beta = x_power[x_power < critical]
ax3.fill_between(x_beta, 0, norm.pdf(x_beta, 2, 1), alpha=0.3, color='orange', label='β (Type II)')
x_power_area = x_power[x_power >= critical]
ax3.fill_between(x_power_area, 0, norm.pdf(x_power_area, 2, 1), alpha=0.3, color='green', label='Power (1-β)')

ax3.axvline(critical, color='red', linestyle='--', linewidth=2, label='Critical Value')
ax3.set_xlabel('Test Statistic', fontsize=11)
ax3.set_ylabel('Probability Density', fontsize=11)
ax3.set_title('Statistical Power Illustration', fontsize=12, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Order Value Boxplots
ax4 = fig.add_subplot(gs[1, 0])
bp = ax4.boxplot([orders_a, orders_b], labels=['Group A', 'Group B'], 
                 patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
    patch.set_facecolor(color)

ax4.set_ylabel('Order Value ($)', fontsize=11)
ax4.set_title(f'Order Value Distribution\nt={t_stat:.2f}, p={p_value_order:.4f}', 
             fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Order Value Histograms
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(orders_a, bins=20, alpha=0.6, color='coral', label='Group A', edgecolor='black')
ax5.hist(orders_b, bins=20, alpha=0.6, color='lightgreen', label='Group B', edgecolor='black')
ax5.axvline(orders_a.mean(), color='darkred', linestyle='--', linewidth=2)
ax5.axvline(orders_b.mean(), color='darkgreen', linestyle='--', linewidth=2)

ax5.set_xlabel('Order Value ($)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('Order Value Distributions', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Effect Size Visualization
ax6 = fig.add_subplot(gs[1, 2])
effect_sizes = [0, cohens_d]
labels = ['Null\n(No Effect)', f"Observed\n(d={cohens_d:.2f})"]
colors_effect = ['gray', 'orange']
bars_effect = ax6.bar(labels, effect_sizes, color=colors_effect, alpha=0.7, 
                      edgecolor='black', linewidth=2)

# Add reference lines for effect size interpretation
ax6.axhline(0.2, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Small')
ax6.axhline(0.5, color='yellow', linestyle=':', linewidth=1, alpha=0.5, label='Medium')
ax6.axhline(0.8, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Large')

ax6.set_ylabel("Cohen's d", fontsize=11)
ax6.set_title('Effect Size (Cohen\'s d)', fontsize=12, fontweight='bold')
ax6.legend(loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Time vs Conversion (Group A)
ax7 = fig.add_subplot(gs[2, 0])
scatter_a = ax7.scatter(df_a['time_on_page'], df_a['converted'], 
                       alpha=0.5, c=df_a['converted'], cmap='RdYlGn', s=30)
r_a, _ = stats.pearsonr(df_a['time_on_page'], df_a['converted'])
ax7.set_xlabel('Time on Page (seconds)', fontsize=11)
ax7.set_ylabel('Converted (0/1)', fontsize=11)
ax7.set_title(f'Group A: Time vs Conversion\nr={r_a:.3f}', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Time vs Conversion (Group B)
ax8 = fig.add_subplot(gs[2, 1])
scatter_b = ax8.scatter(df_b['time_on_page'], df_b['converted'], 
                       alpha=0.5, c=df_b['converted'], cmap='RdYlGn', s=30)
r_b, _ = stats.pearsonr(df_b['time_on_page'], df_b['converted'])
ax8.set_xlabel('Time on Page (seconds)', fontsize=11)
ax8.set_ylabel('Converted (0/1)', fontsize=11)
ax8.set_title(f'Group B: Time vs Conversion\nr={r_b:.3f}', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# Plot 9: Correlation Matrix
ax9 = fig.add_subplot(gs[2, 2])
corr_matrix = df[['converted', 'order_value', 'time_on_page']].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
           square=True, ax=ax9, cbar_kws={'label': 'Correlation'}, vmin=-1, vmax=1)
ax9.set_title('Correlation Matrix (All Data)', fontsize=12, fontweight='bold')

# Plot 10: Summary Statistics Table
ax10 = fig.add_subplot(gs[3, :])
ax10.axis('off')

summary_data = [
    ['Metric', 'Group A', 'Group B', 'Test', 'Statistic', 'p-value', 'Conclusion'],
    ['Conversion Rate', f'{p_a:.1%}', f'{p_b:.1%}', 'Z-test', f'{z_conv:.3f}', 
     f'{p_value_conv:.4f}', '✓ Significant' if p_value_conv < 0.05 else '✗ Not Sig'],
    ['Avg Order Value', f'${orders_a.mean():.2f}', f'${orders_b.mean():.2f}', 
     "Welch's t", f'{t_stat:.3f}', f'{p_value_order:.4f}', 
     '✓ Significant' if p_value_order < 0.05 else '✗ Not Sig'],
    ['Time on Page', f'{df_a["time_on_page"].mean():.1f}s', 
     f'{df_b["time_on_page"].mean():.1f}s', 'Correlation', f'r={r_b:.3f}', 
     f'{p_rho_b:.4f}', '✓ Correlated' if p_rho_b < 0.05 else '✗ Not Corr']
]

table = ax10.table(cellText=summary_data, cellLoc='center', loc='center',
                  colWidths=[0.18, 0.12, 0.12, 0.14, 0.12, 0.12, 0.14])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 3)

for i in range(len(summary_data)):
    for j in range(len(summary_data[0])):
        cell = table[(i, j)]
        if i == 0:
            cell.set_facecolor('#2E7D32')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor(['#E8F5E9', '#FFF9C4', '#FFE0B2'][i-1])
            if j == 6:  # Conclusion column
                if '✓' in summary_data[i][j]:
                    cell.set_facecolor('#C8E6C9')
                    cell.set_text_props(weight='bold', color='green')
                else:
                    cell.set_facecolor('#FFCDD2')
                    cell.set_text_props(color='red')

ax10.set_title('Statistical Test Results Summary', fontsize=14, fontweight='bold', pad=20)

plt.savefig('integrated_ab_testing_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: integrated_ab_testing_analysis.png")
plt.show()


# FINAL RECOMMENDATIONS


print("\n" + "=" * 90)
print("BUSINESS RECOMMENDATIONS")
print("=" * 90)

print(f"""
Based on comprehensive statistical analysis:

1. CONVERSION RATE:
   • Group B shows {(p_b-p_a)*100:.2f} percentage point improvement
   • This difference is {"STATISTICALLY SIGNIFICANT" if p_value_conv < 0.05 else "NOT significant"}
   • Recommendation: {"✓ IMPLEMENT new design" if p_value_conv < 0.05 else "✗ Keep current design"}

2. ORDER VALUE:
   • Group B average: ${orders_b.mean():.2f} vs Group A: ${orders_a.mean():.2f}
   • Difference: ${orders_b.mean() - orders_a.mean():.2f}
   • Effect size: {effect_interp} (Cohen's d = {cohens_d:.3f})
   • This is {"SIGNIFICANT" if p_value_order < 0.05 else "NOT significant"}

3. USER ENGAGEMENT:
   • Time on page correlates with conversion: r = {r_b:.3f}
   • Longer engagement → higher conversion probability
   
4. OVERALL VERDICT:
   {"✓ STRONG RECOMMENDATION: Launch new design (Group B)" if p_value_conv < 0.05 else "⚠ CAUTIOUS: Need more data or refinement"}
   
CONFIDENCE: {"95%" if min(p_value_conv, p_value_order) < 0.05 else "< 95% (inconclusive)"}
""")