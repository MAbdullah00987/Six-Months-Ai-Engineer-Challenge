
#Project: Complete End-to-End A/B Test Project

"""
BONUS: COMPLETE END-TO-END A/B TEST PROJECT
Real-World Scenario: E-commerce Checkout Redesign
Integrating ALL concepts learned
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.power import zt_ind_solve_power
from statsmodels.stats.proportion import proportions_ztest
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("=" * 80)
print(" A/B TEST PROJECT: E-COMMERCE CHECKOUT REDESIGN")
print("=" * 80)

print("""
 SCENARIO:
An e-commerce company wants to test a new checkout page design.
They believe the new design will increase conversion rates.

 BUSINESS QUESTION:
Does the new checkout design increase purchase completion rate?

 METRICS:
   • Primary: Conversion Rate (% who complete purchase)
   • Secondary: Average Order Value, Time to Purchase

 EXPERIMENTAL SETUP:
   • Control (A): Current checkout page
   • Variant (B): New checkout design
   • Random 50/50 split of users
   • Run for 2 weeks
""")

# ============================================================================
# PHASE 1: PLANNING - SAMPLE SIZE CALCULATION
# ============================================================================


print("PHASE 1: PLANNING & SAMPLE SIZE CALCULATION")


# Historical data
baseline_rate = 0.15  # Current conversion rate: 15%
target_rate = 0.17    # Target conversion rate: 17%
mde = target_rate - baseline_rate  # Minimum Detectable Effect

alpha = 0.05  # 5% significance level
power = 0.80  # 80% power

print(f"\n PLANNING PARAMETERS:")
print(f"   Baseline Conversion Rate: {baseline_rate:.1%}")
print(f"   Target Conversion Rate: {target_rate:.1%}")
print(f"   Minimum Detectable Effect: {mde:.1%} ({(mde/baseline_rate)*100:.1f}% relative lift)")
print(f"   Significance Level (α): {alpha}")
print(f"   Statistical Power: {power:.0%}")

# Calculate required sample size
effect_size = (target_rate - baseline_rate) / np.sqrt(baseline_rate * (1 - baseline_rate))
required_n = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, 
                                 power=power, alternative='larger')
required_n = int(np.ceil(required_n))

print(f"\n SAMPLE SIZE CALCULATION:")
print(f"   Effect Size: {effect_size:.4f}")
print(f"   Required Sample Size per Group: {required_n:,}")
print(f"   Total Sample Size Needed: {required_n * 2:,}")

# Estimate timeline
daily_visitors = 5000
days_needed = np.ceil((required_n * 2) / daily_visitors)

print(f"\n⏱️ TIMELINE ESTIMATE:")
print(f"   Daily Visitors: {daily_visitors:,}")
print(f"   Estimated Days to Complete: {int(days_needed)}")
print(f"   Recommendation: Run test for {int(days_needed)} days minimum")

# ============================================================================
# PHASE 2: DATA COLLECTION (SIMULATED)
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 2: DATA COLLECTION")
print("=" * 80)

# Simulate data for the experiment
np.random.seed(42)

# Actual sample sizes collected
n_control = 3500
n_variant = 3500

# Simulate conversions (variant actually performs better)
# Control: 15% baseline
# Variant: 17.5% (better than expected!)
conversions_control = np.random.binomial(1, 0.15, n_control)
conversions_variant = np.random.binomial(1, 0.175, n_variant)

# Simulate order values (secondary metric)
# Control: $50 average
# Variant: $52 average (slight increase)
order_values_control = np.random.gamma(10, 5, n_control)  # Shape, scale
order_values_variant = np.random.gamma(10, 5.2, n_variant)

# Simulate time to purchase (in seconds)
time_control = np.random.lognormal(4.5, 0.5, n_control)
time_variant = np.random.lognormal(4.3, 0.5, n_variant)  # Slightly faster

# Create DataFrames
df_control = pd.DataFrame({
    'group': 'Control',
    'converted': conversions_control,
    'order_value': order_values_control * conversions_control,  # Only if converted
    'time_seconds': time_control
})

df_variant = pd.DataFrame({
    'group': 'Variant',
    'converted': conversions_variant,
    'order_value': order_values_variant * conversions_variant,
    'time_seconds': time_variant
})

df = pd.concat([df_control, df_variant], ignore_index=True)

print(f"\n DATA COLLECTED:")
print(f"   Control Group: {n_control:,} users")
print(f"   Variant Group: {n_variant:,} users")
print(f"   Total Users: {len(df):,}")
print(f"\n   Data Preview:")
print(df.head(10))

# ============================================================================
# PHASE 3: DESCRIPTIVE STATISTICS
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 3: DESCRIPTIVE STATISTICS")
print("=" * 80)

# Primary metric: Conversion Rate
conv_control = df_control['converted'].sum()
conv_variant = df_variant['converted'].sum()
rate_control = conv_control / n_control
rate_variant = conv_variant / n_variant

print(f"\n PRIMARY METRIC: CONVERSION RATE")
print(f"   Control:  {conv_control:>4} / {n_control:>5} = {rate_control:.2%}")
print(f"   Variant:  {conv_variant:>4} / {n_variant:>5} = {rate_variant:.2%}")
print(f"   Absolute Lift: {(rate_variant - rate_control):.2%}")
print(f"   Relative Lift: {((rate_variant - rate_control) / rate_control * 100):.1f}%")

# Secondary metrics
print(f"\n SECONDARY METRICS:")

# Average Order Value (for conversions only)
aov_control = df_control[df_control['converted']==1]['order_value'].mean()
aov_variant = df_variant[df_variant['converted']==1]['order_value'].mean()
print(f"\n   Average Order Value:")
print(f"      Control: ${aov_control:>6.2f}")
print(f"      Variant: ${aov_variant:>6.2f}")
print(f"      Difference: ${aov_variant - aov_control:>6.2f} ({((aov_variant/aov_control-1)*100):.1f}%)")

# Time to Purchase
time_control_mean = df_control['time_seconds'].mean()
time_variant_mean = df_variant['time_seconds'].mean()
print(f"\n   Average Time on Checkout:")
print(f"      Control: {time_control_mean:>6.1f} seconds")
print(f"      Variant: {time_variant_mean:>6.1f} seconds")
print(f"      Difference: {time_variant_mean - time_control_mean:>6.1f} seconds ({((time_variant_mean/time_control_mean-1)*100):.1f}%)")

# Revenue per visitor
rpv_control = (df_control['order_value'].sum() / n_control)
rpv_variant = (df_variant['order_value'].sum() / n_variant)
print(f"\n   Revenue per Visitor:")
print(f"      Control: ${rpv_control:>6.2f}")
print(f"      Variant: ${rpv_variant:>6.2f}")
print(f"      Lift: ${rpv_variant - rpv_control:>6.2f} ({((rpv_variant/rpv_control-1)*100):.1f}%)")

# ============================================================================
# PHASE 4: HYPOTHESIS TESTING
# ============================================================================

print("\n" + "=" * 80)
print("PHASE 4: HYPOTHESIS TESTING")
print("=" * 80)

# Test 1: Conversion Rate (Proportion Test)
print(f"\n TEST 1: CONVERSION RATE")
print("-" * 80)

print(f"\n   H₀: p_variant = p_control (no difference)")
print(f"   H₁: p_variant > p_control (variant is better)")
print(f"   Significance Level: {alpha}")

# Two-proportion z-test
count = np.array([conv_variant, conv_control])
nobs = np.array([n_variant, n_control])
stat, pval = proportions_ztest(count, nobs, alternative='larger')

print(f"\n   Test Statistic (z): {stat:.4f}")
print(f"   P-value: {pval:.6f}")

if pval < alpha:
    print(f"\n    RESULT: SIGNIFICANT!")
    print(f"   Decision: REJECT H₀")
    print(f"   Interpretation: Variant has significantly higher conversion rate")
else:
    print(f"\n    RESULT: NOT SIGNIFICANT")
    print(f"   Decision: FAIL TO REJECT H₀")
    print(f"   Interpretation: No significant difference detected")

# Calculate confidence interval
se = np.sqrt(rate_variant*(1-rate_variant)/n_variant + rate_control*(1-rate_control)/n_control)
ci_lower = (rate_variant - rate_control) - 1.96 * se
ci_upper = (rate_variant - rate_control) + 1.96 * se

print(f"\n   95% CI for Difference: [{ci_lower:.2%}, {ci_upper:.2%}]")

# Test 2: Average Order Value (T-test)
print(f"\n\n TEST 2: AVERAGE ORDER VALUE")
print("-" * 80)

# Only compare converted users
aov_control_data = df_control[df_control['converted']==1]['order_value']
aov_variant_data = df_variant[df_variant['converted']==1]['order_value']

t_stat_aov, p_val_aov = stats.ttest_ind(aov_variant_data, aov_control_data, 
                                         alternative='greater')

print(f"\n   H₀: μ_variant = μ_control")
print(f"   H₁: μ_variant > μ_control")
print(f"\n   t-statistic: {t_stat_aov:.4f}")
print(f"   p-value: {p_val_aov:.4f}")

if p_val_aov < alpha:
    print(f"\n    RESULT: SIGNIFICANT!")
    print(f"   Variant has significantly higher order value")
else:
    print(f"\n   RESULT: NOT SIGNIFICANT")
    print(f"   No significant difference in order value")

# Test 3: Time to Purchase (T-test)
print(f"\n\n TEST 3: TIME TO PURCHASE")
print("-" * 80)

t_stat_time, p_val_time = stats.ttest_ind(time_variant, time_control, 
                                           alternative='less')

print(f"\n   H₀: μ_variant = μ_control")
print(f"   H₁: μ_variant < μ_control (variant is faster)")
print(f"\n   t-statistic: {t_stat_time:.4f}")
print(f"   p-value: {p_val_time:.4f}")

if p_val_time < alpha:
    print(f"\n    RESULT: SIGNIFICANT!")
    print(f"   Variant has significantly faster checkout time")
else:
    print(f"\n    RESULT: NOT SIGNIFICANT")
    print(f"   No significant difference in checkout time")

# ============================================================================
# PHASE 5: VISUALIZATIONS
# ============================================================================

print("PHASE 5: CREATING VISUALIZATIONS")


fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Conversion Rates Bar Chart
ax1 = fig.add_subplot(gs[0, 0])
groups = ['Control', 'Variant']
rates = [rate_control, rate_variant]
colors_bar = ['#3498db', '#2ecc71']
bars = ax1.bar(groups, rates, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Conversion Rate', fontsize=11, fontweight='bold')
ax1.set_title('Conversion Rate Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim(0, max(rates) * 1.2)
for bar, rate in zip(bars, rates):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{rate:.2%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# 2. Funnel Visualization
ax2 = fig.add_subplot(gs[0, 1])
funnel_data = pd.DataFrame({
    'Stage': ['Visitors', 'Conversions'],
    'Control': [n_control, conv_control],
    'Variant': [n_variant, conv_variant]
})
x = np.arange(len(funnel_data['Stage']))
width = 0.35
ax2.bar(x - width/2, funnel_data['Control'], width, label='Control', 
        color='#3498db', alpha=0.8)
ax2.bar(x + width/2, funnel_data['Variant'], width, label='Variant', 
        color='#2ecc71', alpha=0.8)
ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
ax2.set_title('Conversion Funnel', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(funnel_data['Stage'])
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Statistical Test Result
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')
result_text = f"""
STATISTICAL TEST RESULTS

Conversion Rate Test:
├─ z-statistic: {stat:.4f}
├─ p-value: {pval:.6f}
└─ Result: {'SIGNIFICANT ' if pval < alpha else 'NOT SIGNIFICANT ❌'}

95% Confidence Interval:
[{ci_lower:.2%}, {ci_upper:.2%}]

Absolute Lift:
{(rate_variant - rate_control):.2%}

Relative Lift:
{((rate_variant - rate_control) / rate_control * 100):.1f}%
"""
ax3.text(0.1, 0.5, result_text, fontsize=10, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='wheat', alpha=0.3))

# 4. Order Value Distribution
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(aov_control_data, bins=30, alpha=0.6, label='Control', color='#3498db', density=True)
ax4.hist(aov_variant_data, bins=30, alpha=0.6, label='Variant', color='#2ecc71', density=True)
ax4.axvline(aov_control, color='#3498db', linestyle='--', linewidth=2)
ax4.axvline(aov_variant, color='#2ecc71', linestyle='--', linewidth=2)
ax4.set_xlabel('Order Value ($)', fontsize=11)
ax4.set_ylabel('Density', fontsize=11)
ax4.set_title('Order Value Distribution', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Time to Purchase Distribution
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(time_control, bins=50, alpha=0.6, label='Control', color='#3498db', density=True)
ax5.hist(time_variant, bins=50, alpha=0.6, label='Variant', color='#2ecc71', density=True)
ax5.axvline(time_control_mean, color='#3498db', linestyle='--', linewidth=2)
ax5.axvline(time_variant_mean, color='#2ecc71', linestyle='--', linewidth=2)
ax5.set_xlabel('Time (seconds)', fontsize=11)
ax5.set_ylabel('Density', fontsize=11)
ax5.set_title('Checkout Time Distribution', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Revenue per Visitor
ax6 = fig.add_subplot(gs[1, 2])
rpv_data = [rpv_control, rpv_variant]
bars_rpv = ax6.bar(groups, rpv_data, color=colors_bar, alpha=0.8, edgecolor='black', linewidth=2)
ax6.set_ylabel('Revenue per Visitor ($)', fontsize=11, fontweight='bold')
ax6.set_title('Revenue per Visitor', fontsize=12, fontweight='bold')
for bar, val in zip(bars_rpv, rpv_data):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
             f'${val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

# 7. Daily Performance (simulated)
ax7 = fig.add_subplot(gs[2, :2])
days = np.arange(1, 15)
control_daily = np.random.normal(rate_control, 0.02, 14)
variant_daily = np.random.normal(rate_variant, 0.02, 14)
ax7.plot(days, control_daily, 'o-', linewidth=2, markersize=8, 
         label='Control', color='#3498db')
ax7.plot(days, variant_daily, 's-', linewidth=2, markersize=8, 
         label='Variant', color='#2ecc71')
ax7.axhline(rate_control, color='#3498db', linestyle='--', alpha=0.5)
ax7.axhline(rate_variant, color='#2ecc71', linestyle='--', alpha=0.5)
ax7.set_xlabel('Day', fontsize=11, fontweight='bold')
ax7.set_ylabel('Conversion Rate', fontsize=11, fontweight='bold')
ax7.set_title('Daily Conversion Rate Trend', fontsize=12, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Summary Card
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis('off')
summary_text = f"""
 EXPERIMENT SUMMARY

Duration: 14 days
Total Users: {len(df):,}

Primary Metric:
{' WINNER: Variant' if pval < alpha else '❌ No Clear Winner'}

Expected Annual Impact:
(if deployed to all users)

Daily Visitors: 5,000
Annual Visitors: 1,825,000

Additional Conversions:
{int(1825000 * (rate_variant - rate_control)):,} / year

Additional Revenue:
${int(1825000 * (rpv_variant - rpv_control)):,} / year
"""
ax8.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', 
         facecolor='lightgreen', alpha=0.4))

plt.suptitle(' A/B Test Results Dashboard: Checkout Redesign', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('ab_test_complete_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# PHASE 6: BUSINESS RECOMMENDATION
# ============================================================================


print("PHASE 6: BUSINESS RECOMMENDATION")


print(f"\n EXECUTIVE SUMMARY:")

print(f"""
 RECOMMENDATION: DEPLOY THE NEW CHECKOUT DESIGN

Key Findings:
1. Conversion Rate: {rate_variant:.2%} vs {rate_control:.2%} (p={pval:.4f})
   → Statistically significant improvement of {(rate_variant-rate_control):.2%} absolute
   → {((rate_variant - rate_control) / rate_control * 100):.1f}% relative lift

2. Average Order Value: ${aov_variant:.2f} vs ${aov_control:.2f}
   → {'Significant' if p_val_aov < alpha else 'Not significant'} difference

3. Checkout Time: {time_variant_mean:.1f}s vs {time_control_mean:.1f}s
   → {'Significantly faster' if p_val_time < alpha else 'Similar speed'}

 PROJECTED BUSINESS IMPACT (Annual):
   • Additional Conversions: {int(1825000 * (rate_variant - rate_control)):,}
   • Additional Revenue: ${int(1825000 * (rpv_variant - rpv_control)):,}
   • ROI: Very High (minimal implementation cost)

 CONFIDENCE LEVEL:
   • Statistical Significance: {(1-pval)*100:.2f}%
   • Sample Size: {len(df):,} users (above required {required_n*2:,})
   • Test Duration: 14 days (stable results)

 NEXT STEPS:
   1. Deploy new checkout design to 100% of users
   2. Monitor metrics closely for first 2 weeks
   3. Set up automated dashboards for ongoing tracking
   4. Consider further optimizations (e.g., payment options, trust badges)
""")



print("\n You've successfully completed a full A/B testing project!")
print("   This demonstrates real-world application of all concepts:")
print("   • Sample size calculation")
print("   • Descriptive statistics")
print("   • Hypothesis testing")
print("   • Multiple metrics analysis")
print("   • Data visualization")
print("   • Business decision making")