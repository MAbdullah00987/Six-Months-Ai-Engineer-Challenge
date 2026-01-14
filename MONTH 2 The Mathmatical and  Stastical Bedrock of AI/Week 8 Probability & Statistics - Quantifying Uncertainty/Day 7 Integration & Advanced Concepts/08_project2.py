

#Correlation vs. Causation Analysis - Find a dataset showing spurious correlation and write a detailed analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# DATASET: Famous Spurious Correlation Example
# US Spending on Science vs Suicides by Hanging
# ============================================================================

# Data from Tyler Vigen's Spurious Correlations
years = np.arange(1999, 2010)
science_spending = np.array([18.079, 18.594, 19.753, 20.734, 20.831, 
                              23.029, 23.597, 23.584, 25.525, 27.731, 29.449])
suicides_hanging = np.array([5427, 5688, 6198, 6462, 6635, 
                              7336, 7248, 7491, 8161, 8578, 9000])

# Create DataFrame
df = pd.DataFrame({
    'Year': years,
    'Science_Spending_Billions': science_spending,
    'Suicides_by_Hanging': suicides_hanging
})

print("=" * 80)
print("SPURIOUS CORRELATION ANALYSIS")
print("=" * 80)
print("\nDataset: US Spending on Science, Space, and Technology (billions)")
print("         vs. Suicides by Hanging, Strangulation and Suffocation")
print("\nData Preview:")
print(df.head(10))
print("\n" + "=" * 80)

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================

print("\n1. DESCRIPTIVE STATISTICS")
print("-" * 80)
print("\nScience Spending (Billions USD):")
print(f"  Mean: ${science_spending.mean():.2f}B")
print(f"  Std Dev: ${science_spending.std():.2f}B")
print(f"  Min: ${science_spending.min():.2f}B")
print(f"  Max: ${science_spending.max():.2f}B")

print("\nSuicides by Hanging:")
print(f"  Mean: {suicides_hanging.mean():.0f}")
print(f"  Std Dev: {suicides_hanging.std():.0f}")
print(f"  Min: {suicides_hanging.min():.0f}")
print(f"  Max: {suicides_hanging.max():.0f}")

# ============================================================================
# 2. CORRELATION ANALYSIS
# ============================================================================

print("\n\n2. CORRELATION ANALYSIS")
print("-" * 80)

# Pearson correlation
pearson_corr, pearson_pval = pearsonr(science_spending, suicides_hanging)
print(f"\nPearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"P-value: {pearson_pval:.6f}")
print(f"Interpretation: {'Statistically significant' if pearson_pval < 0.05 else 'Not statistically significant'} at α=0.05")

# Spearman correlation (rank-based, less sensitive to outliers)
spearman_corr, spearman_pval = spearmanr(science_spending, suicides_hanging)
print(f"\nSpearman Correlation Coefficient: {spearman_corr:.4f}")
print(f"P-value: {spearman_pval:.6f}")

# Coefficient of determination
r_squared = pearson_corr ** 2
print(f"\nCoefficient of Determination (R²): {r_squared:.4f}")
print(f"Interpretation: {r_squared*100:.2f}% of variance in one variable is 'explained' by the other")

# ============================================================================
# 3. VISUALIZATION
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Dual-axis time series
ax1 = axes[0, 0]
ax1_twin = ax1.twinx()

line1 = ax1.plot(years, science_spending, 'o-', color='#2E86AB', 
                 linewidth=2, markersize=8, label='Science Spending')
ax1.set_xlabel('Year', fontsize=11, fontweight='bold')
ax1.set_ylabel('Science Spending (Billions USD)', fontsize=11, 
               fontweight='bold', color='#2E86AB')
ax1.tick_params(axis='y', labelcolor='#2E86AB')

line2 = ax1_twin.plot(years, suicides_hanging, 's-', color='#A23B72', 
                      linewidth=2, markersize=8, label='Suicides by Hanging')
ax1_twin.set_ylabel('Suicides by Hanging', fontsize=11, 
                    fontweight='bold', color='#A23B72')
ax1_twin.tick_params(axis='y', labelcolor='#A23B72')

ax1.set_title('Time Series: Both Variables Trending Upward', 
              fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3)

# Plot 2: Scatter plot with regression line
ax2 = axes[0, 1]
ax2.scatter(science_spending, suicides_hanging, s=100, alpha=0.6, color='#F18F01')

# Add regression line
z = np.polyfit(science_spending, suicides_hanging, 1)
p = np.poly1d(z)
ax2.plot(science_spending, p(science_spending), "r--", linewidth=2, alpha=0.8)

ax2.set_xlabel('Science Spending (Billions USD)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Suicides by Hanging', fontsize=11, fontweight='bold')
ax2.set_title(f'Scatter Plot with Regression Line\nr = {pearson_corr:.4f}, p = {pearson_pval:.6f}', 
              fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3)

# Plot 3: Normalized comparison
ax3 = axes[1, 0]
science_norm = (science_spending - science_spending.min()) / (science_spending.max() - science_spending.min())
suicides_norm = (suicides_hanging - suicides_hanging.min()) / (suicides_hanging.max() - suicides_hanging.min())

ax3.plot(years, science_norm, 'o-', linewidth=2, markersize=8, 
         label='Science Spending (normalized)', color='#2E86AB')
ax3.plot(years, suicides_norm, 's-', linewidth=2, markersize=8, 
         label='Suicides (normalized)', color='#A23B72')
ax3.set_xlabel('Year', fontsize=11, fontweight='bold')
ax3.set_ylabel('Normalized Value (0-1)', fontsize=11, fontweight='bold')
ax3.set_title('Normalized Comparison: Similar Trends', 
              fontsize=13, fontweight='bold', pad=15)
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

# Plot 4: Residual plot
ax4 = axes[1, 1]
predicted = p(science_spending)
residuals = suicides_hanging - predicted
ax4.scatter(predicted, residuals, s=100, alpha=0.6, color='#C73E1D')
ax4.axhline(y=0, color='black', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
ax4.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax4.set_title('Residual Plot: Checking Linear Relationship', 
              fontsize=13, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spurious_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n\n3. VISUALIZATIONS CREATED")
print("-" * 80)
print("✓ Time series plot (dual-axis)")
print("✓ Scatter plot with regression line")
print("✓ Normalized comparison")
print("✓ Residual plot")

# ============================================================================
# 4. STATISTICAL TESTS FOR CAUSALITY
# ============================================================================

print("\n\n4. TESTS FOR CAUSALITY")
print("-" * 80)

# Granger Causality Test
print("\n4.1 Granger Causality Test")
print("-" * 40)
print("Testing if Science Spending 'Granger-causes' Suicides...")

# Prepare data for Granger test
granger_data = pd.DataFrame({
    'science': science_spending,
    'suicides': suicides_hanging
})

try:
    # Test with lag=1 (we have limited data points)
    max_lag = 2
    granger_result = grangercausalitytests(granger_data[['suicides', 'science']], 
                                           max_lag, verbose=False)
    
    print(f"\nGranger Causality Test Results (max lag = {max_lag}):")
    for lag in range(1, max_lag + 1):
        p_value = granger_result[lag][0]['ssr_ftest'][1]
        print(f"  Lag {lag}: p-value = {p_value:.4f}")
        print(f"    Result: {'Reject null' if p_value < 0.05 else 'Fail to reject null'} (α=0.05)")
    
    print("\nInterpretation: Granger causality tests if past values of one variable")
    print("help predict future values of another. However, this is NOT true causation!")
except Exception as e:
    print(f"Note: Granger test requires more data points. Error: {str(e)[:100]}")

# ============================================================================
# 5. WHY THIS IS SPURIOUS: DETAILED ANALYSIS
# ============================================================================

print("\n\n5. WHY THIS IS A SPURIOUS CORRELATION")
print("=" * 80)

print("\n5.1 Definition of Spurious Correlation:")
print("-" * 40)
print("A spurious correlation occurs when two variables appear to be related")
print("but have no causal connection. The relationship exists due to:")
print("  1. Pure coincidence")
print("  2. A confounding third variable")
print("  3. Both variables following similar time trends")

print("\n5.2 Analysis of This Specific Case:")
print("-" * 40)
print(f"• Correlation coefficient: {pearson_corr:.4f} (very strong!)")
print(f"• Statistical significance: p = {pearson_pval:.6f} (highly significant!)")
print("\nDespite strong statistical measures, there is NO plausible mechanism")
print("by which spending on science could cause suicides, or vice versa.")

print("\n5.3 The Real Explanation: Confounding Variables")
print("-" * 40)
print("Both variables likely influenced by common factors:")
print("  • Population growth (more people → more suicides)")
print("  • Economic growth (larger economy → more science spending)")
print("  • Time trend (both generally increase over time)")
print("  • Improved reporting/data collection methods")

print("\n5.4 Key Statistical Lesson:")
print("-" * 40)
print("⚠️  CORRELATION ≠ CAUSATION")
print("\nJust because two variables are correlated does NOT mean:")
print("  ✗ One causes the other")
print("  ✗ They have any meaningful relationship")
print("  ✗ Changes in one will affect the other")

# ============================================================================
# 6. ADDITIONAL SPURIOUS CORRELATION EXAMPLES
# ============================================================================

print("\n\n6. OTHER FAMOUS SPURIOUS CORRELATIONS")
print("-" * 80)

spurious_examples = [
    "• Number of Nicolas Cage films vs. Drownings in swimming pools (r = 0.67)",
    "• Per capita cheese consumption vs. Deaths by bedsheet tangling (r = 0.95)",
    "• Divorce rate in Maine vs. Per capita margarine consumption (r = 0.99)",
    "• Number of PhDs awarded vs. Arcade revenue (r = 0.98)",
]

for example in spurious_examples:
    print(example)

print("\nAll of these show strong correlations but have NO causal relationship!")

# ============================================================================
# 7. HOW TO IDENTIFY SPURIOUS CORRELATIONS
# ============================================================================

print("\n\n7. HOW TO IDENTIFY SPURIOUS CORRELATIONS")
print("=" * 80)

checklist = [
    ("1. Ask: Is there a plausible mechanism?", 
     "Could one variable realistically affect the other?"),
    ("2. Check for time trends", 
     "Both increasing over time doesn't mean they're related"),
    ("3. Look for confounding variables", 
     "What other factors might influence both?"),
    ("4. Consider the domain knowledge", 
     "Does this make sense in the real world?"),
    ("5. Use controlled experiments", 
     "Observational data alone cannot prove causation"),
    ("6. Apply causal inference methods", 
     "Randomized trials, natural experiments, IV methods"),
]

for i, (title, desc) in enumerate(checklist, 1):
    print(f"\n{title}")
    print(f"   → {desc}")


# 8. CONCLUSION


print("\n\n8. CONCLUSION")
print(f"\nThis analysis demonstrates a textbook spurious correlation:")
print(f"• Strong correlation: r = {pearson_corr:.4f}")
print(f"• High statistical significance: p < 0.001")
print(f"• But ZERO causal relationship")

print("\n📊 Key Takeaways:")
print("-" * 40)
print("1. Always question correlations with domain knowledge")
print("2. Strong statistics don't guarantee meaningful relationships")
print("3. Time-series data is particularly prone to spurious correlations")
print("4. Causation requires theory, mechanism, and rigorous testing")
print("5. Be skeptical of surprising correlations - they're often spurious!")


print("Analysis complete! Check 'spurious_correlation_analysis.png' for visualizations.")
