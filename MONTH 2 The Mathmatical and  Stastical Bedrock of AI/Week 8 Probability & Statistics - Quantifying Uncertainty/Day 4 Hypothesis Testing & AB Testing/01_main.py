
#Day 4: Hypothesis Testing & A/B Testing
#Objective: Determine if a change (e.g., a new model, a new UI) actually made a difference or if it was just luck.
#Concepts: Null Hypothesis ($H_0$), Alternative Hypothesis ($H_1$), P-values, Significance Level ($\alpha$), T-tests.
#Task: Project - A/B Test Analyzer. Generate two fake datasets representing "User Group A" and "User Group B". Use scipy.stats.ttest_ind to calculate the p-value. If $p < 0.05$, the difference is real.
#Descriptive Statistics
#Focus: Measures of central tendency and spread
#Study mean, median, mode, variance, standard deviation
#Learn about quartiles, percentiles, and outliers
#Understand data visualization for statistics
#Project:
#Descriptive Statistics Report - Create a comprehensive script that generates statistical summaries for datasets

#Complete Statistics - Part 1: Descriptive Statistics

"""
PART 1: DESCRIPTIVE STATISTICS
Mastering Central Tendency, Spread, and Visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. MEASURES OF CENTRAL TENDENCY
# ============================================================================

print("=" * 80)
print("1. MEASURES OF CENTRAL TENDENCY")
print("=" * 80)

# Generate sample data: Student exam scores
np.random.seed(42)
scores = np.array([45, 67, 72, 75, 78, 80, 82, 85, 85, 88, 90, 92, 95, 98, 100])
print(f"\nStudent Exam Scores:\n{scores}")

# Mean (Average)
mean_score = np.mean(scores)
print(f"\n📊 Mean: {mean_score:.2f}")
print("   → The average score across all students")
print("   → Sum of all values / Number of values")
print(f"   → Calculation: {scores.sum()} / {len(scores)} = {mean_score:.2f}")

# Median (Middle value)
median_score = np.median(scores)
print(f"\n📊 Median: {median_score:.2f}")
print("   → The middle value when data is sorted")
print("   → Less affected by outliers than mean")
sorted_scores = np.sort(scores)
middle_idx = len(sorted_scores) // 2
print(f"   → Middle position: {sorted_scores[middle_idx]}")

# Mode (Most frequent)
mode_result = stats.mode(scores, keepdims=True)
mode_score = mode_result.mode[0]
print(f"\n📊 Mode: {mode_score}")
print("   → The most frequently occurring value")
print(f"   → Appears {mode_result.count[0]} times")

# Visualizing Central Tendency
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram with central tendency lines
axes[0].hist(scores, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
axes[0].axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
axes[0].axvline(mode_score, color='orange', linestyle='--', linewidth=2, label=f'Mode: {mode_score}')
axes[0].set_xlabel('Scores')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Central Tendency Measures')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Box plot
axes[1].boxplot(scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Scores')
axes[1].set_title('Box Plot Visualization')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('central_tendency.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 2. MEASURES OF SPREAD (DISPERSION)
# ============================================================================

print("\n" + "=" * 80)
print("2. MEASURES OF SPREAD (DISPERSION)")
print("=" * 80)

# Range
data_range = np.ptp(scores)  # Peak to peak (max - min)
print(f"\n📏 Range: {data_range}")
print(f"   → Maximum - Minimum")
print(f"   → {scores.max()} - {scores.min()} = {data_range}")

# Variance (Average squared deviation from mean)
variance = np.var(scores, ddof=1)  # ddof=1 for sample variance
print(f"\n📏 Variance: {variance:.2f}")
print("   → Average of squared differences from mean")
print("   → Shows how spread out the data is")
deviations = scores - mean_score
squared_deviations = deviations ** 2
print(f"   → Sum of squared deviations: {squared_deviations.sum():.2f}")
print(f"   → Divided by (n-1): {squared_deviations.sum() / (len(scores) - 1):.2f}")

# Standard Deviation (Square root of variance)
std_dev = np.std(scores, ddof=1)
print(f"\n Standard Deviation: {std_dev:.2f}")
print("   → Square root of variance")
print("   → In same units as original data")
print(f"   → √{variance:.2f} = {std_dev:.2f}")
print(f"   → About 68% of data falls within 1 std dev of mean")
print(f"   → Range: [{mean_score - std_dev:.2f}, {mean_score + std_dev:.2f}]")

# Coefficient of Variation (Relative variability)
cv = (std_dev / mean_score) * 100
print(f"\n Coefficient of Variation: {cv:.2f}%")
print("   → (Std Dev / Mean) × 100")
print("   → Useful for comparing variability across different datasets")

# Visualizing Spread
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Deviation from mean
axes[0, 0].bar(range(len(scores)), scores, alpha=0.6, color='skyblue', label='Scores')
axes[0, 0].axhline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
axes[0, 0].fill_between(range(len(scores)), mean_score - std_dev, mean_score + std_dev,
                         alpha=0.2, color='green', label=f'±1 Std Dev')
axes[0, 0].set_xlabel('Student Index')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Scores with Mean and Standard Deviation')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Normal distribution overlay
axes[0, 1].hist(scores, bins=10, density=True, alpha=0.7, color='skyblue', edgecolor='black')
xmin, xmax = axes[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mean_score, std_dev)
axes[0, 1].plot(x, p, 'r-', linewidth=2, label='Normal Distribution')
axes[0, 1].set_xlabel('Scores')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Histogram with Normal Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Variance visualization
axes[1, 0].scatter(range(len(scores)), scores, s=100, alpha=0.6, color='blue')
for i, score in enumerate(scores):
    axes[1, 0].plot([i, i], [mean_score, score], 'r--', alpha=0.5)
axes[1, 0].axhline(mean_score, color='green', linewidth=2, label=f'Mean: {mean_score:.2f}')
axes[1, 0].set_xlabel('Student Index')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Deviations from Mean (Variance Components)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Spread comparison
comparison_data = pd.DataFrame({
    'Low Spread': np.random.normal(80, 5, 100),
    'Medium Spread': np.random.normal(80, 15, 100),
    'High Spread': np.random.normal(80, 25, 100)
})
axes[1, 1].hist(comparison_data['Low Spread'], bins=20, alpha=0.5, label='Low Spread (σ=5)')
axes[1, 1].hist(comparison_data['Medium Spread'], bins=20, alpha=0.5, label='Medium Spread (σ=15)')
axes[1, 1].hist(comparison_data['High Spread'], bins=20, alpha=0.5, label='High Spread (σ=25)')
axes[1, 1].set_xlabel('Values')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Comparing Different Spreads')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('measures_of_spread.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 3. QUARTILES, PERCENTILES, AND OUTLIERS
# ============================================================================

print("\n" + "=" * 80)
print("3. QUARTILES, PERCENTILES, AND OUTLIERS")
print("=" * 80)

# Quartiles (divide data into 4 equal parts)
q1 = np.percentile(scores, 25)
q2 = np.percentile(scores, 50)  # Same as median
q3 = np.percentile(scores, 75)

print(f"\n Quartiles:")
print(f"   Q1 (25th percentile): {q1:.2f}")
print(f"   Q2 (50th percentile/Median): {q2:.2f}")
print(f"   Q3 (75th percentile): {q3:.2f}")

# Interquartile Range (IQR)
iqr = q3 - q1
print(f"\n Interquartile Range (IQR): {iqr:.2f}")
print(f"   → Q3 - Q1 = {q3:.2f} - {q1:.2f} = {iqr:.2f}")
print("   → Middle 50% of data spans this range")

# Outlier detection using IQR method
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
print(f"\n Outlier Detection (IQR Method):")
print(f"   Lower Fence: Q1 - 1.5×IQR = {lower_fence:.2f}")
print(f"   Upper Fence: Q3 + 1.5×IQR = {upper_fence:.2f}")

# Add some outliers to demonstrate
scores_with_outliers = np.append(scores, [20, 30, 110, 115])
outliers = scores_with_outliers[(scores_with_outliers < lower_fence) | (scores_with_outliers > upper_fence)]
print(f"   Outliers detected: {outliers}")

# Percentiles (specific examples)
percentiles = [10, 25, 50, 75, 90, 95, 99]
print(f"\n Key Percentiles:")
for p in percentiles:
    value = np.percentile(scores, p)
    print(f"   {p}th percentile: {value:.2f} → {p}% of data is below this value")

# Visualizing Quartiles and Outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Box plot with quartile annotations
bp = axes[0, 0].boxplot(scores_with_outliers, vert=True, patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='red', markersize=8))
axes[0, 0].axhline(q1, color='green', linestyle='--', alpha=0.5, label=f'Q1: {q1:.2f}')
axes[0, 0].axhline(q2, color='orange', linestyle='--', alpha=0.5, label=f'Q2: {q2:.2f}')
axes[0, 0].axhline(q3, color='blue', linestyle='--', alpha=0.5, label=f'Q3: {q3:.2f}')
axes[0, 0].set_ylabel('Scores')
axes[0, 0].set_title('Box Plot with Quartiles and Outliers')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Percentile plot
percentile_values = [np.percentile(scores, p) for p in range(0, 101, 5)]
axes[0, 1].plot(range(0, 101, 5), percentile_values, marker='o', linewidth=2)
axes[0, 1].axhline(q1, color='green', linestyle='--', alpha=0.5, label=f'Q1 (25th)')
axes[0, 1].axhline(q2, color='orange', linestyle='--', alpha=0.5, label=f'Q2 (50th)')
axes[0, 1].axhline(q3, color='blue', linestyle='--', alpha=0.5, label=f'Q3 (75th)')
axes[0, 1].set_xlabel('Percentile')
axes[0, 1].set_ylabel('Score Value')
axes[0, 1].set_title('Percentile Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. IQR visualization
axes[1, 0].hist(scores_with_outliers, bins=15, alpha=0.6, color='skyblue', edgecolor='black')
axes[1, 0].axvline(q1, color='green', linestyle='--', linewidth=2, label=f'Q1: {q1:.2f}')
axes[1, 0].axvline(q3, color='blue', linestyle='--', linewidth=2, label=f'Q3: {q3:.2f}')
axes[1, 0].axvline(lower_fence, color='red', linestyle=':', linewidth=2, label=f'Lower Fence: {lower_fence:.2f}')
axes[1, 0].axvline(upper_fence, color='red', linestyle=':', linewidth=2, label=f'Upper Fence: {upper_fence:.2f}')
axes[1, 0].axvspan(q1, q3, alpha=0.2, color='yellow', label='IQR')
axes[1, 0].set_xlabel('Scores')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('IQR and Outlier Fences')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Scatter plot with outliers highlighted
axes[1, 1].scatter(range(len(scores_with_outliers)), scores_with_outliers, 
                   c=['red' if x in outliers else 'blue' for x in scores_with_outliers],
                   s=100, alpha=0.6)
axes[1, 1].axhline(lower_fence, color='red', linestyle='--', linewidth=2, label='Outlier Bounds')
axes[1, 1].axhline(upper_fence, color='red', linestyle='--', linewidth=2)
axes[1, 1].axhspan(q1, q3, alpha=0.2, color='green', label='IQR (Normal Range)')
axes[1, 1].set_xlabel('Data Point Index')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Outlier Detection Visualization')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quartiles_outliers.png', dpi=150, bbox_inches='tight')
plt.show()


print(" PART 1 COMPLETE: Descriptive Statistics")

print("\nKey Takeaways:")
print("1. Central Tendency: Mean (average), Median (middle), Mode (most frequent)")
print("2. Spread: Range, Variance, Standard Deviation, CV")
print("3. Quartiles: Divide data into 4 parts (Q1, Q2, Q3)")
print("4. IQR: Middle 50% of data (Q3 - Q1)")
print("5. Outliers: Data points beyond 1.5×IQR from quartiles")
print("\nNext: Run Part 2 for Hypothesis Testing!")