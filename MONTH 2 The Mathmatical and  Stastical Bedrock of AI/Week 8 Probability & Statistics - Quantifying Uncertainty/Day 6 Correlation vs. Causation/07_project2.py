
#Project 2: Correlation Heatmap. Load a complex dataset (like the Titanic or Housing data) and plot a sns.
# heatmap() of the correlation matrix. Identify which features are strongly correlated.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Create a sample housing dataset (similar to Boston Housing)
np.random.seed(42)
n_samples = 500

# Generate correlated features
data = {
    'SquareFeet': np.random.randint(800, 4000, n_samples),
    'Bedrooms': np.random.randint(1, 6, n_samples),
    'Bathrooms': np.random.randint(1, 5, n_samples),
    'Age': np.random.randint(0, 100, n_samples),
    'Garage': np.random.randint(0, 4, n_samples),
    'LotSize': np.random.randint(2000, 15000, n_samples),
    'Stories': np.random.randint(1, 4, n_samples),
    'CrimeRate': np.random.uniform(0.5, 15, n_samples),
    'SchoolRating': np.random.randint(1, 11, n_samples),
    'DistanceToCity': np.random.uniform(0.5, 30, n_samples)
}

df = pd.DataFrame(data)

# Create Price with correlations to other features
df['Price'] = (
    df['SquareFeet'] * 150 +
    df['Bedrooms'] * 20000 +
    df['Bathrooms'] * 15000 -
    df['Age'] * 1000 +
    df['Garage'] * 10000 +
    df['LotSize'] * 10 +
    df['Stories'] * 5000 -
    df['CrimeRate'] * 8000 +
    df['SchoolRating'] * 12000 -
    df['DistanceToCity'] * 3000 +
    np.random.normal(0, 50000, n_samples)
)

print("=" * 80)
print("CORRELATION HEATMAP ANALYSIS - HOUSING DATASET")
print("=" * 80)
print("\nDataset Overview:")
print(df.head(10))
print("\n" + "=" * 80)
print("\nDataset Statistics:")
print(df.describe())
print("\n" + "=" * 80)

# Calculate correlation matrix
correlation_matrix = df.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(3))
print("\n" + "=" * 80)

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Full Correlation Heatmap
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            vmin=-1, vmax=1)
plt.title('Complete Correlation Heatmap\n(All Features)', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 2. Correlation with Price (Target Variable)
ax2 = plt.subplot(2, 2, 2)
price_corr = correlation_matrix['Price'].sort_values(ascending=False)
colors = ['green' if x > 0 else 'red' for x in price_corr.values]
bars = plt.barh(range(len(price_corr)), price_corr.values, color=colors, alpha=0.7)
plt.yticks(range(len(price_corr)), price_corr.index)
plt.xlabel('Correlation Coefficient', fontsize=11)
plt.title('Correlation with House Price\n(Target Variable)', fontsize=14, fontweight='bold', pad=15)
plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (v, label) in enumerate(zip(price_corr.values, price_corr.index)):
    plt.text(v + 0.02 if v > 0 else v - 0.02, i, f'{v:.3f}', 
             va='center', ha='left' if v > 0 else 'right', fontsize=9)

# 3. Triangle Heatmap (removing duplicate information)
ax3 = plt.subplot(2, 2, 3)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='RdYlGn', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'Correlation Coefficient'},
            vmin=-1, vmax=1)
plt.title('Triangle Correlation Matrix\n(Unique Pairs Only)', fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

# 4. Scatter plot of strongest correlations
ax4 = plt.subplot(2, 2, 4)
# Find strongest positive correlation (excluding diagonal)
corr_copy = correlation_matrix.copy()
np.fill_diagonal(corr_copy.values, 0)
max_corr_idx = np.unravel_index(corr_copy.values.argmax(), corr_copy.shape)
feature1 = corr_copy.index[max_corr_idx[0]]
feature2 = corr_copy.columns[max_corr_idx[1]]
corr_value = corr_copy.iloc[max_corr_idx[0], max_corr_idx[1]]

plt.scatter(df[feature1], df[feature2], alpha=0.5, s=30)
plt.xlabel(feature1, fontsize=11)
plt.ylabel(feature2, fontsize=11)
plt.title(f'Strongest Positive Correlation\n{feature1} vs {feature2} (r={corr_value:.3f})', 
          fontsize=14, fontweight='bold', pad=15)
plt.grid(alpha=0.3)

# Add trend line
z = np.polyfit(df[feature1], df[feature2], 1)
p = np.poly1d(z)
plt.plot(df[feature1], p(df[feature1]), "r--", alpha=0.8, linewidth=2, label='Trend line')
plt.legend()

plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Identify strongly correlated features
print("\nSTRONGLY CORRELATED FEATURES ANALYSIS")
print("=" * 80)

# Strong positive correlations (excluding diagonal)
print("\n1. STRONG POSITIVE CORRELATIONS (r > 0.7):")
print("-" * 80)
strong_positive = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] > 0.7:
            strong_positive.append((
                correlation_matrix.index[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

if strong_positive:
    for feat1, feat2, corr in sorted(strong_positive, key=lambda x: x[2], reverse=True):
        print(f"   {feat1:20s} <-> {feat2:20s}  |  r = {corr:6.3f}")
else:
    print("   No feature pairs with correlation > 0.7")

# Strong negative correlations
print("\n2. STRONG NEGATIVE CORRELATIONS (r < -0.5):")
print("-" * 80)
strong_negative = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] < -0.5:
            strong_negative.append((
                correlation_matrix.index[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j]
            ))

if strong_negative:
    for feat1, feat2, corr in sorted(strong_negative, key=lambda x: x[2]):
        print(f"   {feat1:20s} <-> {feat2:20s}  |  r = {corr:6.3f}")
else:
    print("   No feature pairs with correlation < -0.5")

# Features most correlated with target (Price)
print("\n3. TOP FEATURES CORRELATED WITH PRICE (Target):")
print("-" * 80)
price_correlations = correlation_matrix['Price'].drop('Price').sort_values(ascending=False)
print("\nPositive correlations:")
for feature, corr in price_correlations[price_correlations > 0].items():
    print(f"   {feature:20s}  |  r = {corr:6.3f}")

print("\nNegative correlations:")
for feature, corr in price_correlations[price_correlations < 0].items():
    print(f"   {feature:20s}  |  r = {corr:6.3f}")

# Statistical significance test for top correlations
print("\n4. STATISTICAL SIGNIFICANCE (P-values for top correlations):")
print("-" * 80)
for feature in price_correlations.abs().nlargest(5).index:
    corr, p_value = stats.pearsonr(df['Price'], df[feature])
    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"   Price vs {feature:20s}  |  r = {corr:6.3f}, p = {p_value:.4f} {significance}")


print("INTERPRETATION GUIDE:")
print("""
Correlation Coefficient (r) ranges from -1 to +1:
  • r > +0.7  : Strong positive correlation
  • r = +0.4 to +0.7 : Moderate positive correlation
  • r = -0.4 to +0.4 : Weak/No correlation
  • r = -0.7 to -0.4 : Moderate negative correlation
  • r < -0.7  : Strong negative correlation

P-value significance:
  • p < 0.001 (***) : Highly significant
  • p < 0.01  (**)  : Very significant
  • p < 0.05  (*)   : Significant
  • p >= 0.05 (ns)  : Not significant

Key Insights from this dataset:
  • Features with high positive correlation with Price are good predictors
  • Features with high negative correlation with Price inversely affect it
  • Highly correlated features (multicollinearity) may be redundant in models
""")

print("\nAnalysis complete! Heatmap saved as 'correlation_heatmap.png'")
