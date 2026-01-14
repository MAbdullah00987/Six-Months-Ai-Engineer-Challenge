

#Day 6: Correlation vs. Causation
#Objective: Quantify relationships between variables without jumping to conclusions.
#Concepts: Covariance, Pearson Correlation Coefficient ($r$), Spurious Correlations.
#Task: Project - Correlation Heatmap. Load a complex dataset (like the Titanic or Housing data) and plot a sns.heatmap() of the correlation matrix. Identify which features are strongly correlated.

#Hypothesis Testing
#Focus: Statistical significance and testing
#Study hypothesis testing framework (null/alternative hypotheses)
#Learn about p-values and significance levels
#Review t-tests, z-tests
#Projects:
#A/B Test Analyzer - Create a script to analyze A/B test results and determine statistical significance

#Part 1. Correlation vs. Causation: Understanding Relationships

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ============================================================================
# PART 1: COVARIANCE - Measuring Linear Relationship
# ============================================================================

print("=" * 80)
print("PART 1: COVARIANCE - Understanding How Variables Move Together")
print("=" * 80)

# Generate data: Ice cream sales and temperature
np.random.seed(42)
temperature = np.linspace(15, 35, 100) + np.random.normal(0, 2, 100)
ice_cream_sales = 50 + 3 * temperature + np.random.normal(0, 20, 100)
drownings = 10 + 0.5 * temperature + np.random.normal(0, 5, 100)

# Calculate covariance manually
def calculate_covariance(x, y):
    """Calculate covariance between two variables"""
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    # Covariance formula: Σ[(xi - x̄)(yi - ȳ)] / (n-1)
    cov = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    return cov

cov_temp_sales = calculate_covariance(temperature, ice_cream_sales)
cov_sales_drownings = calculate_covariance(ice_cream_sales, drownings)

print(f"\n1. Manual Covariance Calculations:")
print(f"   Cov(Temperature, Ice Cream Sales) = {cov_temp_sales:.2f}")
print(f"   Cov(Ice Cream Sales, Drownings) = {cov_sales_drownings:.2f}")

# Using NumPy
cov_matrix = np.cov(temperature, ice_cream_sales)
print(f"\n2. NumPy Covariance Matrix (Temperature vs Sales):")
print(f"   {cov_matrix}")
print(f"   Problem: Covariance units depend on variable scales!")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Temperature vs Ice Cream Sales
axes[0, 0].scatter(temperature, ice_cream_sales, alpha=0.6, color='coral')
axes[0, 0].set_xlabel('Temperature (°C)', fontsize=11)
axes[0, 0].set_ylabel('Ice Cream Sales ($)', fontsize=11)
axes[0, 0].set_title('Temperature vs Ice Cream Sales\n(Strong Positive Covariance)', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Ice Cream Sales vs Drownings (Spurious!)
axes[0, 1].scatter(ice_cream_sales, drownings, alpha=0.6, color='steelblue')
axes[0, 1].set_xlabel('Ice Cream Sales ($)', fontsize=11)
axes[0, 1].set_ylabel('Drowning Incidents', fontsize=11)
axes[0, 1].set_title('Ice Cream Sales vs Drownings\n(SPURIOUS Correlation!)', fontsize=12, fontweight='bold', color='red')
axes[0, 1].grid(True, alpha=0.3)

# ============================================================================
# PART 2: PEARSON CORRELATION COEFFICIENT - Standardized Measure
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: PEARSON CORRELATION COEFFICIENT (r)")
print("=" * 80)

def pearson_correlation(x, y):
    """
    Calculate Pearson correlation coefficient
    Formula: r = Cov(X,Y) / (σx * σy)
    Range: [-1, 1]
    """
    cov = calculate_covariance(x, y)
    std_x = np.std(x, ddof=1)
    std_y = np.std(y, ddof=1)
    
    r = cov / (std_x * std_y)
    return r

r_temp_sales = pearson_correlation(temperature, ice_cream_sales)
r_sales_drownings = pearson_correlation(ice_cream_sales, drownings)
r_temp_drownings = pearson_correlation(temperature, drownings)

print(f"\n1. Pearson Correlation Coefficients:")
print(f"   r(Temperature, Ice Cream) = {r_temp_sales:.4f}")
print(f"   r(Ice Cream, Drownings) = {r_sales_drownings:.4f}")
print(f"   r(Temperature, Drownings) = {r_temp_drownings:.4f}")

# Verify with scipy
scipy_r, scipy_p = stats.pearsonr(temperature, ice_cream_sales)
print(f"\n2. Verification with SciPy:")
print(f"   r = {scipy_r:.4f}, p-value = {scipy_p:.4e}")

# Interpretation
print(f"\n3. Interpretation Guide:")
print(f"   |r| = 0.00 - 0.19  →  Very weak")
print(f"   |r| = 0.20 - 0.39  →  Weak")
print(f"   |r| = 0.40 - 0.59  →  Moderate")
print(f"   |r| = 0.60 - 0.79  →  Strong")
print(f"   |r| = 0.80 - 1.00  →  Very strong")

# Different correlation scenarios
scenarios = {
    'Perfect Positive': (np.arange(50), np.arange(50), 1.0),
    'Strong Positive': (np.arange(50), np.arange(50) + np.random.normal(0, 5, 50), 0.9),
    'Weak Positive': (np.arange(50), np.arange(50) + np.random.normal(0, 20, 50), 0.5),
    'No Correlation': (np.random.randn(50), np.random.randn(50), 0.0),
    'Negative': (np.arange(50), -np.arange(50) + np.random.normal(0, 5, 50), -0.9)
}

# Plot different correlations
for idx, (name, (x, y, expected_r)) in enumerate(scenarios.items()):
    if idx < 2:
        row, col = 1, idx
        actual_r = pearson_correlation(x, y)
        axes[row, col].scatter(x, y, alpha=0.6)
        axes[row, col].set_title(f'{name} (r ≈ {actual_r:.2f})', fontsize=12, fontweight='bold')
        axes[row, col].set_xlabel('X', fontsize=11)
        axes[row, col].set_ylabel('Y', fontsize=11)
        axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: correlation_analysis.png")
plt.show()

# ============================================================================
# PART 3: SPURIOUS CORRELATIONS - Correlation ≠ Causation
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: SPURIOUS CORRELATIONS - When Correlation is Meaningless")
print("=" * 80)

# Create confounding variable example
years = np.arange(2000, 2020)
internet_users = 20 + 3 * (years - 2000) + np.random.normal(0, 2, 20)
obesity_rate = 15 + 0.8 * (years - 2000) + np.random.normal(0, 1, 20)

r_spurious = pearson_correlation(internet_users, obesity_rate)

print(f"\nExample: Internet Users vs Obesity Rate")
print(f"Correlation: r = {r_spurious:.4f}")
print(f"\nWHY THIS IS SPURIOUS:")
print(f"  • Both variables increase with time (confounding variable)")
print(f"  • Internet use doesn't CAUSE obesity")
print(f"  • Time is the hidden common factor")
print(f"  • This is 'Confounding Bias'")

# Create correlation matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Spurious correlation plot
axes[0].plot(years, internet_users, 'o-', label='Internet Users (%)', linewidth=2)
axes[0].plot(years, obesity_rate, 's-', label='Obesity Rate (%)', linewidth=2)
axes[0].set_xlabel('Year', fontsize=11)
axes[0].set_ylabel('Percentage (%)', fontsize=11)
axes[0].set_title(f'Spurious Correlation Example\nr = {r_spurious:.3f} (Meaningless!)', 
                  fontsize=12, fontweight='bold', color='red')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Correlation heatmap
df = pd.DataFrame({
    'Temperature': temperature,
    'Ice Cream Sales': ice_cream_sales,
    'Drownings': drownings
})

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Correlation Matrix\n(All correlations ≠ All causations)', 
                  fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('spurious_correlations.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: spurious_correlations.png")
plt.show()

# ============================================================================
# LOGIC STRENGTHENING EXERCISES
# ============================================================================

print("\n" + "=" * 80)
print("STRENGTHENING YOUR STATISTICAL LOGIC")
print("=" * 80)

print("""
KEY LOGICAL PRINCIPLES:

1. COVARIANCE tells you:
   ✓ Whether variables move together (positive/negative)
   ✗ NOT standardized (hard to compare)
   
2. CORRELATION tells you:
   ✓ Strength of LINEAR relationship (-1 to +1)
   ✓ Standardized measure (comparable)
   ✗ NOT causation!
   
3. CAUSATION requires:
   ✓ Controlled experiments
   ✓ Temporal precedence (cause before effect)
   ✓ Elimination of confounders
   ✓ Mechanism explanation
   
4. CRITICAL THINKING:
   • Always ask: "What else could explain this pattern?"
   • Look for confounding variables
   • Consider reverse causation
   • Demand experimental evidence

FAMOUS SPURIOUS CORRELATIONS:
• Nicholas Cage films ↔ Pool drownings
• Cheese consumption ↔ Engineering PhDs
• Pirates decline ↔ Global warming
""")

print("DATA SUMMARY")
print(df.describe())