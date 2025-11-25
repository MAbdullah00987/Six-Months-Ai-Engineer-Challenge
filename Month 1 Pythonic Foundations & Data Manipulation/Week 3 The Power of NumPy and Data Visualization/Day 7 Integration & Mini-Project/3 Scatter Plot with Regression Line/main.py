
#Scatter Plot with Regression Line:
# Create a scatter plot of two variables and add a linear regression line using Seaborn.

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set style
sns.set_style("whitegrid")

# Create sample data
np.random.seed(42)
n_points = 100
x = np.random.randn(n_points) * 10 + 50  # Mean=50, SD=10
y = 2.5 * x + np.random.randn(n_points) * 20 + 10  # Linear relationship with noise

# Create DataFrame
df = pd.DataFrame({
    'Study Hours': x,
    'Test Score': y
})

# Create figure with multiple subplots to show different options
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Scatter Plot with Regression Line - Different Styles', fontsize=16, fontweight='bold')

# Plot 1: Basic regplot
sns.regplot(data=df, x='Study Hours', y='Test Score', ax=axes[0, 0])
axes[0, 0].set_title('Basic Regression Plot')
axes[0, 0].set_xlabel('Study Hours')
axes[0, 0].set_ylabel('Test Score')

# Plot 2: Customized colors and style
sns.regplot(data=df, x='Study Hours', y='Test Score', 
            scatter_kws={'alpha': 0.5, 'color': 'blue', 's': 50},
            line_kws={'color': 'red', 'linewidth': 2},
            ax=axes[0, 1])
axes[0, 1].set_title('Customized Colors')
axes[0, 1].set_xlabel('Study Hours')
axes[0, 1].set_ylabel('Test Score')

# Plot 3: With confidence interval (default 95%)
sns.regplot(data=df, x='Study Hours', y='Test Score',
            scatter_kws={'alpha': 0.6, 'color': 'green'},
            line_kws={'color': 'darkgreen', 'linewidth': 2},
            ci=95,  # Confidence interval
            ax=axes[1, 0])
axes[1, 0].set_title('With 95% Confidence Interval')
axes[1, 0].set_xlabel('Study Hours')
axes[1, 0].set_ylabel('Test Score')

# Plot 4: Using lmplot style with marker customization
sns.regplot(data=df, x='Study Hours', y='Test Score',
            scatter_kws={'alpha': 0.7, 'color': 'purple', 's': 60, 'edgecolor': 'black'},
            line_kws={'color': 'orange', 'linewidth': 3, 'linestyle': '--'},
            ax=axes[1, 1])
axes[1, 1].set_title('Styled with Markers and Dashed Line')
axes[1, 1].set_xlabel('Study Hours')
axes[1, 1].set_ylabel('Test Score')

plt.tight_layout()
plt.show()

# --- Additional Example: Real-world scenario ---
print("\n" + "="*50)
print("Creating a single publication-ready plot...")
print("="*50 + "\n")

# Create a single, polished plot
plt.figure(figsize=(10, 7))
sns.regplot(data=df, x='Study Hours', y='Test Score',
            scatter_kws={'alpha': 0.6, 'color': '#2E86AB', 's': 80, 'edgecolor': 'white', 'linewidth': 0.5},
            line_kws={'color': '#A23B72', 'linewidth': 2.5})

plt.title('Relationship Between Study Hours and Test Score', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Study Hours per Week', fontsize=12)
plt.ylabel('Test Score (0-100)', fontsize=12)
plt.grid(True, alpha=0.3)

# Add correlation coefficient
correlation = df['Study Hours'].corr(df['Test Score'])
plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
         transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Print statistics
print(f"Number of data points: {len(df)}")
print(f"Correlation coefficient: {correlation:.3f}")
print(f"\nX (Study Hours) - Mean: {df['Study Hours'].mean():.2f}, Std: {df['Study Hours'].std():.2f}")
print(f"Y (Test Score) - Mean: {df['Test Score'].mean():.2f}, Std: {df['Test Score'].std():.2f}")