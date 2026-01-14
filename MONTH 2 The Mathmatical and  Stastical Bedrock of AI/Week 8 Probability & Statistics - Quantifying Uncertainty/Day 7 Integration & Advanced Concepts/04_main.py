

#Phase 4: Statistical Analysis - SciPy & StatsModels
#Topic 4: Statistical Testing with SciPy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from scipy import stats

np.random.seed(42)

print("=" * 70)
print("MATPLOTLIB MASTERY - Professional CLT Visualizations")
print("=" * 70)

# Generate data
population = np.random.exponential(2, 10000)
sample_sizes = [5, 10, 30, 50]
n_simulations = 1000

# Create the masterpiece figure
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

# Custom color palette
colors = {
    'population': '#FF6B6B',
    'samples': '#4ECDC4',
    'normal': '#45B7D1',
    'accent': '#FFA07A'
}

# Main title with custom styling
fig.suptitle('Central Limit Theorem: From Chaos to Order', 
             fontsize=24, fontweight='bold', y=0.98,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# ROW 1: Population + Annotation
ax_pop = fig.add_subplot(gs[0, :2])
counts, bins, patches = ax_pop.hist(population, bins=60, alpha=0.7, 
                                     color=colors['population'], edgecolor='black', linewidth=1.5)

# Color gradient for bars
cm = plt.cm.Reds
for i, patch in enumerate(patches):
    patch.set_facecolor(cm(i / len(patches)))

ax_pop.axvline(population.mean(), color='darkred', linestyle='--', 
               linewidth=3, label=f'μ = {population.mean():.2f}')
ax_pop.set_title('Population Distribution (Exponential)', 
                 fontsize=16, fontweight='bold', pad=15)
ax_pop.set_xlabel('Value', fontsize=13, fontweight='bold')
ax_pop.set_ylabel('Frequency', fontsize=13, fontweight='bold')
ax_pop.legend(fontsize=11, loc='upper right')
ax_pop.grid(True, alpha=0.2, linestyle='--')

# Add statistical annotations
textstr = f'μ = {population.mean():.3f}\nσ = {population.std():.3f}\nn = {len(population)}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax_pop.text(0.75, 0.75, textstr, transform=ax_pop.transAxes, fontsize=12,
            verticalalignment='top', bbox=props, fontfamily='monospace')

# ROW 1: Theory explanation box
ax_theory = fig.add_subplot(gs[0, 2:])
ax_theory.axis('off')
theory_text = """
CENTRAL LIMIT THEOREM

Given:
• Any population with mean μ and std σ
• Take samples of size n
• Calculate mean of each sample

Then as n → ∞:
• Sample means ~ Normal(μ, σ/√n)
• Works regardless of population shape!

Key Formula:
    SE = σ / √n
    
Where:
    SE = Standard Error
    σ  = Population std dev
    n  = Sample size
"""
ax_theory.text(0.1, 0.9, theory_text, transform=ax_theory.transAxes,
              fontsize=12, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

# ROW 2 & 3: Sample distributions for different n
for idx, n in enumerate(sample_sizes):
    row = 1 + idx // 2
    col = idx % 2 * 2
    
    # Generate sample means
    samples = np.random.choice(population, size=(n_simulations, n))
    sample_means = samples.mean(axis=1)
    
    # Histogram
    ax = fig.add_subplot(gs[row, col:col+2])
    counts, bins, patches = ax.hist(sample_means, bins=50, alpha=0.7, 
                                    color=colors['samples'], edgecolor='black',
                                    density=True, label='Sample Means')
    
    # Overlay theoretical normal distribution
    mu, sigma = population.mean(), population.std() / np.sqrt(n)
    x = np.linspace(sample_means.min(), sample_means.max(), 100)
    theoretical_normal = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, theoretical_normal, 'r-', linewidth=3, 
            label=f'Normal({mu:.2f}, {sigma:.2f})', alpha=0.8)
    
    # Vertical line for mean
    ax.axvline(sample_means.mean(), color='darkgreen', linestyle='--', 
               linewidth=2.5, label=f'x̄ = {sample_means.mean():.2f}')
    
    # Styling
    ax.set_title(f'Sample Size n = {n}', fontsize=14, fontweight='bold', pad=10)
    ax.set_xlabel('Sample Mean', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Statistics box
    stats_text = f'x̄ = {sample_means.mean():.3f}\nSE = {sample_means.std():.3f}\nσ/√n = {sigma:.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
            fontfamily='monospace')

# Add watermark
fig.text(0.99, 0.01, 'Created with Matplotlib | CLT Visualization', 
         ha='right', va='bottom', fontsize=10, alpha=0.5, style='italic')

plt.savefig('matplotlib_clt_masterpiece.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# Additional: Interactive comparison plot
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('CLT Convergence: Standard Error Analysis', 
              fontsize=18, fontweight='bold')

sample_range = range(2, 101)
standard_errors = []

for n in sample_range:
    samples = np.random.choice(population, size=(500, n))
    se = samples.mean(axis=1).std()
    standard_errors.append(se)

theoretical_se = [population.std() / np.sqrt(n) for n in sample_range]

# Plot 1: SE vs Sample Size
axes[0, 0].plot(sample_range, standard_errors, 'o-', color='blue', 
                linewidth=2, markersize=3, label='Observed SE', alpha=0.7)
axes[0, 0].plot(sample_range, theoretical_se, '--', color='red', 
                linewidth=2, label='Theoretical SE (σ/√n)')
axes[0, 0].set_xlabel('Sample Size (n)', fontweight='bold')
axes[0, 0].set_ylabel('Standard Error', fontweight='bold')
axes[0, 0].set_title('Standard Error Convergence', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Log scale
axes[0, 1].loglog(sample_range, standard_errors, 'o-', color='blue', 
                  linewidth=2, markersize=3, label='Observed SE', alpha=0.7)
axes[0, 1].loglog(sample_range, theoretical_se, '--', color='red', 
                  linewidth=2, label='Theoretical SE')
axes[0, 1].set_xlabel('Sample Size (n)', fontweight='bold')
axes[0, 1].set_ylabel('Standard Error', fontweight='bold')
axes[0, 1].set_title('Log-Log Scale (Shows √n relationship)', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, which='both')

# Plot 3: Relative Error
relative_error = [(obs - theo) / theo * 100 
                  for obs, theo in zip(standard_errors, theoretical_se)]
axes[1, 0].plot(sample_range, relative_error, color='green', linewidth=2)
axes[1, 0].axhline(0, color='red', linestyle='--', linewidth=1)
axes[1, 0].set_xlabel('Sample Size (n)', fontweight='bold')
axes[1, 0].set_ylabel('Relative Error (%)', fontweight='bold')
axes[1, 0].set_title('Accuracy of CLT Approximation', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Sample size recommendations
axes[1, 1].axis('off')
recommendation_text = """
SAMPLE SIZE GUIDELINES

n < 10:   Unreliable
          High variance
          
n = 30:   Rule of thumb
          Good for most cases
          
n ≥ 50:   Excellent
          Very close to normal
          
n ≥ 100:  Optimal
          Minimal error

Formula: SE = σ / √n
"""
axes[1, 1].text(0.1, 0.9, recommendation_text, transform=axes[1, 1].transAxes,
               fontsize=13, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.6))

plt.tight_layout()
plt.savefig('matplotlib_clt_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("MATPLOTLIB TECHNIQUES USED:")
print("✓ GridSpec for complex layouts")
print("✓ Color gradients and custom palettes")
print("✓ Overlay plots (histogram + theoretical curve)")
print("✓ Text annotations and LaTeX-style math")
print("✓ Multiple subplot configurations")
print("✓ Log-log scales for relationship analysis")
