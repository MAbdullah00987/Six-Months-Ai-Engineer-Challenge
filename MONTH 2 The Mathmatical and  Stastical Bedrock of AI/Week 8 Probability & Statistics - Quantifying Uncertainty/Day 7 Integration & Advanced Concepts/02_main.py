

#Phase 2: Data Handling - Pandas
#Organizing CLT Simulations with Pandas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 70)
print("PANDAS FOR CLT - Organizing Simulation Data")
print("=" * 70)

# 1. Create a DataFrame to store simulation results
print("\n1. STORING CLT SIMULATION DATA")
print("-" * 70)

# Simulate from different distributions
distributions = {
    'Uniform': np.random.uniform(0, 10, 10000),
    'Exponential': np.random.exponential(2, 10000),
    'Binomial': np.random.binomial(10, 0.3, 10000),
    'Poisson': np.random.poisson(3, 10000)
}

# Create DataFrame to store results
results = []
sample_sizes = [5, 10, 30, 50, 100]
n_simulations = 1000

for dist_name, population in distributions.items():
    for n in sample_sizes:
        # Generate sample means
        samples = np.random.choice(population, size=(n_simulations, n))
        sample_means = samples.mean(axis=1)
        
        # Calculate statistics
        for mean in sample_means:
            results.append({
                'Distribution': dist_name,
                'Sample_Size': n,
                'Sample_Mean': mean,
                'Pop_Mean': population.mean(),
                'Pop_Std': population.std()
            })

df = pd.DataFrame(results)
print(df.head(10))
print(f"\nDataFrame Shape: {df.shape}")

# 2. Groupby Analysis - Power of Pandas
print("\n2. GROUPBY ANALYSIS - CLT CONVERGENCE")
print("-" * 70)

summary = df.groupby(['Distribution', 'Sample_Size']).agg({
    'Sample_Mean': ['mean', 'std', 'count'],
    'Pop_Mean': 'first',
    'Pop_Std': 'first'
}).round(4)

summary.columns = ['Mean_of_Means', 'Std_Error', 'Count', 'True_Mean', 'Pop_Std']
summary['Theoretical_SE'] = summary['Pop_Std'] / np.sqrt(summary.index.get_level_values('Sample_Size'))
summary['SE_Ratio'] = (summary['Std_Error'] / summary['Theoretical_SE']).round(3)

print(summary)

# 3. Pivot Tables for Analysis
print("\n3. PIVOT TABLE - STANDARD ERROR BY DISTRIBUTION")
print("-" * 70)

pivot = df.groupby(['Distribution', 'Sample_Size'])['Sample_Mean'].std().reset_index()
pivot_table = pivot.pivot(index='Distribution', columns='Sample_Size', values='Sample_Mean')
print(pivot_table.round(4))

# 4. Calculate Normality Score (Skewness & Kurtosis)
print("\n4. NORMALITY METRICS")
print("-" * 70)

normality = df.groupby(['Distribution', 'Sample_Size'])['Sample_Mean'].agg([
    ('Skewness', lambda x: x.skew()),
    ('Kurtosis', lambda x: x.kurtosis())
]).round(4)

print(normality.head(12))

# 5. Filter and Query Operations
print("\n5. FILTERING DATA FOR SPECIFIC ANALYSIS")
print("-" * 70)

# Get only large sample sizes (n >= 30)
large_samples = df[df['Sample_Size'] >= 30]
print(f"Large samples (n>=30): {len(large_samples)} records")

# Query specific distribution
exponential_data = df.query("Distribution == 'Exponential' and Sample_Size == 30")
print(f"Exponential (n=30): Mean={exponential_data['Sample_Mean'].mean():.4f}")

# 6. Time-Series Style Analysis (Order matters in some experiments)
print("\n6. CUMULATIVE MEAN CONVERGENCE")
print("-" * 70)

# Show how mean converges as we take more samples
sample_data = df[(df['Distribution'] == 'Exponential') & (df['Sample_Size'] == 30)].head(500)
sample_data['Cumulative_Mean'] = sample_data['Sample_Mean'].expanding().mean()
print(sample_data[['Sample_Mean', 'Cumulative_Mean']].tail())

# Visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('Pandas Data Analysis for CLT', fontsize=18, fontweight='bold')

# Plot 1: Standard Error Heatmap
ax1 = fig.add_subplot(gs[0, :2])
pivot_plot = pivot_table.T
for col in pivot_plot.columns:
    ax1.plot(pivot_plot.index, pivot_plot[col], marker='o', linewidth=2, label=col, markersize=8)
ax1.set_xlabel('Sample Size', fontsize=12, fontweight='bold')
ax1.set_ylabel('Standard Error', fontsize=12, fontweight='bold')
ax1.set_title('Standard Error Convergence by Distribution', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Normality Score (Skewness) - FIXED
ax2 = fig.add_subplot(gs[0, 2])
# Reset index to make it accessible as columns
skew_data = normality.reset_index()
for dist in skew_data['Distribution'].unique():
    data = skew_data[skew_data['Distribution'] == dist]
    ax2.plot(data['Sample_Size'], data['Skewness'].abs(), marker='s', label=dist)
ax2.set_xlabel('Sample Size', fontsize=10, fontweight='bold')
ax2.set_ylabel('|Skewness|', fontsize=10, fontweight='bold')
ax2.set_title('Skewness Reduction', fontsize=12, fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Plots 3-6: Distribution histograms for n=30
distributions_list = ['Uniform', 'Exponential', 'Binomial', 'Poisson']
positions = [(1, 0), (1, 1), (1, 2), (2, 0)]

for dist, pos in zip(distributions_list, positions):
    ax = fig.add_subplot(gs[pos])
    data = df[(df['Distribution'] == dist) & (df['Sample_Size'] == 30)]['Sample_Mean']
    ax.hist(data, bins=40, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.set_title(f'{dist} (n=30)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Frequency')
    ax.legend()

# Plot 7: Cumulative Mean Convergence
ax7 = fig.add_subplot(gs[2, 1:])
ax7.plot(sample_data.index[:200], sample_data['Cumulative_Mean'].iloc[:200], 
         linewidth=2, color='darkgreen', label='Cumulative Mean')
ax7.axhline(sample_data['Pop_Mean'].iloc[0], color='red', linestyle='--', 
            linewidth=2, label='True Population Mean')
ax7.set_xlabel('Number of Samples', fontsize=12, fontweight='bold')
ax7.set_ylabel('Cumulative Mean', fontsize=12, fontweight='bold')
ax7.set_title('Law of Large Numbers - Mean Convergence', fontsize=14, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

plt.savefig('pandas_clt_analysis.png', dpi=300, bbox_inches='tight')
plt.show()


print("PANDAS SKILLS DEMONSTRATED:")
print("✓ DataFrame creation from simulations")
print("✓ GroupBy aggregations for statistical analysis")
print("✓ Pivot tables for comparison")
print("✓ Filtering and querying data")
print("✓ Cumulative statistics (expanding window)")
