

#Phase 5: Enhanced Visualization - Seaborn
#Seaborn for Statistical Plots



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_theme(style="whitegrid", palette="husl")

print("=" * 80)
print("SEABORN STATISTICAL VISUALIZATION - CLT Analysis")
print("=" * 80)

# 1. CREATE COMPREHENSIVE DATASET
print("\n1. CREATING CLT SIMULATION DATASET")
print("-" * 80)

distributions = {
    'Uniform': lambda: np.random.uniform(0, 10, 10000),
    'Exponential': lambda: np.random.exponential(2, 10000),
    'Normal': lambda: np.random.normal(5, 2, 10000),
    'Poisson': lambda: np.random.poisson(5, 10000)
}

data_list = []
sample_sizes = [5, 10, 20, 30, 50, 100]
n_simulations = 1000

for dist_name, dist_func in distributions.items():
    population = dist_func()
    
    for n in sample_sizes:
        samples = np.random.choice(population, size=(n_simulations, n))
        sample_means = samples.mean(axis=1)
        
        for mean in sample_means:
            data_list.append({
                'Distribution': dist_name,
                'Sample_Size': n,
                'Sample_Mean': mean,
                'Category': f'n={n}' if n <= 30 else 'n>30'
            })

df = pd.DataFrame(data_list)
print(f"Dataset created: {df.shape[0]} rows")
print(df.groupby(['Distribution', 'Sample_Size']).size())

# Create figure for Seaborn visualizations
fig = plt.figure(figsize=(20, 14))
fig.suptitle('Central Limit Theorem: Seaborn Statistical Visualization Suite', 
             fontsize=22, fontweight='bold', y=0.995)

# 2. DISTRIBUTION PLOTS (violinplot)
print("\n2. VIOLIN PLOTS - DISTRIBUTION SHAPES")
print("-" * 80)

ax1 = plt.subplot(3, 3, 1)
sns.violinplot(data=df[df['Sample_Size'].isin([5, 30, 100])], 
               x='Sample_Size', y='Sample_Mean', hue='Distribution',
               split=False, inner='quartile', ax=ax1)
ax1.set_title('Distribution by Sample Size', fontweight='bold', fontsize=12)
ax1.set_xlabel('Sample Size', fontweight='bold')
ax1.set_ylabel('Sample Mean', fontweight='bold')
ax1.legend(title='Distribution', fontsize=8, loc='upper right')

# 3. BOX PLOTS - VARIANCE REDUCTION (FIXED - removed palette warning)
ax2 = plt.subplot(3, 3, 2)
sns.boxplot(data=df[df['Distribution'] == 'Exponential'], 
            x='Sample_Size', y='Sample_Mean', 
            hue='Sample_Size', legend=False, palette='Set2', ax=ax2)
ax2.set_title('Variance Reduction (Exponential)', fontweight='bold', fontsize=12)
ax2.set_xlabel('Sample Size', fontweight='bold')
ax2.set_ylabel('Sample Mean', fontweight='bold')

# 4. KDE PLOTS - OVERLAID DENSITIES
ax3 = plt.subplot(3, 3, 3)
for n in [5, 20, 50]:
    subset = df[(df['Distribution'] == 'Uniform') & (df['Sample_Size'] == n)]
    sns.kdeplot(data=subset, x='Sample_Mean', label=f'n={n}', 
                linewidth=2.5, ax=ax3)
ax3.set_title('KDE: Convergence to Normal', fontweight='bold', fontsize=12)
ax3.set_xlabel('Sample Mean', fontweight='bold')
ax3.set_ylabel('Density', fontweight='bold')
ax3.legend()

# 5. FACET GRID - MULTIPLE DISTRIBUTIONS
print("\n3. FACET GRID - COMPREHENSIVE VIEW")
print("-" * 80)

g = sns.FacetGrid(df[df['Sample_Size'].isin([10, 30, 50])], 
                  col='Sample_Size', row='Distribution', 
                  height=2.5, aspect=1.2, margin_titles=True)
g.map(sns.histplot, 'Sample_Mean', kde=True, bins=30, color='steelblue')
g.set_titles(row_template='{row_name}', col_template='n = {col_name}', 
             size=10, weight='bold')
g.set_axis_labels('Sample Mean', 'Frequency', fontweight='bold')
g.fig.subplots_adjust(top=0.92)
g.fig.suptitle('CLT Facet Grid: All Distributions', fontweight='bold', fontsize=14)
plt.savefig('seaborn_facet_grid.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. REGRESSION PLOT - STANDARD ERROR
ax4 = plt.subplot(3, 3, 4)
summary_data = df.groupby(['Distribution', 'Sample_Size']).agg({
    'Sample_Mean': 'std'
}).reset_index()
summary_data.columns = ['Distribution', 'Sample_Size', 'Standard_Error']

sns.scatterplot(data=summary_data, x='Sample_Size', y='Standard_Error', 
                hue='Distribution', s=100, ax=ax4)
for dist in summary_data['Distribution'].unique():
    subset = summary_data[summary_data['Distribution'] == dist]
    sns.regplot(data=subset, x='Sample_Size', y='Standard_Error', 
                scatter=False, ax=ax4)
ax4.set_title('Standard Error Decay', fontweight='bold', fontsize=12)
ax4.set_xlabel('Sample Size', fontweight='bold')
ax4.set_ylabel('Standard Error', fontweight='bold')

# 7. HEATMAP - NORMALITY SCORES
ax5 = plt.subplot(3, 3, 5)
normality_data = []
for dist in distributions.keys():
    for n in sample_sizes:
        subset = df[(df['Distribution'] == dist) & (df['Sample_Size'] == n)]
        skewness = abs(subset['Sample_Mean'].skew())
        normality_data.append({'Distribution': dist, 'n': n, 'Skewness': skewness})

normality_df = pd.DataFrame(normality_data)
heatmap_data = normality_df.pivot(index='Distribution', columns='n', values='Skewness')

sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
            cbar_kws={'label': '|Skewness|'}, ax=ax5)
ax5.set_title('Skewness Heatmap (Lower = More Normal)', fontweight='bold', fontsize=12)
ax5.set_xlabel('Sample Size', fontweight='bold')
ax5.set_ylabel('Distribution', fontweight='bold')

# 8. STRIP PLOT WITH SWARM
ax6 = plt.subplot(3, 3, 6)
sample_data = df[(df['Distribution'].isin(['Exponential', 'Uniform'])) & 
                 (df['Sample_Size'] == 30)].sample(200)
sns.stripplot(data=sample_data, x='Distribution', y='Sample_Mean', 
              hue='Distribution', dodge=True, alpha=0.5, size=3, ax=ax6)
sns.boxplot(data=sample_data, x='Distribution', y='Sample_Mean', 
            width=0.3, showcaps=False, boxprops={'facecolor': 'None'},
            showfliers=False, ax=ax6)
ax6.set_title('Strip Plot: Individual Samples (n=30)', fontweight='bold', fontsize=12)
ax6.set_xlabel('Distribution', fontweight='bold')
ax6.set_ylabel('Sample Mean', fontweight='bold')
ax6.legend([],[], frameon=False)

# 9. PAIRPLOT DATA (separate figure)
print("\n4. PAIRPLOT - MULTIVARIATE RELATIONSHIPS")
print("-" * 80)

pairplot_data = df[df['Sample_Size'].isin([10, 30, 50])].copy()
pairplot_data['Log_n'] = np.log(pairplot_data['Sample_Size'])

# Create a smaller sample for pairplot
pairplot_sample = pairplot_data.sample(n=1000, random_state=42)

pair_grid = sns.pairplot(pairplot_sample, 
                         vars=['Sample_Mean', 'Sample_Size'],
                         hue='Distribution', 
                         diag_kind='kde',
                         plot_kws={'alpha': 0.6, 's': 20},
                         height=2.5)
pair_grid.fig.suptitle('Pairplot: CLT Relationships', y=1.02, 
                       fontweight='bold', fontsize=14)
plt.savefig('seaborn_pairplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. JOINT PLOT
print("\n5. JOINT PLOT - BIVARIATE ANALYSIS")
print("-" * 80)

joint_data = df[(df['Distribution'] == 'Exponential') & 
                (df['Sample_Size'].isin([10, 30, 50]))].sample(500)

joint_grid = sns.jointplot(data=joint_data, x='Sample_Size', y='Sample_Mean',
                           kind='hex', color='steelblue', height=7)
joint_grid.fig.suptitle('Joint Distribution: Sample Size vs Mean', 
                        y=1.02, fontweight='bold', fontsize=14)
plt.savefig('seaborn_jointplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. COUNT PLOT (for categorical analysis)
ax7 = plt.subplot(3, 3, 7)
category_data = df.copy()
category_data['Normal_Range'] = pd.cut(category_data['Sample_Mean'], 
                                       bins=[-np.inf, 4, 6, np.inf],
                                       labels=['Low', 'Medium', 'High'])
sns.countplot(data=category_data[category_data['Sample_Size'] == 30], 
              x='Distribution', hue='Normal_Range', ax=ax7)
ax7.set_title('Sample Mean Categories (n=30)', fontweight='bold', fontsize=12)
ax7.set_xlabel('Distribution', fontweight='bold')
ax7.set_ylabel('Count', fontweight='bold')
ax7.legend(title='Range')

# 12. POINT PLOT - MEAN WITH CI (FIXED - updated errwidth parameter)
ax8 = plt.subplot(3, 3, 8)
sns.pointplot(data=df[df['Distribution'].isin(['Exponential', 'Normal'])], 
              x='Sample_Size', y='Sample_Mean', hue='Distribution',
              markers=['o', 's'], linestyles=['-', '--'], 
              capsize=0.1, err_kws={'linewidth': 1.5}, ax=ax8)
ax8.set_title('Point Plot: Mean ± 95% CI', fontweight='bold', fontsize=12)
ax8.set_xlabel('Sample Size', fontweight='bold')
ax8.set_ylabel('Sample Mean', fontweight='bold')

# 13. RESIDUAL PLOT (FIXED - convert range to numpy array)
ax9 = plt.subplot(3, 3, 9)
residual_data = df[(df['Distribution'] == 'Exponential') & 
                   (df['Sample_Size'] == 50)].sample(200).reset_index(drop=True)
expected_mean = residual_data['Sample_Mean'].mean()
residual_data['Residual'] = residual_data['Sample_Mean'] - expected_mean
residual_data['Index'] = np.arange(len(residual_data))

sns.residplot(data=residual_data, x='Index', 
              y='Residual', lowess=True, color='coral', ax=ax9)
ax9.axhline(0, color='black', linestyle='--', linewidth=1)
ax9.set_title('Residual Plot: Deviations from Mean', fontweight='bold', fontsize=12)
ax9.set_xlabel('Sample Index', fontweight='bold')
ax9.set_ylabel('Residual', fontweight='bold')

plt.tight_layout()
plt.savefig('seaborn_clt_comprehensive.png', dpi=300, bbox_inches='tight')
plt.show()


print("SEABORN VISUALIZATIONS CREATED:")

print("✓ Violin plots for distribution shapes")
print("✓ Box plots for variance analysis")
print("✓ KDE plots for density estimation")
print("✓ Facet grids for multi-dimensional views")
print("✓ Heatmaps for normality metrics")
print("✓ Regression plots for trend analysis")
print("✓ Pairplots for multivariate relationships")
print("✓ Joint plots for bivariate distributions")
