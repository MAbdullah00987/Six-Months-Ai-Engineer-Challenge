
#Part 6: Advanced Visualization with Seaborn
#Advanced Seaborn Visualizations for Optimization

#Part 6: Advanced Visualization with Seaborn
#Advanced Seaborn Visualizations for Optimization

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import multivariate_normal

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Define learning rates globally
LEARNING_RATES = [0.01, 0.05, 0.1, 0.3, 0.5]

# GENERATE OPTIMIZATION DATA

def generate_optimization_data():
    """Generate data from multiple optimization runs"""
    
    np.random.seed(42)
    
    data = []
    
    # Multiple runs for each learning rate
    for lr in LEARNING_RATES:
        for run in range(20):
            # Simulate optimization
            start = np.random.randn(2) * 3 + np.array([2, -1])
            
            # Simple quadratic: f(x,y) = (x-2)² + (y+1)²
            point = start.copy()
            
            for iteration in range(50):
                grad = 2 * (point - np.array([2, -1]))
                point = point - lr * grad
                
                value = np.sum((point - np.array([2, -1]))**2)
                distance = np.linalg.norm(point - np.array([2, -1]))
                
                data.append({
                    'learning_rate': lr,
                    'run': run,
                    'iteration': iteration,
                    'x': point[0],
                    'y': point[1],
                    'function_value': value,
                    'distance_to_optimum': distance,
                    'gradient_norm': np.linalg.norm(grad)
                })
    
    return pd.DataFrame(data)


# SEABORN VISUALIZATIONS


def create_seaborn_visualizations():
    """Create comprehensive Seaborn visualizations"""
    
    # Generate data
    print("Generating optimization data...")
    df = generate_optimization_data()
    
    # Figure 1: Multiple plot types
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Line plot: Convergence by learning rate
    ax1 = plt.subplot(3, 3, 1)
    sns.lineplot(data=df, x='iteration', y='function_value',
                hue='learning_rate', style='learning_rate',
                markers=True, dashes=False, ax=ax1, errorbar='sd')
    ax1.set_yscale('log')
    ax1.set_title('Convergence by Learning Rate', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Iteration', fontsize=11)
    ax1.set_ylabel('Function Value (log)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot: Final values distribution
    final_df = df[df['iteration'] == 49]
    ax2 = plt.subplot(3, 3, 2)
    sns.boxplot(data=final_df, x='learning_rate', y='function_value',
               hue='learning_rate', palette='Set2', ax=ax2, legend=False)
    ax2.set_yscale('log')
    ax2.set_title('Final Value Distribution', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Learning Rate', fontsize=11)
    ax2.set_ylabel('Final Function Value (log)', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Violin plot: Distance distribution
    ax3 = plt.subplot(3, 3, 3)
    sns.violinplot(data=final_df, x='learning_rate', y='distance_to_optimum',
                  hue='learning_rate', palette='muted', ax=ax3, inner='box', legend=False)
    ax3.set_yscale('log')
    ax3.set_title('Distance to Optimum Distribution', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Learning Rate', fontsize=11)
    ax3.set_ylabel('Distance (log)', fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Heatmap: Correlation matrix
    ax4 = plt.subplot(3, 3, 4)
    numeric_cols = ['iteration', 'function_value', 'distance_to_optimum', 
                    'gradient_norm', 'learning_rate']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
               center=0, square=True, ax=ax4, cbar_kws={'shrink': 0.8})
    ax4.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    
    # 5. Joint plot style scatter
    ax5 = plt.subplot(3, 3, 5)
    sample_df = df[df['learning_rate'] == 0.1].sample(min(1000, len(df)))
    sns.scatterplot(data=sample_df, x='x', y='y', hue='iteration',
                   palette='viridis', size='function_value', sizes=(20, 200),
                   alpha=0.6, ax=ax5, legend='brief')
    ax5.plot(2, -1, 'r*', markersize=25, markeredgecolor='black',
            markeredgewidth=3, label='Optimum')
    ax5.set_title('Trajectory Scatter (LR=0.1)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('x', fontsize=11)
    ax5.set_ylabel('y', fontsize=11)
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Swarm plot: Iterations needed
    iterations_to_threshold = []
    for (lr, run), group in df.groupby(['learning_rate', 'run']):
        converged = group[group['function_value'] < 0.01]
        if len(converged) > 0:
            iter_needed = converged['iteration'].min()
        else:
            iter_needed = 50
        iterations_to_threshold.append({
            'learning_rate': lr,
            'iterations_to_converge': iter_needed
        })
    
    iter_df = pd.DataFrame(iterations_to_threshold)
    ax6 = plt.subplot(3, 3, 6)
    sns.swarmplot(data=iter_df, x='learning_rate', y='iterations_to_converge',
                 hue='learning_rate', palette='Set1', size=4, ax=ax6, legend=False)
    ax6.set_title('Iterations to Converge (f < 0.01)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Learning Rate', fontsize=11)
    ax6.set_ylabel('Iterations', fontsize=11)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. KDE plot: Function value distribution
    ax7 = plt.subplot(3, 3, 7)
    for lr in LEARNING_RATES:
        subset = df[(df['learning_rate'] == lr) & (df['iteration'] == 49)]
        sns.kdeplot(data=subset, x='function_value', label=f'LR={lr}',
                   ax=ax7, fill=True, alpha=0.5)
    ax7.set_xscale('log')
    ax7.set_title('Final Value KDE by Learning Rate', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Function Value (log)', fontsize=11)
    ax7.set_ylabel('Density', fontsize=11)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Strip plot with box overlay
    ax8 = plt.subplot(3, 3, 8)
    sns.stripplot(data=final_df, x='learning_rate', y='function_value',
                 hue='learning_rate', palette='pastel', alpha=0.5, jitter=True, 
                 ax=ax8, legend=False)
    sns.boxplot(data=final_df, x='learning_rate', y='function_value',
               hue='learning_rate', palette='Set2', ax=ax8, showfliers=False, 
               width=0.3, boxprops=dict(alpha=0.7), legend=False)
    ax8.set_yscale('log')
    ax8.set_title('Value Distribution with Overlay', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Learning Rate', fontsize=11)
    ax8.set_ylabel('Function Value (log)', fontsize=11)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Pair plot for subset (shown as heatmap)
    ax9 = plt.subplot(3, 3, 9)
    pivot_data = df.pivot_table(
        values='function_value',
        index='iteration',
        columns='learning_rate',
        aggfunc='mean'
    )
    sns.heatmap(np.log1p(pivot_data.T), cmap='YlOrRd', ax=ax9,
               cbar_kws={'label': 'log(1 + f(x,y))'})
    ax9.set_title('Convergence Heatmap', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Iteration', fontsize=11)
    ax9.set_ylabel('Learning Rate', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('seaborn_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    

    print("\nSEABORN VISUALIZATION TYPES DEMONSTRATED:")
    print("1. lineplot    - Time series with confidence intervals")
    print("2. boxplot     - Distribution summaries")
    print("3. violinplot  - Distribution shapes")
    print("4. heatmap     - Correlation matrices")
    print("5. scatterplot - Multi-dimensional relationships")
    print("6. swarmplot   - Categorical distributions")
    print("7. kdeplot     - Density estimation")
    print("8. stripplot   - Individual observations")
    print("9. pivot table - Aggregated heatmaps")
    

# ADVANCED STATISTICAL PLOTS


def create_statistical_analysis():
    """Create advanced statistical visualizations"""
    
    df = generate_optimization_data()
    
    fig = plt.figure(figsize=(18, 10))
    
    # 1. Regression plot
    ax1 = plt.subplot(2, 3, 1)
    sample = df[df['learning_rate'] == 0.1].sample(min(500, len(df)))
    sns.regplot(data=sample, x='iteration', y='function_value',
               scatter_kws={'alpha': 0.3}, line_kws={'color': 'red', 'linewidth': 2},
               ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title('Regression: Iteration vs Value', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax2 = plt.subplot(2, 3, 2)
    sns.residplot(data=sample, x='iteration', y='function_value',
                 lowess=True, scatter_kws={'alpha': 0.3},
                 line_kws={'color': 'red', 'linewidth': 2}, ax=ax2)
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Count plot
    converged_df = df[df['function_value'] < 0.01].groupby(
        ['learning_rate', 'run']).size().reset_index(name='converged')
    converged_df['converged'] = converged_df['converged'] > 0
    
    ax3 = plt.subplot(2, 3, 3)
    sns.countplot(data=converged_df, x='learning_rate', hue='converged',
                 palette='Set2', ax=ax3)
    ax3.set_title('Convergence Success Rate', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Learning Rate', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.legend(title='Converged')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Joint KDE
    ax4 = plt.subplot(2, 3, 4)
    final_sample = df[df['iteration'] == 49].sample(min(1000, len(df)))
    sns.kdeplot(data=final_sample, x='x', y='y', fill=True, 
               cmap='Blues', levels=10, ax=ax4)
    ax4.plot(2, -1, 'r*', markersize=25, markeredgecolor='black',
            markeredgewidth=3, label='Optimum')
    ax4.set_title('2D KDE of Final Positions', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Bar plot with error bars
    ax5 = plt.subplot(2, 3, 5)
    summary = df.groupby('learning_rate').agg({
        'function_value': ['mean', 'std']
    }).reset_index()
    summary.columns = ['learning_rate', 'mean_value', 'std_value']
    
    sns.barplot(data=summary, x='learning_rate', y='mean_value',
               hue='learning_rate', palette='viridis', ax=ax5, errorbar=None, legend=False)
    ax5.errorbar(range(len(summary)), summary['mean_value'],
                yerr=summary['std_value'], fmt='none', color='black',
                capsize=5, linewidth=2)
    ax5.set_yscale('log')
    ax5.set_title('Mean Function Value ± Std', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Learning Rate', fontsize=11)
    ax5.set_ylabel('Mean Value (log)', fontsize=11)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. FacetGrid concept (manual)
    ax6 = plt.subplot(2, 3, 6)
    for lr in [0.01, 0.1, 0.5]:
        subset = df[df['learning_rate'] == lr]
        grouped = subset.groupby('iteration')['function_value'].mean()
        ax6.plot(grouped.index, grouped.values, label=f'LR={lr}',
                linewidth=2, marker='o', markersize=4)
    
    ax6.set_yscale('log')
    ax6.set_title('Comparison of Select Learning Rates', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Iteration', fontsize=11)
    ax6.set_ylabel('Mean Function Value (log)', fontsize=11)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('seaborn_statistical.png', dpi=300, bbox_inches='tight')
    plt.show()


# CUSTOM STYLED PLOTS

def create_publication_quality():
    """Create publication-quality plots with Seaborn styling"""
    
    df = generate_optimization_data()
    
    # Use different Seaborn styles
    styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, style in enumerate(styles):
        sns.set_style(style)
        ax = axes[idx]
        
        # Sample data
        sample = df[df['learning_rate'].isin([0.01, 0.1, 0.5])]
        
        sns.lineplot(data=sample, x='iteration', y='distance_to_optimum',
                    hue='learning_rate', style='learning_rate',
                    markers=True, dashes=False, ax=ax, errorbar='sd',
                    palette='deep', linewidth=2.5)
        
        ax.set_yscale('log')
        ax.set_title(f'Style: {style}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=11)
        ax.set_ylabel('Distance to Optimum (log)', fontsize=11)
        ax.legend(title='Learning Rate', fontsize=9)
        
        if style in ['white', 'ticks']:
            sns.despine(ax=ax)
    
    # Last plot: comparison summary
    sns.set_style('whitegrid')
    ax = axes[5]
    
    final = df[df['iteration'] == 49]
    summary = final.groupby('learning_rate')['function_value'].agg(['mean', 'std'])
    
    bars = ax.bar(range(len(summary)), summary['mean'], 
                  color=sns.color_palette('husl', len(summary)),
                  edgecolor='black', linewidth=2)
    ax.errorbar(range(len(summary)), summary['mean'], yerr=summary['std'],
               fmt='none', color='black', capsize=5, linewidth=2)
    
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([f'{lr:.2f}' for lr in summary.index])
    ax.set_yscale('log')
    ax.set_xlabel('Learning Rate', fontsize=11)
    ax.set_ylabel('Final Value (log)', fontsize=11)
    ax.set_title('Final Performance Summary', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('seaborn_styles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Reset to default
    sns.set_style('whitegrid')


# MAIN EXECUTION



if __name__ == "__main__":
    print("ADVANCED SEABORN VISUALIZATIONS FOR OPTIMIZATION")
    print("=" * 70)
    
    print("\n1. Creating comprehensive Seaborn visualizations...")
    create_seaborn_visualizations()
    
    print("\n2. Creating statistical analysis plots...")
    create_statistical_analysis()
    
    print("\n3. Creating publication-quality styled plots...")
    create_publication_quality()
    