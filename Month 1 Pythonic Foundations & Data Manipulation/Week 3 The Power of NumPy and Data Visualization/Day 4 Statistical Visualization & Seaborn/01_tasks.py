
#Day 4: Statistical Visualization & Seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
# Set random seed for reproducibility
np.random.seed(42)

# Basic Seaborn configuration
sns.set_theme()  # Apply default Seaborn theme
print("Seaborn version:", sns.__version__)

#What is Seaborn?
#Built on top of Matplotlib
#Provides high-level interface for statistical graphics
#Beautiful default styles
#Integrates seamlessly with pandas DataFrames

#Exploring Seaborn Styles

x = np.linspace(0, 10, 100)
y = np.sin(x) + np.random.normal(0, 0.1, 100)

# Try different styles
styles = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, style in enumerate(styles):
    sns.set_style(style)
    ax = axes[idx]
    ax.plot(x, y)
    ax.set_title(f'Style: {style}')
    
# Remove extra subplot
axes[-1].axis('off')
plt.tight_layout()
plt.show()

# Reset to default
sns.set_style('darkgrid')

#Understanding Color Palettes

palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind']

fig, axes = plt.subplots(3, 2, figsize=(12, 8))
axes = axes.flatten()

for idx, palette in enumerate(palettes):
    sns.set_palette(palette)
    colors = sns.color_palette()
    ax = axes[idx]
    
    # Show palette as bars
    for i, color in enumerate(colors):
        ax.barh(i, 1, color=color)
    
    ax.set_title(f'Palette: {palette}')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    
plt.tight_layout()
plt.show()

#Creating Sample Dataset

np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'study_hours': np.random.normal(5, 2, n_samples).clip(0, 12),
    'exam_score': np.random.normal(70, 15, n_samples).clip(0, 100),
    'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 10),
    'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
    'attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
    'age': np.random.randint(18, 25, n_samples)
})

# Add correlation: more study hours -> higher scores
data['exam_score'] = data['exam_score'] + data['study_hours'] * 3
data['exam_score'] = data['exam_score'].clip(0, 100)

# Add correlation: more sleep -> lower stress
data['stress_level'] = data['stress_level'] - data['sleep_hours'] * 0.3
data['stress_level'] = data['stress_level'].clip(1, 10)

print(data.head())
print("\nDataset shape:", data.shape)
print("\nBasic statistics:\n", data.describe())



#Practice Tasks Solutions Ideas 
#Task 1: Use sns.set_style() for themes


np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'study_hours': np.random.normal(5, 2, n_samples).clip(0, 12),
    'exam_score': np.random.normal(70, 15, n_samples).clip(0, 100),
    'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 10),
    'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
    'attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
    'age': np.random.randint(18, 25, n_samples)
})

# Add correlations
data['exam_score'] = data['exam_score'] + data['study_hours'] * 3
data['exam_score'] = data['exam_score'].clip(0, 100)
data['stress_level'] = data['stress_level'] - data['sleep_hours'] * 0.3
data['stress_level'] = data['stress_level'].clip(1, 10)

print("Data created successfully!")
print(f"Shape: {data.shape}\n")

# ========================================
# TASK 1: Use sns.set_style() for themes
# ========================================

# Demonstrate all themes with a consistent plot
themes = ['darkgrid', 'whitegrid', 'dark', 'white', 'ticks']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

# Sample the data ONCE outside the loop for consistency
data_sample = data.sample(50, random_state=42)

for idx, theme in enumerate(themes):
    sns.set_style(theme)
    ax = axes[idx]
    
    # Create consistent plot
    sns.scatterplot(data=data_sample, x='study_hours', 
                    y='exam_score', ax=ax, alpha=0.6, s=50)
    sns.regplot(data=data_sample, x='study_hours', 
                y='exam_score', ax=ax, scatter=False, color='red', 
                line_kws={'linewidth': 2})
    
    ax.set_title(f'Theme: {theme}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Study Hours', fontsize=10)
    ax.set_ylabel('Exam Score', fontsize=10)

plt.suptitle('Seaborn Style Themes Comparison', 
             fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

# Reset to default
sns.set_style('darkgrid')
print("\nStyle reset to 'darkgrid'")


#Task 2: Create pairplots with sns.pairplot()


np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'study_hours': np.random.normal(5, 2, n_samples).clip(0, 12),
    'exam_score': np.random.normal(70, 15, n_samples).clip(0, 100),
    'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 10),
    'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
    'attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
    'age': np.random.randint(18, 25, n_samples)
})

# Add correlations
data['exam_score'] = data['exam_score'] + data['study_hours'] * 3
data['exam_score'] = data['exam_score'].clip(0, 100)
data['stress_level'] = data['stress_level'] - data['sleep_hours'] * 0.3
data['stress_level'] = data['stress_level'].clip(1, 10)

# Add performance category (CRITICAL!)
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

print("Data created successfully!")
print(f"Shape: {data.shape}")
print(f"\nPerformance distribution:\n{data['performance'].value_counts()}\n")


g = sns.pairplot(
    data[['study_hours', 'exam_score', 'sleep_hours', 'stress_level', 'performance']],
    hue='performance',
    palette='Set1',
    diag_kind='hist',
    corner=True,  # Only show lower triangle
    plot_kws={'alpha': 0.6, 's': 30},
    diag_kws={'bins': 20, 'alpha': 0.7}
)

g.fig.suptitle('Comprehensive Pairplot Analysis', 
               y=1.02, fontsize=18, fontweight='bold')
plt.show()


np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'study_hours': np.random.normal(5, 2, n_samples).clip(0, 12),
    'exam_score': np.random.normal(70, 15, n_samples).clip(0, 100),
    'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 10),
    'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
    'attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
    'age': np.random.randint(18, 25, n_samples)
})

# Add correlations
data['exam_score'] = data['exam_score'] + data['study_hours'] * 3
data['exam_score'] = data['exam_score'].clip(0, 100)
data['stress_level'] = data['stress_level'] - data['sleep_hours'] * 0.3
data['stress_level'] = data['stress_level'].clip(1, 10)

# Add performance category (CRITICAL!)
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

print("Data created successfully!")
print(f"Shape: {data.shape}")
print(f"\nPerformance distribution:\n{data['performance'].value_counts()}\n")


fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Sequential palette
sns.set_palette('Blues_r')
performance_means = data.groupby('performance')['exam_score'].mean().reset_index()
sns.barplot(data=performance_means,
            x='performance', y='exam_score', ax=axes[0, 0], hue='performance', legend=False)
axes[0, 0].set_title('Sequential Palette (Blues_r)', fontweight='bold', fontsize=12)
axes[0, 0].set_ylabel('Average Exam Score')
axes[0, 0].set_xlabel('Performance Level')

# 2. Diverging palette
sns.set_palette('RdYlGn')
correlation_subset = data[['study_hours', 'exam_score', 'stress_level']].corr()
sns.heatmap(correlation_subset, annot=True, ax=axes[0, 1], 
            cmap='RdYlGn', center=0, vmin=-1, vmax=1, fmt='.2f',
            linewidths=1, square=True)
axes[0, 1].set_title('Diverging Palette (RdYlGn)', fontweight='bold', fontsize=12)

# 3. Qualitative palette
sns.set_palette('Set2')
sns.violinplot(data=data, x='performance', y='study_hours', ax=axes[1, 0], hue='performance', legend=False)
axes[1, 0].set_title('Qualitative Palette (Set2)', fontweight='bold', fontsize=12)
axes[1, 0].set_xlabel('Performance Level')
axes[1, 0].set_ylabel('Study Hours')

# 4. Custom palette
custom_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
sns.set_palette(custom_colors)
sns.boxplot(data=data, x='performance', y='stress_level', ax=axes[1, 1], hue='performance', legend=False)
axes[1, 1].set_title('Custom Color Palette', fontweight='bold', fontsize=12)
axes[1, 1].set_xlabel('Performance Level')
axes[1, 1].set_ylabel('Stress Level')

plt.suptitle('Seaborn Color Palette Comparison', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# Reset palette
sns.set_palette('deep')
print("\nPalette reset to 'deep'")
'''
#Publication-Ready Visualization

np.random.seed(42)
n_samples = 200

data = pd.DataFrame({
    'study_hours': np.random.normal(5, 2, n_samples).clip(0, 12),
    'exam_score': np.random.normal(70, 15, n_samples).clip(0, 100),
    'sleep_hours': np.random.normal(7, 1.5, n_samples).clip(3, 10),
    'stress_level': np.random.normal(5, 2, n_samples).clip(1, 10),
    'attendance': np.random.normal(85, 10, n_samples).clip(0, 100),
    'age': np.random.randint(18, 25, n_samples)
})

# Add correlations
data['exam_score'] = data['exam_score'] + data['study_hours'] * 3
data['exam_score'] = data['exam_score'].clip(0, 100)
data['stress_level'] = data['stress_level'] - data['sleep_hours'] * 0.3
data['stress_level'] = data['stress_level'].clip(1, 10)

# Add performance category (CRITICAL!)
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

print("Data created successfully!")
print(f"Shape: {data.shape}")
print(f"\nPerformance distribution:\n{data['performance'].value_counts()}\n")


# Create a comprehensive, publication-ready figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Set style
sns.set_style('whitegrid')
sns.set_palette('husl')

# 1. Main scatter with regression (large)
ax_main = fig.add_subplot(gs[0:2, 0:2])
sns.scatterplot(data=data, x='study_hours', y='exam_score', 
                hue='performance', size='attendance',
                sizes=(50, 300), alpha=0.6, ax=ax_main)
sns.regplot(data=data, x='study_hours', y='exam_score', 
            scatter=False, color='black', ax=ax_main, 
            line_kws={'linewidth': 2, 'linestyle': '--'})
ax_main.set_title('Primary Analysis: Study Hours vs Exam Performance', 
                  fontsize=14, fontweight='bold', pad=15)
ax_main.set_xlabel('Daily Study Hours', fontsize=12)
ax_main.set_ylabel('Exam Score (%)', fontsize=12)
ax_main.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# 2. Distribution of exam scores
ax_dist = fig.add_subplot(gs[0, 2])
sns.histplot(data=data, y='exam_score', kde=True, ax=ax_dist, color='coral')
ax_dist.set_title('Score Distribution', fontweight='bold', fontsize=11)
ax_dist.set_ylabel('')
ax_dist.set_xlabel('Count')

# 3. Box plot by performance
ax_box = fig.add_subplot(gs[1, 2])
sns.boxplot(data=data, x='performance', y='study_hours', ax=ax_box, 
            palette='Set2', hue='performance', legend=False)
ax_box.set_title('Study Hours by Performance', fontweight='bold', fontsize=11)
ax_box.set_xlabel('Performance Level')
ax_box.set_ylabel('Study Hours')

# 4. Correlation heatmap (bottom)
ax_heat = fig.add_subplot(gs[2, :])
corr_data = data[['study_hours', 'exam_score', 'sleep_hours', 
                   'stress_level', 'attendance']].corr()
sns.heatmap(corr_data, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, ax=ax_heat,
            cbar_kws={'shrink': 0.8})
ax_heat.set_title('Correlation Matrix of Key Variables', 
                  fontweight='bold', pad=15, fontsize=12)

# Add main title
fig.suptitle('Comprehensive Student Performance Analysis Dashboard', 
             fontsize=18, fontweight='bold', y=0.995)

plt.show()

# Reset style
sns.set_style('darkgrid')
print("\nDashboard created successfully!")