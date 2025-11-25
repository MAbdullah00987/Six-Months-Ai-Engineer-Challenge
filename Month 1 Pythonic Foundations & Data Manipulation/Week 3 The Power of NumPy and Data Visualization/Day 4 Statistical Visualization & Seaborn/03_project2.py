
#Heatmap of Correlation Matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''
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

print("Data created successfully!")
print(data.head())
print("\n" + "="*50 + "\n")


correlation_matrix = data[['study_hours', 'exam_score', 'sleep_hours', 
                            'stress_level', 'attendance', 'age']].corr()

print("Correlation Matrix:")
print(correlation_matrix)
print("\n")

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True,  # Show correlation values
            cmap='coolwarm',  # Color scheme
            center=0,  # Center colormap at 0
            square=True,  # Make cells square
            linewidths=1,  # Add lines between cells
            cbar_kws={'shrink': 0.8},
            fmt='.2f')  # Format numbers to 2 decimal places

plt.title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()


#Advanced Heatmap with Custom Styling:

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

print("Data created successfully!\n")


correlation_matrix = data[['study_hours', 'exam_score', 'sleep_hours', 
                            'stress_level', 'attendance', 'age']].corr()

print("Correlation Matrix:")
print(correlation_matrix)
print("\n" + "="*60 + "\n")

# Create mask for upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Create figure with custom size
fig, ax = plt.subplots(figsize=(12, 9))

# Create heatmap with mask
sns.heatmap(correlation_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            square=True,
            linewidths=2,
            cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},
            vmin=-1, vmax=1,
            ax=ax)

plt.title('Correlation Matrix (Lower Triangle)', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

#Distribution Plots

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

# Add performance category (IMPORTANT!)
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

print("Data created successfully!")
print(f"Shape: {data.shape}")
print(f"\nPerformance distribution:\n{data['performance'].value_counts()}\n")


fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Histogram with KDE
sns.histplot(data=data, x='exam_score', kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Exam Score Distribution', fontweight='bold')

# 2. KDE plot
sns.kdeplot(data=data, x='study_hours', ax=axes[0, 1], fill=True, color='coral')
axes[0, 1].set_title('Study Hours Density', fontweight='bold')

# 3. Box plot
sns.boxplot(data=data, y='stress_level', ax=axes[0, 2], color='lightgreen')
axes[0, 2].set_title('Stress Level Box Plot', fontweight='bold')

# 4. Violin plot
sns.violinplot(data=data, x='performance', y='exam_score', 
               ax=axes[1, 0], palette='Set2')
axes[1, 0].set_title('Exam Score by Performance', fontweight='bold')

# 5. Distribution with multiple variables
sns.kdeplot(data=data, x='study_hours', hue='performance', 
            ax=axes[1, 1], fill=True, alpha=0.5, palette='viridis')
axes[1, 1].set_title('Study Hours by Performance Level', fontweight='bold')

# 6. Joint distribution
data_sample = data.sample(100)
sns.scatterplot(data=data_sample, x='study_hours', y='exam_score', 
                ax=axes[1, 2], alpha=0.6, s=50)
axes[1, 2].set_title('Study vs Score (Sample)', fontweight='bold')

plt.tight_layout()
plt.show()

#Pairplot Review

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

# Add performance category (CRITICAL - needed for pairplot!)
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

print("Data created successfully!")
print(f"Shape: {data.shape}")
print(f"\nPerformance distribution:\n{data['performance'].value_counts()}\n")

# Pairplot with hue and custom styling
sns.pairplot(data[['study_hours', 'exam_score', 'sleep_hours', 
                    'stress_level', 'performance']], 
             hue='performance',
             palette='husl',
             diag_kind='kde',  # KDE for diagonal
             plot_kws={'alpha': 0.6},
             diag_kws={'alpha': 0.7, 'linewidth': 2})

plt.suptitle('Pairplot Colored by Performance Level', 
             y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

'''

#Statistical Relationships - Advanced

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

# Add performance category
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

print("Data created successfully!")
print(f"Shape: {data.shape}\n")


# Create a 2x2 grid of advanced plots
fig = plt.figure(figsize=(16, 12))

# 1. LM Plot (combines regplot with FacetGrid)
ax1 = plt.subplot(2, 2, 1)
sns.regplot(data=data, x='study_hours', y='exam_score', 
            scatter_kws={'alpha': 0.5}, ax=ax1)
ax1.set_title('Linear Regression: Study Hours vs Exam Score', fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Residual Plot
ax2 = plt.subplot(2, 2, 2)
sns.residplot(data=data, x='study_hours', y='exam_score', 
              scatter_kws={'alpha': 0.5}, ax=ax2)
ax2.set_title('Residual Plot', fontweight='bold')
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero residual')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Joint Plot alternative visualization (2D Density)
ax3 = plt.subplot(2, 2, 3)
sns.kdeplot(data=data, x='study_hours', y='exam_score', 
            fill=True, cmap='Blues', ax=ax3, levels=10)
ax3.set_title('2D Density Plot', fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Hexbin plot for large datasets
ax4 = plt.subplot(2, 2, 4)
hexbin = ax4.hexbin(data['study_hours'], data['exam_score'], 
                    gridsize=20, cmap='YlOrRd', mincnt=1)
ax4.set_xlabel('Study Hours', fontsize=11)
ax4.set_ylabel('Exam Score', fontsize=11)
ax4.set_title('Hexbin Plot (Density)', fontweight='bold')
plt.colorbar(hexbin, ax=ax4, label='Count')

plt.tight_layout()
plt.show()