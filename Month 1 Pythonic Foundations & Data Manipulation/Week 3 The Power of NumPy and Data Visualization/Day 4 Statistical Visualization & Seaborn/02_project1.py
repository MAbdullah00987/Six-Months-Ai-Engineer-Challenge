
# Scatter Plot with Regression Line



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

print(data.head())
print("\nDataset shape:", data.shape)

# NOW PLOT IT
plt.figure(figsize=(10, 6))
sns.regplot(data=data, x='study_hours', y='exam_score', 
            scatter_kws={'alpha': 0.5}, 
            line_kws={'color': 'red', 'linewidth': 2})

plt.title('Relationship: Study Hours vs Exam Score', fontsize=14, fontweight='bold')
plt.xlabel('Study Hours per Day', fontsize=12)
plt.ylabel('Exam Score (%)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()


#Advanced Scatter Plot with Multiple Features:

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


# Create a more complex visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Study hours vs exam score with confidence interval
sns.regplot(data=data, x='study_hours', y='exam_score', 
            ax=axes[0, 0], scatter_kws={'alpha': 0.6})
axes[0, 0].set_title('Study Hours vs Exam Score', fontweight='bold')

# 2. Sleep hours vs stress level
sns.regplot(data=data, x='sleep_hours', y='stress_level', 
            ax=axes[0, 1], color='green', scatter_kws={'alpha': 0.6})
axes[0, 1].set_title('Sleep Hours vs Stress Level', fontweight='bold')

# 3. Attendance vs exam score
sns.regplot(data=data, x='attendance', y='exam_score', 
            ax=axes[1, 0], color='purple', scatter_kws={'alpha': 0.6})
axes[1, 0].set_title('Attendance vs Exam Score', fontweight='bold')

# 4. Age vs study hours (likely no correlation)
sns.regplot(data=data, x='age', y='study_hours', 
            ax=axes[1, 1], color='orange', scatter_kws={'alpha': 0.6})
axes[1, 1].set_title('Age vs Study Hours', fontweight='bold')

plt.tight_layout()
plt.show()

'''

# ========================================
# CREATE THE DATA FIRST
# ========================================
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

print("Data created!")
print(data.head())

# ========================================
# ADD PERFORMANCE CATEGORY AND PLOT
# ========================================
# Add a category column
data['performance'] = pd.cut(data['exam_score'], 
                              bins=[0, 60, 80, 100], 
                              labels=['Low', 'Medium', 'High'])

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='study_hours', y='exam_score', 
                hue='performance', size='attendance',
                palette='viridis', alpha=0.7, sizes=(20, 200))

plt.title('Study Hours vs Exam Score (by Performance Level)', 
          fontsize=14, fontweight='bold')
plt.xlabel('Study Hours per Day', fontsize=12)
plt.ylabel('Exam Score (%)', fontsize=12)
plt.legend(title='Performance', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
