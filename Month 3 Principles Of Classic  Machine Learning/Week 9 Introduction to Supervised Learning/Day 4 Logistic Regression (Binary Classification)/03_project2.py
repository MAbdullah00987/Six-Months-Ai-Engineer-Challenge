

#Project 2: Iris Flower Classification**
# Start with binary classification (2 classes)
# Then extend to multi-class (3 classes)
# Visualize decision boundaries
# Create confusion matrix


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# =====================================
# PART 1: DATA LOADING AND EXPLORATION
# =====================================

print("=" * 60)
print("IRIS FLOWER CLASSIFICATION PROJECT")
print("=" * 60)

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\n1. Dataset Overview:")
print("-" * 60)
print(df.head(10))
print(f"\nDataset Shape: {df.shape}")
print(f"\nSpecies Distribution:\n{df['species_name'].value_counts()}")

# Statistical Summary
print("\n2. Statistical Summary:")
print("-" * 60)
print(df.describe())

# =====================================
# PART 2: DATA VISUALIZATION
# =====================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Iris Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Pairplot style visualization
features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Plot 1: Sepal Length vs Sepal Width
for species in [0, 1, 2]:
    species_data = df[df['species'] == species]
    axes[0, 0].scatter(species_data['sepal length (cm)'], species_data['sepal width (cm)'], 
                       label=iris.target_names[species], alpha=0.6, s=50)
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Sepal Width (cm)')
axes[0, 0].set_title('Sepal Dimensions')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Petal Length vs Petal Width
for species in [0, 1, 2]:
    species_data = df[df['species'] == species]
    axes[0, 1].scatter(species_data['petal length (cm)'], species_data['petal width (cm)'], 
                       label=iris.target_names[species], alpha=0.6, s=50)
axes[0, 1].set_xlabel('Petal Length (cm)')
axes[0, 1].set_ylabel('Petal Width (cm)')
axes[0, 1].set_title('Petal Dimensions')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Box plot of Petal Length
df_melted = df[['petal length (cm)', 'species_name']].copy()
df_melted.boxplot(by='species_name', ax=axes[1, 0])
axes[1, 0].set_title('Petal Length Distribution by Species')
axes[1, 0].set_xlabel('Species')
axes[1, 0].set_ylabel('Petal Length (cm)')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)

# Plot 4: Correlation Heatmap
corr_matrix = df[features].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[1, 1], 
            square=True, cbar_kws={'shrink': 0.8})
axes[1, 1].set_title('Feature Correlation Matrix')

plt.tight_layout()
plt.show()

# =====================================
# PART 3: BINARY CLASSIFICATION
# =====================================

print("\n" + "=" * 60)
print("PART 3: BINARY CLASSIFICATION (Setosa vs Versicolor)")
print("=" * 60)

# Create binary dataset (only Setosa and Versicolor)
df_binary = df[df['species'] <= 1].copy()
X_binary = df_binary[['petal length (cm)', 'petal width (cm)']].values
y_binary = df_binary['species'].values

# Split the data
X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

# Standardize features
scaler_bin = StandardScaler()
X_train_bin_scaled = scaler_bin.fit_transform(X_train_bin)
X_test_bin_scaled = scaler_bin.transform(X_test_bin)

# Train Logistic Regression
log_reg_bin = LogisticRegression(random_state=42)
log_reg_bin.fit(X_train_bin_scaled, y_train_bin)

# Predictions
y_pred_bin = log_reg_bin.predict(X_test_bin_scaled)

# Evaluation
print("\nBinary Classification Results:")
print("-" * 60)
print(f"Accuracy: {accuracy_score(y_test_bin, y_pred_bin):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_bin, y_pred_bin, target_names=['Setosa', 'Versicolor']))

# Confusion Matrix
cm_bin = confusion_matrix(y_test_bin, y_pred_bin)

# Visualization for Binary Classification
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Binary Classification: Setosa vs Versicolor', fontsize=16, fontweight='bold')

# Decision Boundary
x_min, x_max = X_binary[:, 0].min() - 0.5, X_binary[:, 0].max() + 0.5
y_min, y_max = X_binary[:, 1].min() - 0.5, X_binary[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Transform mesh grid for prediction
Z = log_reg_bin.predict(scaler_bin.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

axes[0].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
axes[0].scatter(X_binary[y_binary == 0, 0], X_binary[y_binary == 0, 1], 
                c='red', label='Setosa', edgecolors='k', s=80)
axes[0].scatter(X_binary[y_binary == 1, 0], X_binary[y_binary == 1, 1], 
                c='blue', label='Versicolor', edgecolors='k', s=80)
axes[0].set_xlabel('Petal Length (cm)')
axes[0].set_ylabel('Petal Width (cm)')
axes[0].set_title('Decision Boundary - Logistic Regression')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Confusion Matrix
sns.heatmap(cm_bin, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Setosa', 'Versicolor'], yticklabels=['Setosa', 'Versicolor'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Confusion Matrix')

plt.tight_layout()
plt.show()

# =====================================
# PART 4: MULTI-CLASS CLASSIFICATION
# =====================================

print("\n" + "=" * 60)
print("PART 4: MULTI-CLASS CLASSIFICATION (All 3 Species)")
print("=" * 60)

# Prepare multi-class data
X_multi = df[['petal length (cm)', 'petal width (cm)']].values
y_multi = df['species'].values

# Split the data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42, stratify=y_multi
)

# Standardize features
scaler_multi = StandardScaler()
X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
X_test_multi_scaled = scaler_multi.transform(X_test_multi)

# Train Multiple Models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=200),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=4)
}

results = {}

for name, model in models.items():
    model.fit(X_train_multi_scaled, y_train_multi)
    y_pred = model.predict(X_test_multi_scaled)
    accuracy = accuracy_score(y_test_multi, y_pred)
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'accuracy': accuracy
    }
    
    print(f"\n{name}:")
    print("-" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test_multi, y_pred, target_names=iris.target_names))

# Confusion Matrices for Multi-class
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Multi-class Classification - Confusion Matrices', fontsize=16, fontweight='bold')

for idx, (name, result) in enumerate(results.items()):
    cm = confusion_matrix(y_test_multi, result['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[idx],
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')
    axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.4f}')

plt.tight_layout()
plt.show()

# Decision Boundaries for Multi-class
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Multi-class Classification - Decision Boundaries', fontsize=16, fontweight='bold')

x_min, x_max = X_multi[:, 0].min() - 0.5, X_multi[:, 0].max() + 0.5
y_min, y_max = X_multi[:, 1].min() - 0.5, X_multi[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

colors = ['red', 'green', 'blue']
for idx, (name, result) in enumerate(results.items()):
    Z = result['model'].predict(scaler_multi.transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    axes[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn')
    for species in [0, 1, 2]:
        species_data = X_multi[y_multi == species]
        axes[idx].scatter(species_data[:, 0], species_data[:, 1], 
                         c=colors[species], label=iris.target_names[species], 
                         edgecolors='k', s=80, alpha=0.7)
    
    axes[idx].set_xlabel('Petal Length (cm)')
    axes[idx].set_ylabel('Petal Width (cm)')
    axes[idx].set_title(f'{name}')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =====================================
# PART 5: STATISTICAL ANALYSIS
# =====================================

print("\n" + "=" * 60)
print("PART 5: STATISTICAL ANALYSIS")
print("=" * 60)

# ANOVA test for each feature
print("\nANOVA Test (comparing means across species):")
print("-" * 60)

for feature in features:
    groups = [df[df['species'] == i][feature].values for i in range(3)]
    f_stat, p_value = stats.f_oneway(*groups)
    print(f"{feature}:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    if p_value < 0.05:
        print(f"  Result: Significant difference (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")
    print()

# Logistic Regression with statsmodels (for binary case)
print("\nStatsmodels Logistic Regression (Binary):")
print("-" * 60)

X_sm = sm.add_constant(X_train_bin_scaled)
logit_model = sm.Logit(y_train_bin, X_sm)
result = logit_model.fit(disp=0)
print(result.summary())

# =====================================
# PART 6: MODEL COMPARISON
# =====================================

print("\n" + "=" * 60)
print("PART 6: MODEL COMPARISON SUMMARY")
print("=" * 60)

comparison_data = {
    'Model': ['Binary Classification (Logistic Regression)', 
              'Multi-class (Logistic Regression)', 
              'Multi-class (Decision Tree)'],
    'Accuracy': [accuracy_score(y_test_bin, y_pred_bin),
                 results['Logistic Regression']['accuracy'],
                 results['Decision Tree']['accuracy']],
    'Classes': [2, 3, 3]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# Visualize model comparison
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(comparison_df['Model'], comparison_df['Accuracy'], 
              color=['steelblue', 'forestgreen', 'coral'], alpha=0.7, edgecolor='black')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_ylim([0.9, 1.0])
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15, ha='right')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()



