#Day 1: The Decision Tree (The Building Block)
#Introduction to Decision Trees
#Objective: Understand how a machine "asks questions" to split data.
#Goal: Understand the fundamentals of Decision Trees

#Concept: Gini Impurity vs. Entropy, Root Node, Leaf Node, Pruning (max_depth).

#What are Decision Trees and how do they work?
#Classification vs Regression Trees
#Impurity measures: Gini Index and Entropy
#Information Gain
#Tree growing process (splitting criteria)

#Resources:

#Hands-On ML: Chapter 6 (pages 1-15)
#Andrew Ng: Course 2, Week 4 - Decision Trees (Part 1)
#StatQuest: "Decision Trees" video


#Task: Project - Visualize a Decision Tree.
#Train a DecisionTreeClassifier on the Iris or Mushroom dataset.
#Use sklearn.tree.plot_tree and graphviz to export a picture of the tree.
#Goal: Trace a single data point down the tree manually to verify the prediction.


#Task: Project - Build your first Decision Tree classifier using scikit-learn
#Use the Iris dataset to classify flower species
#Experiment with max_depth parameter.


#Part 1: Decision Trees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, log, simplify, diff, Sum, Rational
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.api as sm
from statsmodels.formula.api import logit
from sklearn.datasets import load_iris, fetch_openml
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("MODULE 1: MATHEMATICAL FOUNDATION WITH SYMPY")
print("="*80)

# ============================================================================
# 1. SYMBOLIC MATHEMATICS - DERIVE FORMULAS
# ============================================================================

print("\n1.1 Deriving Gini Impurity Formula")
print("-" * 50)

# Define symbols
p1, p2, p = symbols('p1 p2 p', real=True, positive=True)

# Gini for 2 classes
gini_binary = 1 - (p1**2 + p2**2)
print(f"Gini (2 classes): {gini_binary}")

# Substitute p2 = 1 - p1
gini_simplified = gini_binary.subs(p2, 1 - p1)
print(f"Gini simplified: {simplify(gini_simplified)}")

# Find maximum (derivative = 0)
gini_derivative = diff(gini_simplified, p1)
print(f"Derivative: {gini_derivative}")
print(f"Maximum Gini at p1 = 0.5: {gini_simplified.subs(p1, 0.5)}")

print("\n1.2 Deriving Entropy Formula")
print("-" * 50)

# Entropy for binary classification
entropy_binary = -p1 * log(p1, 2) - (1-p1) * log(1-p1, 2)
print(f"Entropy: {entropy_binary}")

# Find maximum
entropy_derivative = diff(entropy_binary, p1)
print(f"Derivative: {simplify(entropy_derivative)}")
print(f"Maximum Entropy at p1 = 0.5: {entropy_binary.subs(p1, 0.5)}")

print("\n1.3 Information Gain Formula")
print("-" * 50)

# Symbols for Information Gain
H_parent, H_left, H_right, n_left, n_right = symbols('H_parent H_left H_right n_left n_right')
n_total = n_left + n_right

# Information Gain = Parent Entropy - Weighted Child Entropy
info_gain = H_parent - (n_left/n_total * H_left + n_right/n_total * H_right)
print(f"Information Gain: {info_gain}")


print("\n" + "="*80)
print("MODULE 2: CORE IMPLEMENTATION WITH NUMPY")
print("="*80)

# ============================================================================
# 2. NUMPY IMPLEMENTATIONS FROM SCRATCH
# ============================================================================

def gini_impurity(y):
    """Calculate Gini Impurity using NumPy"""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    gini = 1 - np.sum(probabilities ** 2)
    return gini

def entropy(y):
    """Calculate Entropy using NumPy"""
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    # Avoid log(0)
    probabilities = probabilities[probabilities > 0]
    ent = -np.sum(probabilities * np.log2(probabilities))
    return ent

def information_gain(y_parent, y_left, y_right, metric='gini'):
    """Calculate Information Gain"""
    n = len(y_parent)
    n_left, n_right = len(y_left), len(y_right)
    
    if metric == 'gini':
        parent_impurity = gini_impurity(y_parent)
        left_impurity = gini_impurity(y_left)
        right_impurity = gini_impurity(y_right)
    else:  # entropy
        parent_impurity = entropy(y_parent)
        left_impurity = entropy(y_left)
        right_impurity = entropy(y_right)
    
    weighted_child = (n_left/n * left_impurity) + (n_right/n * right_impurity)
    ig = parent_impurity - weighted_child
    return ig

def find_best_split(X, y, metric='gini'):
    """Find best feature and threshold to split on"""
    best_gain = -np.inf
    best_feature = None
    best_threshold = None
    
    n_features = X.shape[1]
    
    for feature_idx in range(n_features):
        # Get unique values (potential thresholds)
        thresholds = np.unique(X[:, feature_idx])
        
        for threshold in thresholds:
            # Split data
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                continue
            
            y_left = y[left_mask]
            y_right = y[right_mask]
            
            # Calculate information gain
            gain = information_gain(y, y_left, y_right, metric)
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold
    
    return best_feature, best_threshold, best_gain

print("\n2.1 Testing Gini and Entropy Calculations")
print("-" * 50)

# Test data
y_pure = np.array([0, 0, 0, 0, 0])
y_mixed = np.array([0, 0, 1, 1, 1])
y_impure = np.array([0, 1, 0, 1, 0, 1])

print(f"Pure dataset [0,0,0,0,0]:")
print(f"  Gini: {gini_impurity(y_pure):.4f}")
print(f"  Entropy: {entropy(y_pure):.4f}")

print(f"\nMixed dataset [0,0,1,1,1]:")
print(f"  Gini: {gini_impurity(y_mixed):.4f}")
print(f"  Entropy: {entropy(y_mixed):.4f}")

print(f"\nImpure dataset [0,1,0,1,0,1]:")
print(f"  Gini: {gini_impurity(y_impure):.4f}")
print(f"  Entropy: {entropy(y_impure):.4f}")


print("\n" + "="*80)
print("MODULE 3: DATA ANALYSIS WITH PANDAS")
print("="*80)

# ============================================================================
# 3. LOAD AND EXPLORE DATASETS WITH PANDAS
# ============================================================================

print("\n3.1 Loading Iris Dataset")
print("-" * 50)

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print(iris_df.head())
print(f"\nDataset shape: {iris_df.shape}")
print(f"\nClass distribution:\n{iris_df['species'].value_counts()}")

print("\n3.2 Feature Statistics by Class")
print("-" * 50)

# Groupby analysis
feature_stats = iris_df.groupby('species').agg({
    'sepal length (cm)': ['mean', 'std', 'min', 'max'],
    'sepal width (cm)': ['mean', 'std', 'min', 'max'],
    'petal length (cm)': ['mean', 'std', 'min', 'max'],
    'petal width (cm)': ['mean', 'std', 'min', 'max']
})

print(feature_stats)

print("\n3.3 Finding Best Split Using Pandas")
print("-" * 50)

# For petal length, find best split
feature_name = 'petal length (cm)'
feature_data = iris_df[[feature_name, 'target']].copy()

# Sort by feature
feature_data = feature_data.sort_values(feature_name)

# Calculate Gini for different splits
splits = []
for i in range(10, len(feature_data) - 10, 5):
    threshold = feature_data.iloc[i][feature_name]
    
    left_group = feature_data[feature_data[feature_name] <= threshold]['target']
    right_group = feature_data[feature_data[feature_name] > threshold]['target']
    
    gain = information_gain(
        feature_data['target'].values,
        left_group.values,
        right_group.values,
        metric='gini'
    )
    
    splits.append({
        'threshold': threshold,
        'info_gain': gain,
        'left_size': len(left_group),
        'right_size': len(right_group)
    })

splits_df = pd.DataFrame(splits)
best_split = splits_df.loc[splits_df['info_gain'].idxmax()]

print(f"Best split for {feature_name}:")
print(f"  Threshold: {best_split['threshold']:.2f}")
print(f"  Information Gain: {best_split['info_gain']:.4f}")
print(f"  Left/Right sizes: {int(best_split['left_size'])}/{int(best_split['right_size'])}")


print("\n" + "="*80)
print("MODULE 4: STATISTICAL VALIDATION WITH SCIPY & STATSMODELS")
print("="*80)

# ============================================================================
# 4. STATISTICAL TESTS FOR SPLITS
# ============================================================================

print("\n4.1 ANOVA Test for Feature Importance")
print("-" * 50)

# Test if feature means differ significantly across classes
for feature in iris.feature_names:
    groups = [iris_df[iris_df['target'] == i][feature].values 
              for i in range(3)]
    f_stat, p_value = f_oneway(*groups)
    print(f"{feature:25s} F-stat: {f_stat:8.2f}, p-value: {p_value:.2e}")

print("\n4.2 Chi-Square Test for Categorical Splits")
print("-" * 50)

# Discretize petal length and test independence
iris_df['petal_length_bin'] = pd.cut(iris_df['petal length (cm)'], 
                                      bins=3, labels=['small', 'medium', 'large'])

contingency_table = pd.crosstab(iris_df['petal_length_bin'], iris_df['species'])
print("Contingency Table:")
print(contingency_table)

chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print(f"\nChi-square statistic: {chi2:.2f}")
print(f"P-value: {p_value:.2e}")
print(f"Degrees of freedom: {dof}")

print("\n4.3 Logistic Regression Comparison")
print("-" * 50)

# Binary classification: setosa vs others
iris_binary = iris_df.copy()
iris_binary['is_setosa'] = (iris_binary['target'] == 0).astype(int)

# Fit logistic regression
X_logit = iris_binary[['petal length (cm)', 'petal width (cm)']]
X_logit = sm.add_constant(X_logit)
y_logit = iris_binary['is_setosa']

logit_model = sm.Logit(y_logit, X_logit).fit(disp=0)
print(logit_model.summary().tables[1])


print("\n" + "="*80)
print("MODULE 5: VISUALIZATION WITH MATPLOTLIB & SEABORN")
print("="*80)

# ============================================================================
# 5. COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n5.1 Creating Comprehensive Visualizations...")
print("-" * 50)

fig = plt.figure(figsize=(16, 12))

# 5.1: Gini vs Entropy Comparison
ax1 = plt.subplot(3, 3, 1)
p_vals = np.linspace(0.01, 0.99, 100)
gini_vals = 1 - (p_vals**2 + (1-p_vals)**2)
entropy_vals = -p_vals * np.log2(p_vals) - (1-p_vals) * np.log2(1-p_vals)

ax1.plot(p_vals, gini_vals, label='Gini', linewidth=2)
ax1.plot(p_vals, entropy_vals / 2, label='Entropy/2', linewidth=2)
ax1.set_xlabel('P(class=1)')
ax1.set_ylabel('Impurity')
ax1.set_title('Gini vs Entropy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 5.2: Feature Distributions
ax2 = plt.subplot(3, 3, 2)
for i, species in enumerate(['setosa', 'versicolor', 'virginica']):
    data = iris_df[iris_df['species'] == species]['petal length (cm)']
    ax2.hist(data, alpha=0.6, label=species, bins=15)
ax2.set_xlabel('Petal Length (cm)')
ax2.set_ylabel('Frequency')
ax2.set_title('Petal Length Distribution')
ax2.legend()

# 5.3: Correlation Heatmap
ax3 = plt.subplot(3, 3, 3)
corr_matrix = iris_df[iris.feature_names].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax3)
ax3.set_title('Feature Correlations')

# 5.4: Information Gain Heatmap
ax4 = plt.subplot(3, 3, 4)
ig_matrix = np.zeros((len(iris.feature_names), len(iris.feature_names)))

for i, f1 in enumerate(iris.feature_names):
    for j, f2 in enumerate(iris.feature_names):
        X_temp = iris_df[[f1, f2]].values
        y_temp = iris_df['target'].values
        _, _, gain = find_best_split(X_temp, y_temp, metric='gini')
        ig_matrix[i, j] = gain

sns.heatmap(ig_matrix, annot=True, fmt='.3f', 
            xticklabels=[f.split()[0] for f in iris.feature_names],
            yticklabels=[f.split()[0] for f in iris.feature_names],
            cmap='YlOrRd', ax=ax4)
ax4.set_title('Information Gain by Feature Pair')

# 5.5: Pairplot for first 3 features
ax5 = plt.subplot(3, 3, 5)
for target in [0, 1, 2]:
    mask = iris_df['target'] == target
    ax5.scatter(iris_df.loc[mask, 'petal length (cm)'],
                iris_df.loc[mask, 'petal width (cm)'],
                label=iris.target_names[target],
                alpha=0.6, s=50)
ax5.set_xlabel('Petal Length')
ax5.set_ylabel('Petal Width')
ax5.set_title('Petal Dimensions')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 5.6: Split Quality vs Threshold
ax6 = plt.subplot(3, 3, 6)
ax6.plot(splits_df['threshold'], splits_df['info_gain'], 'o-', linewidth=2)
ax6.axvline(best_split['threshold'], color='red', linestyle='--', 
            label=f'Best: {best_split["threshold"]:.2f}')
ax6.set_xlabel('Threshold (Petal Length)')
ax6.set_ylabel('Information Gain')
ax6.set_title('Split Quality Analysis')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 5.7: Class Distribution
ax7 = plt.subplot(3, 3, 7)
class_counts = iris_df['species'].value_counts()
ax7.bar(class_counts.index, class_counts.values, color=['red', 'green', 'blue'], alpha=0.7)
ax7.set_ylabel('Count')
ax7.set_title('Class Distribution')
ax7.grid(True, alpha=0.3, axis='y')

# 5.8: Boxplot of features
ax8 = plt.subplot(3, 3, 8)
feature_data_list = [iris_df[col].values for col in iris.feature_names]
bp = ax8.boxplot(feature_data_list, labels=[f.split()[0] for f in iris.feature_names])
ax8.set_ylabel('Value (cm)')
ax8.set_title('Feature Distributions')
ax8.grid(True, alpha=0.3, axis='y')

# 5.9: Decision Boundary Preview
ax9 = plt.subplot(3, 3, 9)
threshold = best_split['threshold']
ax9.scatter(iris_df['petal length (cm)'], iris_df['petal width (cm)'],
            c=iris_df['target'], cmap='viridis', alpha=0.6)
ax9.axvline(threshold, color='red', linestyle='--', linewidth=2, label='Split')
ax9.set_xlabel('Petal Length')
ax9.set_ylabel('Petal Width')
ax9.set_title('First Split Visualization')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('decision_tree_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: decision_tree_analysis.png")


print("\n" + "="*80)
print("MODULE 6: SKLEARN IMPLEMENTATION")
print("="*80)

# ============================================================================
# 6. SKLEARN DECISION TREE
# ============================================================================

print("\n6.1 Training Decision Tree with sklearn")
print("-" * 50)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Train trees with different depths
trees = {}
for depth in [2, 3, 5, None]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    trees[depth] = clf
    
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    print(f"Depth={depth}: Train Acc={train_acc:.3f}, Test Acc={test_acc:.3f}")

print("\n6.2 Feature Importance")
print("-" * 50)

clf = trees[3]
feature_importance = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

print("\n6.3 Visualizing Decision Tree")
print("-" * 50)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

for idx, (depth, clf) in enumerate(list(trees.items())[:4]):
    ax = axes[idx // 2, idx % 2]
    plot_tree(clf, 
              feature_names=iris.feature_names,
              class_names=iris.target_names,
              filled=True,
              rounded=True,
              ax=ax,
              fontsize=9)
    ax.set_title(f'Decision Tree (max_depth={depth})', fontsize=14)

plt.tight_layout()
plt.savefig('sklearn_decision_trees.png', dpi=150, bbox_inches='tight')
print("✓ Saved: sklearn_decision_trees.png")

print("\n6.4 Manual Prediction Trace")
print("-" * 50)

# Trace a single prediction
sample_idx = 0
sample = X_test[sample_idx].reshape(1, -1)
true_label = y_test[sample_idx]

clf = trees[3]
prediction = clf.predict(sample)[0]

print(f"Sample features: {sample[0]}")
print(f"True label: {iris.target_names[true_label]}")
print(f"Predicted: {iris.target_names[prediction]}")
print(f"\nDecision path:")

# Get decision path
node_indicator = clf.decision_path(sample)
leaf_id = clf.apply(sample)

feature_names = iris.feature_names
tree_ = clf.tree_

for node_id in node_indicator.indices:
    if leaf_id[0] == node_id:
        print(f"  → Leaf node {node_id} reached")
        print(f"    Class: {iris.target_names[np.argmax(clf.tree_.value[node_id][0])]}")
        break
    
    if sample[0][tree_.feature[node_id]] <= tree_.threshold[node_id]:
        threshold_sign = "<="
        direction = "left"
    else:
        threshold_sign = ">"
        direction = "right"
    
    print(f"  Node {node_id}: {feature_names[tree_.feature[node_id]]} "
          f"{threshold_sign} {tree_.threshold[node_id]:.2f} → go {direction}")


print("\n" + "="*80)
print("MODULE 7: CONFUSION MATRIX & PERFORMANCE")
print("="*80)

print("\n7.1 Detailed Performance Analysis")
print("-" * 50)

clf = trees[3]
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: confusion_matrix.png")


print("""
✓ SymPy: Derived formulas symbolically
✓ NumPy: Implemented Gini, Entropy, Information Gain from scratch
✓ Pandas: Analyzed data, found best splits using groupby
✓ SciPy: Validated splits with ANOVA and Chi-square tests
✓ StatsModels: Compared with logistic regression
✓ Matplotlib/Seaborn: Created comprehensive visualizations
✓ sklearn: Trained trees, visualized structure, traced predictions

NEXT STEPS:
1. Create Manim animation (see separate script)
2. Implement pruning algorithms
3. Build Random Forest (Day 2)
4. Compare with other algorithms
""")


print("ALL VISUALIZATIONS SAVED!")

