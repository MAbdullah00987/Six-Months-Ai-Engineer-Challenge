

#Day 4: Logistic Regression (Binary Classification)
#Objective: Switch from predicting numbers to predicting categories (Yes/No, Spam/Ham).

#Concept: The Sigmoid Function (squashing output between 0 and 1), Decision Boundaries.

#Learning Objectives:
# Understand classification vs regression
# Learn sigmoid function and log loss
# Understand decision boundaries

#Study Materials:
# Géron Chapter 4: "Logistic Regression" (pages 145-152)
# Andrew Ng Course 1, Week 3: "Classification"
# Video: StatQuest "Logistic Regression"


#Project 1: Diabetes Prediction.
# Dataset: Pima Indians Diabetes Dataset (available on Kaggle/Scikit-learn).
# Predict if a patient has diabetes based on health metrics.
# Analyze the model.coef_ to see which feature (e.g., Glucose, BMI) is the strongest predictor.

#Project 2: Iris Flower Classification**
# Start with binary classification (2 classes)
# Then extend to multi-class (3 classes)
# Visualize decision boundaries
# Create confusion matrix

"""
DAY 4: LOGISTIC REGRESSION - BINARY CLASSIFICATION
Complete Guide with NumPy, Pandas, Matplotlib, Seaborn, Scipy, Statsmodels, Sklearn

Author: ML Learning Path
Focus: Building strong logic and deep understanding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("SECTION 1: UNDERSTANDING CLASSIFICATION VS REGRESSION")
print("="*80)

# Create comparison data
np.random.seed(42)
X_reg = np.linspace(0, 10, 100)
y_reg = 2 * X_reg + 1 + np.random.normal(0, 2, 100)

X_class = np.random.normal(5, 2, 100)
y_class = (X_class > 5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Regression plot
axes[0].scatter(X_reg, y_reg, alpha=0.6, color='blue')
axes[0].plot(X_reg, 2*X_reg + 1, 'r-', linewidth=2, label='Best fit line')
axes[0].set_xlabel('X', fontsize=12)
axes[0].set_ylabel('Y (continuous)', fontsize=12)
axes[0].set_title('Regression: Predicting Continuous Values', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Classification plot
axes[1].scatter(X_class[y_class==0], y_class[y_class==0], alpha=0.6, color='red', label='Class 0', s=100)
axes[1].scatter(X_class[y_class==1], y_class[y_class==1], alpha=0.6, color='green', label='Class 1', s=100)
axes[1].axvline(x=5, color='purple', linestyle='--', linewidth=2, label='Decision Boundary')
axes[1].set_xlabel('X', fontsize=12)
axes[1].set_ylabel('Y (discrete)', fontsize=12)
axes[1].set_title('Classification: Predicting Categories', fontsize=14, fontweight='bold')
axes[1].set_yticks([0, 1])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('classification_vs_regression.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📊 KEY DIFFERENCES:")
print("-" * 60)
print("REGRESSION:")
print("  • Predicts continuous values (e.g., house prices, temperature)")
print("  • Output: Any real number")
print("  • Loss Function: Mean Squared Error (MSE)")
print("\nCLASSIFICATION:")
print("  • Predicts discrete categories (e.g., spam/ham, yes/no)")
print("  • Output: Probabilities (0 to 1) converted to classes")
print("  • Loss Function: Log Loss (Cross-Entropy)")

print("\n" + "="*80)
print("SECTION 2: THE SIGMOID FUNCTION - DEEP DIVE")
print("="*80)

# Sigmoid function implementation
def sigmoid(z):
    """
    Sigmoid function: σ(z) = 1 / (1 + e^(-z))
    Maps any real number to (0, 1)
    """
    return 1 / (1 + np.exp(-z))

# Explore sigmoid properties
z_values = np.linspace(-10, 10, 1000)
sigmoid_values = sigmoid(z_values)

# Plot sigmoid function
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Main sigmoid plot
axes[0, 0].plot(z_values, sigmoid_values, 'b-', linewidth=2.5, label='σ(z) = 1/(1+e^(-z))')
axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision threshold (0.5)')
axes[0, 0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0, 0].axhline(y=1, color='gray', linestyle='-', alpha=0.3)
axes[0, 0].axvline(x=0, color='gray', linestyle='-', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xlabel('z (input)', fontsize=12)
axes[0, 0].set_ylabel('σ(z) (output)', fontsize=12)
axes[0, 0].set_title('Sigmoid Function: Squashing to (0, 1)', fontsize=14, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].set_ylim([-0.1, 1.1])

# Derivative of sigmoid
sigmoid_derivative = sigmoid_values * (1 - sigmoid_values)
axes[0, 1].plot(z_values, sigmoid_derivative, 'g-', linewidth=2.5)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xlabel('z', fontsize=12)
axes[0, 1].set_ylabel("σ'(z)", fontsize=12)
axes[0, 1].set_title("Sigmoid Derivative: σ'(z) = σ(z)(1-σ(z))", fontsize=14, fontweight='bold')
axes[0, 1].axvline(x=0, color='r', linestyle='--', alpha=0.7)

# Compare with other activation functions
tanh_values = np.tanh(z_values)
relu_values = np.maximum(0, z_values)

axes[1, 0].plot(z_values, sigmoid_values, 'b-', linewidth=2, label='Sigmoid')
axes[1, 0].plot(z_values, tanh_values, 'r-', linewidth=2, label='Tanh')
axes[1, 0].plot(z_values, relu_values/10, 'g-', linewidth=2, label='ReLU (scaled)')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=10)
axes[1, 0].set_xlabel('z', fontsize=12)
axes[1, 0].set_ylabel('Activation', fontsize=12)
axes[1, 0].set_title('Sigmoid vs Other Activations', fontsize=14, fontweight='bold')

# Sigmoid properties table
key_points = {
    'z': [-5, -2, 0, 2, 5],
    'σ(z)': [sigmoid(z) for z in [-5, -2, 0, 2, 5]]
}
axes[1, 1].axis('off')
table_data = [[f"{z:.1f}", f"{sig:.6f}"] for z, sig in zip(key_points['z'], key_points['σ(z)'])]
table = axes[1, 1].table(cellText=table_data, colLabels=['z', 'σ(z)'],
                         cellLoc='center', loc='center',
                         colWidths=[0.3, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.5)
axes[1, 1].set_title('Key Sigmoid Values', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('sigmoid_function_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎯 SIGMOID PROPERTIES:")
print("-" * 60)
print("1. Range: (0, 1) - Perfect for probabilities")
print("2. σ(0) = 0.5 - Natural decision boundary")
print("3. σ(-z) = 1 - σ(z) - Symmetry around 0.5")
print("4. Derivative: σ'(z) = σ(z)(1-σ(z)) - Easy to compute")
print("5. Smooth and differentiable - Good for gradient descent")

# Numerical examples
print("\n📐 NUMERICAL EXAMPLES:")
print("-" * 60)
test_values = [-10, -2, 0, 2, 10]
for z in test_values:
    prob = sigmoid(z)
    print(f"σ({z:3.0f}) = {prob:.6f} → Class {'1' if prob >= 0.5 else '0'}")

print("\n" + "="*80)
print("SECTION 3: LOG LOSS (BINARY CROSS-ENTROPY)")
print("="*80)

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Binary Cross-Entropy Loss
    L = -[y*log(p) + (1-y)*log(1-p)]
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Visualize log loss
y_true = 1  # True label
predictions = np.linspace(0.01, 0.99, 100)
losses_class1 = -np.log(predictions)
losses_class0 = -np.log(1 - predictions)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss for y=1
axes[0].plot(predictions, losses_class1, 'b-', linewidth=2.5)
axes[0].fill_between(predictions, losses_class1, alpha=0.3)
axes[0].set_xlabel('Predicted Probability', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Log Loss when True Label = 1', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.7)
axes[0].text(0.7, 3, 'Confident & Correct\n(Low Loss)', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[0].text(0.1, 3, 'Confident & Wrong\n(High Loss)', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

# Loss for y=0
axes[1].plot(predictions, losses_class0, 'r-', linewidth=2.5)
axes[1].fill_between(predictions, losses_class0, alpha=0.3)
axes[1].set_xlabel('Predicted Probability', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Log Loss when True Label = 0', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=0.5, color='b', linestyle='--', alpha=0.7)
axes[1].text(0.2, 3, 'Confident & Correct\n(Low Loss)', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
axes[1].text(0.8, 3, 'Confident & Wrong\n(High Loss)', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

plt.tight_layout()
plt.savefig('log_loss_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n💡 LOG LOSS INTUITION:")
print("-" * 60)
print("• Penalizes confident wrong predictions heavily")
print("• Rewards confident correct predictions")
print("• Always positive (0 to ∞)")
print("• Convex function - has global minimum")

# Examples
print("\n📊 NUMERICAL EXAMPLES:")
print("-" * 60)
scenarios = [
    (1, 0.99, "Confident & Correct"),
    (1, 0.51, "Barely Correct"),
    (1, 0.01, "Confident & Wrong"),
    (0, 0.01, "Confident & Correct"),
    (0, 0.99, "Confident & Wrong")
]

for y_true, y_pred, desc in scenarios:
    loss = log_loss(np.array([y_true]), np.array([y_pred]))
    print(f"True={y_true}, Pred={y_pred:.2f} ({desc:20s}): Loss = {loss:.4f}")

print("\n" + "="*80)
print("SECTION 4: LOGISTIC REGRESSION FROM SCRATCH")
print("="*80)

class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch using NumPy
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000, verbose=False):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = sigmoid(linear_model)
            
            # Compute loss
            loss = log_loss(y, y_pred)
            self.losses.append(loss)
            
            # Backward pass (gradients)
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i:4d}: Loss = {loss:.6f}")
        
        return self
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Generate synthetic data
np.random.seed(42)
n_samples = 300

# Class 0
X0 = np.random.randn(n_samples//2, 2) + np.array([2, 2])
y0 = np.zeros(n_samples//2)

# Class 1
X1 = np.random.randn(n_samples//2, 2) + np.array([5, 5])
y1 = np.ones(n_samples//2)

X = np.vstack([X0, X1])
y = np.hstack([y0, y1])

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = LogisticRegressionScratch(learning_rate=0.1, n_iterations=1000, verbose=True)
model.fit(X_scaled, y)

print(f"\n✅ Training completed!")
print(f"Final weights: {model.weights}")
print(f"Final bias: {model.bias:.4f}")
print(f"Final loss: {model.losses[-1]:.6f}")

# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(model.losses, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Log Loss', fontsize=12)
plt.title('Training Progress: Loss vs Iterations', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("SECTION 5: DECISION BOUNDARIES")
print("="*80)

# Create mesh grid for decision boundary
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

# Predict on mesh
Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot decision boundary
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Probability contours
contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
scatter0 = axes[0].scatter(X_scaled[y==0][:, 0], X_scaled[y==0][:, 1], 
                          c='red', marker='o', s=100, edgecolors='black', 
                          linewidths=1.5, label='Class 0', alpha=0.8)
scatter1 = axes[0].scatter(X_scaled[y==1][:, 0], X_scaled[y==1][:, 1], 
                          c='green', marker='s', s=100, edgecolors='black', 
                          linewidths=1.5, label='Class 1', alpha=0.8)
axes[0].set_xlabel('Feature 1', fontsize=12)
axes[0].set_ylabel('Feature 2', fontsize=12)
axes[0].set_title('Decision Boundary with Probability Contours', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
cbar = plt.colorbar(contour, ax=axes[0])
cbar.set_label('P(Class=1)', fontsize=11)

# Predicted classes
predictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
predictions = predictions.reshape(xx.shape)
axes[1].contourf(xx, yy, predictions, levels=1, cmap='RdYlGn', alpha=0.3)
axes[1].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, 
                linestyles='dashed', label='Decision Boundary')
axes[1].scatter(X_scaled[y==0][:, 0], X_scaled[y==0][:, 1], 
               c='red', marker='o', s=100, edgecolors='black', 
               linewidths=1.5, label='Class 0', alpha=0.8)
axes[1].scatter(X_scaled[y==1][:, 0], X_scaled[y==1][:, 1], 
               c='green', marker='s', s=100, edgecolors='black', 
               linewidths=1.5, label='Class 1', alpha=0.8)
axes[1].set_xlabel('Feature 1', fontsize=12)
axes[1].set_ylabel('Feature 2', fontsize=12)
axes[1].set_title('Classification Regions', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n🎯 DECISION BOUNDARY INSIGHTS:")
print("-" * 60)
print("• Linear boundary: h(x) = w₁x₁ + w₂x₂ + b = 0")
print("• Points on one side: Class 0 (red)")
print("• Points on other side: Class 1 (green)")
print("• Distance from boundary indicates confidence")

print("\n" + "="*80)
print("SECTION 6: PROJECT 1 - DIABETES PREDICTION")
print("="*80)

# Load Pima Indians Diabetes Dataset
from sklearn.datasets import fetch_openml

# Note: Using a similar dataset structure
# In practice, you'd load from: pd.read_csv('diabetes.csv')
# Creating synthetic diabetes-like data for demonstration

np.random.seed(42)
n_patients = 768

data = {
    'Pregnancies': np.random.randint(0, 17, n_patients),
    'Glucose': np.random.normal(120, 30, n_patients),
    'BloodPressure': np.random.normal(70, 12, n_patients),
    'SkinThickness': np.random.normal(20, 15, n_patients),
    'Insulin': np.random.normal(80, 115, n_patients),
    'BMI': np.random.normal(32, 7, n_patients),
    'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_patients),
    'Age': np.random.randint(21, 81, n_patients)
}

# Create outcome based on glucose and BMI (main risk factors)
diabetes_prob = sigmoid((data['Glucose'] - 120)/30 + (data['BMI'] - 30)/10)
data['Outcome'] = (np.random.random(n_patients) < diabetes_prob).astype(int)

df_diabetes = pd.DataFrame(data)

print("📊 DATASET OVERVIEW:")
print("-" * 60)
print(df_diabetes.head(10))
print(f"\nShape: {df_diabetes.shape}")
print(f"\nClass distribution:")
print(df_diabetes['Outcome'].value_counts())
print(f"\nClass balance: {df_diabetes['Outcome'].mean():.2%} have diabetes")

# Exploratory Data Analysis
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for idx, feature in enumerate(features):
    axes[idx].hist(df_diabetes[df_diabetes['Outcome']==0][feature], 
                   alpha=0.6, bins=20, color='blue', label='No Diabetes', density=True)
    axes[idx].hist(df_diabetes[df_diabetes['Outcome']==1][feature], 
                   alpha=0.6, bins=20, color='red', label='Diabetes', density=True)
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Density', fontsize=10)
    axes[idx].set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('diabetes_eda.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis
plt.figure(figsize=(10, 8))
correlation = df_diabetes.corr()
sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.savefig('diabetes_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("SECTION 7: TRAIN LOGISTIC REGRESSION MODEL (DIABETES)")
print("="*80)

# Prepare data
X = df_diabetes.drop('Outcome', axis=1).values
y = df_diabetes['Outcome'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train sklearn model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Train statsmodels for detailed statistics
X_train_sm = sm.add_constant(X_train_scaled)
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit(disp=0)

print("\n📊 SKLEARN MODEL RESULTS:")
print("-" * 60)
print(f"Training accuracy: {lr_model.score(X_train_scaled, y_train):.4f}")
print(f"Test accuracy: {lr_model.score(X_test_scaled, y_test):.4f}")

print("\n📈 STATSMODELS DETAILED STATISTICS:")
print("-" * 60)
print(result.summary())

# Feature importance analysis
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': lr_model.coef_[0],
    'Abs_Coefficient': np.abs(lr_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("\n🔍 FEATURE IMPORTANCE (Coefficient Magnitude):")
print("-" * 60)
print(coefficients)

# Visualize coefficients
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coefficients['Coefficient']]
plt.barh(coefficients['Feature'], coefficients['Coefficient'], color=colors, alpha=0.7)
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Coefficients (Positive = Increases Diabetes Risk)', 
          fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('diabetes_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n💡 INTERPRETATION:")
print("-" * 60)
print(f"Strongest positive predictor: {coefficients.iloc[0]['Feature']}")
print(f"  → Higher values increase diabetes probability")
print(f"\nCoefficient = {coefficients.iloc[0]['Coefficient']:.4f}")
print(f"  → 1 SD increase in {coefficients.iloc[0]['Feature']} multiplies odds by e^{coefficients.iloc[0]['Coefficient']:.4f} = {np.exp(coefficients.iloc[0]['Coefficient']):.2f}")

print("\n" + "="*80)
print("SECTION 8: MODEL EVALUATION - CONFUSION MATRIX")
print("="*80)

# Predictions
y_pred = lr_model.predict(X_test_scaled)
y_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted', fontsize=12)
axes[0].set_ylabel('Actual', fontsize=12)
axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
            cbar_kws={'label': 'Percentage'})
axes[1].set_xlabel('Predicted', fontsize=12)
axes[1].set_ylabel('Actual', fontsize=12)
axes[1].set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('diabetes_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * (precision * recall) / (precision + recall)
specificity = tn / (tn + fp)

print("\n📊 DETAILED METRICS:")
print("-" * 60)
print(f"True Negatives (TN):  {tn:4d} - Correctly identified non-diabetic")
print(f"False Positives (FP): {fp:4d} - Incorrectly identified as diabetic")
print(f"False Negatives (FN): {fn:4d} - Missed diabetic cases")
print(f"True Positives (TP):  {tp:4d} - Correctly identified diabetic")
print()
print(f"Accuracy:    {accuracy:.4f} - Overall correctness")
print(f"Precision:   {precision:.4f} - Of predicted diabetic, how many truly are")
print(f"Recall:      {recall:.4f} - Of actual diabetic, how many we caught")
print(f"F1-Score:    {f1:.4f} - Harmonic mean of precision & recall")
print(f"Specificity: {specificity:.4f} - Of actual non-diabetic, how many we caught")

print("\n📋 CLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(y_test, y_pred, 
                          target_names=['No Diabetes', 'Diabetes']))

print("\n" + "="*80)
print("SECTION 9: ROC CURVE AND AUC")
print("="*80)

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
axes[0].plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
axes[0].fill_between(fpr, tpr, alpha=0.2, color='orange')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
axes[0].set_title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=14, fontweight='bold')
axes[0].legend(loc="lower right", fontsize=11)
axes[0].grid(True, alpha=0.3)

# Precision-Recall vs Threshold
precisions = []
recalls = []
for thresh in thresholds:
    y_pred_thresh = (y_pred_proba >= thresh).astype(int)
    if y_pred_thresh.sum() > 0:
        precisions.append(precision_score(y_test, y_pred_thresh, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_thresh))
    else:
        precisions.append(1.0)
        recalls.append(0.0)

axes[1].plot(thresholds, precisions, 'b-', label='Precision', linewidth=2)
axes[1].plot(thresholds, recalls, 'g-', label='Recall', linewidth=2)
axes[1].axvline(x=0.5, color='r', linestyle='--', linewidth=2, 
               label='Default Threshold (0.5)')
axes[1].set_xlabel('Threshold', fontsize=12)
axes[1].set_ylabel('Score', fontsize=12)
axes[1].set_title('Precision & Recall vs Decision Threshold', 
                 fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim([0, 1])
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('diabetes_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

from sklearn.metrics import precision_score, recall_score

print("\n🎯 ROC-AUC INTERPRETATION:")
print("-" * 60)
print(f"AUC = {roc_auc:.4f}")
if roc_auc > 0.9:
    print("  → Excellent model")
elif roc_auc > 0.8:
    print("  → Good model")
elif roc_auc > 0.7:
    print("  → Fair model")
else:
    print("  → Poor model")
print(f"\nThe model correctly ranks a random diabetic patient")
print(f"higher than a random non-diabetic patient {roc_auc:.1%} of the time.")

print("\n" + "="*80)
print("SECTION 10: PROJECT 2 - IRIS FLOWER CLASSIFICATION")
print("="*80)

# Load Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Create DataFrame
df_iris = pd.DataFrame(X_iris, columns=feature_names)
df_iris['species'] = pd.Categorical.from_codes(y_iris, target_names)

print("📊 IRIS DATASET OVERVIEW:")
print("-" * 60)
print(df_iris.head(10))
print(f"\nShape: {df_iris.shape}")
print(f"\nSpecies distribution:")
print(df_iris['species'].value_counts())

# Visualize Iris dataset
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Pairplot style visualization
for idx, (i, j) in enumerate([(0, 1), (0, 2), (0, 3), (2, 3)]):
    ax = axes[idx // 2, idx % 2]
    for species_idx, species in enumerate(target_names):
        mask = y_iris == species_idx
        ax.scatter(X_iris[mask, i], X_iris[mask, j], 
                  label=species, alpha=0.6, s=100, edgecolors='black')
    ax.set_xlabel(feature_names[i], fontsize=11)
    ax.set_ylabel(feature_names[j], fontsize=11)
    ax.set_title(f'{feature_names[i]} vs {feature_names[j]}', 
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_eda.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("SECTION 11: BINARY CLASSIFICATION (IRIS)")
print("="*80)
print("Task: Classify Setosa vs Non-Setosa (Versicolor + Virginica)")
print("-" * 60)

# Binary classification: Setosa (0) vs Others (1)
y_binary = (y_iris != 0).astype(int)
X_iris_binary = X_iris[:, :2]  # Use only first 2 features for visualization

# Split and scale
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris_binary, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

scaler_iris = StandardScaler()
X_train_iris_scaled = scaler_iris.fit_transform(X_train_iris)
X_test_iris_scaled = scaler_iris.transform(X_test_iris)

# Train model
lr_iris = LogisticRegression(random_state=42)
lr_iris.fit(X_train_iris_scaled, y_train_iris)

print(f"Training accuracy: {lr_iris.score(X_train_iris_scaled, y_train_iris):.4f}")
print(f"Test accuracy: {lr_iris.score(X_test_iris_scaled, y_test_iris):.4f}")

# Decision boundary visualization
x_min, x_max = X_train_iris_scaled[:, 0].min() - 1, X_train_iris_scaled[:, 0].max() + 1
y_min, y_max = X_train_iris_scaled[:, 1].min() - 1, X_train_iris_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = lr_iris.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))
contourf = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.6)
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, linestyles='dashed')

# Plot training data
colors = ['blue', 'red']
markers = ['o', 's']
labels = ['Setosa', 'Others (Versicolor + Virginica)']

for idx, (color, marker, label) in enumerate(zip(colors, markers, labels)):
    mask = y_train_iris == idx
    plt.scatter(X_train_iris_scaled[mask, 0], X_train_iris_scaled[mask, 1],
               c=color, marker=marker, s=120, edgecolors='black', 
               linewidths=1.5, label=label, alpha=0.8)

plt.xlabel(feature_names[0], fontsize=12)
plt.ylabel(feature_names[1], fontsize=12)
plt.title('Binary Classification: Setosa vs Others', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
cbar = plt.colorbar(contourf)
cbar.set_label('P(Others)', fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iris_binary_decision_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion matrix
y_pred_iris = lr_iris.predict(X_test_iris_scaled)
cm_iris = confusion_matrix(y_test_iris, y_pred_iris)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_iris, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Setosa', 'Others'],
           yticklabels=['Setosa', 'Others'])
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix: Binary Iris Classification', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("SECTION 12: MULTI-CLASS CLASSIFICATION (IRIS)")
print("="*80)
print("Task: Classify all 3 species (Setosa, Versicolor, Virginica)")
print("-" * 60)

# Use all features for better performance
X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

scaler_mc = StandardScaler()
X_train_mc_scaled = scaler_mc.fit_transform(X_train_mc)
X_test_mc_scaled = scaler_mc.transform(X_test_mc)

# Train multi-class model (One-vs-Rest by default)
lr_multiclass = LogisticRegression(multi_class='ovr', random_state=42, max_iter=1000)
lr_multiclass.fit(X_train_mc_scaled, y_train_mc)

print(f"Training accuracy: {lr_multiclass.score(X_train_mc_scaled, y_train_mc):.4f}")
print(f"Test accuracy: {lr_multiclass.score(X_test_mc_scaled, y_test_mc):.4f}")

# Multi-class predictions
y_pred_mc = lr_multiclass.predict(X_test_mc_scaled)
y_pred_proba_mc = lr_multiclass.predict_proba(X_test_mc_scaled)

# Confusion matrix
cm_multiclass = confusion_matrix(y_test_mc, y_pred_mc)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_multiclass, annot=True, fmt='d', cmap='YlGnBu',
           xticklabels=target_names,
           yticklabels=target_names,
           cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Species', fontsize=12)
plt.ylabel('Actual Species', fontsize=12)
plt.title('Multi-class Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('iris_multiclass_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n📋 MULTI-CLASS CLASSIFICATION REPORT:")
print("-" * 60)
print(classification_report(y_test_mc, y_pred_mc, target_names=target_names))

# Visualize decision boundaries (2D projection using first 2 features)
X_2d = X_iris[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_iris, test_size=0.3, random_state=42, stratify=y_iris
)

scaler_2d = StandardScaler()
X_train_2d_scaled = scaler_2d.fit_transform(X_train_2d)
X_test_2d_scaled = scaler_2d.transform(X_test_2d)

lr_2d = LogisticRegression(multi_class='ovr', random_state=42)
lr_2d.fit(X_train_2d_scaled, y_train_2d)

# Create mesh
x_min, x_max = X_train_2d_scaled[:, 0].min() - 1, X_train_2d_scaled[:, 0].max() + 1
y_min, y_max = X_train_2d_scaled[:, 1].min() - 1, X_train_2d_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z_mc = lr_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z_mc = Z_mc.reshape(xx.shape)

# Plot
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z_mc, alpha=0.4, cmap='viridis', levels=2)

colors_mc = ['red', 'green', 'blue']
markers_mc = ['o', 's', '^']

for idx, (color, marker, name) in enumerate(zip(colors_mc, markers_mc, target_names)):
    mask = y_train_2d == idx
    plt.scatter(X_train_2d_scaled[mask, 0], X_train_2d_scaled[mask, 1],
               c=color, marker=marker, s=120, edgecolors='black',
               linewidths=1.5, label=name, alpha=0.8)

plt.xlabel(feature_names[0], fontsize=12)
plt.ylabel(feature_names[1], fontsize=12)
plt.title('Multi-class Decision Boundaries (3 Species)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iris_multiclass_boundaries.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("SECTION 13: COMPARING OPTIMIZATION METHODS")
print("="*80)

# Compare different solvers
solvers = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
results = []

for solver in solvers:
    try:
        model = LogisticRegression(solver=solver, max_iter=1000, random_state=42)
        model.fit(X_train_mc_scaled, y_train_mc)
        train_acc = model.score(X_train_mc_scaled, y_train_mc)
        test_acc = model.score(X_test_mc_scaled, y_test_mc)
        results.append({
            'Solver': solver,
            'Train Accuracy': train_acc,
            'Test Accuracy': test_acc
        })
    except Exception as e:
        print(f"⚠️  {solver} failed: {str(e)}")

df_solvers = pd.DataFrame(results)
print("\n🔧 SOLVER COMPARISON:")
print("-" * 60)
print(df_solvers.to_string(index=False))

# Visualize
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(df_solvers))
width = 0.35

bars1 = ax.bar(x_pos - width/2, df_solvers['Train Accuracy'], width, 
              label='Train', alpha=0.8, color='skyblue')
bars2 = ax.bar(x_pos + width/2, df_solvers['Test Accuracy'], width,
              label='Test', alpha=0.8, color='lightcoral')

ax.set_xlabel('Solver', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Comparison of Optimization Solvers', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(df_solvers['Solver'])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.9, 1.0])

plt.tight_layout()
plt.savefig('solver_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("SECTION 14: REGULARIZATION (L1 vs L2)")
print("="*80)

# Compare L1 (Lasso) and L2 (Ridge) regularization
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
l1_scores = []
l2_scores = []

for C in C_values:
    # L1
    model_l1 = LogisticRegression(penalty='l1', C=C, solver='liblinear', 
                                 random_state=42, max_iter=1000)
    model_l1.fit(X_train_mc_scaled, y_train_mc)
    l1_scores.append(model_l1.score(X_test_mc_scaled, y_test_mc))
    
    # L2
    model_l2 = LogisticRegression(penalty='l2', C=C, solver='lbfgs',
                                 random_state=42, max_iter=1000)
    model_l2.fit(X_train_mc_scaled, y_train_mc)
    l2_scores.append(model_l2.score(X_test_mc_scaled, y_test_mc))

plt.figure(figsize=(10, 6))
plt.plot(C_values, l1_scores, 'o-', linewidth=2, markersize=8, 
        label='L1 (Lasso)', color='blue')
plt.plot(C_values, l2_scores, 's-', linewidth=2, markersize=8,
        label='L2 (Ridge)', color='red')
plt.xscale('log')
plt.xlabel('Regularization Parameter (C)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('L1 vs L2 Regularization', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('regularization_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nREGULARIZATION INSIGHTS:")
print("-" * 60)
print("L1 (Lasso):")
print("  • Performs feature selection (drives some coefficients to exactly 0)")
print("  • Better when you have many irrelevant features")
print("  • Creates sparse models")
print("\nL2 (Ridge):")
print("  • Shrinks all coefficients but doesn't zero them out")
print("  • Better when all features are somewhat relevant")
print("  • More stable with correlated features")

print("\n" + "="*80)
print("SECTION 15: KEY TAKEAWAYS & SUMMARY")
print("="*80)

summary = """
WHAT YOU LEARNED TODAY:

1. CLASSIFICATION vs REGRESSION
   ✓ Regression: Continuous outputs (prices, temperatures)
   ✓ Classification: Discrete categories (spam/ham, yes/no)

2. SIGMOID FUNCTION
   ✓ Squashes any input to (0, 1) range
   ✓ σ(z) = 1 / (1 + e^(-z))
   ✓ Perfect for converting linear outputs to probabilities

3. LOG LOSS (CROSS-ENTROPY)
   ✓ Measures prediction quality for classification
   ✓ Penalizes confident wrong predictions heavily
   ✓ Convex function → guaranteed convergence

4. DECISION BOUNDARIES
   ✓ Linear boundaries separate classes
   ✓ Defined by: w₁x₁ + w₂x₂ + ... + b = 0
   ✓ Distance from boundary = prediction confidence

5. MODEL EVALUATION
   ✓ Confusion Matrix: TN, FP, FN, TP
   ✓ Accuracy, Precision, Recall, F1-Score
   ✓ ROC Curve and AUC for threshold analysis

6. FEATURE IMPORTANCE
   ✓ Coefficient magnitude shows feature impact
   ✓ Positive coefficients increase probability
   ✓ Negative coefficients decrease probability

7. MULTI-CLASS CLASSIFICATION
   ✓ One-vs-Rest (OvR): Train N binary classifiers
   ✓ Multinomial: Direct multi-class prediction
   ✓ Softmax for probability distribution

8. REGULARIZATION
   ✓ L1 (Lasso): Feature selection, sparse models
   ✓ L2 (Ridge): Coefficient shrinkage, stable
   ✓ C parameter: Lower C = stronger regularization

LIBRARIES MASTERED:
   ✓ NumPy: Vectorized operations, sigmoid, math
   ✓ Pandas: Data manipulation, exploratory analysis
   ✓ Matplotlib: Plots, decision boundaries, visualizations
   ✓ Seaborn: Statistical plots, heatmaps, correlation
   ✓ Scipy: Optimization, statistical functions
   ✓ Statsmodels: Detailed statistical analysis
   ✓ Sklearn: Model training, evaluation, metrics

 NEXT STEPS:
   • Practice with more datasets
   • Experiment with different thresholds
   • Try feature engineering
   • Explore polynomial features for non-linear boundaries
   • Study multi-class strategies in depth
"""


print("ALL VISUALIZATIONS SAVED!")
print("Files created:")
print("  1. classification_vs_regression.png")
print("  2. sigmoid_function_analysis.png")
print("  3. log_loss_visualization.png")
print("  4. training_progress.png")
print("  5. decision_boundaries.png")
print("  6. diabetes_eda.png")
print("  7. diabetes_correlation.png")
print("  8. diabetes_feature_importance.png")
print("  9. diabetes_confusion_matrix.png")
print(" 10. diabetes_roc_curve.png")
print(" 11. iris_eda.png")
print(" 12. iris_binary_decision_boundary.png")
print(" 13. iris_binary_confusion_matrix.png")
print(" 14. iris_multiclass_confusion_matrix.png")
print(" 15. iris_multiclass_boundaries.png")
print(" 16. solver_comparison.png")
print(" 17. regularization_comparison.png")

