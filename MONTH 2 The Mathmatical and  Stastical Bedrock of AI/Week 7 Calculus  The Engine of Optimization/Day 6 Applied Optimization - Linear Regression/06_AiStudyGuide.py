'''
# Applied Optimization & Linear Regression - Complete Learning Roadmap

## 🎯 Learning Objectives

By completing this comprehensive guide, you will:
- Understand the mathematical foundations of optimization and calculus
- Master NumPy for efficient numerical computations
- Use Pandas for data manipulation and analysis
- Create publication-quality visualizations with Matplotlib and Seaborn
- Implement machine learning algorithms from scratch
- Apply advanced optimization techniques (Momentum, RMSprop, Adam)
- Build complete end-to-end ML pipelines

---

## 📚 8 Core Topics Covered

### **Topic 1: Mathematical Foundation - Derivatives with SymPy**
**File:** `1_sympy_derivatives.py`

**What You'll Learn:**
- Understanding derivatives geometrically
- Computing partial derivatives (gradients)
- Mean Squared Error loss function
- Symbolic mathematics with SymPy
- Visualizing loss functions

**Key Concepts:**
```python
# Loss function: L = (y - (wx + b))²
# Gradients: ∂L/∂w and ∂L/∂b
# Critical points: where gradient = 0
```

**Practice Exercises:**
1. Derive gradients for polynomial loss functions
2. Find minimum of f(x) = x³ - 6x² + 9x + 1
3. Compute Hessian matrix for convexity analysis

---

### **Topic 2: NumPy Operations - Vectorized Gradient Computation**
**File:** `2_numpy_data_ops.py`

**What You'll Learn:**
- Generating synthetic datasets
- Vectorized operations (10-100x faster than loops!)
- Broadcasting in NumPy
- Computing MSE loss efficiently
- Gradient computation for multiple data points

**Key NumPy Operations:**
```python
# Vectorized prediction
y_pred = X @ weights + bias  # Matrix multiplication

# Vectorized gradients
grad_w = (2/n) * (X.T @ (y_pred - y))
grad_b = (2/n) * np.sum(y_pred - y)
```

**Practice Exercises:**
1. Implement batch gradient computation
2. Profile loop vs vectorized code
3. Handle missing data with np.nan

---

### **Topic 3: Gradient Descent Implementation**
**File:** `3_gradient_descent_impl.py`

**What You'll Learn:**
- Complete gradient descent algorithm
- Learning rate selection
- Convergence monitoring
- Parameter space visualization
- Training history tracking

**Algorithm Structure:**
```python
for iteration in range(n_iterations):
    # 1. Forward pass
    y_pred = predict(X, w, b)
    
    # 2. Compute loss
    loss = mse_loss(y, y_pred)
    
    # 3. Compute gradients
    dw, db = compute_gradients(X, y, w, b)
    
    # 4. Update parameters
    w -= learning_rate * dw
    b -= learning_rate * db
```

**Practice Exercises:**
1. Implement mini-batch gradient descent
2. Add early stopping criterion
3. Experiment with learning rate schedules

---

### **Topic 4: Pandas for Data Analysis and Preprocessing**
**File:** `4_pandas_data_analysis.py`

**What You'll Learn:**
- DataFrame operations
- Statistical analysis (describe, corr)
- Feature engineering
- Train-test splitting
- Feature scaling (StandardScaler)
- Multiple linear regression

**Key Pandas Operations:**
```python
# Correlation analysis
df.corr()

# Feature engineering
df['total_rooms'] = df['bedrooms'] + df['bathrooms']

# One-hot encoding
pd.get_dummies(df, columns=['location'])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Practice Exercises:**
1. Handle categorical variables
2. Create polynomial features
3. Detect and remove outliers

---

### **Topic 5: Seaborn for Statistical Visualization**
**File:** `5_seaborn_visualization.py`

**What You'll Learn:**
- Distribution analysis (histplot, kde)
- Categorical comparisons (boxplot, violinplot)
- Regression visualization (regplot)
- Correlation heatmaps
- Pairplots for relationships
- Regression diagnostics

**Essential Seaborn Plots:**
```python
# Distribution
sns.histplot(data, kde=True)

# Relationships
sns.scatterplot(x='feature', y='target', hue='category')
sns.regplot(x='x', y='y')

# Correlation
sns.heatmap(df.corr(), annot=True)

# Comprehensive
sns.pairplot(df, hue='category')
```

**Practice Exercises:**
1. Create custom color palettes
2. Analyze residual patterns
3. Build regression diagnostic suite

---

### **Topic 6: Advanced Optimization Algorithms**
**File:** `6_advanced_optimization.py`

**What You'll Learn:**
- Momentum (exponential moving average)
- RMSprop (adaptive learning rates)
- Adam optimizer (combines best of both)
- Comparing convergence rates
- Hyperparameter sensitivity

**Algorithm Comparison:**

| Algorithm | Key Idea | Best For |
|-----------|----------|----------|
| Standard GD | Follow negative gradient | Small, simple problems |
| Momentum | Build velocity | Noisy gradients |
| RMSprop | Adaptive per-parameter LR | Non-stationary objectives |
| Adam | Momentum + RMSprop | General purpose (most popular) |

**Update Rules:**
```python
# Momentum
v = beta * v + (1 - beta) * gradient
param -= lr * v

# RMSprop
s = beta * s + (1 - beta) * gradient²
param -= lr * gradient / (√s + ε)

# Adam (combines both)
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient²
param -= lr * m_hat / (√v_hat + ε)
```

**Practice Exercises:**
1. Implement learning rate schedules
2. Compare on different loss landscapes
3. Tune hyperparameters (betas, epsilon)

---

### **Topic 7: Complete ML Pipeline - Real Estate Prediction**
**File:** `7_complete_ml_pipeline.py`

**What You'll Learn:**
- End-to-end workflow
- Data generation and EDA
- Feature engineering
- Model training with Adam
- Comprehensive evaluation
- Feature importance analysis

**Pipeline Phases:**
```
1. Data Generation → 2. EDA → 3. Feature Engineering →
4. Preprocessing → 5. Model Training → 6. Evaluation → 7. Visualization
```

**Evaluation Metrics:**
```python
R² Score:  Proportion of variance explained (0-1, higher better)
RMSE:      Root Mean Squared Error (same units as target)
MAE:       Mean Absolute Error (robust to outliers)
```

**Practice Exercises:**
1. Add cross-validation
2. Implement regularization (L1/L2)
3. Build prediction intervals

---

### **Topic 8: Mathematical Foundations - Complete Derivations**
**File:** `8_mathematical_derivations.py`

**What You'll Learn:**
- Symbolic derivation of gradients
- Matrix calculus
- Normal equations (analytical solution)
- Numerical verification
- Loss surface visualization

**Key Mathematical Results:**

**Single Variable:**
```
L = (y - (wx + b))²
∂L/∂w = 2(wx + b - y)·x
∂L/∂b = 2(wx + b - y)
```

**Multiple Variables (Matrix Form):**
```
MSE = (1/n)||y - Xw||²
∂MSE/∂w = (2/n)Xᵀ(Xw - y)

Normal Equation: w = (XᵀX)⁻¹Xᵀy
```

**Practice Exercises:**
1. Derive gradients for different loss functions
2. Prove convexity of MSE
3. Implement Newton's method

---

## 🛠️ Required Libraries Installation

```bash
# Core scientific computing
pip install numpy pandas matplotlib seaborn

# Symbolic mathematics
pip install sympy

# Machine learning
pip install scikit-learn

# Optional: Animation (Manim)
pip install manim
```

---

## 📊 Suggested Learning Path

### Week 1: Foundations
- **Day 1-2:** Topic 1 (SymPy, derivatives)
- **Day 3-4:** Topic 2 (NumPy, vectorization)
- **Day 5-7:** Topic 3 (Gradient descent)

### Week 2: Data Science Tools
- **Day 1-3:** Topic 4 (Pandas, data analysis)
- **Day 4-7:** Topic 5 (Seaborn, visualization)

### Week 3: Advanced Optimization
- **Day 1-4:** Topic 6 (Advanced optimizers)
- **Day 5-7:** Topic 7 (Complete pipeline)

### Week 4: Deep Dive
- **Day 1-3:** Topic 8 (Mathematical derivations)
- **Day 4-7:** Final project and practice

---

## 🎓 Practice Projects

### Beginner Level
1. **Simple Linear Regression:** Predict house prices from square footage
2. **Polynomial Regression:** Fit curves to non-linear data
3. **Gradient Visualization:** Animate gradient descent

### Intermediate Level
4. **Multiple Regression:** Predict with many features
5. **Regularization:** Implement Ridge and Lasso
6. **Cross-Validation:** K-fold evaluation

### Advanced Level
7. **Custom Optimizer:** Build your own Adam variant
8. **Feature Selection:** Automated feature engineering
9. **Production Pipeline:** Deploy model with REST API

---

## 📖 Mathematical Concepts Quick Reference

### Calculus
- **Derivative:** Rate of change
- **Partial Derivative:** Change w.r.t. one variable
- **Gradient:** Vector of partial derivatives
- **Chain Rule:** Derivative of compositions

### Linear Algebra
- **Matrix Multiplication:** `(m×n) @ (n×p) → (m×p)`
- **Transpose:** Flip rows and columns
- **Inverse:** `A⁻¹` such that `AA⁻¹ = I`
- **Eigenvalues:** Stretching factors

### Statistics
- **Mean:** Average value
- **Variance:** Spread around mean
- **Covariance:** Joint variability
- **Correlation:** Normalized covariance (-1 to 1)

### Optimization
- **Convex Function:** Single global minimum
- **Learning Rate:** Step size
- **Convergence:** Approaching optimal solution
- **Local Minimum:** Not necessarily global

---

## 🎯 Key Formulas Cheat Sheet

### Loss Functions
```
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²
MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|
R² = 1 - (SS_res / SS_tot)
```

### Gradients
```
∂MSE/∂w = (2/n) Σᵢ xᵢ(ŷᵢ - yᵢ)
∂MSE/∂b = (2/n) Σᵢ (ŷᵢ - yᵢ)
```

### Update Rules
```
Standard GD:  θ ← θ - α∇L
Momentum:     θ ← θ - α·v, where v ← βv + (1-β)∇L
Adam:         θ ← θ - α·m̂/(√v̂ + ε)
```

---

## 💡 Tips for Success

### Understanding
1. **Visualize Everything:** Plot data, loss, gradients
2. **Start Simple:** Master 1D before moving to multi-D
3. **Code from Scratch:** Don't rely on libraries initially
4. **Verify Numerically:** Check symbolic derivations

### Implementation
5. **Vectorize Operations:** Avoid Python loops
6. **Check Shapes:** Print array shapes frequently
7. **Start with Small Data:** Debug with 10 samples
8. **Plot Often:** Visualize intermediate results

### Debugging
9. **Check Gradients:** Compare numerical vs analytical
10. **Monitor Loss:** Should decrease consistently
11. **Watch for NaN:** Often from large learning rates
12. **Normalize Features:** Scale to similar ranges

---

## 🔗 Next Steps

After completing this guide:

1. **Explore Scikit-learn:** Compare your implementations
2. **Neural Networks:** Extend to deep learning
3. **Other Loss Functions:** Huber, LogCosh, etc.
4. **Regularization:** Prevent overfitting
5. **Bayesian Inference:** Probabilistic approaches
6. **Time Series:** Specialized regression techniques

---

## 📚 Recommended Resources

### Books
- "Pattern Recognition and Machine Learning" - Bishop
- "The Elements of Statistical Learning" - Hastie et al.
- "Deep Learning" - Goodfellow et al.

### Online
- 3Blue1Brown (YouTube): Calculus & Linear Algebra
- StatQuest (YouTube): Statistics & ML
- Fast.ai: Practical Deep Learning
- Distill.pub: Interactive ML explanations

### Practice
- Kaggle: Real datasets and competitions
- UCI ML Repository: Classic datasets
- Google Colab: Free GPU for experiments

---

## ✅ Self-Assessment Checklist

### Calculus & Optimization
- [ ] Can derive gradients by hand
- [ ] Understand chain rule applications
- [ ] Know when functions are convex
- [ ] Can visualize loss surfaces

### Programming
- [ ] Comfortable with NumPy operations
- [ ] Can vectorize computations
- [ ] Understand broadcasting rules
- [ ] Profile code for performance

### Machine Learning
- [ ] Implement gradient descent from scratch
- [ ] Understand train/test split
- [ ] Can interpret R² and RMSE
- [ ] Know when to use different optimizers

### Data Science
- [ ] Perform exploratory data analysis
- [ ] Engineer meaningful features
- [ ] Create publication-quality plots
- [ ] Diagnose model problems

---


**Remember:** The best way to learn is by doing. Run every example, modify the code, break things, fix them, and build your own projects!
'''