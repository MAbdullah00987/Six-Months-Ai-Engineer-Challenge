
# Week 9: Supervised Learning - Linear Regression from Scratch

## 📋 Project Overview

This project implements **Linear Regression from scratch** using gradient descent and compares it with scikit-learn's implementation. It includes comprehensive data preprocessing, visualization, and model evaluation.

## 🎯 Learning Objectives

- Implement gradient descent optimization algorithm
- Build linear regression without using ML libraries
- Understand the mathematics behind linear regression
- Compare custom implementation with industry-standard libraries
- Visualize model convergence and performance
- Perform comprehensive model evaluation

## 📁 Project Structure

```
week-9-supervised-learning/
├── data/
│   ├── raw/                    # Original datasets
│   │   ├── salary_data.csv
│   │   └── housing_data.csv
│   └── processed/              # Cleaned and processed data
│
├── notebooks/
│   ├── 01_salary_prediction.ipynb      # Salary prediction project
│   ├── 02_housing_prices.ipynb         # Housing price prediction
│   └── 03_model_comparison.ipynb       # Advanced comparisons
│
├── src/
│   ├── models.py               # Linear regression implementation
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── visualization.py        # Plotting utilities
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Sample Data

```python
from src.preprocessing import generate_salary_dataset, generate_housing_dataset

# Generate salary dataset
salary_df = generate_salary_dataset(n_samples=500, save_path='data/raw/salary_data.csv')

# Generate housing dataset
housing_df = generate_housing_dataset(n_samples=1000, save_path='data/raw/housing_data.csv')
```

### 3. Run Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open notebooks in the notebooks/ folder
```

## 💡 Key Features

### Linear Regression Implementation

The custom implementation includes:

- **Gradient Descent Optimization**
  - Configurable learning rate
  - Convergence detection
  - Cost history tracking
  
- **Regularization Support**
  - L1 regularization (Lasso)
  - L2 regularization (Ridge)
  
- **Comprehensive Metrics**
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score (Coefficient of Determination)

### Data Preprocessing

- Missing value handling (mean, median, mode, drop)
- Categorical encoding (one-hot encoding)
- Feature scaling (standardization, normalization)
- Outlier detection and removal (IQR, Z-score)
- Polynomial feature creation

### Visualizations

- Cost function convergence plots
- Predictions vs actual scatter plots
- Residual analysis plots
- Feature importance charts
- Correlation heatmaps

## 📊 Example Usage

### Basic Linear Regression

```python
from src.models import LinearRegressionScratch
from src.preprocessing import DataPreprocessor
import pandas as pd

# Load data
df = pd.read_csv('data/raw/salary_data.csv')

# Preprocess
preprocessor = DataPreprocessor(scaling_method='standard')
X, y = preprocessor.prepare_features(df, target_column='Salary')

# Train model
model = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
r2_score = model.score(X_test, y_test)
print(f"R² Score: {r2_score:.4f}")
```

### Compare with Scikit-Learn

```python
from sklearn.linear_model import LinearRegression as SklearnLR
from src.models import calculate_metrics

# Train scikit-learn model
sklearn_model = SklearnLR()
sklearn_model.fit(X_train, y_train)

# Compare predictions
y_pred_scratch = model.predict(X_test)
y_pred_sklearn = sklearn_model.predict(X_test)

# Calculate metrics
metrics_scratch = calculate_metrics(y_test, y_pred_scratch)
metrics_sklearn = calculate_metrics(y_test, y_pred_sklearn)

print("Comparison:")
print(f"Custom R²: {metrics_scratch['R²']:.4f}")
print(f"Sklearn R²: {metrics_sklearn['R²']:.4f}")
```

## 📈 Visualization Examples

### Convergence Plot

```python
from src.models import ModelVisualizer

visualizer = ModelVisualizer()
visualizer.plot_cost_history(model)
```

### Predictions vs Actual

```python
visualizer.plot_predictions(y_test, y_pred, title='Model Predictions')
```

### Residual Analysis

```python
visualizer.plot_residuals(y_test, y_pred)
```

## 🔬 Mathematical Foundation

### Cost Function (Mean Squared Error)

```
J(θ) = (1/2m) Σ(h(x^(i)) - y^(i))²
```

### Gradient Descent Update Rule

```
θ_j := θ_j - α * (1/m) * Σ(h(x^(i)) - y^(i)) * x_j^(i)
```

Where:
- `θ` = model parameters (weights)
- `α` = learning rate
- `m` = number of training examples
- `h(x)` = hypothesis function (predictions)

### Regularization

**L2 (Ridge):**
```
J(θ) = MSE + (λ/2m) * Σθ_j²
```

**L1 (Lasso):**
```
J(θ) = MSE + (λ/m) * Σ|θ_j|
```

## 📝 Project Tasks Completed

- [x] Implement gradient descent from Week 7
- [x] Build linear regression from scratch
- [x] Compare with scikit-learn implementation
- [x] Visualize convergence and performance
- [x] Create comprehensive project structure
- [x] Add data preprocessing utilities
- [x] Implement multiple evaluation metrics
- [x] Create detailed documentation

## 🎓 Learning Resources

### Key Concepts Covered

1. **Linear Regression Theory**
   - Cost functions
   - Gradient descent optimization
   - Normal equation vs gradient descent

2. **Model Evaluation**
   - Train/test split
   - Cross-validation
   - Bias-variance tradeoff

3. **Feature Engineering**
   - Feature scaling
   - Polynomial features
   - Feature selection

4. **Regularization**
   - Ridge regression (L2)
   - Lasso regression (L1)
   - Elastic Net

## 🐛 Troubleshooting

### Model Not Converging

- **Reduce learning rate**: Try `learning_rate=0.001`
- **Increase iterations**: Set `n_iterations=5000`
- **Check feature scaling**: Ensure features are normalized

### Poor Performance

- **Add polynomial features**: Try higher-degree features
- **Check for outliers**: Use outlier detection
- **Try regularization**: Add L1/L2 regularization

### Memory Issues

- **Reduce dataset size**: Use sampling
- **Use batch gradient descent**: Implement mini-batch updates

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Gradient Descent Visualization](https://www.coursera.org/learn/machine-learning)
- [StatsModels Documentation](https://www.statsmodels.org/)

## 🤝 Contributing

Feel free to enhance this project by:

- Adding more datasets
- Implementing advanced algorithms
- Improving visualizations
- Adding unit tests

## 📄 License

This project is for educational purposes.

## 👤 Author

Created as part of Week 9 - Supervised Learning curriculum

