
#Part 2. Implementing from Scratch with NumPy

import numpy as np
import pandas as pd

class LinearRegressionScratch:
    """
    Multiple Linear Regression implemented from scratch using NumPy
    Supports both Normal Equation and Gradient Descent
    """
    
    def __init__(self, method='normal_equation', learning_rate=0.01, n_iterations=1000):
        """
        Parameters:
        - method: 'normal_equation' or 'gradient_descent'
        - learning_rate: Step size for gradient descent
        - n_iterations: Number of iterations for gradient descent
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []
        
    def add_intercept(self, X):
        """Add column of ones for intercept term"""
        m = X.shape[0]
        return np.c_[np.ones((m, 1)), X]
    
    def compute_cost(self, X, y, theta):
        """
        Compute Mean Squared Error cost
        J(θ) = (1/2m) * Σ(h(x) - y)²
        """
        m = len(y)
        predictions = X @ theta
        errors = predictions - y
        cost = (1/(2*m)) * np.sum(errors**2)
        return cost
    
    def normal_equation(self, X, y):
        """
        Closed-form solution: θ = (XᵀX)⁻¹Xᵀy
        """
        print("Using Normal Equation...")
        theta = np.linalg.inv(X.T @ X) @ X.T @ y
        return theta
    
    def gradient_descent(self, X, y):
        """
        Iterative optimization using gradient descent
        θⱼ := θⱼ - α(∂J/∂θⱼ)
        """
        print(f"Using Gradient Descent (α={self.learning_rate}, iterations={self.n_iterations})...")
        m, n = X.shape
        theta = np.zeros((n, 1))
        
        for i in range(self.n_iterations):
            # Compute predictions
            predictions = X @ theta
            
            # Compute errors (residuals)
            errors = predictions - y
            
            # Compute gradient: ∂J/∂θ = (1/m)Xᵀ(Xθ - y)
            gradient = (1/m) * X.T @ errors
            
            # Update parameters
            theta = theta - self.learning_rate * gradient
            
            # Track cost
            cost = self.compute_cost(X, y, theta)
            self.cost_history.append(cost)
            
            # Print progress every 100 iterations
            if (i+1) % 100 == 0:
                print(f"  Iteration {i+1}/{self.n_iterations}, Cost: {cost:.4f}")
        
        return theta
    
    def fit(self, X, y):
        """
        Train the model
        """
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Add intercept term
        X_b = self.add_intercept(X)
        
        # Choose method
        if self.method == 'normal_equation':
            self.theta = self.normal_equation(X_b, y)
        elif self.method == 'gradient_descent':
            self.theta = self.gradient_descent(X_b, y)
        else:
            raise ValueError("Method must be 'normal_equation' or 'gradient_descent'")
        
        print(f"\nLearned Parameters:")
        print(f"  θ₀ (intercept) = {self.theta[0, 0]:.4f}")
        for i in range(1, len(self.theta)):
            print(f"  θ_{i} = {self.theta[i, 0]:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_b = self.add_intercept(X)
        return X_b @ self.theta
    
    def get_residuals(self, X, y):
        """
        Compute residuals: ε = y - ŷ
        """
        predictions = self.predict(X)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y - predictions


# DEMONSTRATION
print("="*70)
print("LINEAR REGRESSION FROM SCRATCH - DEMONSTRATION")
print("="*70)

# Generate synthetic data
np.random.seed(42)
m = 100  # samples
n = 2    # features

X = np.random.randn(m, n)
true_theta = np.array([[4.0], [3.0], [2.0]])  # [intercept, coef1, coef2]
X_b = np.c_[np.ones((m, 1)), X]
y = X_b @ true_theta + np.random.randn(m, 1) * 0.5  # Add noise

print(f"\nDataset: {m} samples, {n} features")
print(f"True parameters: θ = {true_theta.T}")

# Method 1: Normal Equation
print("\n" + "="*70)
print("METHOD 1: NORMAL EQUATION")
print("="*70)
model_ne = LinearRegressionScratch(method='normal_equation')
model_ne.fit(X, y)

# Method 2: Gradient Descent
print("\n" + "="*70)
print("METHOD 2: GRADIENT DESCENT")
print("="*70)
model_gd = LinearRegressionScratch(
    method='gradient_descent', 
    learning_rate=0.1, 
    n_iterations=500
)
model_gd.fit(X, y)

# Compare predictions
print("\n" + "="*70)
print("COMPARISON OF METHODS")
print("="*70)
X_test = np.array([[1.0, 2.0], [2.0, 3.0], [-1.0, 1.0]])
pred_ne = model_ne.predict(X_test)
pred_gd = model_gd.predict(X_test)

print("\nTest Data:")
print(X_test)
print("\nPredictions (Normal Equation):")
print(pred_ne)
print("\nPredictions (Gradient Descent):")
print(pred_gd)
print("\nDifference:")
print(np.abs(pred_ne - pred_gd))

# Compute evaluation metrics
def compute_metrics(y_true, y_pred):
    """Compute MAE, MSE, RMSE, R²"""
    y_true = y_true.reshape(-1, 1) if y_true.ndim == 1 else y_true
    y_pred = y_pred.reshape(-1, 1) if y_pred.ndim == 1 else y_pred
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred)**2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # R² Score
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

print("\n" + "="*70)
print("EVALUATION METRICS (on training data)")
print("="*70)

metrics_ne = compute_metrics(y, model_ne.predict(X))
metrics_gd = compute_metrics(y, model_gd.predict(X))

print("\nNormal Equation:")
for metric, value in metrics_ne.items():
    print(f"  {metric}: {value:.6f}")

print("\nGradient Descent:")
for metric, value in metrics_gd.items():
    print(f"  {metric}: {value:.6f}")


print("Implementation complete!")
