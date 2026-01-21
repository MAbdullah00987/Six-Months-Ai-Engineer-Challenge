"""
Linear Regression Implementation from Scratch
models.py - Contains custom linear regression implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using gradient descent.
    
    Parameters:
    -----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    n_iterations : int, default=1000
        Number of iterations for gradient descent
    regularization : str, default=None
        Type of regularization ('l1', 'l2', or None)
    lambda_reg : float, default=0.01
        Regularization parameter
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: Optional[str] = None, lambda_reg: float = 0.01):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _initialize_parameters(self, n_features: int):
        """Initialize weights and bias"""
        self.weights = np.zeros(n_features)
        self.bias = 0
        
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost function
        
        J(w,b) = (1/2m) * Σ(h(x) - y)²
        """
        m = len(y)
        predictions = self.predict(X)
        cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        
        # Add regularization term
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(self.weights))
            
        return cost
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias
        
        ∂J/∂w = (1/m) * X^T * (h(x) - y)
        ∂J/∂b = (1/m) * Σ(h(x) - y)
        """
        m = len(y)
        predictions = self.predict(X)
        error = predictions - y
        
        # Gradient for weights
        dw = (1 / m) * X.T.dot(error)
        
        # Add regularization gradient
        if self.regularization == 'l2':
            dw += (self.lambda_reg / m) * self.weights
        elif self.regularization == 'l1':
            dw += (self.lambda_reg / m) * np.sign(self.weights)
        
        # Gradient for bias
        db = (1 / m) * np.sum(error)
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LinearRegressionScratch':
        """
        Train the model using gradient descent
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values
        verbose : bool, default=True
            Print cost every 100 iterations
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self._initialize_parameters(n_features)
        self.cost_history = []
        
        # Gradient descent
        for iteration in range(self.n_iterations):
            # Compute cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if verbose and (iteration % 100 == 0 or iteration == self.n_iterations - 1):
                print(f"Iteration {iteration:4d}: Cost = {cost:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the linear model
        
        h(x) = w^T * x + b
        """
        X = np.array(X)
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score
        
        R² = 1 - (SS_res / SS_tot)
        """
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Return model parameters"""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'learning_rate': self.learning_rate,
            'n_iterations': self.n_iterations
        }
    
    def plot_cost_history(self, figsize: Tuple[int, int] = (10, 6)):
        """Plot cost function convergence"""
        plt.figure(figsize=figsize)
        plt.plot(self.cost_history, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Cost (MSE)', fontsize=12)
        plt.title('Cost Function Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt


class LinearRegressionNormalEquation:
    """
    Linear Regression using Normal Equation (Closed-form solution)
    
    θ = (X^T * X)^(-1) * X^T * y
    """
    
    def __init__(self):
        self.weights = None
        self.bias = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionNormalEquation':
        """
        Train model using normal equation
        """
        X = np.array(X)
        y = np.array(y)
        
        # Add bias term (column of ones)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal equation: θ = (X^T * X)^(-1) * X^T * y
        theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = np.array(X)
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def compare_models(X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   learning_rates: List[float] = [0.001, 0.01, 0.1],
                   n_iterations: int = 1000):
    """
    Compare different learning rates and visualize results
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Test data
    learning_rates : List of learning rates to compare
    n_iterations : Number of iterations for gradient descent
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    results = []
    
    for lr in learning_rates:
        # Train model
        model = LinearRegressionScratch(learning_rate=lr, n_iterations=n_iterations)
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        results.append({
            'learning_rate': lr,
            'train_r2': train_score,
            'test_r2': test_score,
            'final_cost': model.cost_history[-1],
            'model': model
        })
        
        # Plot cost history
        axes[0, 0].plot(model.cost_history, label=f'LR={lr}', linewidth=2)
    
    # Cost history comparison
    axes[0, 0].set_xlabel('Iteration', fontsize=11)
    axes[0, 0].set_ylabel('Cost', fontsize=11)
    axes[0, 0].set_title('Cost Function Convergence', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² scores comparison
    lrs = [r['learning_rate'] for r in results]
    train_scores = [r['train_r2'] for r in results]
    test_scores = [r['test_r2'] for r in results]
    
    x_pos = np.arange(len(lrs))
    width = 0.35
    
    axes[0, 1].bar(x_pos - width/2, train_scores, width, label='Train R²', alpha=0.8)
    axes[0, 1].bar(x_pos + width/2, test_scores, width, label='Test R²', alpha=0.8)
    axes[0, 1].set_xlabel('Learning Rate', fontsize=11)
    axes[0, 1].set_ylabel('R² Score', fontsize=11)
    axes[0, 1].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels([str(lr) for lr in lrs])
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Predictions vs Actual (best model)
    best_model = max(results, key=lambda x: x['test_r2'])['model']
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    
    axes[1, 0].scatter(y_train, y_pred_train, alpha=0.5, label='Train', s=30)
    axes[1, 0].scatter(y_test, y_pred_test, alpha=0.5, label='Test', s=30)
    
    # Perfect prediction line
    all_y = np.concatenate([y_train, y_test])
    axes[1, 0].plot([all_y.min(), all_y.max()], [all_y.min(), all_y.max()], 
                    'r--', linewidth=2, label='Perfect Prediction')
    
    axes[1, 0].set_xlabel('Actual Values', fontsize=11)
    axes[1, 0].set_ylabel('Predicted Values', fontsize=11)
    axes[1, 0].set_title(f'Predictions vs Actual (LR={best_model.learning_rate})', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals_train = y_train - y_pred_train
    residuals_test = y_test - y_pred_test
    
    axes[1, 1].scatter(y_pred_train, residuals_train, alpha=0.5, label='Train', s=30)
    axes[1, 1].scatter(y_pred_test, residuals_test, alpha=0.5, label='Test', s=30)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('Predicted Values', fontsize=11)
    axes[1, 1].set_ylabel('Residuals', fontsize=11)
    axes[1, 1].set_title('Residual Plot', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return results, fig


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate various regression metrics
    
    Returns:
    --------
    dict with MSE, RMSE, MAE, R², and Adjusted R²
    """
    n = len(y_true)
    
    # Mean Squared Error
    mse = np.mean((y_true - y_pred) ** 2)
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # R² Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }