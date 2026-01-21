
"""
Linear Regression from Scratch
Week 9 - Supervised Learning Project
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Optional

class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using gradient descent.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, 
                 tolerance: float = 1e-6, regularization: Optional[str] = None,
                 lambda_reg: float = 0.1):
        """
        Initialize Linear Regression model.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence threshold for cost function
        regularization : str, optional
            Type of regularization ('l1', 'l2', or None)
        lambda_reg : float
            Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        # Model parameters
        self.weights = None
        self.bias = None
        
        # Training history
        self.cost_history = []
        self.weight_history = []
        
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the cost function (Mean Squared Error with optional regularization).
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix (m samples, n features)
        y : np.ndarray
            Target values (m samples,)
            
        Returns:
        --------
        float : Cost value
        """
        m = len(y)
        predictions = X.dot(self.weights) + self.bias
        mse = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
        
        # Add regularization term
        if self.regularization == 'l2':
            reg_term = (self.lambda_reg / (2 * m)) * np.sum(self.weights ** 2)
            return mse + reg_term
        elif self.regularization == 'l1':
            reg_term = (self.lambda_reg / m) * np.sum(np.abs(self.weights))
            return mse + reg_term
        
        return mse
    
    def _compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients for weights and bias.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
            
        Returns:
        --------
        tuple : (weight_gradients, bias_gradient)
        """
        m = len(y)
        predictions = X.dot(self.weights) + self.bias
        errors = predictions - y
        
        # Compute gradients
        dw = (1 / m) * X.T.dot(errors)
        db = (1 / m) * np.sum(errors)
        
        # Add regularization gradient
        if self.regularization == 'l2':
            dw += (self.lambda_reg / m) * self.weights
        elif self.regularization == 'l1':
            dw += (self.lambda_reg / m) * np.sign(self.weights)
        
        return dw, db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> 'LinearRegressionScratch':
        """
        Train the linear regression model using gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features (m samples, n features)
        y : np.ndarray
            Training targets (m samples,)
        verbose : bool
            Print training progress
            
        Returns:
        --------
        self : Returns the instance itself
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Reset history
        self.cost_history = []
        self.weight_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Compute gradients
            dw, db = self._compute_gradients(X, y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store cost
            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)
            self.weight_history.append(self.weights.copy())
            
            # Print progress
            if verbose and (i % 100 == 0 or i == self.n_iterations - 1):
                print(f"Iteration {i}: Cost = {cost:.6f}")
            
            # Check for convergence
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {i}")
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Parameters:
        -----------
        X : np.ndarray
            Features to predict (m samples, n features)
            
        Returns:
        --------
        np.ndarray : Predictions
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
        
        return X.dot(self.weights) + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            True values
            
        Returns:
        --------
        float : R² score
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'weights': self.weights,
            'bias': self.bias,
            'cost_history': self.cost_history
        }


class ModelVisualizer:
    """
    Visualization utilities for linear regression analysis.
    """
    
    @staticmethod
    def plot_cost_history(model: LinearRegressionScratch, figsize: Tuple[int, int] = (12, 5)):
        """Plot cost function convergence."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Full history
        ax1.plot(model.cost_history, 'b-', linewidth=2)
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Cost (MSE)', fontsize=12)
        ax1.set_title('Cost Function Convergence', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Log scale (if cost decreases significantly)
        if len(model.cost_history) > 10:
            ax2.plot(model.cost_history, 'r-', linewidth=2)
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Cost (MSE)', fontsize=12)
            ax2.set_title('Cost Function (Log Scale)', fontsize=14, fontweight='bold')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = 'Predictions vs Actual',
                        figsize: Tuple[int, int] = (10, 6)):
        """Plot predictions vs actual values."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                      figsize: Tuple[int, int] = (12, 5)):
        """Plot residual analysis."""
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residual Plot', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residual distribution
        ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def compare_models(models: dict, X_test: np.ndarray, y_test: np.ndarray,
                      figsize: Tuple[int, int] = (12, 6)):
        """Compare predictions from multiple models."""
        n_models = len(models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, model) in zip(axes, models.items()):
            y_pred = model.predict(X_test)
            r2 = model.score(X_test, y_test)
            
            ax.scatter(y_test, y_pred, alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
            
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
            
            ax.set_xlabel('Actual Values', fontsize=10)
            ax.set_ylabel('Predicted Values', fontsize=10)
            ax.set_title(f'{name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate various regression metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values
        
    Returns:
    --------
    dict : Dictionary containing various metrics
    """
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
    
    # Adjusted R² (requires number of features)
    n = len(y_true)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    print("=" * 60)
    print("LINEAR REGRESSION FROM SCRATCH - DEMONSTRATION")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print("\nTraining Linear Regression from Scratch...")
    model = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE METRICS")
    print("=" * 60)
    metrics = calculate_metrics(y_test, y_pred)
    for metric, value in metrics.items():
        print(f"{metric:10s}: {value:.6f}")
    
    print(f"\nModel Parameters:")
    print(f"Weight: {model.weights[0]:.6f}")
    print(f"Bias: {model.bias:.6f}")
    print("\nTrue relationship: y = 4 + 3*x + noise")