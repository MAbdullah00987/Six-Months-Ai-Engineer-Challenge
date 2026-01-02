
#Project 1: Linear Regression with Gradient Descent (2.5 hours)
#Generate synthetic 2D data (y = 3x + 2 + noise)
#Implement gradient descent to find best-fit line
#Use MSE loss function: L = (1/n)Σ(y_pred - y_true)²
#Derive and implement gradients for slope and intercept
#Visualize:
#Data points and evolving fit line
#Loss over iterations
#Parameter convergence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Set random seed for reproducibility
np.random.seed(42)

# ==================== DATA GENERATION ====================
def generate_data(n_samples=100, slope=3, intercept=2, noise_std=1.5):
    """Generate synthetic 2D data: y = 3x + 2 + noise"""
    X = np.random.uniform(-10, 10, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = slope * X + intercept + noise
    return X, y

# ==================== GRADIENT DESCENT ====================
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.slope = 0
        self.intercept = 0
        self.history = {
            'slope': [],
            'intercept': [],
            'loss': []
        }
    
    def mse_loss(self, X, y):
        """Calculate Mean Squared Error loss"""
        y_pred = self.slope * X + self.intercept
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def compute_gradients(self, X, y):
        """
        Derive gradients for MSE loss:
        L = (1/n)Σ(y_pred - y_true)²
        where y_pred = slope * x + intercept
        
        ∂L/∂slope = (2/n)Σ(y_pred - y_true) * x
        ∂L/∂intercept = (2/n)Σ(y_pred - y_true)
        """
        n = len(X)
        y_pred = self.slope * X + self.intercept
        error = y_pred - y
        
        grad_slope = (2/n) * np.sum(error * X)
        grad_intercept = (2/n) * np.sum(error)
        
        return grad_slope, grad_intercept
    
    def fit(self, X, y):
        """Train the model using gradient descent"""
        # Initialize parameters randomly
        self.slope = np.random.randn()
        self.intercept = np.random.randn()
        
        for i in range(self.n_iterations):
            # Compute gradients
            grad_slope, grad_intercept = self.compute_gradients(X, y)
            
            # Update parameters
            self.slope -= self.lr * grad_slope
            self.intercept -= self.lr * grad_intercept
            
            # Record history
            loss = self.mse_loss(X, y)
            self.history['slope'].append(self.slope)
            self.history['intercept'].append(self.intercept)
            self.history['loss'].append(loss)
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iterations} - Loss: {loss:.4f}, "
                      f"Slope: {self.slope:.4f}, Intercept: {self.intercept:.4f}")
    
    def predict(self, X):
        """Make predictions"""
        return self.slope * X + self.intercept

# ==================== VISUALIZATION ====================
def visualize_results(X, y, model):
    """Create comprehensive visualizations"""
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Data points with final fit line
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(X, y, alpha=0.6, s=50, color='blue', label='Data points')
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line)
    ax1.plot(X_line, y_line, 'r-', linewidth=2, label=f'Fit: y={model.slope:.2f}x+{model.intercept:.2f}')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Linear Regression - Final Fit', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss over iterations
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(model.history['loss'], color='red', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.set_title('Loss Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Slope convergence
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(model.history['slope'], color='green', linewidth=2)
    ax3.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='True slope (3)')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Slope', fontsize=12)
    ax3.set_title('Slope Parameter Convergence', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Intercept convergence
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(model.history['intercept'], color='purple', linewidth=2)
    ax4.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='True intercept (2)')
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Intercept', fontsize=12)
    ax4.set_title('Intercept Parameter Convergence', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Residuals plot
    ax5 = plt.subplot(2, 3, 5)
    y_pred = model.predict(X)
    residuals = y - y_pred
    ax5.scatter(y_pred, residuals, alpha=0.6, s=50, color='blue')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('Predicted Values', fontsize=12)
    ax5.set_ylabel('Residuals', fontsize=12)
    ax5.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Evolution of fit line (animation-style)
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(X, y, alpha=0.6, s=50, color='blue', label='Data points')
    
    # Show fit line at different stages
    checkpoints = [0, len(model.history['slope'])//4, len(model.history['slope'])//2, 
                   3*len(model.history['slope'])//4, -1]
    colors = ['gray', 'orange', 'yellow', 'pink', 'red']
    alphas = [0.3, 0.4, 0.5, 0.7, 1.0]
    
    for checkpoint, color, alpha in zip(checkpoints, colors, alphas):
        slope_at = model.history['slope'][checkpoint]
        intercept_at = model.history['intercept'][checkpoint]
        y_line_at = slope_at * X_line + intercept_at
        label = f'Iter {checkpoint if checkpoint >= 0 else len(model.history["slope"])}'
        ax6.plot(X_line, y_line_at, color=color, linewidth=2, alpha=alpha, label=label)
    
    ax6.set_xlabel('X', fontsize=12)
    ax6.set_ylabel('y', fontsize=12)
    ax6.set_title('Evolution of Fit Line', fontsize=14, fontweight='bold')
    ax6.legend(loc='best', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'linear_regression_results.png'")
    plt.show()

# ==================== MAIN EXECUTION ====================
def main():
    print("="*60)
    print("LINEAR REGRESSION WITH GRADIENT DESCENT")
    print("="*60)
    
    # Generate data
    print("\n1. Generating synthetic data (y = 3x + 2 + noise)...")
    X, y = generate_data(n_samples=100)
    print(f"   Generated {len(X)} data points")
    
    # Create DataFrame for better display
    df = pd.DataFrame({'X': X, 'y': y})
    print("\nFirst 5 data points:")
    print(df.head())
    print(f"\nData statistics:\n{df.describe()}")
    
    # Train model
    print("\n2. Training Linear Regression with Gradient Descent...")
    print("-"*60)
    model = LinearRegressionGD(learning_rate=0.01, n_iterations=1000)
    model.fit(X, y)
    
    # Display results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"True parameters:      slope = 3.00, intercept = 2.00")
    print(f"Learned parameters:   slope = {model.slope:.4f}, intercept = {model.intercept:.4f}")
    print(f"Final loss (MSE):     {model.history['loss'][-1]:.4f}")
    
    # Calculate R² score
    y_pred = model.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    print(f"R² Score:             {r2_score:.4f}")
    
    # Visualize
    print("\n3. Creating visualizations...")
    visualize_results(X, y, model)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()