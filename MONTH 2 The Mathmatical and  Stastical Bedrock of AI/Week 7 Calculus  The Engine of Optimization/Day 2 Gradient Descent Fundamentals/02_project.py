
#Project: Gradient Descent from Scratch (2-3 hours)
#Implement gradient descent to find the minimum of f(x) = x²
#* Write the algorithm from scratch (no ML libraries)
#* Track the path of convergence
#* Visualize the function and the optimization path
#* Print iteration number, x value, and function value at each step
#Deliverable: Clean implementation with visualization showing convergence


"""
Gradient Descent from Scratch
Objective: Find the minimum of f(x) = x²

Clean implementation without any warnings
"""

import warnings
warnings.filterwarnings('ignore')  # Suppress any non-critical warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, diff, lambdify

# Set style for better visualizations
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (14, 10)

class GradientDescent:
    """
    Gradient Descent optimizer for finding function minima
    """
    
    def __init__(self, learning_rate=0.1, max_iterations=100, tolerance=1e-6):
        """
        Initialize the optimizer
        
        Parameters:
        -----------
        learning_rate : float
            Step size for each iteration
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence threshold for gradient magnitude
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.history = []
        
    def objective_function(self, x):
        """
        The function we want to minimize: f(x) = x²
        
        Parameters:
        -----------
        x : float
            Input value
            
        Returns:
        --------
        float : f(x) = x²
        """
        return x ** 2
    
    def gradient(self, x):
        """
        Derivative of f(x) = x²: f'(x) = 2x
        
        Parameters:
        -----------
        x : float
            Input value
            
        Returns:
        --------
        float : f'(x) = 2x
        """
        return 2 * x
    
    def optimize(self, initial_x):
        """
        Run gradient descent optimization
        
        Parameters:
        -----------
        initial_x : float
            Starting point for optimization
            
        Returns:
        --------
        pd.DataFrame : History of optimization steps
        """
        x = initial_x
        self.history = []
        
        print(f"{'Iteration':<10} {'x':<15} {'f(x)':<15} {'Gradient':<15} {'Step Size':<15}")
        print("=" * 75)
        
        for iteration in range(self.max_iterations):
            # Calculate gradient and function value
            grad = self.gradient(x)
            fx = self.objective_function(x)
            
            # Store history
            self.history.append({
                'iteration': iteration,
                'x': x,
                'f(x)': fx,
                'gradient': grad,
                'step_size': self.learning_rate * abs(grad)
            })
            
            # Print progress
            print(f"{iteration:<10} {x:<15.6f} {fx:<15.6f} {grad:<15.6f} {self.learning_rate * abs(grad):<15.6f}")
            
            # Check convergence
            if abs(grad) < self.tolerance:
                print(f"\n✓ Converged after {iteration} iterations!")
                print(f"✓ Minimum found at x = {x:.6f}, f(x) = {fx:.6f}")
                break
            
            # Update x using gradient descent rule
            x = x - self.learning_rate * grad
        
        else:
            print(f"\n⚠ Reached maximum iterations ({self.max_iterations})")
            print(f"Current position: x = {x:.6f}, f(x) = {fx:.6f}")
        
        return pd.DataFrame(self.history)
    
    def visualize_static(self, df, initial_x):
        """
        Create static visualizations of the optimization process
        
        Parameters:
        -----------
        df : pd.DataFrame
            History of optimization steps
        initial_x : float
            Starting point
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Gradient Descent Optimization: f(x) = x²', fontsize=16, fontweight='bold')
        
        # 1. Function with optimization path
        ax1 = axes[0, 0]
        x_range = np.linspace(-initial_x * 1.2, initial_x * 1.2, 1000)
        y_range = self.objective_function(x_range)
        
        ax1.plot(x_range, y_range, 'b-', linewidth=2, label='f(x) = x²', alpha=0.7)
        ax1.plot(df['x'], df['f(x)'], 'ro-', markersize=6, linewidth=1.5, 
                label='Optimization Path', alpha=0.6)
        ax1.scatter(df['x'].iloc[0], df['f(x)'].iloc[0], color='green', s=200, 
                   marker='*', label='Start', zorder=5, edgecolors='black', linewidths=2)
        ax1.scatter(df['x'].iloc[-1], df['f(x)'].iloc[-1], color='red', s=200, 
                   marker='*', label='End', zorder=5, edgecolors='black', linewidths=2)
        
        ax1.set_xlabel('x', fontsize=12, fontweight='bold')
        ax1.set_ylabel('f(x)', fontsize=12, fontweight='bold')
        ax1.set_title('Function and Optimization Path', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Convergence of x value
        ax2 = axes[0, 1]
        ax2.plot(df['iteration'], df['x'], 'g-', linewidth=2, marker='o', markersize=4)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='True Minimum (x=0)')
        ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax2.set_ylabel('x value', fontsize=12, fontweight='bold')
        ax2.set_title('Convergence of x to Minimum', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Function value convergence
        ax3 = axes[1, 0]
        ax3.plot(df['iteration'], df['f(x)'], 'purple', linewidth=2, marker='s', markersize=4)
        ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax3.set_ylabel('f(x)', fontsize=12, fontweight='bold')
        ax3.set_title('Function Value Convergence', fontsize=13, fontweight='bold')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Gradient magnitude
        ax4 = axes[1, 1]
        ax4.plot(df['iteration'], abs(df['gradient']), 'orange', linewidth=2, marker='^', markersize=4)
        ax4.axhline(y=self.tolerance, color='r', linestyle='--', linewidth=2, 
                   alpha=0.7, label=f'Tolerance ({self.tolerance})')
        ax4.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax4.set_ylabel('|Gradient|', fontsize=12, fontweight='bold')
        ax4.set_title('Gradient Magnitude (Convergence Indicator)', fontsize=13, fontweight='bold')
        ax4.set_yscale('log')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_descent_optimization.png', dpi=300, bbox_inches='tight')
        print("✓ Static visualization saved as 'gradient_descent_optimization.png'")
        plt.close()
    
    def visualize_heatmap(self, df):
        """
        Create a heatmap showing the relationship between iterations and metrics
        
        Parameters:
        -----------
        df : pd.DataFrame
            History of optimization steps
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create normalized data for heatmap
        metrics = df[['x', 'f(x)', 'gradient', 'step_size']].T
        
        sns.heatmap(metrics, annot=False, cmap='RdYlGn_r', cbar_kws={'label': 'Value'}, 
                   xticklabels=df['iteration'], yticklabels=['x', 'f(x)', 'gradient', 'step_size'],
                   ax=ax)
        
        ax.set_title('Optimization Metrics Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('gradient_descent_heatmap.png', dpi=300, bbox_inches='tight')
        print("✓ Heatmap saved as 'gradient_descent_heatmap.png'")
        plt.close()


def verify_with_sympy():
    """
    Verify our gradient calculation using SymPy
    """
    print("\n" + "="*75)
    print("SYMBOLIC VERIFICATION WITH SYMPY")
    print("="*75)
    
    x = symbols('x')
    f = x**2
    
    # Calculate derivative symbolically
    df_dx = diff(f, x)
    
    print(f"Function: f(x) = {f}")
    print(f"Derivative: f'(x) = {df_dx}")
    
    # Verify at specific points
    f_numeric = lambdify(x, f, 'numpy')
    df_numeric = lambdify(x, df_dx, 'numpy')
    
    test_points = [0, 1, 5, 10, -5]
    print(f"\n{'x':<10} {'f(x)':<15} {'f\'(x)':<15}")
    print("-"*40)
    for point in test_points:
        print(f"{point:<10} {f_numeric(point):<15.6f} {df_numeric(point):<15.6f}")


def compare_learning_rates(initial_x=10):
    """
    Compare different learning rates
    
    Parameters:
    -----------
    initial_x : float
        Starting point
    """
    
    print("COMPARING DIFFERENT LEARNING RATES")
    
    
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for idx, lr in enumerate(learning_rates):
        print(f"Testing Learning Rate: {lr}")
        optimizer = GradientDescent(learning_rate=lr, max_iterations=50, tolerance=1e-6)
        
        # Suppress iteration output for comparison
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        df = optimizer.optimize(initial_x)
        
        sys.stdout = old_stdout
        
        ax = axes[idx]
        x_range = np.linspace(-initial_x * 1.2, initial_x * 1.2, 1000)
        y_range = optimizer.objective_function(x_range)
        
        ax.plot(x_range, y_range, 'b-', linewidth=2, alpha=0.5)
        ax.plot(df['x'], df['f(x)'], 'ro-', markersize=4, linewidth=1.5, alpha=0.7)
        ax.scatter(df['x'].iloc[0], df['f(x)'].iloc[0], color='green', s=150, 
                  marker='*', zorder=5)
        ax.scatter(df['x'].iloc[-1], df['f(x)'].iloc[-1], color='red', s=150, 
                  marker='*', zorder=5)
        
        ax.set_title(f'Learning Rate = {lr} ({len(df)} iterations)', fontweight='bold')
        ax.set_xlabel('x', fontweight='bold')
        ax.set_ylabel('f(x)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        print(f"  → Converged in {len(df)} iterations\n")
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Learning rate comparison saved as 'learning_rate_comparison.png'")
    plt.close()


def main():
    """
    Main execution function
    """
    print(" GRADIENT DESCENT OPTIMIZATION PROJECT")
    print(" Objective: Find the minimum of f(x) = x²")
    
    
    # Verify with SymPy first
    verify_with_sympy()
    
    # Initialize optimizer
   
    print("RUNNING GRADIENT DESCENT")
    
    
    initial_x = 10.0
    optimizer = GradientDescent(learning_rate=0.1, max_iterations=100, tolerance=1e-6)
    
    # Run optimization
    df_history = optimizer.optimize(initial_x)
    
    # Display summary statistics
  
    print("OPTIMIZATION SUMMARY STATISTICS")
    
    print(df_history.describe())
    
    # Save history to CSV
    df_history.to_csv('gradient_descent_history.csv', index=False)
    print("\n✓ Optimization history saved to 'gradient_descent_history.csv'")
    
    # Create visualizations
   
    print("GENERATING VISUALIZATIONS")
   
    
    optimizer.visualize_static(df_history, initial_x)
    optimizer.visualize_heatmap(df_history)
    
    # Compare learning rates
    compare_learning_rates(initial_x)
    
    print("\n Main Points Covered:")
    print("  1. gradient_descent_history.csv")
    print("  2. gradient_descent_optimization.png")
    print("  3. gradient_descent_heatmap.png")
    print("  4. learning_rate_comparison.png")
    print("\n All visualizations have been saved to the current directory!")
    print("="*75)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n Error occurred: {e}")
        import traceback
        traceback.print_exc()


