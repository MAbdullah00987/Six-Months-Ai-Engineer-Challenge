
#Day 7: Advanced Topics and Integration
#Focus: Local vs. global optimization and comprehensive review
#Learning Activities 
#Review: Week's key concepts
#Study: Optimization challenges (local minima, saddle points)
#Read: Brief introduction to second-order methods
#Projects
##Project 1: Local vs. Global Minima (2 hours)
#Create a function with multiple minima: f(x) = sin(x) + sin(1.5x) + 0.1x
#Run gradient descent from different starting points
#Show how algorithm gets stuck in local minima
#Create visualization showing:
#Function with multiple minima
#Different convergence paths
#Final solutions from different initializations
#Discuss implications for neural networks

#Project 2: Visualizing Optimization (2 hours)
#Create an animated visualization of gradient descent
#Use 2D function: f(x,y) = (x-2)² + (y+1)²
#Generate animation showing:
#Contour plot of the function
#Path of gradient descent ball rolling downhill
#Current gradient vector at each step
# #Iteration counter and loss value
#Save as GIF or MP4
#Project 3: Week Summary Document (1 hour)
#Create a comprehensive summary including:
#Key concepts learned
#All project results and visualizations
#Connections to neural networks
#Challenges faced and solutions
#Questions for further exploration

#Part 1: Core Concepts
#Gradient Descent Fundamentals

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# TOPIC 1: GRADIENT DESCENT REVIEW


class GradientDescentReview:
    """Complete review of gradient descent concepts"""
    
    def __init__(self):
        self.history = []
    
    def objective_function(self, x, y):
        """Simple quadratic function: f(x,y) = x² + y²"""
        return x**2 + y**2
    
    def gradient(self, x, y):
        """Analytical gradient: ∇f = [2x, 2y]"""
        return np.array([2*x, 2*y])
    
    def gradient_descent(self, start_point, learning_rate=0.1, iterations=50):
        """Basic gradient descent implementation"""
        point = np.array(start_point, dtype=float)
        self.history = [point.copy()]
        
        for i in range(iterations):
            grad = self.gradient(point[0], point[1])
            point = point - learning_rate * grad
            self.history.append(point.copy())
        
        return np.array(self.history)
    
    def visualize_descent(self):
        """Visualize the gradient descent process"""
        fig = plt.figure(figsize=(18, 5))
        
        # Create mesh for contour plot
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)
        
        # Plot 1: Contour plot with path
        ax1 = fig.add_subplot(131)
        contour = ax1.contour(X, Y, Z, levels=20, cmap='viridis')
        ax1.clabel(contour, inline=True, fontsize=8)
        
        history = np.array(self.history)
        ax1.plot(history[:, 0], history[:, 1], 'r.-', linewidth=2, 
                markersize=8, label='Gradient Descent Path')
        ax1.plot(history[0, 0], history[0, 1], 'go', markersize=12, 
                label='Start')
        ax1.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, 
                label='End')
        
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title('Contour Plot with Descent Path', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: 3D surface with path
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis')
        
        z_history = [self.objective_function(p[0], p[1]) for p in history]
        ax2.plot(history[:, 0], history[:, 1], z_history, 
                'r.-', linewidth=2, markersize=6)
        ax2.scatter(history[0, 0], history[0, 1], z_history[0], 
                   color='green', s=100, label='Start')
        ax2.scatter(history[-1, 0], history[-1, 1], z_history[-1], 
                   color='red', s=150, marker='*', label='End')
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_zlabel('f(x,y)')
        ax2.set_title('3D Surface with Descent Path')
        ax2.legend()
        
        # Plot 3: Convergence curve
        ax3 = fig.add_subplot(133)
        ax3.plot(z_history, 'b-', linewidth=2)
        ax3.set_xlabel('Iteration', fontsize=12)
        ax3.set_ylabel('Function Value', fontsize=12)
        ax3.set_title('Convergence Curve', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('gradient_descent_review.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("=" * 60)
        print("GRADIENT DESCENT REVIEW RESULTS")
        print("=" * 60)
        print(f"Starting point: {self.history[0]}")
        print(f"Final point: {self.history[-1]}")
        print(f"Starting value: {z_history[0]:.6f}")
        print(f"Final value: {z_history[-1]:.6f}")
        print(f"Total iterations: {len(self.history) - 1}")
        print(f"Improvement: {z_history[0] - z_history[-1]:.6f}")
        print("=" * 60)

# TOPIC 2: LEARNING RATE COMPARISON


def compare_learning_rates():
    """Compare different learning rates"""
    learning_rates = [0.01, 0.1, 0.5, 0.9]
    start = [4.0, 3.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Create mesh for background
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2
    
    for idx, lr in enumerate(learning_rates):
        gd = GradientDescentReview()
        history = gd.gradient_descent(start, learning_rate=lr, iterations=30)
        
        ax = axes[idx]
        contour = ax.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
        ax.plot(history[:, 0], history[:, 1], 'r.-', linewidth=2, markersize=6)
        ax.plot(history[0, 0], history[0, 1], 'go', markersize=12)
        ax.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15)
        
        ax.set_xlabel('x', fontsize=11)
        ax.set_ylabel('y', fontsize=11)
        ax.set_title(f'Learning Rate = {lr}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add text box with info
        final_val = history[-1][0]**2 + history[-1][1]**2
        textstr = f'Final value: {final_val:.4f}\nIterations: {len(history)-1}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.8), fontsize=10)
    
    plt.tight_layout()
    plt.savefig('learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nLearning Rate Effects:")
    print("-" * 50)
    print("Small LR (0.01): Slow but stable convergence")
    print("Medium LR (0.1): Good balance")
    print("Large LR (0.5): Fast but may overshoot")
    print("Very Large LR (0.9): Risk of divergence")
    print("-" * 50)


# MAIN EXECUTION
if __name__ == "__main__":
    print("ADVANCED OPTIMIZATION REVIEW")
    print("=" * 60)
    
    # Part 1: Basic gradient descent
    print("\n1. BASIC GRADIENT DESCENT")
    gd = GradientDescentReview()
    gd.gradient_descent(start_point=[4.0, 3.0], learning_rate=0.1, iterations=50)
    gd.visualize_descent()
    
    # Part 2: Learning rate comparison
    print("\n2. LEARNING RATE COMPARISON")
    compare_learning_rates()
    
    print("\n✓ Week review completed!")
    print("Key takeaway: Gradient descent follows the negative gradient to minimize functions")