#Day 2: Gradient Descent Fundamentals
#Focus: Understanding optimization and implementing basic gradient descent
#Learning Activities (3-4 hours)

#Read: Mathematics for Machine Learning, Chapter 5 - Section on optimization
#Watch: Coursera lectures on gradient descent and optimization
#Theory: Study the gradient descent update rule: x_{n+1} = x_n - α∇f(x_n)

##Project: Gradient Descent from Scratch (2-3 hours)
#Implement gradient descent to find the minimum of f(x) = x²

#Write the algorithm from scratch (no ML libraries)
#Track the path of convergence
#Visualize the function and the optimization path
#Print iteration number, x value, and function value at each step

#Deliverable: Clean implementation with visualization showing convergence


import sympy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

#1. Mathematical Foundation with SymPy
'''
# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Define symbolic variables
x, y, alpha = sp.symbols('x y alpha')

# Example 1: Simple quadratic function
f1 = x**2 + 2*x + 1
print("Function: f(x) =", f1)
print("Gradient: ∇f(x) =", sp.diff(f1, x))
print("At x=3:", sp.diff(f1, x).subs(x, 3))
print("\n" + "="*60 + "\n")

# Example 2: 2D quadratic function
f2 = x**2 + y**2
print("Function: f(x,y) =", f2)
print("Gradient: ∇f(x,y) =", [sp.diff(f2, x), sp.diff(f2, y)])
print("\n" + "="*60 + "\n")

# Example 3: More complex function
f3 = x**2 + 3*x*y + y**2
print("Function: f(x,y) =", f3)
grad_f3 = [sp.diff(f3, x), sp.diff(f3, y)]
print("Gradient: ∇f(x,y) =", grad_f3)
print("At x=2, y=3:", grad_f3[0].subs({x: 2, y: 3}), grad_f3[1].subs({x: 2, y: 3}))
print("\n" + "="*60 + "\n")
'''
#2. Implementing Gradient Descent from Scratch

'''
def gradient_descent_1d(f, df, x0, learning_rate=0.1, n_iterations=50):
    """
    Perform gradient descent on a 1D function
    
    Parameters:
    - f: function to minimize
    - df: derivative of f
    - x0: starting point
    - learning_rate: step size (α)
    - n_iterations: number of iterations
    """
    x_history = [x0]
    f_history = [f(x0)]
    
    x = x0
    for i in range(n_iterations):
        # Gradient descent update rule: x_{n+1} = x_n - α∇f(x_n)
        gradient = df(x)
        x = x - learning_rate * gradient
        
        x_history.append(x)
        f_history.append(f(x))
    
    return np.array(x_history), np.array(f_history)

# Define a simple quadratic function
def f(x):
    return x**2 + 4*x + 4

def df(x):
    return 2*x + 4

# Run gradient descent
x_hist, f_hist = gradient_descent_1d(f, df, x0=5, learning_rate=0.1, n_iterations=50)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Function and optimization path
x_range = np.linspace(-3, 6, 300)
ax1.plot(x_range, f(x_range), 'b-', linewidth=2, label='f(x) = x² + 4x + 4')
ax1.plot(x_hist, f_hist, 'ro-', markersize=4, label='GD path', alpha=0.6)
ax1.plot(x_hist[0], f_hist[0], 'go', markersize=10, label='Start')
ax1.plot(x_hist[-1], f_hist[-1], 'r*', markersize=15, label='End')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('Gradient Descent on 1D Function', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Convergence
ax2.plot(f_hist, 'b-', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('Convergence: Function Value vs Iteration', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Starting point: x₀ = {x_hist[0]:.4f}, f(x₀) = {f_hist[0]:.4f}")
print(f"Final point: x = {x_hist[-1]:.4f}, f(x) = {f_hist[-1]:.4f}")
print(f"True minimum: x = -2, f(x) = 0")

'''

#2: 2D Optimization with Contour Plots
'''

def gradient_descent_2d(f, grad_f, x0, y0, learning_rate=0.1, n_iterations=100):
    """
    Perform gradient descent on a 2D function
    """
    path = [(x0, y0)]
    
    x, y = x0, y0
    for i in range(n_iterations):
        # Compute gradient
        grad_x, grad_y = grad_f(x, y)
        
        # Update rule: (x,y)_{n+1} = (x,y)_n - α∇f(x,y)_n
        x = x - learning_rate * grad_x
        y = y - learning_rate * grad_y
        
        path.append((x, y))
    
    return np.array(path)

# Example: Rosenbrock function (banana function)
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_rosenbrock(x, y):
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return dx, dy

# Run gradient descent
path = gradient_descent_2d(rosenbrock, grad_rosenbrock, 
                           x0=-1, y0=1, 
                           learning_rate=0.001, 
                           n_iterations=1000)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Contour plot with optimization path
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

# Use logarithmic scale for better visualization
levels = np.logspace(-1, 3.5, 20)
contour = ax1.contour(X, Y, Z, levels=levels, cmap='viridis', alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)

# Plot optimization path
ax1.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, alpha=0.7, label='GD path')
ax1.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start')
ax1.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15, label='End')
ax1.plot(1, 1, 'b*', markersize=15, label='True minimum')

ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Gradient Descent on Rosenbrock Function', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Function value over iterations
f_values = [rosenbrock(p[0], p[1]) for p in path]
ax2.semilogy(f_values, 'b-', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('f(x, y) [log scale]', fontsize=12)
ax2.set_title('Convergence (Log Scale)', fontsize=14)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Starting: ({path[0, 0]:.4f}, {path[0, 1]:.4f}), f = {f_values[0]:.4f}")
print(f"Final: ({path[-1, 0]:.4f}, {path[-1, 1]:.4f}), f = {f_values[-1]:.4f}")
print(f"True minimum: (1, 1), f = 0")

'''

#Real Machine Learning Example: Linear Regression

'''

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)
y = 4 + 3 * X + np.random.randn(n_samples, 1)

# Add bias term
X_b = np.c_[np.ones((n_samples, 1)), X]

# Gradient descent for linear regression
def compute_cost(X, y, theta):
    """Mean Squared Error"""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

def compute_gradient(X, y, theta):
    """Gradient of MSE"""
    m = len(y)
    predictions = X.dot(theta)
    gradient = (1/m) * X.T.dot(predictions - y)
    return gradient

def gradient_descent_linear_regression(X, y, learning_rate=0.1, n_iterations=1000):
    """Gradient descent for linear regression"""
    m, n = X.shape
    theta = np.random.randn(n, 1)  # Random initialization
    
    cost_history = []
    theta_history = []
    
    for i in range(n_iterations):
        # Compute gradient
        gradient = compute_gradient(X, y, theta)
        
        # Update parameters: θ_{n+1} = θ_n - α∇J(θ_n)
        theta = theta - learning_rate * gradient
        
        # Track progress
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        theta_history.append(theta.copy())
        
        if i % 100 == 0:
            print(f"Iteration {i}: Cost = {cost:.4f}")
    
    return theta, np.array(cost_history), theta_history

# Run gradient descent
print("Training Linear Regression with Gradient Descent...")
print("="*60)
theta_final, cost_history, theta_history = gradient_descent_linear_regression(
    X_b, y, learning_rate=0.1, n_iterations=1000
)

# Compare with analytical solution
theta_analytical = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("\n" + "="*60)
print("Results:")
print(f"GD Solution: θ₀ = {theta_final[0][0]:.4f}, θ₁ = {theta_final[1][0]:.4f}")
print(f"Analytical: θ₀ = {theta_analytical[0][0]:.4f}, θ₁ = {theta_analytical[1][0]:.4f}")
print(f"True values: θ₀ = 4.0000, θ₁ = 3.0000")

# Visualization
fig = plt.figure(figsize=(18, 5))

# Plot 1: Data and fitted line
ax1 = plt.subplot(131)
ax1.scatter(X, y, alpha=0.6, s=50, label='Data points')
ax1.plot(X, X_b.dot(theta_final), 'r-', linewidth=3, label='GD fit')
ax1.plot(X, X_b.dot(theta_analytical), 'g--', linewidth=2, label='Analytical fit')
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Linear Regression: Data and Fit', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost function convergence
ax2 = plt.subplot(132)
ax2.plot(cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Cost J(θ)', fontsize=12)
ax2.set_title('Cost Function Convergence', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Parameter space trajectory
ax3 = plt.subplot(133)
theta0_history = [th[0][0] for th in theta_history]
theta1_history = [th[1][0] for th in theta_history]
ax3.plot(theta0_history, theta1_history, 'b-', linewidth=2, alpha=0.7)
ax3.plot(theta0_history[0], theta1_history[0], 'go', markersize=12, label='Start')
ax3.plot(theta0_history[-1], theta1_history[-1], 'r*', markersize=15, label='End')
ax3.plot(theta_analytical[0][0], theta_analytical[1][0], 'y^', 
         markersize=12, label='Analytical')
ax3.set_xlabel('θ₀ (intercept)', fontsize=12)
ax3.set_ylabel('θ₁ (slope)', fontsize=12)
ax3.set_title('Parameter Space Trajectory', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

'''
#Interactive Learning Rate Explorer

def learning_rate_sensitivity_analysis():
    """Analyze how different learning rates affect convergence"""
    
    def f(x):
        return x**2 + 4*x + 4
    
    def df(x):
        return 2*x + 4
    
    learning_rates = np.logspace(-2, 0, 20)  # 0.01 to 1
    final_values = []
    iterations_to_converge = []
    
    for lr in learning_rates:
        x_hist, f_hist = gradient_descent_1d(f, df, x0=5, 
                                             learning_rate=lr, 
                                             n_iterations=100)
        final_values.append(x_hist[-1])
        
        # Count iterations to reach near-optimal (within 0.01 of minimum)
        converged = np.where(np.abs(x_hist - (-2)) < 0.01)[0]
        if len(converged) > 0:
            iterations_to_converge.append(converged[0])
        else:
            iterations_to_converge.append(100)
    
    # Visualize
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Final value vs learning rate
    ax1.semilogx(learning_rates, final_values, 'bo-', linewidth=2, markersize=6)
    ax1.axhline(y=-2, color='r', linestyle='--', linewidth=2, label='True minimum')
    ax1.fill_between(learning_rates, -2.1, -1.9, alpha=0.2, color='green',
                     label='Acceptable range')
    ax1.set_xlabel('Learning Rate (α)', fontsize=12)
    ax1.set_ylabel('Final x value', fontsize=12)
    ax1.set_title('Final Value vs Learning Rate', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Iterations to converge
    ax2.semilogx(learning_rates, iterations_to_converge, 'go-', 
                 linewidth=2, markersize=6)
    ax2.set_xlabel('Learning Rate (α)', fontsize=12)
    ax2.set_ylabel('Iterations to Converge', fontsize=12)
    ax2.set_title('Convergence Speed vs Learning Rate', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Heatmap of convergence behavior
    test_lrs = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    test_x0s = [5, 4, 3, 2, 1, 0, -1]
    convergence_matrix = np.zeros((len(test_x0s), len(test_lrs)))
    
    for i, x0 in enumerate(test_x0s):
        for j, lr in enumerate(test_lrs):
            x_hist, _ = gradient_descent_1d(f, df, x0=x0, 
                                           learning_rate=lr, 
                                           n_iterations=50)
            # Check if converged to minimum
            if abs(x_hist[-1] - (-2)) < 0.1:
                convergence_matrix[i, j] = 1  # Converged
            else:
                convergence_matrix[i, j] = 0  # Did not converge
    
    # Visualize heatmap
    ax3.imshow(convergence_matrix, extent=[0, 1, 0, 1], 
               aspect='auto', cmap='viridis', 
               origin='lower', alpha=0.8)
    ax3.set_xlabel('Learning Rate (α)', fontsize=12)
    ax3.set_ylabel('Initial x₀', fontsize=12)
    ax3.set_title('Convergence Behavior', fontsize=14, fontweight='bold')
    ax3.grid(False)
    
    plt.tight_layout()
    plt.show()

    