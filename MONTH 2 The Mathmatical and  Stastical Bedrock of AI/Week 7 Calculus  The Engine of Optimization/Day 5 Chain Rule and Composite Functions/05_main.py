

#CHAIN RULE PRACTICE EXERCISES

import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import pandas as pd
import seaborn as sns


print("CHAIN RULE MASTERY: PRACTICE EXERCISES")


# EXERCISE No. 1 

def exercise_1_1():
    """Chain rule with polynomial"""
    print("\nExercise 1.1: Compute d/dx[(2x + 3)^4]")
    print("Solution steps:")
    
    x = sp.Symbol('x')
    g = 2*x + 3
    f = g**4
    
    print(f"  1. Identify: g(x) = {g}, f(u) = u^4")
    
    dg_dx = sp.diff(g, x)
    df_dg = 4 * g**3
    result = sp.diff(f, x)
    
    print(f"  2. dg/dx = {dg_dx}")
    print(f"  3. df/dg = {df_dg.subs(g, sp.Symbol('u'))}")
    print(f"  4. df/dx = (df/dg)(dg/dx) = {result}")
    print(f"  5. Simplified: {sp.simplify(result)}")
    
    # Numerical verification
    x_val = 2.0
    numerical = float(result.subs(x, x_val))
    print(f"\n  Verification at x={x_val}: df/dx = {numerical:.2f}")

def exercise_1_2():
    """Chain rule with trigonometric functions"""
    print("\nExercise 1.2: Compute d/dx[sin(x^2)]")
    print("Solution steps:")
    
    x = sp.Symbol('x')
    g = x**2
    f = sp.sin(g)
    
    print(f"  1. Identify: g(x) = {g}, f(u) = sin(u)")
    
    dg_dx = sp.diff(g, x)
    df_dg = sp.cos(g)
    result = sp.diff(f, x)
    
    print(f"  2. dg/dx = {dg_dx}")
    print(f"  3. df/dg = {df_dg.subs(g, sp.Symbol('u'))}")
    print(f"  4. df/dx = {result}")
    
    # Plot
    x_vals = np.linspace(-3, 3, 100)
    y_vals = np.sin(x_vals**2)
    dy_vals = 2*x_vals*np.cos(x_vals**2)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, y_vals, 'b-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('sin(x²)')
    plt.title('Function')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, dy_vals, 'r-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('d/dx[sin(x²)]')
    plt.title('Derivative')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_1_2.png', dpi=100, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved as 'exercise_1_2.png'")
    plt.close()

def exercise_1_3():
    """Multiple compositions"""
    print("\nExercise 1.3: Compute d/dx[e^(sin(x))]")
    print("Solution steps:")
    
    x = sp.Symbol('x')
    u = sp.sin(x)
    f = sp.exp(u)
    
    print(f"  1. Identify: u(x) = {u}, f(u) = e^u")
    
    du_dx = sp.diff(u, x)
    df_du = sp.exp(u)
    result = sp.diff(f, x)
    
    print(f"  2. du/dx = {du_dx}")
    print(f"  3. df/du = {df_du.subs(u, sp.Symbol('u'))}")
    print(f"  4. df/dx = (df/du)(du/dx) = {result}")
    print(f"  5. Simplified: {sp.simplify(result)}")

# Run Level 1 exercises
exercise_1_1()
exercise_1_2()
exercise_1_3()


#EXERCISE No. 2


def exercise_2_1():
    """Multivariable chain rule"""
    print("\nExercise 2.1: Find dz/dt for z = x²y, x = cos(t), y = sin(t)")
    print("Solution steps:")
    
    x, y, t = sp.symbols('x y t')
    z = x**2 * y
    x_t = sp.cos(t)
    y_t = sp.sin(t)
    
    print(f"  1. z = {z}")
    print(f"  2. x(t) = {x_t}, y(t) = {y_t}")
    
    # Method 1: Chain rule
    dz_dx = sp.diff(z, x)
    dz_dy = sp.diff(z, y)
    dx_dt = sp.diff(x_t, t)
    dy_dt = sp.diff(y_t, t)
    
    dz_dt_chain = (dz_dx.subs([(x, x_t), (y, y_t)]) * dx_dt + 
                   dz_dy.subs([(x, x_t), (y, y_t)]) * dy_dt)
    
    print(f"  3. ∂z/∂x = {dz_dx}")
    print(f"  4. ∂z/∂y = {dz_dy}")
    print(f"  5. dx/dt = {dx_dt}")
    print(f"  6. dy/dt = {dy_dt}")
    print(f"  7. dz/dt = (∂z/∂x)(dx/dt) + (∂z/∂y)(dy/dt)")
    print(f"  8. dz/dt = {sp.simplify(dz_dt_chain)}")
    
    # Method 2: Direct substitution
    z_t = z.subs([(x, x_t), (y, y_t)])
    dz_dt_direct = sp.diff(z_t, t)
    
    print(f"\n  Verification (direct method): {sp.simplify(dz_dt_direct)}")

def exercise_2_2():
    """Implement gradient descent manually"""
    print("\nExercise 2.2: Implement gradient descent for f(x) = x⁴ - 3x³ + 2")
    
    def f(x):
        return x**4 - 3*x**3 + 2
    
    def df_dx(x):
        return 4*x**3 - 9*x**2
    
    # Gradient descent
    x = 3.0  # Starting point
    learning_rate = 0.01
    iterations = 100
    
    history = {'x': [x], 'f(x)': [f(x)], 'gradient': [df_dx(x)]}
    
    print(f"  Starting at x = {x}")
    
    for i in range(iterations):
        grad = df_dx(x)
        x_new = x - learning_rate * grad
        
        history['x'].append(x_new)
        history['f(x)'].append(f(x_new))
        history['gradient'].append(grad)
        
        if i % 20 == 0:
            print(f"  Iteration {i:3d}: x = {x_new:.4f}, f(x) = {f(x_new):.4f}, grad = {grad:.4f}")
        
        if abs(grad) < 1e-5:
            print(f"  Converged at iteration {i}")
            break
        
        x = x_new
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Function
    x_plot = np.linspace(-1, 4, 200)
    axes[0].plot(x_plot, f(x_plot), 'b-', linewidth=2)
    axes[0].plot(history['x'], history['f(x)'], 'ro-', markersize=4, alpha=0.6)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('f(x)')
    axes[0].set_title('Function with GD Path')
    axes[0].grid(True, alpha=0.3)
    
    # Convergence
    axes[1].plot(history['f(x)'], 'g-', linewidth=2)
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('f(x)')
    axes[1].set_title('Convergence')
    axes[1].grid(True, alpha=0.3)
    
    # Gradient
    axes[2].plot(history['gradient'], 'r-', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--')
    axes[2].set_xlabel('Iteration')
    axes[2].set_ylabel('Gradient')
    axes[2].set_title('Gradient Magnitude')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('exercise_2_2.png', dpi=100, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved as 'exercise_2_2.png'")
    plt.close()

def exercise_2_3():
    """Implement simple neural network from scratch"""
    print("\nExercise 2.3: Build 2-layer network for y = x₁² + 2x₂")
    
    class SimpleNet:
        def __init__(self):
            np.random.seed(42)
            self.W1 = np.random.randn(2, 4) * 0.5
            self.b1 = np.zeros((1, 4))
            self.W2 = np.random.randn(4, 1) * 0.5
            self.b2 = np.zeros((1, 1))
            
        def forward(self, X):
            self.X = X
            self.Z1 = X @ self.W1 + self.b1
            self.A1 = np.maximum(0, self.Z1)  # ReLU
            self.Z2 = self.A1 @ self.W2 + self.b2
            return self.Z2
        
        def backward(self, y, lr=0.01):
            m = self.X.shape[0]
            
            dZ2 = (self.Z2 - y) / m
            dW2 = self.A1.T @ dZ2
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * (self.Z1 > 0)
            dW1 = self.X.T @ dZ1
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
    
    # Generate data
    n = 200
    X = np.random.randn(n, 2)
    y = (X[:, 0]**2 + 2*X[:, 1]).reshape(-1, 1)
    
    # Train
    net = SimpleNet()
    losses = []
    
    for epoch in range(500):
        y_pred = net.forward(X)
        loss = np.mean((y - y_pred)**2)
        losses.append(loss)
        net.backward(y, lr=0.01)
        
        if epoch % 100 == 0:
            print(f"  Epoch {epoch:3d}: Loss = {loss:.6f}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('exercise_2_3.png', dpi=100, bbox_inches='tight')
    print(f"\n  ✓ Visualization saved as 'exercise_2_3.png'")
    plt.close()

# Run Level 2 exercises
exercise_2_1()
exercise_2_2()
exercise_2_3()


#ADVANCED No. 3
print("LEVEL 3: ADVANCED EXERCISES")


def exercise_3_1():
    """Analyze vanishing/exploding gradients"""
    print("\nExercise 3.1: Demonstrate vanishing gradients in deep network")
    
    def analyze_gradient_flow(n_layers, activation='sigmoid'):
        """Analyze gradient flow through multiple layers"""
        np.random.seed(42)
        
        # Initialize weights
        weights = [np.random.randn(10, 10) * 0.5 for _ in range(n_layers)]
        
        # Forward pass
        x = np.random.randn(1, 10)
        activations = [x]
        
        for W in weights:
            z = activations[-1] @ W
            if activation == 'sigmoid':
                a = 1 / (1 + np.exp(-z))
            else:  # relu
                a = np.maximum(0, z)
            activations.append(a)
        
        # Backward pass
        gradient = np.ones((1, 10))
        gradient_norms = [np.linalg.norm(gradient)]
        
        for i in range(n_layers - 1, -1, -1):
            if activation == 'sigmoid':
                a = activations[i+1]
                gradient = gradient * (a * (1 - a)) @ weights[i].T
            else:
                gradient = gradient * (activations[i+1] > 0) @ weights[i].T
            
            gradient_norms.append(np.linalg.norm(gradient))
        
        return gradient_norms[::-1]
    
    # Compare different depths
    depths = [5, 10, 20, 30]
    
    plt.figure(figsize=(12, 5))
    
    for depth in depths:
        norms = analyze_gradient_flow(depth, 'sigmoid')
        plt.plot(range(len(norms)), norms, marker='o', label=f'{depth} layers')
    
    plt.xlabel('Layer (from output to input)')
    plt.ylabel('Gradient Norm')
    plt.title('Vanishing Gradients in Deep Networks (Sigmoid)')
    plt.legend()
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('exercise_3_1.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Visualization saved as 'exercise_3_1.png'")
    plt.close()
    
    print("  Observation: Gradients vanish exponentially with depth!")

def exercise_3_2():
    """Implement batch normalization"""
    print("\nExercise 3.2: Implement and visualize batch normalization")
    
    def batch_norm(X, gamma=1.0, beta=0.0, epsilon=1e-8):
        """Apply batch normalization"""
        mean = np.mean(X, axis=0, keepdims=True)
        var = np.var(X, axis=0, keepdims=True)
        X_norm = (X - mean) / np.sqrt(var + epsilon)
        X_scaled = gamma * X_norm + beta
        return X_scaled, mean, var
    
    # Generate data
    X = np.random.randn(1000, 3) * np.array([5, 0.5, 2]) + np.array([10, -5, 0])
    X_norm, mean, var = batch_norm(X)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i in range(3):
        # Before normalization
        axes[0, i].hist(X[:, i], bins=50, alpha=0.7, edgecolor='black')
        axes[0, i].set_title(f'Feature {i+1} (Before)\nμ={mean[0,i]:.2f}, σ²={var[0,i]:.2f}')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # After normalization
        axes[1, i].hist(X_norm[:, i], bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[1, i].set_title(f'Feature {i+1} (After)\nμ≈0, σ²≈1')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('exercise_3_2.png', dpi=100, bbox_inches='tight')
    print(f"  ✓ Visualization saved as 'exercise_3_2.png'")
    plt.close()

# Run Level 3 exercises
exercise_3_1()
exercise_3_2()


# STUDY ROADMAP
print("COMPLETE STUDY ROADMAP")
print("="*70)

roadmap = """
WEEK 1-2: FOUNDATIONS
├── Day 1-3: Single Variable Chain Rule
│   ✓ Theory: Composite functions, derivative rules
│   ✓ Practice: Exercises 1.1-1.3
│   ✓ Code: SymPy symbolic differentiation
│
├── Day 4-7: Numerical Computation
│   ✓ Theory: Finite differences, numerical stability
│   ✓ Practice: Implement derivative approximations
│   ✓ Code: NumPy gradient computation
│
└── Day 8-14: Visualization Skills
    ✓ Matplotlib: Basic plots, subplots, customization
    ✓ Seaborn: Statistical visualizations
    ✓ Practice: Recreate all visualizations from examples

WEEK 3-4: MULTIVARIABLE CALCULUS
├── Day 15-18: Partial Derivatives
│   ✓ Theory: Gradient vectors, Jacobian matrices
│   ✓ Practice: Exercises 2.1-2.3
│   ✓ Code: Multidimensional gradient computation
│
├── Day 19-22: Computational Graphs
│   ✓ Theory: Forward/backward mode differentiation
│   ✓ Practice: Draw graphs for complex functions
│   ✓ Code: Implement simple computational graph
│
└── Day 23-28: Optimization
    ✓ Theory: Gradient descent variants
    ✓ Practice: Exercise 2.2 with different functions
    ✓ Code: SGD, momentum, Adam implementations

WEEK 5-6: NEURAL NETWORKS
├── Day 29-33: Backpropagation Theory
│   ✓ Study: Matrix calculus for neural networks
│   ✓ Practice: Hand-calculate gradients
│   ✓ Code: Exercise 2.3 variations
│
├── Day 34-38: Deep Learning Challenges
│   ✓ Theory: Vanishing/exploding gradients
│   ✓ Practice: Exercise 3.1, analyze different activations
│   ✓ Code: Implement gradient clipping, batch norm
│
└── Day 39-42: Advanced Topics
    ✓ Automatic differentiation
    ✓ Higher-order derivatives
    ✓ Hessian-based optimization

WEEK 7-8: MASTERY & PROJECTS
├── Real Datasets
│   ✓ Apply to classification problems
│   ✓ Regression with custom architectures
│   ✓ Time series prediction
│
├── Visualization Portfolio
│   ✓ Create Manim animations
│   ✓ Interactive dashboards
│   ✓ Publication-quality figures
│
└── Final Project
    ✓ Build complete ML pipeline
    ✓ Document with theory + code
    ✓ Present findings with visualizations

DAILY PRACTICE ROUTINE (1-2 hours)
1. Theory review (15 min): Read concepts
2. Symbolic work (20 min): SymPy exercises
3. Coding (30 min): NumPy/Pandas implementations
4. Visualization (15 min): Create one plot
5. Review (10 min): Check understanding

RESOURCES TO USE:
• Coursera: Mathematics for Machine Learning
• 3Blue1Brown: Essence of Calculus
• Anthropic documentation on Claude API
• PyTorch/JAX source code study
"""

print(roadmap)
print("\n" + "="*70)
print("PRACTICE TRACKER")
print("="*70)

# Create practice log template
practice_log = pd.DataFrame({
    'Week': range(1, 9),
    'Topic': [
        'Chain Rule Basics',
        'Numerical Methods',
        'Partial Derivatives',
        'Computational Graphs',
        'Backpropagation',
        'Deep Learning',
        'Advanced Topics',
        'Final Project'
    ],
    'Hours_Planned': [10, 10, 14, 14, 14, 14, 14, 14],
    'Hours_Completed': [0] * 8,
    'Exercises_Completed': [0] * 8,
    'Status': ['Not Started'] * 8
})

print("\nPractice Log Template:")
print(practice_log.to_string(index=False))

# Save to CSV
practice_log.to_csv('chain_rule_practice_log.csv', index=False)
print(f"\nPractice log saved as 'chain_rule_practice_log.csv'")


