
#Day 2: The Multi-Dimensional World (Partial Derivatives)Objective: Learn to handle functions with multiple inputs (like a neural network with 1000 weights).Concept: Partial derivatives ($\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}$). Holding all variables constant except one.Resource:Read: Chapter 5.2 of Mathematics for Machine Learning.Watch: Khan Academy "Partial Derivatives" (first 2 videos).

#Probability Distributions Deep Dive
#Focus: Understanding and working with various distributions
#Bishop Chapter 2: Focus on probability distributions section
#eep dive into Normal distribution properties
#Study Binomial and other discrete distributions

#Task: Calculate partial derivatives by hand, then verify with SymPy.Code Snippet:Pythonx, y = sp.symbols('x y')
#f = x**2 + y**2  # A simple bowl shape
#df_dx = sp.diff(f, x) # Slope in x direction
#df_dy = sp.diff(f, y) # Slope in y direction


#Probability Distribution Plotter - Build an interactive tool to visualize different distributions with adjustable parameters
#Practice

#Experiment with distribution parameters
#Calculate probabilities for real-world scenarios

#Probability Distributions Deep Dive: From Theory to Python

"""
DAY 2 COMPLETE: PARTIAL DERIVATIVES + PROBABILITY DISTRIBUTIONS
Multi-Dimensional Calculus meets Statistical Understanding
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sympy as sp
from sympy import symbols, diff, Eq, sin, cos, exp, log, sqrt
from sympy.stats import Bernoulli, Binomial, Poisson, Normal, Uniform, Exponential, E, variance, P, density
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

print("="*80)
print("DAY 2: MULTI-DIMENSIONAL WORLD + PROBABILITY DISTRIBUTIONS")
print("="*80)

# ============================================================================
# PART 1: PARTIAL DERIVATIVES (Multi-Variable Calculus)
# ============================================================================

print("\n" + "="*80)
print("PART 1: PARTIAL DERIVATIVES - THE FOUNDATION OF OPTIMIZATION")
print("="*80)
print("\nWhy this matters: Neural networks have THOUSANDS of parameters!")
print("Partial derivatives tell us how changing ONE parameter affects the output.\n")

# ----------------------------------------------------------------------------
# 1.1 Basic Partial Derivatives
# ----------------------------------------------------------------------------
print("--- 1.1 BASIC EXAMPLES ---\n")

# Define symbolic variables
x, y, z = symbols('x y z')

# Example 1: Simple bowl shape (common in optimization)
f1 = x**2 + y**2
print(f"Function: f(x,y) = {f1}")
print(f"  This is a paraboloid - like a bowl!")

df1_dx = diff(f1, x)
df1_dy = diff(f1, y)
print(f"\n  ∂f/∂x = {df1_dx}  (slope in x-direction)")
print(f"  ∂f/∂y = {df1_dy}  (slope in y-direction)")

# Evaluate at a point
point_x, point_y = 3, 4
print(f"\n  At point ({point_x}, {point_y}):")
print(f"    ∂f/∂x = {df1_dx.subs([(x, point_x), (y, point_y)])}")
print(f"    ∂f/∂y = {df1_dy.subs([(x, point_x), (y, point_y)])}")

# Example 2: More complex function
f2 = x**2*y + y**3
print(f"\n\nFunction: f(x,y) = {f2}")
df2_dx = diff(f2, x)
df2_dy = diff(f2, y)
print(f"  ∂f/∂x = {df2_dx}")
print(f"  ∂f/∂y = {df2_dy}")

# Example 3: Function with 3 variables (like a small neural network!)
f3 = x**2 + 2*y**2 + 3*z**2 + x*y*z
print(f"\n\nFunction: f(x,y,z) = {f3}")
print(f"  ∂f/∂x = {diff(f3, x)}")
print(f"  ∂f/∂y = {diff(f3, y)}")
print(f"  ∂f/∂z = {diff(f3, z)}")

# Example 4: Neural network activation functions
print("\n\n--- 1.2 NEURAL NETWORK EXAMPLES ---")

# Sigmoid activation
sigmoid = 1 / (1 + exp(-x))
print(f"\nSigmoid: σ(x) = {sigmoid}")
sigmoid_derivative = diff(sigmoid, x)
print(f"  dσ/dx = {sigmoid_derivative}")
print(f"  Simplified: {sp.simplify(sigmoid_derivative)}")

# ReLU (Rectified Linear Unit) - using piecewise
print("\nReLU: ReLU(x) = max(0, x)")
print("  dReLU/dx = 1 if x > 0, else 0")

# Loss function (Mean Squared Error with 2 parameters)
w1, w2, y_true = symbols('w1 w2 y_true')
y_pred = w1*x + w2*y
loss = (y_pred - y_true)**2
print(f"\n\nLoss Function: L = (ŷ - y_true)² = {loss}")
print(f"  where ŷ = {y_pred}")
print(f"\n  ∂L/∂w1 = {diff(loss, w1)}")
print(f"  ∂L/∂w2 = {diff(loss, w2)}")

# ----------------------------------------------------------------------------
# 1.3 Visualizing Partial Derivatives
# ----------------------------------------------------------------------------
print("\n\n--- 1.3 VISUALIZATION ---")

# Create meshgrid for 3D plotting
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Function: f(x,y) = x² + y²
Z1 = X**2 + Y**2

# Function: f(x,y) = sin(x) * cos(y)
Z2 = np.sin(X) * np.cos(Y)

# Function: f(x,y) = x² - y² (saddle point)
Z3 = X**2 - Y**2

fig = plt.figure(figsize=(18, 12))

# Plot 1: Paraboloid
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z1, cmap=cm.viridis, alpha=0.8)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('f(x,y) = x² + y² (Bowl Shape)')
fig.colorbar(surf1, ax=ax1, shrink=0.5)

# Plot 2: Contour of Paraboloid
ax2 = fig.add_subplot(2, 3, 2)
contour1 = ax2.contour(X, Y, Z1, levels=20, cmap='viridis')
ax2.clabel(contour1, inline=True, fontsize=8)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Plot: x² + y²')
ax2.grid(True, alpha=0.3)

# Plot 3: Gradient vectors
ax3 = fig.add_subplot(2, 3, 3)
# Calculate gradient at sample points
x_sample = np.linspace(-3, 3, 10)
y_sample = np.linspace(-3, 3, 10)
X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
U = 2*X_sample  # ∂f/∂x = 2x
V = 2*Y_sample  # ∂f/∂y = 2y
ax3.quiver(X_sample, Y_sample, U, V, alpha=0.7)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Gradient Vectors (∇f)')
ax3.grid(True, alpha=0.3)

# Plot 4: Wave function
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
surf2 = ax4.plot_surface(X, Y, Z2, cmap=cm.coolwarm, alpha=0.8)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('f(x,y)')
ax4.set_title('f(x,y) = sin(x)·cos(y)')
fig.colorbar(surf2, ax=ax4, shrink=0.5)

# Plot 5: Saddle point
ax5 = fig.add_subplot(2, 3, 5, projection='3d')
surf3 = ax5.plot_surface(X, Y, Z3, cmap=cm.RdYlBu, alpha=0.8)
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_zlabel('f(x,y)')
ax5.set_title('f(x,y) = x² - y² (Saddle Point)')
fig.colorbar(surf3, ax=ax5, shrink=0.5)

# Plot 6: Gradient descent path on paraboloid
ax6 = fig.add_subplot(2, 3, 6)
contour2 = ax6.contour(X, Y, Z1, levels=20, cmap='viridis', alpha=0.5)

# Simulate gradient descent
x_start, y_start = 4, 4
learning_rate = 0.1
path_x, path_y = [x_start], [y_start]

for i in range(50):
    grad_x = 2 * path_x[-1]  # ∂f/∂x
    grad_y = 2 * path_y[-1]  # ∂f/∂y
    
    new_x = path_x[-1] - learning_rate * grad_x
    new_y = path_y[-1] - learning_rate * grad_y
    
    path_x.append(new_x)
    path_y.append(new_y)

ax6.plot(path_x, path_y, 'r-o', linewidth=2, markersize=4, label='Gradient Descent Path')
ax6.plot(path_x[0], path_y[0], 'go', markersize=10, label='Start')
ax6.plot(path_x[-1], path_y[-1], 'r*', markersize=15, label='End (minimum)')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_title('Gradient Descent in Action')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('Partial Derivatives & Gradient Descent Visualization', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('partial_derivatives_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Saved: partial_derivatives_visualization.png")
plt.close()

# ----------------------------------------------------------------------------
# 1.4 Practical Exercises
# ----------------------------------------------------------------------------
print("\n\n--- 1.4 PRACTICE EXERCISES ---\n")

exercises = [
    ("f(x,y) = 3x² + 4xy + 2y²", 3*x**2 + 4*x*y + 2*y**2),
    ("f(x,y) = e^(x+y)", exp(x + y)),
    ("f(x,y) = ln(x² + y²)", log(x**2 + y**2)),
    ("f(x,y) = xy/(x²+y²)", (x*y)/(x**2 + y**2)),
]

for i, (desc, func) in enumerate(exercises, 1):
    print(f"Exercise {i}: {desc}")
    print(f"  ∂f/∂x = {diff(func, x)}")
    print(f"  ∂f/∂y = {diff(func, y)}")
    print()

# ============================================================================
# PART 2: PROBABILITY DISTRIBUTIONS
# ============================================================================

print("\n" + "="*80)
print("PART 2: PROBABILITY DISTRIBUTIONS - THE VOCABULARY OF REALITY")
print("="*80)

# ----------------------------------------------------------------------------
# 2.1 BERNOULLI DISTRIBUTION
# ----------------------------------------------------------------------------
print("\n--- 2.1 BERNOULLI DISTRIBUTION ---")
print("Single trial: Success (1) or Failure (0)\n")

p_success = 0.7
n_trials = 1000
bernoulli_samples = np.random.binomial(n=1, p=p_success, size=n_trials)

print(f"Parameter p: {p_success}")
print(f"Theoretical Mean: {p_success}")
print(f"Empirical Mean: {np.mean(bernoulli_samples):.4f}")
print(f"Theoretical Variance: {p_success * (1-p_success):.4f}")
print(f"Empirical Variance: {np.var(bernoulli_samples):.4f}")

# Theoretical Analysis (simplified approach)
print("\nTheoretical Analysis:")
print(f"P(X = 1) = {p_success}")
print(f"P(X = 0) = {1 - p_success}")
print(f"E(X) = {p_success}")
print(f"Var(X) = {p_success * (1 - p_success):.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

unique, counts = np.unique(bernoulli_samples, return_counts=True)
axes[0].bar(unique, counts/n_trials, color=['salmon', 'lightgreen'], alpha=0.7, edgecolor='black', linewidth=2)
axes[0].axhline(y=p_success, color='green', linestyle='--', linewidth=2, label=f'Expected P(X=1) = {p_success}')
axes[0].axhline(y=1-p_success, color='red', linestyle='--', linewidth=2, label=f'Expected P(X=0) = {1-p_success}')
axes[0].set_xlabel('Outcome', fontsize=12)
axes[0].set_ylabel('Probability', fontsize=12)
axes[0].set_title('Bernoulli Distribution (p=0.7)', fontsize=14, fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

running_avg = np.cumsum(bernoulli_samples) / np.arange(1, n_trials + 1)
axes[1].plot(running_avg, alpha=0.7, linewidth=1.5)
axes[1].axhline(y=p_success, color='red', linestyle='--', linewidth=2, label=f'True mean (p={p_success})')
axes[1].set_xlabel('Number of Trials', fontsize=12)
axes[1].set_ylabel('Running Average', fontsize=12)
axes[1].set_title('Law of Large Numbers: Convergence to p', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bernoulli_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: bernoulli_distribution.png")
plt.close()

# ----------------------------------------------------------------------------
# 2.2 BINOMIAL DISTRIBUTION
# ----------------------------------------------------------------------------
print("\n\n--- 2.2 BINOMIAL DISTRIBUTION ---")
print("Sum of n independent Bernoulli trials\n")

n_flips = 20
p_heads = 0.6
n_experiments = 10000
binomial_samples = np.random.binomial(n=n_flips, p=p_heads, size=n_experiments)

x_values = np.arange(0, n_flips + 1)
theoretical_pmf = stats.binom.pmf(x_values, n=n_flips, p=p_heads)

print(f"Parameters: n={n_flips}, p={p_heads}")
print(f"Theoretical Mean: n*p = {n_flips * p_heads}")
print(f"Empirical Mean: {np.mean(binomial_samples):.4f}")
print(f"Theoretical Variance: n*p*(1-p) = {n_flips * p_heads * (1-p_heads):.4f}")
print(f"Empirical Variance: {np.var(binomial_samples):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(binomial_samples, bins=30, density=True, alpha=0.6, color='skyblue', edgecolor='black')
axes[0, 0].plot(x_values, theoretical_pmf, 'ro-', linewidth=2, markersize=6, label='Theoretical PMF')
axes[0, 0].set_xlabel('Number of Successes')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].set_title(f'Binomial Distribution: n={n_flips}, p={p_heads}')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

sns.histplot(binomial_samples, bins=30, kde=True, ax=axes[0, 1], color='purple', stat='density')
axes[0, 1].set_xlabel('Number of Successes')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Seaborn: Binomial with KDE')

theoretical_cdf = stats.binom.cdf(x_values, n=n_flips, p=p_heads)
axes[1, 0].step(x_values, theoretical_cdf, 'b-', linewidth=2, where='post', label='Theoretical CDF')
axes[1, 0].set_xlabel('Number of Successes')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Cumulative Distribution Function (CDF)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

for p_val in [0.2, 0.5, 0.8]:
    pmf = stats.binom.pmf(x_values, n=n_flips, p=p_val)
    axes[1, 1].plot(x_values, pmf, 'o-', label=f'p={p_val}', linewidth=2, markersize=5)
axes[1, 1].set_xlabel('Number of Successes')
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_title(f'Effect of p on Binomial Distribution (n={n_flips})')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binomial_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: binomial_distribution.png")
plt.close()

# ----------------------------------------------------------------------------
# 2.3 NORMAL DISTRIBUTION
# ----------------------------------------------------------------------------
print("\n\n--- 2.3 NORMAL (GAUSSIAN) DISTRIBUTION ---")
print("The most important distribution in statistics\n")

mu, sigma = 100, 15
normal_samples = np.random.normal(loc=mu, scale=sigma, size=10000)

x_normal = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
theoretical_normal = stats.norm.pdf(x_normal, loc=mu, scale=sigma)

print(f"Parameters: μ={mu}, σ={sigma}")
print(f"Empirical Mean: {np.mean(normal_samples):.4f}")
print(f"Empirical Std Dev: {np.std(normal_samples, ddof=1):.4f}")

# Empirical Rule
within_1_sigma = np.sum((normal_samples >= mu - sigma) & (normal_samples <= mu + sigma)) / len(normal_samples)
within_2_sigma = np.sum((normal_samples >= mu - 2*sigma) & (normal_samples <= mu + 2*sigma)) / len(normal_samples)
within_3_sigma = np.sum((normal_samples >= mu - 3*sigma) & (normal_samples <= mu + 3*sigma)) / len(normal_samples)

print(f"\nEmpirical Rule (68-95-99.7):")
print(f"Within 1σ: {within_1_sigma:.4f} (Expected: 0.6827)")
print(f"Within 2σ: {within_2_sigma:.4f} (Expected: 0.9545)")
print(f"Within 3σ: {within_3_sigma:.4f} (Expected: 0.9973)")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(normal_samples, bins=50, density=True, alpha=0.6, color='lightblue', edgecolor='black')
axes[0, 0].plot(x_normal, theoretical_normal, 'r-', linewidth=2, label='Theoretical PDF')
axes[0, 0].axvline(mu, color='black', linestyle='--', linewidth=2, label='μ')
axes[0, 0].axvline(mu + sigma, color='orange', linestyle='--', linewidth=1.5, label='μ±σ')
axes[0, 0].axvline(mu - sigma, color='orange', linestyle='--', linewidth=1.5)
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Normal Distribution (μ={mu}, σ={sigma})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

sns.histplot(normal_samples, kde=True, ax=axes[0, 1], color='purple', stat='density')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Seaborn: Normal with KDE')

stats.probplot(normal_samples, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Testing Normality')
axes[1, 0].grid(True, alpha=0.3)

z_scores = (normal_samples - mu) / sigma
axes[1, 1].hist(z_scores, bins=50, density=True, alpha=0.6, color='lightgreen', edgecolor='black')
z_x = np.linspace(-4, 4, 1000)
z_pdf = stats.norm.pdf(z_x, loc=0, scale=1)
axes[1, 1].plot(z_x, z_pdf, 'r-', linewidth=2, label='Standard Normal')
axes[1, 1].set_xlabel('Z-score')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Standardized Normal Distribution (μ=0, σ=1)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: normal_distribution.png")
plt.close()

# ----------------------------------------------------------------------------
# 2.4 POISSON DISTRIBUTION
# ----------------------------------------------------------------------------
print("\n\n--- 2.4 POISSON DISTRIBUTION ---")
print("Counts of events in fixed time/space intervals\n")

lambda_param = 5
n_samples = 10000
poisson_samples = np.random.poisson(lam=lambda_param, size=n_samples)

x_poisson = np.arange(0, 20)
theoretical_poisson = stats.poisson.pmf(x_poisson, mu=lambda_param)

print(f"Parameter λ: {lambda_param}")
print(f"Theoretical Mean: λ = {lambda_param}")
print(f"Empirical Mean: {np.mean(poisson_samples):.4f}")
print(f"Theoretical Variance: λ = {lambda_param}")
print(f"Empirical Variance: {np.var(poisson_samples):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(poisson_samples, bins=range(0, 20), density=True, alpha=0.6, color='lightcoral', edgecolor='black')
axes[0, 0].plot(x_poisson, theoretical_poisson, 'bo-', linewidth=2, markersize=6, label='Theoretical PMF')
axes[0, 0].set_xlabel('Number of Events')
axes[0, 0].set_ylabel('Probability')
axes[0, 0].set_title(f'Poisson Distribution (λ={lambda_param})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

for lam in [2, 5, 10]:
    pmf = stats.poisson.pmf(x_poisson, mu=lam)
    axes[0, 1].plot(x_poisson, pmf, 'o-', label=f'λ={lam}', linewidth=2, markersize=5)
axes[0, 1].set_xlabel('Number of Events')
axes[0, 1].set_ylabel('Probability')
axes[0, 1].set_title('Effect of λ on Poisson Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

running_mean = np.cumsum(poisson_samples) / np.arange(1, n_samples + 1)
axes[1, 0].plot(running_mean, alpha=0.7, linewidth=1.5)
axes[1, 0].axhline(y=lambda_param, color='red', linestyle='--', linewidth=2, label=f'True mean (λ={lambda_param})')
axes[1, 0].set_xlabel('Number of Samples')
axes[1, 0].set_ylabel('Running Mean')
axes[1, 0].set_title('Convergence to λ')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

theoretical_cdf = stats.poisson.cdf(x_poisson, mu=lambda_param)
axes[1, 1].step(x_poisson, theoretical_cdf, 'g-', linewidth=2, where='post', label='Theoretical CDF')
axes[1, 1].set_xlabel('Number of Events')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('Cumulative Distribution Function')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('poisson_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: poisson_distribution.png")
plt.close()

# ----------------------------------------------------------------------------
# 2.5 EXPONENTIAL DISTRIBUTION
# ----------------------------------------------------------------------------
print("\n\n--- 2.5 EXPONENTIAL DISTRIBUTION ---")
print("Time between events in a Poisson process\n")

rate = 0.5  # λ (rate parameter)
n_samples = 10000
exponential_samples = np.random.exponential(scale=1/rate, size=n_samples)

x_exp = np.linspace(0, 15, 1000)
theoretical_exp = stats.expon.pdf(x_exp, scale=1/rate)

print(f"Parameter λ (rate): {rate}")
print(f"Theoretical Mean: 1/λ = {1/rate}")
print(f"Empirical Mean: {np.mean(exponential_samples):.4f}")
print(f"Theoretical Variance: 1/λ² = {1/(rate**2)}")
print(f"Empirical Variance: {np.var(exponential_samples):.4f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(exponential_samples, bins=50, density=True, alpha=0.6, color='lightsalmon', edgecolor='black')
axes[0, 0].plot(x_exp, theoretical_exp, 'r-', linewidth=2, label='Theoretical PDF')
axes[0, 0].set_xlabel('Time Between Events')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title(f'Exponential Distribution (λ={rate})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

for r in [0.5, 1.0, 2.0]:
    pdf = stats.expon.pdf(x_exp, scale=1/r)
    axes[0, 1].plot(x_exp, pdf, linewidth=2, label=f'λ={r}')
axes[0, 1].set_xlabel('Time Between Events')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Effect of λ on Exponential Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

theoretical_cdf = stats.expon.cdf(x_exp, scale=1/rate)
axes[1, 0].plot(x_exp, theoretical_cdf, 'b-', linewidth=2, label='Theoretical CDF')
axes[1, 0].set_xlabel('Time Between Events')
axes[1, 0].set_ylabel('Cumulative Probability')
axes[1, 0].set_title('Cumulative Distribution Function')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

stats.probplot(exponential_samples, dist="expon", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Testing Exponential Fit')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('exponential_distribution.png', dpi=150, bbox_inches='tight')
print("✓ Saved: exponential_distribution.png")
plt.close()

