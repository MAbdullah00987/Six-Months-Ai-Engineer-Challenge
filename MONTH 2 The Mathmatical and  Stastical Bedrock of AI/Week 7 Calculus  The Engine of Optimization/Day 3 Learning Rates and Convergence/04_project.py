
#Project: Learning Rate Explorer (3-4 hours)
#Experiment with different learning rates in gradient descent

#Test learning rates: [0.001, 0.01, 0.1, 0.5, 0.9, 1.1]
#Use function: f(x) = x² + 5sin(x)
#Create subplots showing:

#Convergence paths for each learning rate
#umber of iterations to converge
#Cases of divergence (learning rate too high)
#Cases of slow convergence (learning rate too low)


"""
Learning Rate Explorer - Comprehensive Gradient Descent Analysis
This code demonstrates Python, NumPy, Matplotlib, SymPy, and Pandas concepts
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sympy import symbols, diff, lambdify, sin
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Set styling for better visualizations
sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (16, 12)

# PART 1: SYMPY - Symbolic Mathematics
print("=" * 80)
print("PART 1: SYMPY - Symbolic Math & Derivatives")
print("=" * 80)

# Define symbolic variable
x = symbols('x')

# Define the function symbolically: f(x) = x² + 5sin(x)
f_symbolic = x**2 + 5*sin(x)
print(f"\nOriginal Function: f(x) = {f_symbolic}")

# Calculate derivative symbolically (automatic differentiation)
f_prime_symbolic = diff(f_symbolic, x)
print(f"Derivative: f'(x) = {f_prime_symbolic}")

# Convert symbolic expressions to numerical functions
f = lambdify(x, f_symbolic, 'numpy')  # Function
f_prime = lambdify(x, f_prime_symbolic, 'numpy')  # Derivative

# Test the functions
test_x = 2.0
print(f"\nTest at x = {test_x}:")
print(f"  f({test_x}) = {f(test_x):.4f}")
print(f"  f'({test_x}) = {f_prime(test_x):.4f}")
# PART 2: NUMPY - Numerical Computin
print("\n" + "=" * 80)
print("PART 2: NUMPY - Efficient Numerical Operations")
print("=" * 80)

# Create arrays for visualization (vectorized operations)
x_range = np.linspace(-5, 5, 1000)  # 1000 points from -5 to 5
y_range = f(x_range)  # Vectorized function evaluation

print(f"\nArray operations:")
print(f"  x_range shape: {x_range.shape}")
print(f"  y_range shape: {y_range.shape}")
print(f"  Min value: {np.min(y_range):.4f} at x ≈ {x_range[np.argmin(y_range)]:.4f}")
print(f"  Max value: {np.max(y_range):.4f} at x ≈ {x_range[np.argmax(y_range)]:.4f}")


# PART 3: GRADIENT DESCENT IMPLEMENTATION

print("\n" + "=" * 80)
print("PART 3: GRADIENT DESCENT ALGORITHM")
print("=" * 80)

def gradient_descent(f, f_prime, x_start, learning_rate, max_iterations=1000, 
                     tolerance=1e-6):
    """
    Perform gradient descent optimization
    
    Parameters:
    -----------
    f : function
        The objective function to minimize
    f_prime : function
        Derivative of the objective function
    x_start : float
        Starting point
    learning_rate : float
        Step size for each iteration
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold
    
    Returns:
    --------
    dict : Dictionary containing history and statistics
    """
    # Initialize with NumPy arrays for efficiency
    x_history = np.zeros(max_iterations + 1)
    y_history = np.zeros(max_iterations + 1)
    gradient_history = np.zeros(max_iterations + 1)
    
    x_current = x_start
    x_history[0] = x_current
    y_history[0] = f(x_current)
    gradient_history[0] = f_prime(x_current)
    
    converged = False
    diverged = False
    iterations = 0
    
    for i in range(max_iterations):
        # Calculate gradient at current point
        gradient = f_prime(x_current)
        gradient_history[i] = gradient
        
        # Gradient descent update rule: x_new = x_old - learning_rate * gradient
        x_new = x_current - learning_rate * gradient
        
        # Store history
        x_history[i + 1] = x_new
        y_history[i + 1] = f(x_new)
        
        # Check for divergence (values becoming too large)
        if np.abs(x_new) > 1e10 or np.isnan(x_new):
            diverged = True
            iterations = i + 1
            break
        
        # Check for convergence (gradient near zero)
        if np.abs(x_new - x_current) < tolerance:
            converged = True
            iterations = i + 1
            break
        
        x_current = x_new
        iterations = i + 1
    
    # Trim arrays to actual iterations
    return {
        'x_history': x_history[:iterations + 1],
        'y_history': y_history[:iterations + 1],
        'gradient_history': gradient_history[:iterations],
        'final_x': x_current,
        'final_y': f(x_current),
        'iterations': iterations,
        'converged': converged,
        'diverged': diverged,
        'learning_rate': learning_rate
    }


# PART 4: EXPERIMENT WITH DIFFERENT LEARNING RATES

print("\n" + "=" * 80)
print("PART 4: RUNNING EXPERIMENTS")
print("=" * 80)

learning_rates = [0.001, 0.01, 0.1, 0.5, 0.9, 1.1]
x_start = 4.0  # Starting point for all experiments
results = []

print(f"\nStarting point: x = {x_start}")
print(f"Function value at start: f({x_start}) = {f(x_start):.4f}\n")

for lr in learning_rates:
    print(f"Testing learning rate = {lr}...")
    result = gradient_descent(f, f_prime, x_start, lr, max_iterations=1000)
    results.append(result)
    
    status = "CONVERGED" if result['converged'] else ("DIVERGED" if result['diverged'] else "MAX ITER")
    print(f"  Status: {status}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Final x: {result['final_x']:.6f}")
    print(f"  Final f(x): {result['final_y']:.6f}\n")

# PART 5: PANDAS - Data Analysis
print("=" * 80)
print("PART 5: PANDAS - Data Analysis & Tabulation")
print("=" * 80)

# Create a DataFrame for easy analysis
df = pd.DataFrame({
    'Learning Rate': [r['learning_rate'] for r in results],
    'Iterations': [r['iterations'] for r in results],
    'Final X': [r['final_x'] for r in results],
    'Final f(X)': [r['final_y'] for r in results],
    'Converged': [r['converged'] for r in results],
    'Diverged': [r['diverged'] for r in results]
})

# Add status column
df['Status'] = df.apply(lambda row: 'Converged' if row['Converged'] 
                        else ('Diverged' if row['Diverged'] else 'Max Iter'), axis=1)

print("\n" + df.to_string(index=False))

# Statistical summary
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY")
print("=" * 80)
print(df.describe())


# PART 6: MATPLOTLIB - Comprehensive Visualization

print("\n" + "=" * 80)
print("PART 6: MATPLOTLIB - Creating Visualizations")
print("=" * 80)

fig = plt.figure(figsize=(20, 14))

# Color palette for different learning rates
colors = sns.color_palette("husl", len(learning_rates))

# --------- Subplot 1: Function and All Convergence Paths ---------
ax1 = plt.subplot(3, 3, 1)
ax1.plot(x_range, y_range, 'k-', linewidth=2, label='f(x) = x² + 5sin(x)', alpha=0.3)

for result, color in zip(results, colors):
    if not result['diverged']:
        ax1.plot(result['x_history'], result['y_history'], 'o-', 
                color=color, alpha=0.7, markersize=3,
                label=f"lr={result['learning_rate']}")

ax1.scatter([x_start], [f(x_start)], color='red', s=200, 
           marker='*', zorder=5, label='Start', edgecolors='black', linewidths=2)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('All Convergence Paths', fontsize=14, fontweight='bold')
ax1.legend(fontsize=8, loc='upper right')
ax1.grid(True, alpha=0.3)

# --------- Individual Convergence Paths (Subplots 2-7) ---------
for idx, (result, color, lr) in enumerate(zip(results, colors, learning_rates)):
    ax = plt.subplot(3, 3, idx + 2)
    
    # Plot function
    ax.plot(x_range, y_range, 'k-', linewidth=1.5, alpha=0.3)
    
    if result['diverged']:
        # For diverged cases, show the explosion
        ax.plot(result['x_history'], result['y_history'], 'o-', 
               color=color, alpha=0.8, markersize=4, linewidth=2)
        ax.set_title(f'lr={lr} - DIVERGED ({result["iterations"]} iter)', 
                    fontsize=11, fontweight='bold', color='red')
        ax.set_ylim([min(f(x_start), np.min(result['y_history'][:10])), 
                     max(f(x_start) * 2, np.max(result['y_history'][:10]))])
    else:
        # For converged/slow cases
        ax.plot(result['x_history'], result['y_history'], 'o-', 
               color=color, alpha=0.8, markersize=4, linewidth=2)
        
        # Mark start and end points
        ax.scatter([x_start], [f(x_start)], color='red', s=150, 
                  marker='*', zorder=5, edgecolors='black', linewidths=1.5)
        ax.scatter([result['final_x']], [result['final_y']], 
                  color='green', s=150, marker='X', zorder=5, 
                  edgecolors='black', linewidths=1.5)
        
        status = "CONVERGED" if result['converged'] else "SLOW"
        color_status = 'green' if result['converged'] else 'orange'
        ax.set_title(f'lr={lr} - {status} ({result["iterations"]} iter)', 
                    fontsize=11, fontweight='bold', color=color_status)
    
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('f(x)', fontsize=10)
    ax.grid(True, alpha=0.3)

# --------- Subplot 8: Iterations Comparison ---------
ax8 = plt.subplot(3, 3, 8)
bars = ax8.bar(range(len(learning_rates)), 
               [r['iterations'] for r in results], 
               color=colors, edgecolor='black', linewidth=1.5)

# Color code bars by status
for bar, result in zip(bars, results):
    if result['diverged']:
        bar.set_edgecolor('red')
        bar.set_linewidth(3)
    elif result['converged']:
        bar.set_edgecolor('green')
        bar.set_linewidth(3)

ax8.set_xlabel('Learning Rate', fontsize=12)
ax8.set_ylabel('Iterations to Converge', fontsize=12)
ax8.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
ax8.set_xticks(range(len(learning_rates)))
ax8.set_xticklabels([str(lr) for lr in learning_rates], rotation=45)
ax8.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (bar, result) in enumerate(zip(bars, results)):
    height = bar.get_height()
    status = "DIV" if result['diverged'] else ("CONV" if result['converged'] else "SLOW")
    ax8.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n{status}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# --------- Subplot 9: Final Values Comparison ---------
ax9 = plt.subplot(3, 3, 9)
valid_results = [r for r in results if not r['diverged']]
valid_colors = [c for r, c in zip(results, colors) if not r['diverged']]
valid_lrs = [r['learning_rate'] for r in valid_results]

if valid_results:
    bars = ax9.bar(range(len(valid_results)), 
                   [r['final_y'] for r in valid_results],
                   color=valid_colors, edgecolor='black', linewidth=1.5)
    
    ax9.set_xlabel('Learning Rate', fontsize=12)
    ax9.set_ylabel('Final f(x) Value', fontsize=12)
    ax9.set_title('Final Function Values (Converged Only)', fontsize=14, fontweight='bold')
    ax9.set_xticks(range(len(valid_results)))
    ax9.set_xticklabels([str(lr) for lr in valid_lrs], rotation=45)
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, result in zip(bars, valid_results):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('learning_rate_explorer.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'learning_rate_explorer.png'")
plt.show()


# PART 7: SEABORN - Advanced Statistical Visualization

print("\n" + "=" * 80)
print("PART 7: SEABORN - Statistical Visualizations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Iterations vs Learning Rate (Seaborn style)
sns.barplot(data=df, x='Learning Rate', y='Iterations', 
           hue='Status', palette=['green', 'red', 'orange'], ax=axes[0, 0])
axes[0, 0].set_title('Iterations by Learning Rate and Status', 
                     fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Number of Iterations', fontsize=12)
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Learning Rate Categories
df['Category'] = pd.cut(df['Learning Rate'], 
                        bins=[0, 0.05, 0.5, 2.0],
                        labels=['Too Small', 'Good', 'Too Large'])
category_counts = df['Status'].value_counts()
sns.countplot(data=df, x='Status', palette=['green', 'orange', 'red'], ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Outcomes', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Count', fontsize=12)

# Plot 3: Heatmap of convergence characteristics
convergence_matrix = df[['Learning Rate', 'Iterations', 'Final f(X)']].copy()
convergence_matrix = convergence_matrix[df['Status'] != 'Diverged']
if not convergence_matrix.empty:
    pivot = convergence_matrix.set_index('Learning Rate')
    sns.heatmap(pivot.T, annot=True, fmt='.2f', cmap='RdYlGn_r', 
               cbar_kws={'label': 'Value'}, ax=axes[1, 0])
    axes[1, 0].set_title('Convergence Characteristics Heatmap', 
                        fontsize=14, fontweight='bold')

# Plot 4: Trajectory length analysis
for result, color, lr in zip(results, colors, learning_rates):
    if not result['diverged'] and len(result['x_history']) > 1:
        # Calculate distances between consecutive points
        distances = np.sqrt(np.diff(result['x_history'])**2 + 
                          np.diff(result['y_history'])**2)
        axes[1, 1].plot(distances, label=f'lr={lr}', color=color, alpha=0.7)

axes[1, 1].set_xlabel('Step Number', fontsize=12)
axes[1, 1].set_ylabel('Step Size', fontsize=12)
axes[1, 1].set_title('Step Size Over Iterations', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('learning_rate_analysis_seaborn.png', dpi=300, bbox_inches='tight')
print("\nSeaborn visualization saved as 'learning_rate_analysis_seaborn.png'")
plt.show
# PART 8: ANIMATED VISUALIZATION.
print("\n" + "=" * 80)
print("PART 8: CREATING ANIMATION (Example Code)")
print("=" * 80)
print("""
To create an animation of the gradient descent process, you can use:

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots(figsize=(12, 8))

def animate(frame):
    ax.clear()
    ax.plot(x_range, y_range, 'k-', linewidth=2, alpha=0.3)
    
    for result, color in zip(results, colors):
        if not result['diverged'] and frame < len(result['x_history']):
            ax.plot(result['x_history'][:frame+1], 
                   result['y_history'][:frame+1], 
                   'o-', color=color, alpha=0.7)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-10, 30)
    ax.set_title(f'Gradient Descent Progress (Step {frame})', fontsize=14)
    ax.grid(True, alpha=0.3)

max_frames = max(len(r['x_history']) for r in results if not r['diverged'])
anim = FuncAnimation(fig, animate, frames=max_frames, interval=50, repeat=True)

# Save as GIF or MP4
# anim.save('gradient_descent.gif', writer='pillow', fps=20)
""")


# Main Points
'''
CONCEPTS COVERED:

1. SYMPY:
   - Symbolic variable definition: symbols('x')
   - Symbolic function creation: x**2 + 5*sin(x)
   - Automatic differentiation: diff(function, variable)
   - Converting symbolic to numerical: lambdify()

2. NUMPY:
   - Array creation: np.linspace(), np.zeros()
   - Vectorized operations: f(x_range) applies to all elements
   - Array operations: np.min(), np.max(), np.argmin(), np.argmax()
   - Boolean indexing and array slicing

3. MATPLOTLIB:
   - Figure and subplot management: plt.subplot()
   - Multiple plot types: plot(), scatter(), bar()
   - Customization: colors, markers, labels, legends
   - Layout management: tight_layout()
   - Saving figures: savefig()

4. PANDAS:
   - DataFrame creation from dictionaries
   - Data analysis: describe(), apply()
   - Column operations and filtering
   - Data presentation: to_string()

5. SEABORN:
   - Statistical plots: barplot(), countplot()
   - Heatmaps for matrix visualization
   - Color palettes: color_palette()
   - Enhanced styling: set_style()

6. GRADIENT DESCENT:
   - Update rule: x_new = x_old - learning_rate * gradient
   - Convergence criteria: |x_new - x_old| < tolerance
   - Divergence detection: values growing too large
   - History tracking for analysis

EXPERIMENT RESULTS:
""")

for _, row in df.iterrows():
    print(f"\n  Learning Rate {row['Learning Rate']:>5.3f}:")
    print(f"    Status: {row['Status']}")
    print(f"    Iterations: {row['Iterations']}")
    if not row['Diverged']:
        print(f"    Final X: {row['Final X']:.6f}")
        print(f"    Final f(X): {row['Final f(X)']:.6f}")

print("\n" + "=" * 80)
print(" INSIGHTS:")
print("=" * 80)
print("""
• Too Small (0.001, 0.01): Converge slowly, need many iterations
• Optimal (0.1, 0.5): Balance speed and stability
• Too Large (0.9, 1.1): May overshoot or diverge completely

The choice of learning rate is crucial in optimization!
'''