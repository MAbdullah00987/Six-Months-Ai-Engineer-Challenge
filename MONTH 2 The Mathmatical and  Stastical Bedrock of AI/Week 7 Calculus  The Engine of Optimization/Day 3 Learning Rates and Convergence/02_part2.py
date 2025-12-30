

#Part 2: Comparing Different Learning Rates

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Define a simple quadratic function: f(x) = x^2
def f(x):
    return x**2

def gradient_f(x):
    return 2*x

# Gradient descent function
def gradient_descent(start_x, learning_rate, iterations):
    x_values = [start_x]
    loss_values = [f(start_x)]
    
    for i in range(iterations):
        grad = gradient_f(x_values[-1])
        x_new = x_values[-1] - learning_rate * grad
        x_values.append(x_new)
        loss_values.append(f(x_new))
    
    return np.array(x_values), np.array(loss_values)

# Test different learning rates
learning_rates = {
    'Too Small (0.01)': 0.01,
    'Good (0.1)': 0.1,
    'Good (0.3)': 0.3,
    'Too Large (0.6)': 0.6,
    'Divergent (1.1)': 1.1
}

start_point = 2.0
iterations = 30

# Create comprehensive visualization
fig = plt.figure(figsize=(18, 10))

# Plot 1: Convergence curves
ax1 = fig.add_subplot(2, 3, 1)
colors = plt.cm.rainbow(np.linspace(0, 1, len(learning_rates)))

for (name, lr), color in zip(learning_rates.items(), colors):
    x_path, loss_path = gradient_descent(start_point, lr, iterations)
    ax1.plot(loss_path, label=name, linewidth=2, color=color, marker='o', markersize=3)

ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Loss (f(x) = x²)', fontsize=11)
ax1.set_title('Convergence Speed Comparison', fontsize=12, fontweight='bold')
ax1.legend(loc='best')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Plot 2: Parameter trajectory
ax2 = fig.add_subplot(2, 3, 2)
for (name, lr), color in zip(learning_rates.items(), colors):
    x_path, _ = gradient_descent(start_point, lr, iterations)
    ax2.plot(x_path, label=name, linewidth=2, color=color, marker='o', markersize=3)

ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Optimal x=0')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Parameter Value (x)', fontsize=11)
ax2.set_title('Parameter Updates Over Time', fontsize=12, fontweight='bold')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Function landscape with paths
ax3 = fig.add_subplot(2, 3, 3)
x_range = np.linspace(-2.5, 2.5, 200)
y_range = f(x_range)
ax3.plot(x_range, y_range, 'k-', linewidth=2, label='f(x) = x²')

for (name, lr), color in zip(learning_rates.items(), colors):
    if lr <= 0.6:  # Only plot non-divergent cases
        x_path, loss_path = gradient_descent(start_point, lr, 15)
        ax3.plot(x_path, loss_path, 'o-', color=color, markersize=6, 
                linewidth=2, alpha=0.7, label=name)

ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('f(x)', fontsize=11)
ax3.set_title('Paths on Loss Landscape', fontsize=12, fontweight='bold')
ax3.legend(loc='best', fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Step sizes over iterations
ax4 = fig.add_subplot(2, 3, 4)
for (name, lr), color in zip(learning_rates.items(), colors):
    x_path, _ = gradient_descent(start_point, lr, iterations)
    step_sizes = np.abs(np.diff(x_path))
    ax4.plot(step_sizes, label=name, linewidth=2, color=color, marker='o', markersize=3)

ax4.set_xlabel('Iteration', fontsize=11)
ax4.set_ylabel('Step Size |Δx|', fontsize=11)
ax4.set_title('Step Size Magnitude', fontsize=12, fontweight='bold')
ax4.set_yscale('log')
ax4.legend(loc='best')
ax4.grid(True, alpha=0.3)

# Plot 5: Gradient magnitude over time
ax5 = fig.add_subplot(2, 3, 5)
for (name, lr), color in zip(learning_rates.items(), colors):
    x_path, _ = gradient_descent(start_point, lr, iterations)
    grad_magnitudes = np.abs([gradient_f(x) for x in x_path])
    ax5.plot(grad_magnitudes, label=name, linewidth=2, color=color, marker='o', markersize=3)

ax5.set_xlabel('Iteration', fontsize=11)
ax5.set_ylabel('|Gradient|', fontsize=11)
ax5.set_title('Gradient Magnitude Decay', fontsize=12, fontweight='bold')
ax5.set_yscale('log')
ax5.legend(loc='best')
ax5.grid(True, alpha=0.3)

# Plot 6: Summary statistics table
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')

summary_data = []
for name, lr in learning_rates.items():
    x_path, loss_path = gradient_descent(start_point, lr, iterations)
    final_loss = loss_path[-1]
    converged = "✓" if final_loss < 0.01 else "✗"
    summary_data.append([name, f"{lr:.2f}", f"{final_loss:.6f}", converged])

table = ax6.table(cellText=summary_data,
                  colLabels=['Learning Rate', 'Value', 'Final Loss', 'Converged'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.3, 0.2, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax6.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n" + "=" * 70)
print("LEARNING RATE ANALYSIS")
for name, lr in learning_rates.items():
    x_path, loss_path = gradient_descent(start_point, lr, iterations)
    print(f"\n{name} (lr={lr}):")
    print(f"  Final position: x = {x_path[-1]:.6f}")
    print(f"  Final loss: {loss_path[-1]:.6f}")
    print(f"  Iterations to convergence: {np.argmax(loss_path < 0.01) if any(loss_path < 0.01) else 'N/A'}")
    if lr > 1.0:
        print(f"  Status: DIVERGENT - Learning rate exceeds stability threshold!")
    elif lr >= 0.5:
        print(f"  Status: UNSTABLE - Oscillations present")
    elif lr >= 0.2:
        print(f"  Status: GOOD - Fast convergence")
    else:
        print(f"  Status: SLOW - Many iterations needed")
