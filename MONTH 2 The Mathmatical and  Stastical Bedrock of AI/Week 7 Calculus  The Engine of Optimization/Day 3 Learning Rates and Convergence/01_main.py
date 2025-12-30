#Day 3: Learning Rates and Convergence
#Focus: Understanding how hyperparameters affect optimization
#Learning Activities (2-3 hours)

#Read: Study convergence criteria and learning rate selection
#Review: Revisit gradient descent theory with focus on step size
#Research: Read about adaptive learning rates (brief overview)

#Project: Learning Rate Explorer (3-4 hours)
#Experiment with different learning rates in gradient descent

#Test learning rates: [0.001, 0.01, 0.1, 0.5, 0.9, 1.1]
#Use function: f(x) = x² + 5sin(x)
#Create subplots showing:

#Convergence paths for each learning rate
#Number of iterations to converge
#Cases of divergence (learning rate too high)
#Cases of slow convergence (learning rate too low)


#Mastering Learning Rates & Convergence Through Python

#Part 1: Understanding Gradients Visually

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a grid of points
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Define a function: f(x,y) = x^2 + y^2 (a paraboloid)
Z = X**2 + Y**2

# Calculate gradient: ∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]
gradient_x = 2 * X
gradient_y = 2 * Y

# Create figure with subplots
fig = plt.figure(figsize=(16, 5))

# Plot 1: Contour plot with gradient vectors (quiver plot)
ax1 = fig.add_subplot(131)
contour = ax1.contour(X, Y, Z, levels=15, cmap='viridis', alpha=0.6)
ax1.clabel(contour, inline=True, fontsize=8)
# Quiver plot - arrows show gradient direction (steepest ascent)
quiver = ax1.quiver(X, Y, gradient_x, gradient_y, alpha=0.7, color='red')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Gradient Vectors (Point Uphill)')
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: 3D surface with gradient descent path
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

# Simulate gradient descent from a starting point
start_x, start_y = 2.5, 2.5
learning_rate = 0.1
path_x, path_y, path_z = [start_x], [start_y], [start_x**2 + start_y**2]

for _ in range(20):
    # Gradient at current point
    grad_x = 2 * path_x[-1]
    grad_y = 2 * path_y[-1]
    
    # Move OPPOSITE to gradient (descent, not ascent)
    new_x = path_x[-1] - learning_rate * grad_x
    new_y = path_y[-1] - learning_rate * grad_y
    
    path_x.append(new_x)
    path_y.append(new_y)
    path_z.append(new_x**2 + new_y**2)

# Plot the descent path
ax2.plot(path_x, path_y, path_z, 'ro-', linewidth=2, markersize=5, label='Gradient Descent Path')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('3D Surface with Gradient Descent')
ax2.legend()

# Plot 3: Convergence plot (loss over iterations)
ax3 = fig.add_subplot(133)
ax3.plot(path_z, 'bo-', linewidth=2, markersize=6)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Loss Value')
ax3.set_title('Convergence: Loss vs Iterations')
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='r', linestyle='--', label='Global Minimum')
ax3.legend()

plt.tight_layout()
plt.show()


