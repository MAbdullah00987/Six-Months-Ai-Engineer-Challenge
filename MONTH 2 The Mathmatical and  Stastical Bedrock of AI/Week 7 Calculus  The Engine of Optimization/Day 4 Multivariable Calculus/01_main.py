
#Day 4: Multivariable Calculus
#Focus: Partial derivatives and gradients in multiple dimensions
#Learning Activities (3-4 hours)

#Read: Mathematics for Machine Learning, Chapter 5 - Multivariate calculus sections
#Watch: Coursera lectures on partial derivatives and gradients
#Practice: Calculate partial derivatives by hand for 5-7 functions

#Project: Gradient of a Multivariable Function (2-3 hours)
#Calculate and visualize gradient vector fields

#Choose function: f(x,y) = x² + y² or f(x,y) = x² - y²
#Compute partial derivatives ∂f/∂x and ∂f/∂y
#Create visualizations:

#3D surface plot
#Contour plot with gradient vectors overlaid
#Quiver plot showing gradient direction and magnitude



#Deliverable: Interactive visualizations of gradient vector fields

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#1. Understanding Partial Derivatives
##Concept: A partial derivative measures how a function changes when you vary just ONE variable while keeping others constant.
#or a function f(x, y), we have:

#∂f/∂x: rate of change in the x-direction
#∂f/∂y: rate of change in the y-direction

'''
# Define symbolic variables
x, y = sp.symbols('x y')

# Function 1: f(x,y) = x^2 + y^2
f1 = x**2 + y**2
print("Function 1:", f1)
print("∂f/∂x =", sp.diff(f1, x))  # 2x
print("∂f/∂y =", sp.diff(f1, y))  # 2y
print()

# Function 2: f(x,y) = x*y + 3x^2
f2 = x*y + 3*x**2
print("Function 2:", f2)
print("∂f/∂x =", sp.diff(f2, x))  # y + 6x
print("∂f/∂y =", sp.diff(f2, y))  # x
print()

# Function 3: f(x,y) = sin(x) * cos(y)
f3 = sp.sin(x) * sp.cos(y)
print("Function 3:", f3)
print("∂f/∂x =", sp.diff(f3, x))  # cos(x)*cos(y)
print("∂f/∂y =", sp.diff(f3, y))  # -sin(x)*sin(y)
print()

# Function 4: f(x,y) = e^(x*y)
f4 = sp.exp(x*y)
print("Function 4:", f4)
print("∂f/∂x =", sp.diff(f4, x))  # y*e^(xy)
print("∂f/∂y =", sp.diff(f4, y))  # x*e^(xy)
print()

# Function 5: f(x,y) = x^3 - 3xy + y^2
f5 = x**3 - 3*x*y + y**2
print("Function 5:", f5)
print("∂f/∂x =", sp.diff(f5, x))  # 3x^2 - 3y
print("∂f/∂y =", sp.diff(f5, y))  # -3x + 2y

'''

#Visualizing Functions and Their Partial Derivatives

'''
# Let's visualize f(x,y) = x^2 + y^2 and its partial derivatives
X = np.linspace(-3, 3, 100)
Y = np.linspace(-3, 3, 100)
X_grid, Y_grid = np.meshgrid(X, Y)

# Function
Z = X_grid**2 + Y_grid**2

# Partial derivatives
dZ_dx = 2 * X_grid  # ∂f/∂x = 2x
dZ_dy = 2 * Y_grid  # ∂f/∂y = 2y

# Create visualizations
fig = plt.figure(figsize=(15, 5))

# Plot 1: Original function
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('f(x,y) = x² + y²')

# Plot 2: Partial derivative with respect to x
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(X_grid, Y_grid, dZ_dx, cmap='plasma', alpha=0.8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('∂f/∂x')
ax2.set_title('∂f/∂x = 2x')

# Plot 3: Partial derivative with respect to y
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(X_grid, Y_grid, dZ_dy, cmap='inferno', alpha=0.8)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('∂f/∂y')
ax3.set_title('∂f/∂y = 2y')

plt.tight_layout()
plt.savefig('partial_derivatives_visualization.png', dpi=300)
plt.show()
'''

#2. Gradients: Combining All Partial Derivatives
#Concept: The gradient ∇f is a vector containing all partial derivatives. It points in the direction of steepest ascent.
#For f(x, y): ∇f = [∂f/∂x, ∂f/∂y]

# Function: f(x,y) = x^2 + y^2

'''
def f(x, y):
    return x**2 + y**2

# Gradient: ∇f = [2x, 2y]
def gradient_f(x, y):
    df_dx = 2 * x
    df_dy = 2 * y
    return np.array([df_dx, df_dy])

# Create grid
x = np.linspace(-3, 3, 20)
y = np.linspace(-3, 3, 20)
X, Y = np.meshgrid(x, y)

# Calculate function values
Z = f(X, Y)

# Calculate gradient at each point
U = 2 * X  # x-component of gradient
V = 2 * Y  # y-component of gradient

# Visualization
plt.figure(figsize=(12, 5))

# Left plot: Contour with gradient vectors
plt.subplot(121)
plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.quiver(X, Y, U, V, color='red', alpha=0.6)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Vectors on Contour Plot')
plt.colorbar(label='f(x,y)')
plt.grid(True, alpha=0.3)

# Right plot: 3D surface with gradient at specific points
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(122, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# Show gradient vectors at specific points
points = [(-2, -2), (-1, 1), (1, -1), (2, 2)]
for px, py in points:
    grad = gradient_f(px, py)
    ax.quiver(px, py, f(px, py), grad[0], grad[1], 0, 
              color='red', arrow_length_ratio=0.3, linewidth=2)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Gradient Vectors in 3D')

plt.tight_layout()
plt.savefig('gradient_visualization.png', dpi=300)
plt.show()
'''

#3. Gradient Descent: Finding Minima
#Concept: Follow the negative gradient to find the minimum of a function.
#Implementing Gradient Descent from Scratch

'''
# Function: f(x,y) = x^2 + y^2
def f(x, y):
    return x**2 + y**2

# Gradient
def gradient(x, y):
    return np.array([2*x, 2*y])

# Gradient Descent Algorithm
def gradient_descent(start_point, learning_rate=0.1, num_iterations=50):
    position = np.array(start_point, dtype=float)
    path = [position.copy()]
    
    for i in range(num_iterations):
        grad = gradient(position[0], position[1])
        position = position - learning_rate * grad  # Move opposite to gradient
        path.append(position.copy())
    
    return np.array(path)

# Run gradient descent from different starting points
start_points = [(3, 3), (-2, 3), (3, -2), (-3, -3)]
colors = ['red', 'blue', 'green', 'orange']

# Visualization
plt.figure(figsize=(14, 6))

# Left: Contour plot with paths
plt.subplot(121)
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
plt.colorbar(label='f(x,y)')

for start, color in zip(start_points, colors):
    path = gradient_descent(start)
    plt.plot(path[:, 0], path[:, 1], 'o-', color=color, 
             label=f'Start: {start}', markersize=3, linewidth=2)
    plt.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=10)
    plt.plot(path[-1, 0], path[-1, 1], '*', color=color, markersize=15)

plt.plot(0, 0, 'k*', markersize=20, label='Global Minimum')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Paths')
plt.legend()
plt.grid(True, alpha=0.3)

# Right: Function value over iterations
plt.subplot(122)
for start, color in zip(start_points, colors):
    path = gradient_descent(start)
    values = [f(p[0], p[1]) for p in path]
    plt.plot(values, 'o-', color=color, label=f'Start: {start}')

plt.xlabel('Iteration')
plt.ylabel('f(x, y)')
plt.title('Function Value During Descent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plt.savefig('gradient_descent.png', dpi=300)
plt.show()

'''

#4. Second-Order Partial Derivatives and Hessian Matrix
#Concept: Second derivatives tell us about the curvature of a function.
#Hessian Matrix with SymPy

'''
x, y = sp.symbols('x y')

# Function: f(x,y) = x^2*y + y^3 - 2*x
f = x**2 * y + y**3 - 2*x

print("Function:", f)
print()

# First-order partial derivatives
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

print("First-order partial derivatives:")
print(f"∂f/∂x = {df_dx}")
print(f"∂f/∂y = {df_dy}")
print()

# Second-order partial derivatives
d2f_dx2 = sp.diff(df_dx, x)
d2f_dy2 = sp.diff(df_dy, y)
d2f_dxdy = sp.diff(df_dx, y)
d2f_dydx = sp.diff(df_dy, x)

print("Second-order partial derivatives:")
print(f"∂²f/∂x² = {d2f_dx2}")
print(f"∂²f/∂y² = {d2f_dy2}")
print(f"∂²f/∂x∂y = {d2f_dxdy}")
print(f"∂²f/∂y∂x = {d2f_dydx}")
print()

# Hessian matrix
hessian = sp.Matrix([[d2f_dx2, d2f_dxdy],
                     [d2f_dydx, d2f_dy2]])

print("Hessian Matrix:")
print(hessian)
print()

# Find critical points (where gradient = 0)
critical_points = sp.solve([df_dx, df_dy], [x, y])
print("Critical points:", critical_points)
print()

# Evaluate Hessian at critical points
for point in critical_points:
    H = hessian.subs({x: point[0], y: point[1]})
    print(f"Hessian at {point}:")
    print(H)
    print(f"Determinant: {H.det()}")
    print(f"Trace: {H.trace()}")
    print()


'''

#5. Chain Rule in Multiple Variables
#Concept: When functions are composed, we use the chain rule to find derivatives.
#Chain Rule with SymPy

'''
# Define variables
x, y, t = sp.symbols('x y t')

# x and y are functions of t
x_t = sp.cos(t)
y_t = sp.sin(t)

# f is a function of x and y
f = x**2 + y**2

print("Given:")
print(f"x(t) = {x_t}")
print(f"y(t) = {y_t}")
print(f"f(x,y) = {f}")
print()

# Substitute to get f as a function of t
f_t = f.subs({x: x_t, y: y_t})
print(f"f(t) = {f_t}")
print(f"Simplified: {sp.simplify(f_t)}")
print()

# Method 1: Direct differentiation
df_dt_direct = sp.diff(f_t, t)
print("Method 1 - Direct differentiation:")
print(f"df/dt = {df_dt_direct}")
print(f"Simplified: {sp.simplify(df_dt_direct)}")
print()

# Method 2: Chain rule
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)
dx_dt = sp.diff(x_t, t)
dy_dt = sp.diff(y_t, t)

df_dt_chain = df_dx.subs({x: x_t, y: y_t}) * dx_dt + \
              df_dy.subs({x: x_t, y: y_t}) * dy_dt

print("Method 2 - Chain rule:")
print(f"∂f/∂x = {df_dx}")
print(f"∂f/∂y = {df_dy}")
print(f"dx/dt = {dx_dt}")
print(f"dy/dt = {dy_dt}")
print(f"df/dt = (∂f/∂x)(dx/dt) + (∂f/∂y)(dy/dt)")
print(f"df/dt = {df_dt_chain}")
print(f"Simplified: {sp.simplify(df_dt_chain)}")

'''

#6. Directional Derivatives
#Concept: Rate of change of a function in ANY direction (not just along axes).
#Directional Derivatives with NumPy

# Function: f(x,y) = x^2 + 2*y^2

'''
def f(x, y):
    return x**2 + 2*y**2

# Gradient
def gradient(x, y):
    return np.array([2*x, 4*y])

# Directional derivative: D_u(f) = ∇f · u (where u is unit vector)
def directional_derivative(x, y, direction):
    # Normalize direction vector
    u = direction / np.linalg.norm(direction)
    grad = gradient(x, y)
    return np.dot(grad, u)

# Point of interest
point = (2, 1)

# Different directions
directions = [
    (1, 0),      # East
    (0, 1),      # North
    (1, 1),      # Northeast
    (-1, 1),     # Northwest
    (1, -1),     # Southeast
]

print(f"At point {point}:")
print(f"Gradient = {gradient(*point)}")
print()

for direction in directions:
    dir_deriv = directional_derivative(*point, direction)
    normalized = np.array(direction) / np.linalg.norm(direction)
    print(f"Direction {normalized}: D_u(f) = {dir_deriv:.3f}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Contour plot with directional derivatives
x = np.linspace(-1, 4, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

ax1.contour(X, Y, Z, levels=20, cmap='viridis')
ax1.plot(*point, 'r*', markersize=20, label='Point (2,1)')

# Plot gradient
grad = gradient(*point)
ax1.quiver(*point, grad[0], grad[1], color='red', 
           scale=20, width=0.01, label='Gradient')

# Plot directional derivatives
colors = ['blue', 'green', 'orange', 'purple', 'brown']
for direction, color in zip(directions, colors):
    u = np.array(direction) / np.linalg.norm(direction)
    dir_deriv = directional_derivative(*point, direction)
    ax1.quiver(*point, u[0]*abs(dir_deriv), u[1]*abs(dir_deriv), 
               color=color, scale=10, width=0.008, alpha=0.7)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Directional Derivatives at (2,1)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Directional derivative for all angles
angles = np.linspace(0, 2*np.pi, 100)
dir_derivs = []

for angle in angles:
    direction = (np.cos(angle), np.sin(angle))
    dir_deriv = directional_derivative(*point, direction)
    dir_derivs.append(dir_deriv)

ax2.plot(angles, dir_derivs, 'b-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('Direction (radians)')
ax2.set_ylabel('Directional Derivative')
ax2.set_title('Directional Derivative in All Directions')
ax2.grid(True, alpha=0.3)

# Mark maximum (gradient direction)
max_idx = np.argmax(dir_derivs)
ax2.plot(angles[max_idx], dir_derivs[max_idx], 'r*', 
         markersize=15, label='Max (gradient direction)')
ax2.legend()

plt.tight_layout()
plt.savefig('directional_derivatives.png', dpi=300)
plt.show()

print(f"\nMaximum directional derivative: {max(dir_derivs):.3f}")
print(f"This equals |∇f| = {np.linalg.norm(gradient(*point)):.3f}")
'''

#7. Level Curves and Contour Plots
#Concept: Level curves show where a function has constant value. The gradient is perpendicular to level curves.
#Level Curves with Matplotlib and Seaborn

'''
sns.set_style("whitegrid")

# Function: f(x,y) = x^2 - y^2 (saddle point)
def f(x, y):
    return x**2 - y**2

# Gradient
def gradient(x, y):
    return np.array([2*x, -2*y])

# Create grid
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Create figure
fig = plt.figure(figsize=(16, 6))

# Plot 1: Contour plot with gradients
ax1 = plt.subplot(131)
levels = np.linspace(-8, 8, 17)
contours = ax1.contour(X, Y, Z, levels=levels, cmap='RdBu_r')
ax1.clabel(contours, inline=True, fontsize=8)

# Add gradient vectors at select points
sample_points = [(-2, -2), (-1, 1), (0, 2), (1, -1), (2, 2)]
for px, py in sample_points:
    grad = gradient(px, py)
    ax1.quiver(px, py, grad[0], grad[1], color='green', 
               scale=15, width=0.01, alpha=0.8)

ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Level Curves with Gradients\n(Gradients ⊥ to curves)')
ax1.set_aspect('equal')

# Plot 2: Filled contour (heatmap style)
ax2 = plt.subplot(132)
im = ax2.contourf(X, Y, Z, levels=30, cmap='RdBu_r')
plt.colorbar(im, ax=ax2, label='f(x,y)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Heatmap of f(x,y) = x² - y²')

# Plot 3: 3D surface
ax3 = plt.subplot(133, projection='3d')
ax3.plot_surface(X, Y, Z, cmap='RdBu_r', alpha=0.8)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('3D Surface')

plt.tight_layout()
plt.savefig('level_curves_analysis.png', dpi=300)
plt.show()
'''

#8. Machine Learning Application: Linear Regression with Gradient Descent
#Concept: Use gradients to minimize the cost function in machine learning.
#Linear Regression from Scratch

'''
# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add bias term
X_b = np.c_[np.ones((100, 1)), X]

# Cost function: MSE = (1/m) * Σ(prediction - actual)²
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum((predictions - y)**2)
    return cost

# Gradient of cost function
def compute_gradient(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    gradient = (1/m) * X.T.dot(predictions - y)
    return gradient

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []
    theta_history = [theta.copy()]
    
    for i in range(num_iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
        
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        theta_history.append(theta.copy())
    
    return theta, cost_history, theta_history

# Initialize parameters
theta_init = np.random.randn(2, 1)
learning_rate = 0.1
num_iterations = 100

# Run gradient descent
theta_final, cost_history, theta_history = gradient_descent(
    X_b, y, theta_init, learning_rate, num_iterations
)

print("Final parameters:")
print(f"θ0 (intercept) = {theta_final[0][0]:.3f}")
print(f"θ1 (slope) = {theta_final[1][0]:.3f}")
print(f"Final cost = {cost_history[-1]:.3f}")

# Visualization
fig = plt.figure(figsize=(16, 5))

# Plot 1: Data and fitted line
ax1 = plt.subplot(131)
ax1.scatter(X, y, alpha=0.5, label='Data')
ax1.plot(X, X_b.dot(theta_final), 'r-', linewidth=2, label='Fitted line')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression Result')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost over iterations
ax2 = plt.subplot(132)
ax2.plot(cost_history, 'b-', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost (MSE)')
ax2.set_title('Cost Function During Training')
ax2.grid(True, alpha=0.3)

# Plot 3: Parameter trajectory
ax3 = plt.subplot(133)
theta_history = np.array(theta_history).squeeze()

# Create contour plot of cost function
theta0_vals = np.linspace(-2, 8, 100)
theta1_vals = np.linspace(0, 6, 100)
Theta0, Theta1 = np.meshgrid(theta0_vals, theta1_vals)
Cost_vals = np.zeros(Theta0.shape)

for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        theta_temp = np.array([[Theta0[j, i]], [Theta1[j, i]]])
        Cost_vals[j, i] = compute_cost(X_b, y, theta_temp)

ax3.contour(Theta0, Theta1, Cost_vals, levels=30, cmap='viridis', alpha=0.6)
ax3.plot(theta_history[:, 0], theta_history[:, 1], 'ro-', 
         markersize=3, linewidth=1, label='Gradient descent path')
ax3.plot(theta_history[0, 0], theta_history[0, 1], 'g*', 
         markersize=15, label='Start')
ax3.plot(theta_history[-1, 0], theta_history[-1, 1], 'r*', 
         markersize=15, label='End')
ax3.set_xlabel('θ0 (intercept)')
ax3.set_ylabel('θ1 (slope)')
ax3.set_title('Parameter Space Optimization')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_gradient_descent.png', dpi=300)
plt.show()
'''

#9. Advanced: Multivariate Functions (3+ Variables)
#Working with Higher Dimensions

'''
# Define variables
x, y, z, w = sp.symbols('x y z w')

# 3-variable function
f_3d = x**2 + y**2 + z**2 - 2*x*y + 3*z

print("3D Function:", f_3d)
print("\nGradient (3D):")
gradient_3d = [sp.diff(f_3d, var) for var in [x, y, z]]
for i, (var, deriv) in enumerate(zip([x, y, z], gradient_3d)):
    print(f"∂f/∂{var} = {deriv}")

# 4-variable function
f_4d = x**2 + y**2 + z**2 + w**2 - x*y - y*z

print("\n4D Function:", f_4d)
print("\nGradient (4D):")
gradient_4d = [sp.diff(f_4d, var) for var in [x, y, z, w]]
for var, deriv in zip([x, y, z, w], gradient_4d):
    print(f"∂f/∂{var} = {deriv}")

# Laplacian (sum of second derivatives)
laplacian = sum([sp.diff(f_4d, var, 2) for var in [x, y, z, w]])
print(f"\nLaplacian (∇²f) = {laplacian}")

# Evaluate gradient at a specific point
point = {x: 1, y: 2, z: 3, w: 4}
print(f"\nGradient at {point}:")
for var, deriv in zip([x, y, z, w], gradient_4d):
    value = deriv.subs(point)
    print(f"∂f/∂{var}|_point = {value}")

# Magnitude of gradient
gradient_values = [float(deriv.subs(point)) for deriv in gradient_4d]
magnitude = np.sqrt(sum([v**2 for v in gradient_values]))
print(f"\n|∇f| = {magnitude:.3f}")

'''

