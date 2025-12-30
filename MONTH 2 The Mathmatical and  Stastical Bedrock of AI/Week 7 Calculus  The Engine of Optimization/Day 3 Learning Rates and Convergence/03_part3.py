
#Part 3: Symbolic Math with SymPy - Computing Gradients

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define symbolic variables
x, y = sp.symbols('x y')

print("=" * 70)
print("SYMBOLIC GRADIENT COMPUTATION WITH SYMPY")
print("=" * 70)

# Example 1: Simple quadratic function
print("\n### EXAMPLE 1: Quadratic Function f(x,y) = x² + y² ###")
f1 = x**2 + y**2
print(f"Function: f(x,y) = {f1}")

# Compute partial derivatives
df1_dx = sp.diff(f1, x)
df1_dy = sp.diff(f1, y)
print(f"∂f/∂x = {df1_dx}")
print(f"∂f/∂y = {df1_dy}")
print(f"Gradient: ∇f = [{df1_dx}, {df1_dy}]")

# Evaluate at a specific point
point1 = {x: 2, y: 3}
grad_at_point1 = [df1_dx.subs(point1), df1_dy.subs(point1)]
print(f"At point (2, 3): ∇f = {grad_at_point1}")
# Convert SymPy integers to Python floats before using NumPy
grad_magnitude = np.sqrt(sum([float(g)**2 for g in grad_at_point1]))
print(f"Gradient magnitude: {grad_magnitude:.4f}")

# Example 2: Rosenbrock function (banana function)
print("\n### EXAMPLE 2: Rosenbrock Function (Optimization Challenge) ###")
a, b = 1, 100
f2 = (a - x)**2 + b * (y - x**2)**2
print(f"Function: f(x,y) = (1-x)² + 100(y-x²)²")

df2_dx = sp.diff(f2, x)
df2_dy = sp.diff(f2, y)
print(f"∂f/∂x = {sp.simplify(df2_dx)}")
print(f"∂f/∂y = {sp.simplify(df2_dy)}")

# Example 3: Directional derivative
print("\n### EXAMPLE 3: Directional Derivative ###")
f3 = x**2 + 2*y**2
print(f"Function: f(x,y) = {f3}")

# Gradient
grad_f3 = [sp.diff(f3, x), sp.diff(f3, y)]
print(f"Gradient: ∇f = [{grad_f3[0]}, {grad_f3[1]}]")

# Direction vector (unit vector)
direction = sp.Matrix([1/sp.sqrt(2), 1/sp.sqrt(2)])  # 45-degree direction
print(f"Direction vector (normalized): {direction.T}")

# Directional derivative = ∇f · direction
grad_vec = sp.Matrix(grad_f3)
directional_deriv = grad_vec.dot(direction)
print(f"Directional derivative: D_v f = ∇f · v = {directional_deriv}")

# Evaluate at point (1, 1)
point3 = {x: 1, y: 1}
dir_deriv_value = directional_deriv.subs(point3)
print(f"At point (1, 1): D_v f = {dir_deriv_value} = {float(dir_deriv_value):.4f}")

# Example 4: Hessian matrix (second derivatives)
print("\n### EXAMPLE 4: Hessian Matrix (Curvature Information) ###")
f4 = x**3 + y**3 - 3*x*y
print(f"Function: f(x,y) = {f4}")

# First derivatives
fx = sp.diff(f4, x)
fy = sp.diff(f4, y)
print(f"First derivatives: fx = {fx}, fy = {fy}")

# Second derivatives (Hessian)
fxx = sp.diff(fx, x)
fxy = sp.diff(fx, y)
fyy = sp.diff(fy, y)
hessian = sp.Matrix([[fxx, fxy], [fxy, fyy]])
print(f"Hessian matrix H:")
print(hessian)

# Evaluate at critical point
critical_point = {x: 1, y: 1}
H_at_critical = hessian.subs(critical_point)
print(f"At point (1, 1):")
print(H_at_critical)
eigenvalues = H_at_critical.eigenvals()
print(f"Eigenvalues: {list(eigenvalues.keys())}")

# Visualize Example 2: Rosenbrock function
print("\n### VISUALIZATION: Rosenbrock Function ###")

# Convert symbolic to numerical function
f2_numerical = sp.lambdify((x, y), f2, 'numpy')
grad_x_numerical = sp.lambdify((x, y), df2_dx, 'numpy')
grad_y_numerical = sp.lambdify((x, y), df2_dy, 'numpy')

# Create meshgrid
x_vals = np.linspace(-2, 2, 100)
y_vals = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f2_numerical(X, Y)

# Create plots
fig = plt.figure(figsize=(15, 5))

# Plot 1: Contour with gradient vectors
ax1 = fig.add_subplot(131)
contour = ax1.contour(X, Y, np.log(Z + 1), levels=20, cmap='viridis')
ax1.clabel(contour, inline=True, fontsize=8)

# Sample points for gradient vectors
x_sample = np.linspace(-1.5, 1.5, 10)
y_sample = np.linspace(0, 2.5, 10)
X_sample, Y_sample = np.meshgrid(x_sample, y_sample)
U = grad_x_numerical(X_sample, Y_sample)
V = grad_y_numerical(X_sample, Y_sample)

# Normalize for better visualization
magnitude = np.sqrt(U**2 + V**2)
U_norm = U / (magnitude + 1e-8)
V_norm = V / (magnitude + 1e-8)

ax1.quiver(X_sample, Y_sample, U_norm, V_norm, alpha=0.6, color='red')
ax1.plot(1, 1, 'r*', markersize=20, label='Global Minimum (1,1)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Rosenbrock: Contour + Gradients')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: 3D surface
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8, 
                        linewidth=0, antialiased=True)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('f(x,y)')
ax2.set_title('Rosenbrock Function 3D')
ax2.view_init(elev=25, azim=45)

# Plot 3: Gradient magnitude
ax3 = fig.add_subplot(133)
grad_magnitude = np.sqrt(grad_x_numerical(X, Y)**2 + grad_y_numerical(X, Y)**2)
im = ax3.contourf(X, Y, np.log(grad_magnitude + 1), levels=20, cmap='hot')
plt.colorbar(im, ax=ax3, label='log(|∇f| + 1)')
ax3.plot(1, 1, 'c*', markersize=20, label='Minimum')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Gradient Magnitude (log scale)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("KEY TAKEAWAYS:")
print("=" * 70)
print("1. SymPy computes exact symbolic derivatives")
print("2. Gradient = vector of all partial derivatives")
print("3. Directional derivative = ∇f · direction_vector")
print("4. Hessian matrix contains second derivatives (curvature info)")
print("5. Rosenbrock function is hard to optimize (narrow valley)")
print("=" * 70)