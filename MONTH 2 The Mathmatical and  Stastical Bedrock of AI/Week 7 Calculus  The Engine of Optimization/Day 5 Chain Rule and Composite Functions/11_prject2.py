
#Project 2: Jacobian Matrix (1.5 hours)

#For vector-valued function F: ℝ² → ℝ²
#Example: F(x,y) = [x² + y, xy]
#Compute the Jacobian matrix manually
#Verify with code
#Visualize the transformation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

print("="*60)
print("JACOBIAN MATRIX PROJECT")
print("="*60)

# Define the vector-valued function F: ℝ² → ℝ²
# F(x,y) = [x² + y, xy]
def F(x, y):
    """
    Vector-valued function F: ℝ² → ℝ²
    F(x,y) = [x² + y, xy]
    """
    f1 = x**2 + y
    f2 = x * y
    return np.array([f1, f2])

# Partial derivatives (computed manually)
def partial_f1_x(x, y):
    """∂f₁/∂x = 2x"""
    return 2 * x

def partial_f1_y(x, y):
    """∂f₁/∂y = 1"""
    return 1

def partial_f2_x(x, y):
    """∂f₂/∂x = y"""
    return y

def partial_f2_y(x, y):
    """∂f₂/∂y = x"""
    return x

# Jacobian matrix function
def jacobian_manual(x, y):
    """
    Compute Jacobian matrix manually using partial derivatives
    J = [[∂f₁/∂x, ∂f₁/∂y],
         [∂f₂/∂x, ∂f₂/∂y]]
    """
    J = np.array([
        [partial_f1_x(x, y), partial_f1_y(x, y)],
        [partial_f2_x(x, y), partial_f2_y(x, y)]
    ])
    return J

# Numerical Jacobian (for verification)
def jacobian_numerical(x, y, h=1e-5):
    """
    Compute Jacobian numerically using finite differences
    """
    J = np.zeros((2, 2))
    
    # ∂f₁/∂x
    J[0, 0] = (F(x + h, y)[0] - F(x - h, y)[0]) / (2 * h)
    # ∂f₁/∂y
    J[0, 1] = (F(x, y + h)[0] - F(x, y - h)[0]) / (2 * h)
    # ∂f₂/∂x
    J[1, 0] = (F(x + h, y)[1] - F(x - h, y)[1]) / (2 * h)
    # ∂f₂/∂y
    J[1, 1] = (F(x, y + h)[1] - F(x, y - h)[1]) / (2 * h)
    
    return J

# ============================================================
# PART 1: Manual Computation at Specific Point
# ============================================================
print("\n" + "="*60)
print("PART 1: MANUAL JACOBIAN COMPUTATION")
print("="*60)

# Test point
x0, y0 = 2, 3

print(f"\nGiven function F(x,y) = [x² + y, xy]")
print(f"\nAt point ({x0}, {y0}):")
print(f"F({x0}, {y0}) = {F(x0, y0)}")

print("\n--- Manual Partial Derivatives ---")
print(f"∂f₁/∂x = 2x = 2({x0}) = {partial_f1_x(x0, y0)}")
print(f"∂f₁/∂y = 1")
print(f"∂f₂/∂x = y = {y0}")
print(f"∂f₂/∂y = x = {x0}")

J_manual = jacobian_manual(x0, y0)
print(f"\nJacobian Matrix (Manual):")
print(J_manual)

# ============================================================
# PART 2: Numerical Verification
# ============================================================
print("\n" + "="*60)
print("PART 2: NUMERICAL VERIFICATION")
print("="*60)

J_numerical = jacobian_numerical(x0, y0)
print(f"\nJacobian Matrix (Numerical):")
print(J_numerical)

print(f"\nDifference between manual and numerical:")
print(J_manual - J_numerical)
print(f"\nMax absolute error: {np.max(np.abs(J_manual - J_numerical)):.2e}")

# ============================================================
# PART 3: Jacobian at Multiple Points
# ============================================================
print("\n" + "="*60)
print("PART 3: JACOBIAN AT MULTIPLE POINTS")
print("="*60)

test_points = [(0, 0), (1, 1), (2, 3), (-1, 2)]
for point in test_points:
    x, y = point
    J = jacobian_manual(x, y)
    det_J = np.linalg.det(J)
    print(f"\nAt ({x}, {y}):")
    print(f"Jacobian:\n{J}")
    print(f"Determinant: {det_J:.4f}")

# ============================================================
# PART 4: VISUALIZATION
# ============================================================
print("\n" + "="*60)
print("PART 4: CREATING VISUALIZATIONS")
print("="*60)

fig = plt.figure(figsize=(18, 12))

# Plot 1: Vector field in input space
ax1 = fig.add_subplot(2, 3, 1)
x = np.linspace(-3, 3, 15)
y = np.linspace(-3, 3, 15)
X, Y = np.meshgrid(x, y)

# Compute function values
F_vals = F(X, Y)
U = F_vals[0]
V = F_vals[1]

ax1.quiver(X, Y, U, V, alpha=0.6, color='blue')
ax1.scatter([x0], [y0], color='red', s=100, zorder=5, label=f'Point ({x0}, {y0})')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Vector Field: F(x,y)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.axis('equal')

# Plot 2: Transformation visualization
ax2 = fig.add_subplot(2, 3, 2)
# Input grid
grid_x = np.linspace(-2, 2, 10)
grid_y = np.linspace(-2, 2, 10)
for gx in grid_x:
    y_line = np.linspace(-2, 2, 50)
    x_line = np.full_like(y_line, gx)
    ax2.plot(x_line, y_line, 'b-', alpha=0.3, linewidth=0.5)
for gy in grid_y:
    x_line = np.linspace(-2, 2, 50)
    y_line = np.full_like(x_line, gy)
    ax2.plot(x_line, y_line, 'b-', alpha=0.3, linewidth=0.5)
ax2.scatter([x0], [y0], color='red', s=100, zorder=5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Input Space', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axis('equal')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)

# Plot 3: Transformed grid
ax3 = fig.add_subplot(2, 3, 3)
for gx in grid_x:
    y_line = np.linspace(-2, 2, 50)
    x_line = np.full_like(y_line, gx)
    F_line = F(x_line, y_line)
    ax3.plot(F_line[0], F_line[1], 'g-', alpha=0.3, linewidth=0.5)
for gy in grid_y:
    x_line = np.linspace(-2, 2, 50)
    y_line = np.full_like(x_line, gy)
    F_line = F(x_line, y_line)
    ax3.plot(F_line[0], F_line[1], 'g-', alpha=0.3, linewidth=0.5)
F_point = F(x0, y0)
ax3.scatter([F_point[0]], [F_point[1]], color='red', s=100, zorder=5)
ax3.set_xlabel('f₁', fontsize=12)
ax3.set_ylabel('f₂', fontsize=12)
ax3.set_title('Output Space (Transformed)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.axis('equal')

# Plot 4: Jacobian heatmap
ax4 = fig.add_subplot(2, 3, 4)
J = jacobian_manual(x0, y0)
sns.heatmap(J, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            xticklabels=['∂/∂x', '∂/∂y'],
            yticklabels=['f₁', 'f₂'],
            cbar_kws={'label': 'Value'},
            ax=ax4, square=True, linewidths=2)
ax4.set_title(f'Jacobian Matrix at ({x0}, {y0})', fontsize=14, fontweight='bold')

# Plot 5: Determinant across space
ax5 = fig.add_subplot(2, 3, 5)
x_det = np.linspace(-3, 3, 100)
y_det = np.linspace(-3, 3, 100)
X_det, Y_det = np.meshgrid(x_det, y_det)
Det = np.zeros_like(X_det)
for i in range(X_det.shape[0]):
    for j in range(X_det.shape[1]):
        J_temp = jacobian_manual(X_det[i, j], Y_det[i, j])
        Det[i, j] = np.linalg.det(J_temp)

contour = ax5.contourf(X_det, Y_det, Det, levels=20, cmap='RdYlBu_r')
plt.colorbar(contour, ax=ax5, label='det(J)')
ax5.scatter([x0], [y0], color='red', s=100, zorder=5, edgecolor='black', linewidth=2)
ax5.set_xlabel('x', fontsize=12)
ax5.set_ylabel('y', fontsize=12)
ax5.set_title('Jacobian Determinant', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Linear approximation
ax6 = fig.add_subplot(2, 3, 6)
# Small displacement vectors around the point
delta = 0.5
arrows = [
    ([1, 0], 'e₁'),
    ([0, 1], 'e₂'),
    ([1, 1]/np.sqrt(2), 'diagonal')
]
for vec, label in arrows:
    vec = np.array(vec) * delta
    # Apply Jacobian
    J_vec = J_manual.dot(vec)
    ax6.arrow(F_point[0], F_point[1], J_vec[0], J_vec[1],
              head_width=0.3, head_length=0.2, fc='blue', ec='blue', alpha=0.7)
    ax6.text(F_point[0] + J_vec[0], F_point[1] + J_vec[1], f'  J·{label}',
             fontsize=10)

ax6.scatter([F_point[0]], [F_point[1]], color='red', s=100, zorder=5)
ax6.set_xlabel('f₁', fontsize=12)
ax6.set_ylabel('f₂', fontsize=12)
ax6.set_title('Linear Approximation (J·Δx)', fontsize=14, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.axis('equal')

plt.tight_layout()
plt.savefig('jacobian_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'jacobian_analysis.png'")
plt.show()

# ============================================================
# PART 5: Summary Statistics
# ============================================================
print("\n" + "="*60)
print("PART 5: SUMMARY")
print("="*60)

print(f"\nFunction: F(x,y) = [x² + y, xy]")
print(f"\nJacobian Matrix Formula:")
print("J(x,y) = [[2x,  1],")
print("          [ y,  x]]")
print(f"\nAt point ({x0}, {y0}):")
print(f"J = \n{J_manual}")
print(f"\nDeterminant: {np.linalg.det(J_manual):.4f}")
print(f"Trace: {np.trace(J_manual):.4f}")
eigenvalues = np.linalg.eigvals(J_manual)
print(f"Eigenvalues: {eigenvalues}")
print(f"\nInterpretation:")
if np.linalg.det(J_manual) > 0:
    print("- det(J) > 0: Transformation preserves orientation")
else:
    print("- det(J) < 0: Transformation reverses orientation")
print(f"- |det(J)| = {abs(np.linalg.det(J_manual)):.4f}: Area scaling factor")
