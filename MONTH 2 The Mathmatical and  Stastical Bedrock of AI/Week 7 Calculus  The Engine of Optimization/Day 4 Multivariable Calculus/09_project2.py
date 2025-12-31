
#Project: Gradient of a Multivariable Function (2-3 hours)
#Calculate and visualize gradient vector fields
#* Choose function: f(x,y) = x² + y² or f(x,y) = x² - y²
#* Compute partial derivatives ∂f/∂x and ∂f/∂y
#* Create visualizations:
#   * 3D surface plot
#   * Contour plot with gradient vectors overlaid
#   * Quiver plot showing gradient direction and magnitude
#Deliverable: Interactive visualizations of gradient vector fields


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sympy as sp

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)

# Define symbolic variables
x_sym, y_sym = sp.symbols('x y')

# Define the two functions symbolically
f1_sym = x_sym**2 + y_sym**2  # Paraboloid (bowl shape)
f2_sym = x_sym**2 - y_sym**2  # Saddle point

# Compute partial derivatives symbolically
df1_dx = sp.diff(f1_sym, x_sym)
df1_dy = sp.diff(f1_sym, y_sym)

df2_dx = sp.diff(f2_sym, x_sym)
df2_dy = sp.diff(f2_sym, y_sym)

print("=" * 80)
print("GRADIENT VECTOR FIELD ANALYSIS")
print("=" * 80)

print("\n1. FUNCTION 1: f(x,y) = x² + y² (Paraboloid)")
print("-" * 80)
print(f"   Function:       f(x,y) = {f1_sym}")
print(f"   ∂f/∂x:          {df1_dx}")
print(f"   ∂f/∂y:          {df1_dy}")
print(f"   Gradient:       ∇f = ({df1_dx}, {df1_dy})")

print("\n2. FUNCTION 2: f(x,y) = x² - y² (Saddle Point)")
print("-" * 80)
print(f"   Function:       f(x,y) = {f2_sym}")
print(f"   ∂f/∂x:          {df2_dx}")
print(f"   ∂f/∂y:          {df2_dy}")
print(f"   Gradient:       ∇f = ({df2_dx}, {df2_dy})")

# Convert symbolic expressions to numerical functions
f1_func = sp.lambdify((x_sym, y_sym), f1_sym, 'numpy')
f1_dx_func = sp.lambdify((x_sym, y_sym), df1_dx, 'numpy')
f1_dy_func = sp.lambdify((x_sym, y_sym), df1_dy, 'numpy')

f2_func = sp.lambdify((x_sym, y_sym), f2_sym, 'numpy')
f2_dx_func = sp.lambdify((x_sym, y_sym), df2_dx, 'numpy')
f2_dy_func = sp.lambdify((x_sym, y_sym), df2_dy, 'numpy')

# Create meshgrid for plotting
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Compute function values
Z1 = f1_func(X, Y)
Z2 = f2_func(X, Y)

# Compute gradients
U1 = f1_dx_func(X, Y)  # ∂f1/∂x
V1 = f1_dy_func(X, Y)  # ∂f1/∂y

U2 = f2_dx_func(X, Y)  # ∂f2/∂x
V2 = f2_dy_func(X, Y)  # ∂f2/∂y

# Compute gradient magnitudes
mag1 = np.sqrt(U1**2 + V1**2)
mag2 = np.sqrt(U2**2 + V2**2)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Gradient Vector Field Analysis', fontsize=18, fontweight='bold', y=0.98)

# ============================================================================
# FUNCTION 1: f(x,y) = x² + y² (PARABOLOID)
# ============================================================================

# Plot 1: 3D Surface
ax1 = fig.add_subplot(2, 3, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.8, 
                         edgecolor='none', antialiased=True)
ax1.set_xlabel('x', fontsize=11, fontweight='bold')
ax1.set_ylabel('y', fontsize=11, fontweight='bold')
ax1.set_zlabel('f(x,y)', fontsize=11, fontweight='bold')
ax1.set_title('3D Surface: f(x,y) = x² + y²', fontsize=13, fontweight='bold', pad=10)
ax1.view_init(elev=25, azim=45)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

# Plot 2: Contour plot with gradient vectors
ax2 = fig.add_subplot(2, 3, 2)
contour1 = ax2.contour(X, Y, Z1, levels=15, cmap='viridis', linewidths=1.5)
ax2.clabel(contour1, inline=True, fontsize=8)

# Overlay gradient vectors (sampled to avoid clutter)
step = 10
X_sample = X[::step, ::step]
Y_sample = Y[::step, ::step]
U1_sample = U1[::step, ::step]
V1_sample = V1[::step, ::step]

quiver1 = ax2.quiver(X_sample, Y_sample, U1_sample, V1_sample, 
                     mag1[::step, ::step], cmap='Reds', 
                     scale=50, width=0.004, alpha=0.8)
ax2.set_xlabel('x', fontsize=11, fontweight='bold')
ax2.set_ylabel('y', fontsize=11, fontweight='bold')
ax2.set_title('Contour + Gradient Vectors: x² + y²', fontsize=13, fontweight='bold')
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
fig.colorbar(quiver1, ax=ax2, label='Gradient Magnitude')

# Plot 3: Quiver plot (gradient field)
ax3 = fig.add_subplot(2, 3, 3)
quiver_plot1 = ax3.quiver(X_sample, Y_sample, U1_sample, V1_sample,
                          mag1[::step, ::step], cmap='plasma',
                          scale=50, width=0.005, alpha=0.9)
ax3.set_xlabel('x', fontsize=11, fontweight='bold')
ax3.set_ylabel('y', fontsize=11, fontweight='bold')
ax3.set_title('Gradient Vector Field: ∇f = (2x, 2y)', fontsize=13, fontweight='bold')
ax3.set_aspect('equal')
ax3.grid(True, alpha=0.3)
fig.colorbar(quiver_plot1, ax=ax3, label='Gradient Magnitude')

# ============================================================================
# FUNCTION 2: f(x,y) = x² - y² (SADDLE POINT)
# ============================================================================

# Plot 4: 3D Surface
ax4 = fig.add_subplot(2, 3, 4, projection='3d')
surf2 = ax4.plot_surface(X, Y, Z2, cmap='coolwarm', alpha=0.8,
                         edgecolor='none', antialiased=True)
ax4.set_xlabel('x', fontsize=11, fontweight='bold')
ax4.set_ylabel('y', fontsize=11, fontweight='bold')
ax4.set_zlabel('f(x,y)', fontsize=11, fontweight='bold')
ax4.set_title('3D Surface: f(x,y) = x² - y²', fontsize=13, fontweight='bold', pad=10)
ax4.view_init(elev=25, azim=45)
fig.colorbar(surf2, ax=ax4, shrink=0.5, aspect=5)

# Plot 5: Contour plot with gradient vectors
ax5 = fig.add_subplot(2, 3, 5)
contour2 = ax5.contour(X, Y, Z2, levels=20, cmap='coolwarm', linewidths=1.5)
ax5.clabel(contour2, inline=True, fontsize=8)

# Overlay gradient vectors
U2_sample = U2[::step, ::step]
V2_sample = V2[::step, ::step]

quiver2 = ax5.quiver(X_sample, Y_sample, U2_sample, V2_sample,
                     mag2[::step, ::step], cmap='Reds',
                     scale=50, width=0.004, alpha=0.8)
ax5.set_xlabel('x', fontsize=11, fontweight='bold')
ax5.set_ylabel('y', fontsize=11, fontweight='bold')
ax5.set_title('Contour + Gradient Vectors: x² - y²', fontsize=13, fontweight='bold')
ax5.set_aspect('equal')
ax5.grid(True, alpha=0.3)
fig.colorbar(quiver2, ax=ax5, label='Gradient Magnitude')

# Plot 6: Quiver plot (gradient field)
ax6 = fig.add_subplot(2, 3, 6)
quiver_plot2 = ax6.quiver(X_sample, Y_sample, U2_sample, V2_sample,
                          mag2[::step, ::step], cmap='plasma',
                          scale=50, width=0.005, alpha=0.9)
ax6.set_xlabel('x', fontsize=11, fontweight='bold')
ax6.set_ylabel('y', fontsize=11, fontweight='bold')
ax6.set_title('Gradient Vector Field: ∇f = (2x, -2y)', fontsize=13, fontweight='bold')
ax6.set_aspect('equal')
ax6.grid(True, alpha=0.3)
fig.colorbar(quiver_plot2, ax=ax6, label='Gradient Magnitude')

plt.tight_layout()
plt.show()

# ============================================================================
# DETAILED GRADIENT ANALYSIS AT SPECIFIC POINTS
# ============================================================================

print("\n" + "=" * 80)
print("GRADIENT ANALYSIS AT SPECIFIC POINTS")
print("=" * 80)

test_points = [
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (-1, -1),
    (2, 2)
]

print("\nFUNCTION 1: f(x,y) = x² + y²")
print("-" * 80)
print(f"{'Point (x,y)':<15} {'f(x,y)':<12} {'∂f/∂x':<12} {'∂f/∂y':<12} {'|∇f|':<12} {'Direction'}")
print("-" * 80)

for (x_pt, y_pt) in test_points:
    f_val = f1_func(x_pt, y_pt)
    dx = f1_dx_func(x_pt, y_pt)
    dy = f1_dy_func(x_pt, y_pt)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if magnitude > 0:
        angle = np.arctan2(dy, dx) * 180 / np.pi
        direction = f"{angle:.1f}°"
    else:
        direction = "N/A (zero)"
    
    print(f"({x_pt:>3}, {y_pt:>3})     {f_val:>8.3f}    {dx:>8.3f}    {dy:>8.3f}    {magnitude:>8.3f}    {direction}")

print("\nFUNCTION 2: f(x,y) = x² - y²")
print("-" * 80)
print(f"{'Point (x,y)':<15} {'f(x,y)':<12} {'∂f/∂x':<12} {'∂f/∂y':<12} {'|∇f|':<12} {'Direction'}")
print("-" * 80)

for (x_pt, y_pt) in test_points:
    f_val = f2_func(x_pt, y_pt)
    dx = f2_dx_func(x_pt, y_pt)
    dy = f2_dy_func(x_pt, y_pt)
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if magnitude > 0:
        angle = np.arctan2(dy, dx) * 180 / np.pi
        direction = f"{angle:.1f}°"
    else:
        direction = "N/A (zero)"
    
    print(f"({x_pt:>3}, {y_pt:>3})     {f_val:>8.3f}    {dx:>8.3f}    {dy:>8.3f}    {magnitude:>8.3f}    {direction}")

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("\n" + "=" * 80)
print("KEY INSIGHTS ABOUT GRADIENTS")
print("=" * 80)

insights = """
1. GRADIENT INTERPRETATION:
   • The gradient ∇f points in the direction of steepest ascent
   • The magnitude |∇f| indicates how steep the slope is
   • Gradient is perpendicular to contour lines

2. FUNCTION 1: f(x,y) = x² + y² (PARABOLOID)
   • Gradient: ∇f = (2x, 2y)
   • At origin (0,0): gradient is zero (critical point - minimum)
   • Gradients point radially outward from origin
   • Magnitude increases linearly with distance from origin
   • Contours are circles centered at origin

3. FUNCTION 2: f(x,y) = x² - y² (SADDLE POINT)
   • Gradient: ∇f = (2x, -2y)
   • At origin (0,0): gradient is zero (critical point - saddle)
   • Along x-axis: gradient points horizontally (uphill in x)
   • Along y-axis: gradient points vertically downward (downhill in y)
   • Contours are hyperbolas
   • This is a classic saddle point topology

4. APPLICATIONS:
   • Optimization: follow negative gradient to find minimum (gradient descent)
   • Machine learning: gradients used to update neural network weights
   • Physics: negative gradient of potential energy gives force
   • Image processing: gradient indicates edges and features
"""

print(insights)
