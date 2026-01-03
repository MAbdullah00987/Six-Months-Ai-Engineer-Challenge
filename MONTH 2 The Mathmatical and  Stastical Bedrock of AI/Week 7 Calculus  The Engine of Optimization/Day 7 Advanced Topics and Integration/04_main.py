
#Part 4: Second-Order Methods (Newton's Method)
#Second-Order Optimization Methods

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp

sns.set_style("whitegrid")

# NEWTON'S METHOD IMPLEMENTATION


class SecondOrderMethods:
    """Compare first-order (gradient descent) vs second-order (Newton) methods"""
    
    def __init__(self):
        self.history_gd = []
        self.history_newton = []
    
    def rosenbrock(self, x, y):
        """
        Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
        Global minimum at (1, 1) with f(1,1) = 0
        """
        return (1 - x)**2 + 100 * (y - x**2)**2
    
    def rosenbrock_gradient(self, x, y):
        """Gradient of Rosenbrock function"""
        df_dx = -2*(1 - x) - 400*x*(y - x**2)
        df_dy = 200*(y - x**2)
        return np.array([df_dx, df_dy])
    
    def rosenbrock_hessian(self, x, y):
        """Hessian matrix of Rosenbrock function"""
        d2f_dx2 = 2 + 1200*x**2 - 400*y
        d2f_dxdy = -400*x
        d2f_dy2 = 200
        
        return np.array([
            [d2f_dx2, d2f_dxdy],
            [d2f_dxdy, d2f_dy2]
        ])
    
    def gradient_descent(self, start, lr=0.001, max_iter=1000, tol=1e-6):
        """Standard gradient descent"""
        point = np.array(start, dtype=float)
        self.history_gd = [point.copy()]
        
        for i in range(max_iter):
            grad = self.rosenbrock_gradient(point[0], point[1])
            
            if np.linalg.norm(grad) < tol:
                print(f"  GD converged in {i} iterations")
                break
            
            point = point - lr * grad
            self.history_gd.append(point.copy())
        
        return np.array(self.history_gd)
    
    def newton_method(self, start, max_iter=50, tol=1e-6):
        """Newton's method with Hessian"""
        point = np.array(start, dtype=float)
        self.history_newton = [point.copy()]
        
        for i in range(max_iter):
            grad = self.rosenbrock_gradient(point[0], point[1])
            
            if np.linalg.norm(grad) < tol:
                print(f"  Newton converged in {i} iterations")
                break
            
            hess = self.rosenbrock_hessian(point[0], point[1])
            
            # Newton update: x_new = x - H^(-1) * grad
            try:
                delta = np.linalg.solve(hess, grad)
                point = point - delta
            except np.linalg.LinAlgError:
                print("  Hessian is singular, stopping")
                break
            
            self.history_newton.append(point.copy())
        
        return np.array(self.history_newton)
    
    def compare_methods(self):
        """Compare gradient descent vs Newton's method"""
        
        fig = plt.figure(figsize=(18, 12))
        
        # Create mesh for contour
        x = np.linspace(-2, 2, 200)
        y = np.linspace(-1, 3, 200)
        X, Y = np.meshgrid(x, y)
        Z = self.rosenbrock(X, Y)
        
        # Use log scale for better visualization
        Z_log = np.log1p(Z)
        
        # Starting point
        start = [-1.5, 2.5]
        
        # Run both methods
        print("\nRunning Gradient Descent...")
        path_gd = self.gradient_descent(start, lr=0.001, max_iter=5000)
        
        print("Running Newton's Method...")
        path_newton = self.newton_method(start, max_iter=50)
        
        # Plot 1: 3D surface with paths
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, Z_log, cmap='viridis', alpha=0.6,
                                linewidth=0, antialiased=True)
        
        # Plot paths
        z_gd = [np.log1p(self.rosenbrock(p[0], p[1])) for p in path_gd[::10]]
        ax1.plot(path_gd[::10, 0], path_gd[::10, 1], z_gd, 
                'r-', linewidth=2, label='Gradient Descent')
        
        z_newton = [np.log1p(self.rosenbrock(p[0], p[1])) for p in path_newton]
        ax1.plot(path_newton[:, 0], path_newton[:, 1], z_newton,
                'b-', linewidth=3, label='Newton Method')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('log(1 + f(x,y))')
        ax1.set_title('Rosenbrock Function (Log Scale)', fontweight='bold')
        ax1.legend()
        
        # Plot 2: Contour with full GD path
        ax2 = fig.add_subplot(2, 3, 2)
        contour = ax2.contour(X, Y, Z_log, levels=30, cmap='viridis', alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        
        ax2.plot(path_gd[:, 0], path_gd[:, 1], 'r-', linewidth=1.5, 
                alpha=0.7, label=f'GD ({len(path_gd)} iter)')
        ax2.plot(start[0], start[1], 'go', markersize=12, 
                markeredgecolor='black', markeredgewidth=2, label='Start')
        ax2.plot(1, 1, 'y*', markersize=20, markeredgecolor='black',
                markeredgewidth=2, label='Global Min')
        
        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('y', fontsize=11)
        ax2.set_title('Gradient Descent Path', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Contour with Newton path
        ax3 = fig.add_subplot(2, 3, 3)
        contour = ax3.contour(X, Y, Z_log, levels=30, cmap='viridis', alpha=0.6)
        ax3.clabel(contour, inline=True, fontsize=8)
        
        ax3.plot(path_newton[:, 0], path_newton[:, 1], 'b-o', linewidth=2.5,
                markersize=6, label=f'Newton ({len(path_newton)} iter)')
        ax3.plot(start[0], start[1], 'go', markersize=12,
                markeredgecolor='black', markeredgewidth=2, label='Start')
        ax3.plot(1, 1, 'y*', markersize=20, markeredgecolor='black',
                markeredgewidth=2, label='Global Min')
        
        ax3.set_xlabel('x', fontsize=11)
        ax3.set_ylabel('y', fontsize=11)
        ax3.set_title('Newton Method Path', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Convergence comparison
        ax4 = fig.add_subplot(2, 3, 4)
        
        values_gd = [self.rosenbrock(p[0], p[1]) for p in path_gd]
        values_newton = [self.rosenbrock(p[0], p[1]) for p in path_newton]
        
        ax4.semilogy(values_gd, 'r-', linewidth=2, label='Gradient Descent')
        ax4.semilogy(values_newton, 'b-o', linewidth=2, markersize=6, 
                    label='Newton Method')
        ax4.axhline(y=1e-6, color='g', linestyle='--', linewidth=2,
                   label='Tolerance')
        
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Function Value (log scale)', fontsize=11)
        ax4.set_title('Convergence Comparison', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Distance to optimum
        ax5 = fig.add_subplot(2, 3, 5)
        
        optimum = np.array([1.0, 1.0])
        dist_gd = [np.linalg.norm(p - optimum) for p in path_gd]
        dist_newton = [np.linalg.norm(p - optimum) for p in path_newton]
        
        ax5.semilogy(dist_gd, 'r-', linewidth=2, label='Gradient Descent')
        ax5.semilogy(dist_newton, 'b-o', linewidth=2, markersize=6,
                    label='Newton Method')
        
        ax5.set_xlabel('Iteration', fontsize=11)
        ax5.set_ylabel('Distance to Optimum (log scale)', fontsize=11)
        ax5.set_title('Distance Convergence', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Gradient magnitude
        ax6 = fig.add_subplot(2, 3, 6)
        
        grad_norm_gd = [np.linalg.norm(self.rosenbrock_gradient(p[0], p[1])) 
                        for p in path_gd]
        grad_norm_newton = [np.linalg.norm(self.rosenbrock_gradient(p[0], p[1]))
                           for p in path_newton]
        
        ax6.semilogy(grad_norm_gd, 'r-', linewidth=2, label='Gradient Descent')
        ax6.semilogy(grad_norm_newton, 'b-o', linewidth=2, markersize=6,
                    label='Newton Method')
        
        ax6.set_xlabel('Iteration', fontsize=11)
        ax6.set_ylabel('Gradient Magnitude (log scale)', fontsize=11)
        ax6.set_title('Gradient Norm Reduction', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('first_vs_second_order.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print comparison table
        print("\n" + "=" * 70)
        print("FIRST-ORDER VS SECOND-ORDER METHODS COMPARISON")
        print("=" * 70)
        print(f"{'Method':<20} {'Iterations':<15} {'Final Value':<20} {'Final Distance':<15}")
        print("-" * 70)
        print(f"{'Gradient Descent':<20} {len(path_gd):<15} {values_gd[-1]:<20.10f} {dist_gd[-1]:<15.10f}")
        print(f"{'Newton Method':<20} {len(path_newton):<15} {values_newton[-1]:<20.10f} {dist_newton[-1]:<15.10f}")
        print("-" * 70)
        print(f"Speed-up: {len(path_gd) / len(path_newton):.1f}x faster")
        print("=" * 70)


# SYMBOLIC COMPUTATION WITH SYMPY


def symbolic_optimization_analysis():
    """Use SymPy for symbolic optimization analysis"""
    
    print("\n" + "=" * 70)
    print("SYMBOLIC OPTIMIZATION ANALYSIS WITH SYMPY")
    print("=" * 70)
    
    # Define symbols
    x, y = sp.symbols('x y', real=True)
    
    # Define function
    f = (x - 2)**2 + (y + 1)**2
    
    print("\nFunction: f(x,y) = (x-2)² + (y+1)²")
    print(f"SymPy representation: {f}")
    
    # Compute gradient symbolically
    grad_x = sp.diff(f, x)
    grad_y = sp.diff(f, y)
    
    print("\n1. GRADIENT (First Derivatives):")
    print(f"   ∂f/∂x = {grad_x}")
    print(f"   ∂f/∂y = {grad_y}")
    
    # Solve for critical points
    critical_points = sp.solve([grad_x, grad_y], [x, y])
    print(f"\n2. CRITICAL POINT:")
    print(f"   Setting gradient = 0: {critical_points}")
    
    # Compute Hessian symbolically
    hess_xx = sp.diff(grad_x, x)
    hess_xy = sp.diff(grad_x, y)
    hess_yy = sp.diff(grad_y, y)
    
    print("\n3. HESSIAN MATRIX (Second Derivatives):")
    print(f"   ∂²f/∂x² = {hess_xx}")
    print(f"   ∂²f/∂x∂y = {hess_xy}")
    print(f"   ∂²f/∂y² = {hess_yy}")
    
    # Create Hessian matrix
    H = sp.Matrix([
        [hess_xx, hess_xy],
        [hess_xy, hess_yy]
    ])
    
    print(f"\n   H = {H}")
    
    # Compute eigenvalues
    eigenvals = H.eigenvals()
    print(f"\n4. EIGENVALUES: {eigenvals}")
    
    # Classify critical point
    cp = critical_points[0]
    H_at_cp = H.subs([(x, cp[0]), (y, cp[1])])
    eigs = list(H_at_cp.eigenvals().keys())
    
    print("\n5. CLASSIFICATION:")
    if all(float(e) > 0 for e in eigs):
        print("   ✓ All eigenvalues positive → LOCAL MINIMUM")
    elif all(float(e) < 0 for e in eigs):
        print("   ✗ All eigenvalues negative → LOCAL MAXIMUM")
    else:
        print("   ⚠ Mixed eigenvalues → SADDLE POINT")
    
    print("\n6. FUNCTION VALUE AT CRITICAL POINT:")
    f_min = f.subs([(x, cp[0]), (y, cp[1])])
    print(f"   f({cp[0]}, {cp[1]}) = {f_min}")
    
    print("=" * 70)
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert to numerical function
    f_num = sp.lambdify((x, y), f, 'numpy')
    
    # Create mesh
    x_vals = np.linspace(-1, 5, 100)
    y_vals = np.linspace(-4, 2, 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = f_num(X, Y)
    
    # Plot 1: 3D surface
    ax1 = axes[0] if not hasattr(axes[0], 'plot_surface') else plt.subplot(121, projection='3d')
    if hasattr(ax1, 'plot_surface'):
        surf = ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
        ax1.scatter([float(cp[0])], [float(cp[1])], [float(f_min)],
                   color='red', s=200, marker='*', edgecolors='black', linewidths=2)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        ax1.set_title('Function Surface with Minimum')
    
    # Plot 2: Contour
    ax2 = axes[1] if len(axes) > 1 else axes[0]
    contour = ax2.contour(X, Y, Z, levels=20, cmap='coolwarm')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(float(cp[0]), float(cp[1]), 'r*', markersize=25,
            markeredgecolor='black', markeredgewidth=2, label='Minimum')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('Contour Plot', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('symbolic_optimization.png', dpi=300, bbox_inches='tight')
    plt.show()


# MAIN EXECUTION


if __name__ == "__main__":
    print("SECOND-ORDER OPTIMIZATION METHODS")
    print("=" * 70)
    
    # Part 1: Compare methods
    print("\nPart 1: Gradient Descent vs Newton's Method")
    optimizer = SecondOrderMethods()
    optimizer.compare_methods()
    
    # Part 2: Symbolic analysis
    print("\nPart 2: Symbolic Optimization with SymPy")
    symbolic_optimization_analysis()
    
    print("\n✓ Second-order methods analysis complete!")
    print("\nKEY INSIGHTS:")
    print("- Newton's method uses second-order information (Hessian)")
    print("- Much faster convergence near the optimum (quadratic vs linear)")
    print("- Requires computing and inverting Hessian (expensive)")
    print("- SymPy enables symbolic computation for exact analysis")