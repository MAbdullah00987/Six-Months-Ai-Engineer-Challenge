
#Part 3: Saddle Points Challenge
#Saddle Points and Optimization Challenges


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sns.set_style("whitegrid")


# SADDLE POINT FUNCTIONS


class SaddlePointAnalysis:
    """Analyze saddle points and their impact on optimization"""
    
    def saddle_function(self, x, y):
        """
        Classic saddle point function: f(x,y) = x² - y²
        Saddle point at (0, 0)
        """
        return x**2 - y**2
    
    def saddle_gradient(self, x, y):
        """Gradient: [2x, -2y]"""
        return np.array([2*x, -2*y])
    
    def monkey_saddle(self, x, y):
        """
        Monkey saddle: f(x,y) = x³ - 3xy²
        More complex saddle point at origin
        """
        return x**3 - 3*x*y**2
    
    def monkey_gradient(self, x, y):
        """Gradient of monkey saddle"""
        df_dx = 3*x**2 - 3*y**2
        df_dy = -6*x*y
        return np.array([df_dx, df_dy])
    
    def himmelblau_function(self, x, y):
        """
        Himmelblau's function - has 4 minima and 1 saddle point
        f(x,y) = (x² + y - 11)² + (x + y² - 7)²
        """
        return (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    def himmelblau_gradient(self, x, y):
        """Gradient of Himmelblau's function"""
        df_dx = 4*x*(x**2 + y - 11) + 2*(x + y**2 - 7)
        df_dy = 2*(x**2 + y - 11) + 4*y*(x + y**2 - 7)
        return np.array([df_dx, df_dy])
    
    def visualize_saddle_types(self):
        """Visualize different types of saddle points"""
        fig = plt.figure(figsize=(18, 12))
        
        x = np.linspace(-3, 3, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        # 1. Classic Saddle Point
        Z1 = self.saddle_function(X, Y)
        
        ax1 = fig.add_subplot(3, 3, 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, Z1, cmap='coolwarm', alpha=0.8)
        ax1.scatter([0], [0], [0], color='red', s=200, marker='*', 
                   edgecolors='black', linewidths=2)
        ax1.set_title('Classic Saddle: f(x,y) = x² - y²', 
                     fontsize=11, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('f(x,y)')
        
        ax2 = fig.add_subplot(3, 3, 2)
        contour1 = ax2.contour(X, Y, Z1, levels=20, cmap='coolwarm')
        ax2.clabel(contour1, inline=True, fontsize=8)
        ax2.plot(0, 0, 'r*', markersize=20, markeredgecolor='black', 
                markeredgewidth=2, label='Saddle Point')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Contour View')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cross-sections through saddle
        ax3 = fig.add_subplot(3, 3, 3)
        x_line = np.linspace(-3, 3, 100)
        ax3.plot(x_line, self.saddle_function(x_line, 0), 'b-', 
                linewidth=2, label='Along x (y=0): x²')
        ax3.plot(x_line, self.saddle_function(0, x_line), 'r-', 
                linewidth=2, label='Along y (x=0): -y²')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Function Value')
        ax3.set_title('Cross-Sections')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 2. Monkey Saddle
        Z2 = self.monkey_saddle(X, Y)
        
        ax4 = fig.add_subplot(3, 3, 4, projection='3d')
        surf2 = ax4.plot_surface(X, Y, Z2, cmap='viridis', alpha=0.8)
        ax4.scatter([0], [0], [0], color='red', s=200, marker='*',
                   edgecolors='black', linewidths=2)
        ax4.set_title('Monkey Saddle: f(x,y) = x³ - 3xy²', 
                     fontsize=11, fontweight='bold')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_zlabel('f(x,y)')
        
        ax5 = fig.add_subplot(3, 3, 5)
        contour2 = ax5.contour(X, Y, Z2, levels=20, cmap='viridis')
        ax5.clabel(contour2, inline=True, fontsize=8)
        ax5.plot(0, 0, 'r*', markersize=20, markeredgecolor='black',
                markeredgewidth=2, label='Saddle Point')
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        ax5.set_title('Contour View')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(3, 3, 6)
        ax6.plot(x_line, self.monkey_saddle(x_line, 0), 'b-',
                linewidth=2, label='Along x (y=0)')
        ax6.plot(x_line, self.monkey_saddle(0, x_line), 'r-',
                linewidth=2, label='Along y (x=0)')
        ax6.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax6.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Position')
        ax6.set_ylabel('Function Value')
        ax6.set_title('Cross-Sections')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 3. Himmelblau's Function
        x_h = np.linspace(-5, 5, 100)
        y_h = np.linspace(-5, 5, 100)
        X_h, Y_h = np.meshgrid(x_h, y_h)
        Z3 = self.himmelblau_function(X_h, Y_h)
        
        ax7 = fig.add_subplot(3, 3, 7, projection='3d')
        surf3 = ax7.plot_surface(X_h, Y_h, np.log1p(Z3), cmap='plasma', alpha=0.8)
        ax7.set_title('Himmelblau Function (log scale)', 
                     fontsize=11, fontweight='bold')
        ax7.set_xlabel('x')
        ax7.set_ylabel('y')
        ax7.set_zlabel('log(1+f(x,y))')
        
        ax8 = fig.add_subplot(3, 3, 8)
        contour3 = ax8.contour(X_h, Y_h, Z3, levels=30, cmap='plasma')
        ax8.clabel(contour3, inline=True, fontsize=8)
        
        # Mark known minima
        minima = [(3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)]
        for xm, ym in minima:
            ax8.plot(xm, ym, 'g*', markersize=15, markeredgecolor='black',
                    markeredgewidth=2)
        
        ax8.set_xlabel('x')
        ax8.set_ylabel('y')
        ax8.set_title('4 Minima + Saddle Points')
        ax8.grid(True, alpha=0.3)
        
        # Gradient magnitude heatmap
        ax9 = fig.add_subplot(3, 3, 9)
        grad_x = 2*X - 2*Y
        grad_y = -2*X + 2*Y
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        im = ax9.imshow(grad_magnitude, extent=[-3, 3, -3, 3], 
                       origin='lower', cmap='hot', aspect='auto')
        ax9.plot(0, 0, 'b*', markersize=20, markeredgecolor='white',
                markeredgewidth=2)
        plt.colorbar(im, ax=ax9)
        ax9.set_xlabel('x')
        ax9.set_ylabel('y')
        ax9.set_title('Gradient Magnitude\n(Zero at saddle)')
        
        plt.tight_layout()
        plt.savefig('saddle_point_types.png', dpi=300, bbox_inches='tight')
        plt.show()


# DEMONSTRATE STUCK AT SADDLE

def demonstrate_saddle_trap():
    """Show how gradient descent behaves near saddle points"""
    
    saddle = SaddlePointAnalysis()
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Create mesh
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = saddle.saddle_function(X, Y)
    
    # Different starting points near saddle
    start_points = [
        (0.1, 0.1, 'Near Saddle (+,+)'),
        (0.1, -0.1, 'Near Saddle (+,-)'),
        (-0.1, 0.1, 'Near Saddle (-,+)'),
        (-0.1, -0.1, 'Near Saddle (-,-)'),
        (0.01, 0.01, 'Very Close'),
        (1.0, 1.0, 'Far from Saddle'),
    ]
    
    for idx, (start_x, start_y, label) in enumerate(start_points):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Run gradient descent
        path = []
        current = np.array([start_x, start_y], dtype=float)
        path.append(current.copy())
        
        lr = 0.1
        for i in range(100):
            grad = saddle.saddle_gradient(current[0], current[1])
            
            # Add small noise to escape saddle
            if np.linalg.norm(grad) < 0.01:
                grad += np.random.normal(0, 0.001, 2)
            
            current = current - lr * grad
            path.append(current.copy())
            
            # Stop if diverging
            if np.abs(current[0]) > 5 or np.abs(current[1]) > 5:
                break
        
        path = np.array(path)
        
        # Plot
        contour = ax.contour(X, Y, Z, levels=20, cmap='coolwarm', alpha=0.6)
        ax.plot(path[:, 0], path[:, 1], 'b.-', linewidth=2, markersize=4)
        ax.plot(start_x, start_y, 'go', markersize=12, 
               markeredgecolor='black', markeredgewidth=2, label='Start')
        ax.plot(path[-1, 0], path[-1, 1], 'r*', markersize=15,
               markeredgecolor='black', markeredgewidth=2, label='End')
        ax.plot(0, 0, 'y*', markersize=20, markeredgecolor='black',
               markeredgewidth=2, label='Saddle')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{label}\n{len(path)} iterations')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        
        # Add text with final position
        final_dist = np.sqrt(path[-1, 0]**2 + path[-1, 1]**2)
        ax.text(0.02, 0.98, f'Final dist: {final_dist:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=9)
    
    plt.tight_layout()
    plt.savefig('saddle_point_behavior.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("SADDLE POINT BEHAVIOR ANALYSIS")
    print("=" * 70)
    print("Observations:")
    print("- Starting near a saddle point leads to very slow convergence")
    print("- The gradient approaches zero at the saddle")
    print("- Small numerical noise can help escape saddle points")
    print("- Direction of escape depends on the starting quadrant")
    print("=" * 70)

# HESSIAN ANALYSIS


def analyze_critical_points():
    """Analyze critical points using Hessian matrix"""
    
    print("\n" + "=" * 70)
    print("CRITICAL POINT ANALYSIS WITH HESSIAN")
    print("=" * 70)
    
    # For f(x,y) = x² - y²
    print("\n1. Classic Saddle f(x,y) = x² - y²")
    print("   Critical point: (0, 0)")
    print("   Gradient: [2x, -2y] → [0, 0] at origin")
    print("\n   Hessian matrix:")
    print("   H = | 2   0 |")
    print("       | 0  -2 |")
    print("\n   Eigenvalues: λ₁ = 2, λ₂ = -2")
    print("   → Mixed signs → SADDLE POINT")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Eigenvalue visualization
    ax1 = axes[0]
    eigenvalues = [2, -2]
    colors = ['green' if e > 0 else 'red' for e in eigenvalues]
    bars = ax1.bar(['λ₁', 'λ₂'], eigenvalues, color=colors, 
                   edgecolor='black', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Eigenvalue', fontsize=12)
    ax1.set_title('Hessian Eigenvalues\n(Mixed → Saddle)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # For minimum: f(x,y) = x² + y²
    ax2 = axes[1]
    eigenvalues_min = [2, 2]
    bars = ax2.bar(['λ₁', 'λ₂'], eigenvalues_min, color='green',
                   edgecolor='black', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('Minimum: f = x² + y²\n(Both + → Minimum)', 
                 fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    print("\n2. Local Minimum f(x,y) = x² + y²")
    print("   Hessian eigenvalues: λ₁ = 2, λ₂ = 2")
    print("   → Both positive → LOCAL MINIMUM")
    
    # For maximum: f(x,y) = -x² - y²
    ax3 = axes[2]
    eigenvalues_max = [-2, -2]
    bars = ax3.bar(['λ₁', 'λ₂'], eigenvalues_max, color='red',
                   edgecolor='black', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Eigenvalue', fontsize=12)
    ax3.set_title('Maximum: f = -x² - y²\n(Both - → Maximum)', 
                 fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    print("\n3. Local Maximum f(x,y) = -x² - y²")
    print("   Hessian eigenvalues: λ₁ = -2, λ₂ = -2")
    print("   → Both negative → LOCAL MAXIMUM")
    
    plt.tight_layout()
    plt.savefig('hessian_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "-" * 70)
    print("CLASSIFICATION RULE:")
    print("- Both eigenvalues positive → Local Minimum")
    print("- Both eigenvalues negative → Local Maximum")
    print("- Mixed sign eigenvalues → Saddle Point")
    print("- Zero eigenvalue → Inconclusive (need higher order)")
    print("=" * 70)

# MAIN EXECUTION

if __name__ == "__main__":
    print("SADDLE POINTS AND OPTIMIZATION CHALLENGES")
    print("=" * 70)
    
    # Part 1: Types of saddle points
    print("\nPart 1: Visualizing Different Saddle Point Types")
    saddle = SaddlePointAnalysis()
    saddle.visualize_saddle_types()
    
    # Part 2: Getting stuck at saddles
    print("\nPart 2: Demonstrating Saddle Point Traps")
    demonstrate_saddle_trap()
    
    # Part 3: Hessian analysis
    print("\nPart 3: Critical Point Classification")
    analyze_critical_points()
    
    print("\n✓ Saddle point analysis complete!")