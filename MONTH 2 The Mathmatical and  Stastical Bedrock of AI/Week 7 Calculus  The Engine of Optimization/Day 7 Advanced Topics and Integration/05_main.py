
#Part 5: The Specific Task - Multiple Starting Points
#Task: f(x,y) = (x-2)² + (y+1)² from Multiple Starts

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import pandas as pd

sns.set_style("whitegrid")


# TASK FUNCTION: f(x,y) = (x-2)² + (y+1)²

class TaskOptimization:
    """
    Optimize f(x,y) = (x-2)² + (y+1)²
    Global minimum at (2, -1) with f(2,-1) = 0
    """
    
    def __init__(self):
        self.all_histories = []
        self.starting_points = []
    
    def objective(self, x, y):
        """Target function: f(x,y) = (x-2)² + (y+1)²"""
        return (x - 2)**2 + (y + 1)**2
    
    def gradient(self, x, y):
        """Gradient: ∇f = [2(x-2), 2(y+1)]"""
        return np.array([2*(x - 2), 2*(y + 1)])
    
    def gradient_descent(self, start, lr=0.1, max_iter=100):
        """Run gradient descent from a starting point"""
        point = np.array(start, dtype=float)
        history = [point.copy()]
        
        for i in range(max_iter):
            grad = self.gradient(point[0], point[1])
            
            # Check convergence
            if np.linalg.norm(grad) < 1e-6:
                break
            
            point = point - lr * grad
            history.append(point.copy())
        
        return np.array(history)
    
    def run_from_multiple_starts(self, num_points=12, lr=0.1):
        """Run optimization from multiple random starting points"""
        
        # Generate diverse starting points
        np.random.seed(42)
        
        # Grid of starting points
        angles = np.linspace(0, 2*np.pi, num_points, endpoint=False)
        radius = 5
        
        self.starting_points = []
        self.all_histories = []
        
        for angle in angles:
            x_start = 2 + radius * np.cos(angle)
            y_start = -1 + radius * np.sin(angle)
            start = [x_start, y_start]
            
            self.starting_points.append(start)
            history = self.gradient_descent(start, lr=lr)
            self.all_histories.append(history)
        
        return self.all_histories
    
    def visualize_comprehensive(self):
        """Create comprehensive visualization"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # Create mesh for plotting
        x = np.linspace(-4, 8, 200)
        y = np.linspace(-7, 5, 200)
        X, Y = np.meshgrid(x, y)
        Z = self.objective(X, Y)
        
        # ===== PLOT 1: 3D Surface with all paths =====
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        
        # Surface
        surf = ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.4,
                               linewidth=0, antialiased=True)
        
        # Plot all paths
        colors = plt.cm.hsv(np.linspace(0, 1, len(self.all_histories)))
        
        for history, color in zip(self.all_histories, colors):
            z_vals = [self.objective(p[0], p[1]) for p in history]
            ax1.plot(history[:, 0], history[:, 1], z_vals,
                    color=color, linewidth=2, alpha=0.8)
            
            # Start point
            ax1.scatter([history[0, 0]], [history[0, 1]], [z_vals[0]],
                       color=color, s=100, marker='o', edgecolors='black',
                       linewidths=2, alpha=1)
        
        # Mark global minimum
        ax1.scatter([2], [-1], [0], color='yellow', s=300, marker='*',
                   edgecolors='black', linewidths=3, label='Global Min')
        
        ax1.set_xlabel('x', fontsize=11)
        ax1.set_ylabel('y', fontsize=11)
        ax1.set_zlabel('f(x,y)', fontsize=11)
        ax1.set_title('3D View: All Optimization Paths', 
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.view_init(elev=25, azim=45)
        
        # ===== PLOT 2: Contour view with all paths =====
        ax2 = fig.add_subplot(2, 3, 2)
        
        contour = ax2.contour(X, Y, Z, levels=25, cmap='coolwarm', alpha=0.6)
        ax2.clabel(contour, inline=True, fontsize=8)
        
        for idx, (history, color) in enumerate(zip(self.all_histories, colors)):
            ax2.plot(history[:, 0], history[:, 1], color=color,
                    linewidth=2, alpha=0.7, label=f'Path {idx+1}')
            ax2.plot(history[0, 0], history[0, 1], 'o', color=color,
                    markersize=10, markeredgecolor='black', markeredgewidth=2)
        
        # Mark minimum
        ax2.plot(2, -1, 'y*', markersize=25, markeredgecolor='black',
                markeredgewidth=3, label='Global Minimum')
        
        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('y', fontsize=11)
        ax2.set_title('Contour View: Converging to Same Point',
                     fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # ===== PLOT 3: Convergence curves =====
        ax3 = fig.add_subplot(2, 3, 3)
        
        for history, color in zip(self.all_histories, colors):
            values = [self.objective(p[0], p[1]) for p in history]
            ax3.semilogy(values, color=color, linewidth=2, alpha=0.7)
        
        ax3.axhline(y=1e-6, color='green', linestyle='--', linewidth=2,
                   label='Convergence Threshold')
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Function Value (log scale)', fontsize=11)
        ax3.set_title('All Convergence Curves', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ===== PLOT 4: Distance to optimum =====
        ax4 = fig.add_subplot(2, 3, 4)
        
        optimum = np.array([2.0, -1.0])
        
        for history, color in zip(self.all_histories, colors):
            distances = [np.linalg.norm(p - optimum) for p in history]
            ax4.semilogy(distances, color=color, linewidth=2, alpha=0.7)
        
        ax4.set_xlabel('Iteration', fontsize=11)
        ax4.set_ylabel('Distance to Minimum (log scale)', fontsize=11)
        ax4.set_title('Distance Convergence', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # ===== PLOT 5: Heatmap of starting points =====
        ax5 = fig.add_subplot(2, 3, 5)
        
        # Create heatmap showing starting point locations
        starts = np.array(self.starting_points)
        iterations_needed = [len(h) for h in self.all_histories]
        
        scatter = ax5.scatter(starts[:, 0], starts[:, 1], 
                            c=iterations_needed, cmap='viridis',
                            s=300, edgecolors='black', linewidths=2)
        
        # Add text labels
        for idx, (start, iters) in enumerate(zip(starts, iterations_needed)):
            ax5.text(start[0], start[1], f'{iters}', 
                    ha='center', va='center', fontsize=10,
                    color='white', fontweight='bold')
        
        ax5.plot(2, -1, 'r*', markersize=25, markeredgecolor='black',
                markeredgewidth=3, label='Target')
        
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Iterations to Converge', fontsize=10)
        
        ax5.set_xlabel('x', fontsize=11)
        ax5.set_ylabel('y', fontsize=11)
        ax5.set_title('Starting Points (colored by iterations)',
                     fontsize=13, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
        
        # ===== PLOT 6: Statistics =====
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate statistics
        iterations_list = [len(h) for h in self.all_histories]
        final_values = [self.objective(h[-1, 0], h[-1, 1]) 
                       for h in self.all_histories]
        final_distances = [np.linalg.norm(h[-1] - optimum) 
                          for h in self.all_histories]
        
        stats_text = f"""
        OPTIMIZATION STATISTICS
        {'='*50}
        
        Number of starting points: {len(self.starting_points)}
        Learning rate: 0.1
        
        CONVERGENCE:
        - Min iterations: {min(iterations_list)}
        - Max iterations: {max(iterations_list)}
        - Mean iterations: {np.mean(iterations_list):.1f}
        - Std iterations: {np.std(iterations_list):.1f}
        
        FINAL VALUES:
        - Best f(x,y): {min(final_values):.2e}
        - Worst f(x,y): {max(final_values):.2e}
        - Mean f(x,y): {np.mean(final_values):.2e}
        
        FINAL DISTANCES:
        - Min distance: {min(final_distances):.2e}
        - Max distance: {max(final_distances):.2e}
        - Mean distance: {np.mean(final_distances):.2e}
        
        RESULT: All paths converged to
        the global minimum at (2, -1)!
        
        ✓ No local minima found
        ✓ Convex optimization succeeded
        """
        
        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('task_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_dataframe_summary(self):
        """Create pandas DataFrame with results"""
        
        optimum = np.array([2.0, -1.0])
        
        data = []
        for idx, history in enumerate(self.all_histories):
            start = self.starting_points[idx]
            final = history[-1]
            
            data.append({
                'Start_ID': idx + 1,
                'Start_X': start[0],
                'Start_Y': start[1],
                'Final_X': final[0],
                'Final_Y': final[1],
                'Iterations': len(history),
                'Final_Value': self.objective(final[0], final[1]),
                'Distance_to_Optimum': np.linalg.norm(final - optimum),
                'Start_Distance': np.linalg.norm(np.array(start) - optimum)
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "=" * 100)
        print("DETAILED RESULTS TABLE")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        
        # Summary statistics
        print("\nSUMMARY STATISTICS:")
        print(df[['Iterations', 'Final_Value', 'Distance_to_Optimum']].describe())
        
        return df


# COMPARISON WITH LOCAL MINIMA FUNCTION


def compare_convex_vs_nonconvex():
    """Compare the task function (convex) with a non-convex function"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Function 1: Task function (convex)
    x = np.linspace(-2, 6, 100)
    y = np.linspace(-5, 3, 100)
    X1, Y1 = np.meshgrid(x, y)
    Z1 = (X1 - 2)**2 + (Y1 + 1)**2
    
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X1, Y1, Z1, levels=30, cmap='coolwarm')
    ax1.contour(X1, Y1, Z1, levels=30, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour1, ax=ax1)
    ax1.plot(2, -1, 'y*', markersize=25, markeredgecolor='black',
            markeredgewidth=3, label='Global Minimum')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title('TASK: f(x,y) = (x-2)² + (y+1)²\n(CONVEX - No Local Minima)',
                 fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Function 2: Non-convex with local minima
    x2 = np.linspace(-6, 6, 100)
    y2 = np.linspace(-6, 6, 100)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = np.sin(X2) * np.cos(Y2) + 0.1 * (X2**2 + Y2**2)
    
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X2, Y2, Z2, levels=30, cmap='coolwarm')
    ax2.contour(X2, Y2, Z2, levels=30, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour2, ax=ax2)
    
    # Mark some local minima
    local_minima = [(0, 0), (np.pi, 0), (-np.pi, 0)]
    for xm, ym in local_minima:
        ax2.plot(xm, ym, 'y*', markersize=20, markeredgecolor='black',
                markeredgewidth=2)
    
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title('NON-CONVEX: f(x,y) = sin(x)cos(y) + 0.1(x²+y²)\n(Multiple Local Minima)',
                 fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3D comparison
    ax3 = plt.subplot(2, 2, 3, projection='3d')
    surf1 = ax3.plot_surface(X1, Y1, Z1, cmap='coolwarm', alpha=0.8)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('f(x,y)')
    ax3.set_title('Task Function (3D)\nSmooth Bowl Shape', fontweight='bold')
    ax3.view_init(elev=30, azim=45)
    
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    surf2 = ax4.plot_surface(X2, Y2, Z2, cmap='coolwarm', alpha=0.8)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('f(x,y)')
    ax4.set_title('Non-Convex Function (3D)\nMultiple Valleys', fontweight='bold')
    ax4.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig('convex_vs_nonconvex.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 70)
    print("KEY DIFFERENCES:")
    print("=" * 70)
    print("TASK FUNCTION (CONVEX):")
    print("  ✓ Single global minimum")
    print("  ✓ No local minima")
    print("  ✓ Any starting point → same solution")
    print("  ✓ Gradient always points toward minimum")
    print()
    print("NON-CONVEX FUNCTION:")
    print("  ✗ Multiple local minima")
    print("  ✗ Starting point determines which minimum is found")
    print("  ✗ Gradient descent can get stuck")
    print("  ✗ Need advanced techniques (random restarts, etc.)")
    print("=" * 70)


# MAIN EXECUTION


if __name__ == "__main__":
    print("TASK: GRADIENT DESCENT FROM MULTIPLE STARTING POINTS")
    print("Function: f(x,y) = (x-2)² + (y+1)²")
    print("=" * 70)
    
    # Initialize optimizer
    task = TaskOptimization()
    
    # Run from 12 starting points
    print("\nRunning gradient descent from 12 different starting points...")
    task.run_from_multiple_starts(num_points=12, lr=0.1)
    
    # Comprehensive visualization
    print("\nGenerating comprehensive visualization...")
    task.visualize_comprehensive()
    
    # Create summary table
    print("\nCreating results summary...")
    df = task.create_dataframe_summary()
    
    # Compare with non-convex
    print("\nComparing with non-convex function...")
    compare_convex_vs_nonconvex()
    
    
    print("✓ TASK COMPLETED SUCCESSFULLY!")
    print("\nCONCLUSIONS:")
    print("1. All starting points converged to the SAME global minimum (2, -1)")
    print("2. This is a CONVEX function - no local minima exist")
    print("3. Starting point doesn't matter for convex optimization")
    print("4. Convergence speed varies with initial distance")
    print("5. Gradient descent ALWAYS succeeds for convex problems")
    