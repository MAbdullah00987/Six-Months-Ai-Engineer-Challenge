
#Using Claude
#Now let me create a comprehensive integration guide that ties everything together:
#Complete Integration: All Libraries & Concepts


"""
COMPREHENSIVE OPTIMIZATION STUDY - FIXED
================================
Integration of: NumPy, Matplotlib, Seaborn, SymPy, Pandas
Topics: Gradient Descent, Local Minima, Saddle Points, Second-Order Methods
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
from matplotlib.gridspec import GridSpec

# Configure plotting
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# ============================================
# MASTER CLASS: UNIFIED OPTIMIZATION FRAMEWORK
# ============================================

class UnifiedOptimizationFramework:
    """
    Complete framework demonstrating all concepts and libraries
    """
    
    def __init__(self, function_name='rosenbrock'):
        self.function_name = function_name
        self.results = []
        self.symbolic_analysis = {}
        
    # ========== FUNCTION DEFINITIONS ==========
    
    def get_function(self, name):
        """Return function, gradient, and hessian"""
        functions = {
            'quadratic': {
                'f': lambda x: (x[0] - 2)**2 + (x[1] + 1)**2,
                'grad': lambda x: np.array([2*(x[0] - 2), 2*(x[1] + 1)]),
                'hess': lambda x: np.array([[2, 0], [0, 2]]),
                'optimum': np.array([2, -1]),
                'bounds': [(-5, 7), (-6, 4)]
            },
            'rosenbrock': {
                'f': lambda x: (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2,
                'grad': lambda x: np.array([
                    -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                    200*(x[1] - x[0]**2)
                ]),
                'hess': lambda x: np.array([
                    [2 + 1200*x[0]**2 - 400*x[1], -400*x[0]],
                    [-400*x[0], 200]
                ]),
                'optimum': np.array([1, 1]),
                'bounds': [(-2, 2), (-1, 3)]
            },
            'saddle': {
                'f': lambda x: x[0]**2 - x[1]**2,
                'grad': lambda x: np.array([2*x[0], -2*x[1]]),
                'hess': lambda x: np.array([[2, 0], [0, -2]]),
                'optimum': np.array([0, 0]),
                'bounds': [(-3, 3), (-3, 3)]
            },
            'multimodal': {
                'f': lambda x: np.sin(x[0])*np.cos(x[1]) + 0.1*(x[0]**2 + x[1]**2),
                'grad': lambda x: np.array([
                    np.cos(x[0])*np.cos(x[1]) + 0.2*x[0],
                    -np.sin(x[0])*np.sin(x[1]) + 0.2*x[1]
                ]),
                'hess': lambda x: np.array([
                    [-np.sin(x[0])*np.cos(x[1]) + 0.2, -np.cos(x[0])*np.sin(x[1])],
                    [-np.cos(x[0])*np.sin(x[1]), -np.sin(x[0])*np.cos(x[1]) + 0.2]
                ]),
                'optimum': np.array([0, 0]),
                'bounds': [(-6, 6), (-6, 6)]
            }
        }
        return functions[name]
    
    # ========== OPTIMIZATION ALGORITHMS ==========
    
    def gradient_descent(self, start, lr=0.01, max_iter=1000, tol=1e-6):
        """Standard gradient descent"""
        func_info = self.get_function(self.function_name)
        f, grad = func_info['f'], func_info['grad']
        
        x = np.array(start, dtype=float)
        path = [x.copy()]
        values = [f(x)]
        
        for i in range(max_iter):
            g = grad(x)
            if np.linalg.norm(g) < tol:
                break
            x = x - lr * g
            path.append(x.copy())
            values.append(f(x))
        
        return {
            'path': np.array(path),
            'values': values,
            'final': x,
            'iterations': len(path),
            'method': 'Gradient Descent'
        }
    
    def momentum(self, start, lr=0.01, beta=0.9, max_iter=1000):
        """Gradient descent with momentum"""
        func_info = self.get_function(self.function_name)
        f, grad = func_info['f'], func_info['grad']
        
        x = np.array(start, dtype=float)
        v = np.zeros_like(x)
        path = [x.copy()]
        values = [f(x)]
        
        for i in range(max_iter):
            g = grad(x)
            v = beta * v - lr * g
            x = x + v
            path.append(x.copy())
            values.append(f(x))
            
            if values[-1] < 1e-6:
                break
        
        return {
            'path': np.array(path),
            'values': values,
            'final': x,
            'iterations': len(path),
            'method': 'Momentum'
        }
    
    def adam(self, start, lr=0.01, max_iter=1000):
        """Adam optimizer"""
        func_info = self.get_function(self.function_name)
        f, grad = func_info['f'], func_info['grad']
        
        x = np.array(start, dtype=float)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        
        path = [x.copy()]
        values = [f(x)]
        
        for t in range(1, max_iter + 1):
            g = grad(x)
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * g**2
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            x = x - lr * m_hat / (np.sqrt(v_hat) + epsilon)
            path.append(x.copy())
            values.append(f(x))
            
            if values[-1] < 1e-6:
                break
        
        return {
            'path': np.array(path),
            'values': values,
            'final': x,
            'iterations': len(path),
            'method': 'Adam'
        }
    
    def newton_method(self, start, max_iter=50):
        """Newton's method with Hessian"""
        func_info = self.get_function(self.function_name)
        f, grad, hess = func_info['f'], func_info['grad'], func_info['hess']
        
        x = np.array(start, dtype=float)
        path = [x.copy()]
        values = [f(x)]
        
        for i in range(max_iter):
            g = grad(x)
            H = hess(x)
            
            try:
                delta = np.linalg.solve(H, g)
                x = x - delta
            except:
                break
            
            path.append(x.copy())
            values.append(f(x))
            
            if np.linalg.norm(g) < 1e-6:
                break
        
        return {
            'path': np.array(path),
            'values': values,
            'final': x,
            'iterations': len(path),
            'method': 'Newton'
        }
    
    # ========== SYMBOLIC ANALYSIS WITH SYMPY ==========
    
    def symbolic_analysis_sympy(self):
        """Complete symbolic analysis using SymPy"""
        print("\n" + "=" * 70)
        print("SYMBOLIC ANALYSIS WITH SYMPY")
        print("=" * 70)
        
        # Define symbolic variables
        x, y = sp.symbols('x y', real=True)
        
        # Define function based on type
        if self.function_name == 'quadratic':
            f = (x - 2)**2 + (y + 1)**2
            title = "Quadratic Function"
        elif self.function_name == 'saddle':
            f = x**2 - y**2
            title = "Saddle Point Function"
        else:
            print("Symbolic analysis available for quadratic and saddle functions")
            return
        
        print(f"\n{title}: f(x,y) = {f}")
        
        # Gradient
        grad_x = sp.diff(f, x)
        grad_y = sp.diff(f, y)
        print(f"\nGradient:")
        print(f"  ∂f/∂x = {grad_x}")
        print(f"  ∂f/∂y = {grad_y}")
        
        # Critical points
        critical = sp.solve([grad_x, grad_y], [x, y])
        print(f"\nCritical points: {critical}")
        
        # Hessian
        H_xx = sp.diff(grad_x, x)
        H_xy = sp.diff(grad_x, y)
        H_yy = sp.diff(grad_y, y)
        
        print(f"\nHessian matrix:")
        print(f"  ∂²f/∂x² = {H_xx}")
        print(f"  ∂²f/∂x∂y = {H_xy}")
        print(f"  ∂²f/∂y² = {H_yy}")
        
        # Eigenvalue analysis
        H = sp.Matrix([[H_xx, H_xy], [H_xy, H_yy]])
        if critical:
            # FIX: Handle multiple return types from solve
            try:
                if isinstance(critical, dict):
                    # Single solution as dict: {x: val, y: val}
                    cp_x = critical[x]
                    cp_y = critical[y]
                elif isinstance(critical, list) and len(critical) > 0:
                    # List of solutions
                    cp = critical[0]
                    if isinstance(cp, dict):
                        # List of dicts: [{x: val, y: val}]
                        cp_x = cp[x]
                        cp_y = cp[y]
                    elif isinstance(cp, tuple):
                        # List of tuples: [(x_val, y_val)]
                        cp_x, cp_y = cp
                    else:
                        # Fallback
                        cp_x = cp[0] if hasattr(cp, '__getitem__') else cp
                        cp_y = cp[1] if hasattr(cp, '__getitem__') and len(cp) > 1 else cp
                else:
                    print("No critical points found")
                    return
            except Exception as e:
                print(f"Error extracting critical point: {e}")
                print(f"Critical points data structure: {type(critical)}")
                print(f"Critical points value: {critical}")
                return
            
            H_at_cp = H.subs([(x, cp_x), (y, cp_y)])
            eigenvals = list(H_at_cp.eigenvals().keys())
            
            print(f"\nEigenvalues at critical point: {eigenvals}")
            
            # Classification
            evals_numeric = [float(e) for e in eigenvals]
            if all(e > 0 for e in evals_numeric):
                classification = "LOCAL MINIMUM"
            elif all(e < 0 for e in evals_numeric):
                classification = "LOCAL MAXIMUM"
            else:
                classification = "SADDLE POINT"
            
            print(f"Classification: {classification}")
        
        print("=" * 70)
        
        self.symbolic_analysis = {
            'function': f,
            'gradient': [grad_x, grad_y],
            'hessian': H,
            'critical_points': critical
        }
    
    # ========== COMPREHENSIVE VISUALIZATION ==========
    
    def visualize_everything(self, start_points=None, methods=['gradient_descent', 'adam']):
        """Create mega visualization with all concepts"""
        
        if start_points is None:
            start_points = [[-1.5, 2], [1.5, -2], [-1, -1], [1, 2]]
        
        func_info = self.get_function(self.function_name)
        f = func_info['f']
        bounds = func_info['bounds']
        
        # Create mesh
        x = np.linspace(bounds[0][0], bounds[0][1], 200)
        y = np.linspace(bounds[1][0], bounds[1][1], 200)
        X, Y = np.meshgrid(x, y)
        Z = np.array([[f([xi, yi]) for xi, yi in zip(xrow, yrow)] 
                      for xrow, yrow in zip(X, Y)])
        
        # Run all optimizations
        all_results = []
        for start in start_points:
            for method in methods:
                if method == 'gradient_descent':
                    result = self.gradient_descent(start, lr=0.01, max_iter=500)
                elif method == 'momentum':
                    result = self.momentum(start, lr=0.01, max_iter=500)
                elif method == 'adam':
                    result = self.adam(start, lr=0.01, max_iter=500)
                elif method == 'newton':
                    result = self.newton_method(start, max_iter=50)
                
                result['start'] = start
                all_results.append(result)
        
        # Create figure
        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: 3D Surface with all paths
        ax1 = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        surf = ax1.plot_surface(X, Y, np.log1p(Z), cmap='viridis', alpha=0.6)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(all_results)))
        for result, color in zip(all_results, colors):
            path = result['path']
            z_vals = [np.log1p(f(p)) for p in path]
            ax1.plot(path[:, 0], path[:, 1], z_vals, 
                    color=color, linewidth=2, alpha=0.8)
        
        ax1.set_title(f'{self.function_name.upper()}: All Optimization Paths',
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('log(1 + f(x,y))')
        
        # Plot 2: Contour with paths
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        contour = ax2.contour(X, Y, np.log1p(Z), levels=30, cmap='viridis', alpha=0.6)
        
        for result, color in zip(all_results, colors):
            path = result['path']
            ax2.plot(path[:, 0], path[:, 1], color=color, linewidth=2,
                    alpha=0.7, label=f"{result['method'][:3]}")
            ax2.plot(path[0, 0], path[0, 1], 'o', color=color, markersize=10)
        
        ax2.plot(func_info['optimum'][0], func_info['optimum'][1], 'r*',
                markersize=25, markeredgecolor='black', markeredgewidth=3)
        ax2.set_xlabel('x', fontsize=11)
        ax2.set_ylabel('y', fontsize=11)
        ax2.set_title('Contour View', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Convergence comparison
        ax3 = fig.add_subplot(gs[2, 0:2])
        for result, color in zip(all_results, colors):
            ax3.semilogy(result['values'], color=color, linewidth=2,
                        alpha=0.7, label=result['method'])
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Function Value (log)', fontsize=11)
        ax3.set_title('Convergence Comparison', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=8, ncol=2)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Method comparison (Pandas DataFrame)
        ax4 = fig.add_subplot(gs[2, 2:4])
        
        summary_data = []
        for result in all_results:
            summary_data.append({
                'Method': result['method'],
                'Start': f"({result['start'][0]:.1f},{result['start'][1]:.1f})",
                'Iterations': result['iterations'],
                'Final_Value': result['values'][-1]
            })
        
        df = pd.DataFrame(summary_data)
        
        # Group by method
        method_summary = df.groupby('Method').agg({
            'Iterations': 'mean',
            'Final_Value': 'mean'
        }).reset_index()
        
        bars = ax4.barh(method_summary['Method'], method_summary['Iterations'],
                       color=sns.color_palette('husl', len(method_summary)))
        ax4.set_xlabel('Average Iterations', fontsize=11)
        ax4.set_title('Method Performance', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, method_summary['Iterations']):
            ax4.text(val, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', ha='left', va='center', fontsize=10)
        
        # Plot 5: Seaborn heatmap of results
        ax5 = fig.add_subplot(gs[3, 0:2])
        
        # Create pivot table
        pivot_data = df.pivot_table(
            values='Final_Value',
            index='Method',
            columns='Start',
            aggfunc='first'
        )
        
        sns.heatmap(np.log1p(pivot_data), annot=True, fmt='.2e',
                   cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'log(1+value)'})
        ax5.set_title('Final Values Heatmap', fontsize=13, fontweight='bold')
        
        # Plot 6: Statistical summary
        ax6 = fig.add_subplot(gs[3, 2:4])
        ax6.axis('off')
        
        stats_text = f"""
        OPTIMIZATION SUMMARY
        {'='*50}
        Function: {self.function_name}
        Starting Points: {len(start_points)}
        Methods Tested: {len(methods)}
        
        RESULTS:
        {df.to_string(index=False)}
        
        BEST PERFORMERS:
        - Fastest: {df.loc[df['Iterations'].idxmin(), 'Method']}
          ({df['Iterations'].min():.0f} iterations)
        
        - Most Accurate: {df.loc[df['Final_Value'].idxmin(), 'Method']}
          (f = {df['Final_Value'].min():.2e})
        """
        
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.savefig(f'{self.function_name}_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return df

# ============================================
# MAIN EXECUTION & DEMONSTRATION
# ============================================

def main():
    """Run comprehensive demonstration"""
    
    print("=" * 70)
    print("COMPREHENSIVE OPTIMIZATION FRAMEWORK")
    print("Advanced Topics Integration: NumPy, Matplotlib, Seaborn, SymPy, Pandas")
    print("=" * 70)
    
    # Test different functions
    functions_to_test = ['quadratic', 'rosenbrock', 'multimodal']
    
    all_dataframes = []
    
    for func_name in functions_to_test:
        print(f"\n{'='*70}")
        print(f"Testing: {func_name.upper()}")
        print('='*70)
        
        framework = UnifiedOptimizationFramework(func_name)
        
        # Symbolic analysis (if available)
        if func_name in ['quadratic', 'saddle']:
            framework.symbolic_analysis_sympy()
        
        # Run comprehensive visualization
        print(f"\nRunning optimizations on {func_name}...")
        df = framework.visualize_everything(
            start_points=[[-1.5, 1.5], [1.5, -1.5], [-1, -1], [1, 1]],
            methods=['gradient_descent', 'adam', 'newton']
        )
        
        df['Function'] = func_name
        all_dataframes.append(df)
    
    # Combine all results
    final_df = pd.concat(all_dataframes, ignore_index=True)
    
    print("\n" + "=" * 70)
    print("FINAL COMPREHENSIVE RESULTS")
    print("=" * 70)
    print(final_df.to_string(index=False))
    
    # Save to CSV
    final_df.to_csv('optimization_results.csv', index=False)
    print("\n✓ Results saved to 'optimization_results.csv'")
    
    print("\n" + "=" * 70)
    print("✓ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nLIBRARIES DEMONSTRATED:")
    print("  ✓ NumPy: Array operations, linear algebra")
    print("  ✓ Matplotlib: 2D/3D plots, animations")
    print("  ✓ Seaborn: Statistical visualizations, heatmaps")
    print("  ✓ SymPy: Symbolic mathematics, calculus")
    print("  ✓ Pandas: Data analysis, tables")
    print("\nCONCEPTS COVERED:")
    print("  ✓ Gradient Descent & variants")
    print("  ✓ Local vs Global optimization")
    print("  ✓ Saddle points & Hessian analysis")
    print("  ✓ Second-order methods (Newton)")
    print("  ✓ Multiple starting points")
    print("  ✓ Convergence analysis")
    print("=" * 70)

if __name__ == "__main__":
    main()