#Project 2: Stochastic Gradient Descent (SGD) (1.5 hours)

#Implement basic SGD (update with single random sample per iteration)
#Compare with batch gradient descent on same dataset
#Visualize convergence paths side by side
#Analyze trade-offs (speed vs. stability)

#Deliverable: Two implementations with comparative analysis and visualizations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Set random seed for reproducibility
np.random.seed(42)

# ==================== DATA GENERATION ====================
def generate_data(n_samples=200, slope=3, intercept=2, noise_std=2.0):
    """Generate synthetic 2D data: y = 3x + 2 + noise"""
    X = np.random.uniform(-10, 10, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = slope * X + intercept + noise
    return X, y

# ==================== BATCH GRADIENT DESCENT ====================
class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.slope = 0
        self.intercept = 0
        self.history = {
            'slope': [],
            'intercept': [],
            'loss': [],
            'time': []
        }
    
    def mse_loss(self, X, y):
        """Calculate Mean Squared Error loss"""
        y_pred = self.slope * X + self.intercept
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def compute_gradients(self, X, y):
        """Compute gradients using ALL data points (batch)"""
        n = len(X)
        y_pred = self.slope * X + self.intercept
        error = y_pred - y
        
        grad_slope = (2/n) * np.sum(error * X)
        grad_intercept = (2/n) * np.sum(error)
        
        return grad_slope, grad_intercept
    
    def fit(self, X, y):
        """Train using Batch Gradient Descent"""
        # Initialize parameters
        self.slope = np.random.randn()
        self.intercept = np.random.randn()
        
        start_time = time.time()
        
        for i in range(self.n_iterations):
            # Compute gradients using ALL samples
            grad_slope, grad_intercept = self.compute_gradients(X, y)
            
            # Update parameters
            self.slope -= self.lr * grad_slope
            self.intercept -= self.lr * grad_intercept
            
            # Record history
            loss = self.mse_loss(X, y)
            current_time = time.time() - start_time
            self.history['slope'].append(self.slope)
            self.history['intercept'].append(self.intercept)
            self.history['loss'].append(loss)
            self.history['time'].append(current_time)
            
            if (i + 1) % 200 == 0:
                print(f"  Iter {i+1}/{self.n_iterations} - Loss: {loss:.4f}")
        
        return time.time() - start_time
    
    def predict(self, X):
        """Make predictions"""
        return self.slope * X + self.intercept

# ==================== STOCHASTIC GRADIENT DESCENT ====================
class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.slope = 0
        self.intercept = 0
        self.history = {
            'slope': [],
            'intercept': [],
            'loss': [],
            'time': []
        }
    
    def mse_loss(self, X, y):
        """Calculate Mean Squared Error loss"""
        y_pred = self.slope * X + self.intercept
        loss = np.mean((y_pred - y) ** 2)
        return loss
    
    def compute_gradients_single(self, x_i, y_i):
        """Compute gradients using a SINGLE random sample"""
        y_pred = self.slope * x_i + self.intercept
        error = y_pred - y_i
        
        grad_slope = 2 * error * x_i
        grad_intercept = 2 * error
        
        return grad_slope, grad_intercept
    
    def fit(self, X, y):
        """Train using Stochastic Gradient Descent"""
        # Initialize parameters
        self.slope = np.random.randn()
        self.intercept = np.random.randn()
        
        start_time = time.time()
        n_samples = len(X)
        
        for i in range(self.n_iterations):
            # Pick a random sample
            random_idx = np.random.randint(0, n_samples)
            x_i = X[random_idx]
            y_i = y[random_idx]
            
            # Compute gradients using ONLY this sample
            grad_slope, grad_intercept = self.compute_gradients_single(x_i, y_i)
            
            # Update parameters
            self.slope -= self.lr * grad_slope
            self.intercept -= self.lr * grad_intercept
            
            # Record history (compute loss on full dataset for comparison)
            loss = self.mse_loss(X, y)
            current_time = time.time() - start_time
            self.history['slope'].append(self.slope)
            self.history['intercept'].append(self.intercept)
            self.history['loss'].append(loss)
            self.history['time'].append(current_time)
            
            if (i + 1) % 200 == 0:
                print(f"  Iter {i+1}/{self.n_iterations} - Loss: {loss:.4f}")
        
        return time.time() - start_time
    
    def predict(self, X):
        """Make predictions"""
        return self.slope * X + self.intercept

# ==================== VISUALIZATION ====================
def visualize_comparison(X, y, batch_model, sgd_model):
    """Create comprehensive comparison visualizations"""
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Loss Convergence Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(batch_model.history['loss'], label='Batch GD', linewidth=2, color='blue', alpha=0.8)
    ax1.plot(sgd_model.history['loss'], label='SGD', linewidth=1.5, color='red', alpha=0.6)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('MSE Loss', fontsize=12)
    ax1.set_title('Loss Convergence: Batch GD vs SGD', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Convergence (Log Scale) - Better for seeing SGD noise
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(batch_model.history['loss'], label='Batch GD', linewidth=2, color='blue', alpha=0.8)
    ax2.plot(sgd_model.history['loss'], label='SGD', linewidth=1.5, color='red', alpha=0.6)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('MSE Loss (log scale)', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Loss Convergence (Log Scale)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 3. Slope Convergence Path
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(batch_model.history['slope'], label='Batch GD', linewidth=2, color='blue', alpha=0.8)
    ax3.plot(sgd_model.history['slope'], label='SGD', linewidth=1.5, color='red', alpha=0.6)
    ax3.axhline(y=3, color='orange', linestyle='--', linewidth=2, label='True slope (3)')
    ax3.set_xlabel('Iteration', fontsize=12)
    ax3.set_ylabel('Slope Parameter', fontsize=12)
    ax3.set_title('Slope Parameter Convergence', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Intercept Convergence Path
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(batch_model.history['intercept'], label='Batch GD', linewidth=2, color='blue', alpha=0.8)
    ax4.plot(sgd_model.history['intercept'], label='SGD', linewidth=1.5, color='red', alpha=0.6)
    ax4.axhline(y=2, color='orange', linestyle='--', linewidth=2, label='True intercept (2)')
    ax4.set_xlabel('Iteration', fontsize=12)
    ax4.set_ylabel('Intercept Parameter', fontsize=12)
    ax4.set_title('Intercept Parameter Convergence', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    # 5. Parameter Space Trajectory (2D path)
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(batch_model.history['slope'], batch_model.history['intercept'], 
             'b-', linewidth=2, alpha=0.8, label='Batch GD')
    ax5.plot(sgd_model.history['slope'], sgd_model.history['intercept'], 
             'r-', linewidth=1.5, alpha=0.6, label='SGD')
    ax5.scatter([3], [2], color='orange', s=200, marker='*', 
                label='True parameters', zorder=5, edgecolors='black', linewidth=2)
    ax5.scatter([batch_model.history['slope'][0]], [batch_model.history['intercept'][0]], 
                color='green', s=100, marker='o', label='Start', zorder=5)
    ax5.set_xlabel('Slope', fontsize=12)
    ax5.set_ylabel('Intercept', fontsize=12)
    ax5.set_title('Parameter Space Trajectory', fontsize=14, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # 6. Final Fits Comparison
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(X, y, alpha=0.4, s=30, color='gray', label='Data')
    X_line = np.linspace(X.min(), X.max(), 100)
    
    # True line
    y_true = 3 * X_line + 2
    ax6.plot(X_line, y_true, 'orange', linestyle='--', linewidth=2.5, 
             label=f'True: y=3.0x+2.0', alpha=0.8)
    
    # Batch GD line
    y_batch = batch_model.predict(X_line)
    ax6.plot(X_line, y_batch, 'blue', linewidth=2.5, 
             label=f'Batch: y={batch_model.slope:.2f}x+{batch_model.intercept:.2f}')
    
    # SGD line
    y_sgd = sgd_model.predict(X_line)
    ax6.plot(X_line, y_sgd, 'red', linewidth=2.5, 
             label=f'SGD: y={sgd_model.slope:.2f}x+{sgd_model.intercept:.2f}')
    
    ax6.set_xlabel('X', fontsize=12)
    ax6.set_ylabel('y', fontsize=12)
    ax6.set_title('Final Model Fits Comparison', fontsize=14, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sgd_vs_batch_comparison.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'sgd_vs_batch_comparison.png'")
    plt.show()

# ==================== ANALYSIS ====================
def analyze_results(X, y, batch_model, sgd_model, batch_time, sgd_time):
    """Perform detailed comparative analysis"""
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: BATCH GD vs SGD")
    print("="*70)
    
    # Calculate metrics
    y_pred_batch = batch_model.predict(X)
    y_pred_sgd = sgd_model.predict(X)
    
    # R² scores
    ss_res_batch = np.sum((y - y_pred_batch) ** 2)
    ss_res_sgd = np.sum((y - y_pred_sgd) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2_batch = 1 - (ss_res_batch / ss_tot)
    r2_sgd = 1 - (ss_res_sgd / ss_tot)
    
    # Parameter errors
    slope_error_batch = abs(batch_model.slope - 3)
    slope_error_sgd = abs(sgd_model.slope - 3)
    intercept_error_batch = abs(batch_model.intercept - 2)
    intercept_error_sgd = abs(sgd_model.intercept - 2)
    
    # Loss statistics
    final_loss_batch = batch_model.history['loss'][-1]
    final_loss_sgd = sgd_model.history['loss'][-1]
    avg_loss_last_100_batch = np.mean(batch_model.history['loss'][-100:])
    avg_loss_last_100_sgd = np.mean(sgd_model.history['loss'][-100:])
    std_loss_last_100_batch = np.std(batch_model.history['loss'][-100:])
    std_loss_last_100_sgd = np.std(sgd_model.history['loss'][-100:])
    
    # Create comparison table
    comparison_data = {
        'Metric': [
            'Final Slope',
            'Final Intercept',
            'Slope Error',
            'Intercept Error',
            'Final Loss (MSE)',
            'Avg Loss (last 100 iter)',
            'Loss Std Dev (last 100)',
            'R² Score',
            'Training Time (seconds)',
            'Time per Iteration (ms)'
        ],
        'Batch GD': [
            f"{batch_model.slope:.4f}",
            f"{batch_model.intercept:.4f}",
            f"{slope_error_batch:.4f}",
            f"{intercept_error_batch:.4f}",
            f"{final_loss_batch:.4f}",
            f"{avg_loss_last_100_batch:.4f}",
            f"{std_loss_last_100_batch:.4f}",
            f"{r2_batch:.4f}",
            f"{batch_time:.4f}",
            f"{(batch_time/len(batch_model.history['loss']))*1000:.4f}"
        ],
        'SGD': [
            f"{sgd_model.slope:.4f}",
            f"{sgd_model.intercept:.4f}",
            f"{slope_error_sgd:.4f}",
            f"{intercept_error_sgd:.4f}",
            f"{final_loss_sgd:.4f}",
            f"{avg_loss_last_100_sgd:.4f}",
            f"{std_loss_last_100_sgd:.4f}",
            f"{r2_sgd:.4f}",
            f"{sgd_time:.4f}",
            f"{(sgd_time/len(sgd_model.history['loss']))*1000:.4f}"
        ]
    }
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    # Trade-offs Analysis
    print("\n" + "="*70)
    print("TRADE-OFFS ANALYSIS")
    print("="*70)
    
    print("\n📊 BATCH GRADIENT DESCENT:")
    print("   ✓ Pros:")
    print("     • Smooth, stable convergence path")
    print("     • Deterministic updates (same result every run)")
    print(f"     • Lower loss variance: {std_loss_last_100_batch:.4f}")
    print("     • Better for small datasets")
    print("   ✗ Cons:")
    print("     • Slower per iteration (processes all data)")
    print("     • Memory intensive for large datasets")
    print("     • Can get stuck in local minima")
    print(f"     • Training time: {batch_time:.4f}s")
    
    print("\n⚡ STOCHASTIC GRADIENT DESCENT:")
    print("   ✓ Pros:")
    print("     • Faster iterations (single sample)")
    print(f"     • More efficient: {sgd_time/batch_time:.2f}x faster")
    print("     • Can escape local minima (due to noise)")
    print("     • Scales well to large datasets")
    print("   ✗ Cons:")
    print("     • Noisy, zigzag convergence path")
    print(f"     • Higher loss variance: {std_loss_last_100_sgd:.4f}")
    print("     • Non-deterministic (varies between runs)")
    print("     • May oscillate around optimum")
    
    print("\n🎯 RECOMMENDATIONS:")
    print("   • Small datasets (<10K): Use Batch GD")
    print("   • Large datasets (>100K): Use SGD or Mini-batch GD")
    print("   • Need stability: Use Batch GD")
    print("   • Need speed: Use SGD")
    print("   • Best of both worlds: Mini-batch GD (future improvement)")
    
    # Save analysis to CSV
    df_comparison.to_csv('sgd_vs_batch_analysis.csv', index=False)
    print("\n✓ Analysis saved to 'sgd_vs_batch_analysis.csv'")
    
    return df_comparison

# ==================== MAIN EXECUTION ====================
def main():
    print("="*70)
    print("STOCHASTIC GRADIENT DESCENT vs BATCH GRADIENT DESCENT")
    print("="*70)
    
    # Generate data
    print("\n1. Generating synthetic data (y = 3x + 2 + noise)...")
    X, y = generate_data(n_samples=200)
    print(f"   Generated {len(X)} data points")
    
    # Train Batch GD
    print("\n2. Training with BATCH GRADIENT DESCENT...")
    print("-"*70)
    batch_model = BatchGradientDescent(learning_rate=0.01, n_iterations=1000)
    batch_time = batch_model.fit(X, y)
    print(f"   Training completed in {batch_time:.4f} seconds")
    
    # Train SGD
    print("\n3. Training with STOCHASTIC GRADIENT DESCENT...")
    print("-"*70)
    sgd_model = StochasticGradientDescent(learning_rate=0.01, n_iterations=1000)
    sgd_time = sgd_model.fit(X, y)
    print(f"   Training completed in {sgd_time:.4f} seconds")
    
    # Analyze results
    df_analysis = analyze_results(X, y, batch_model, sgd_model, batch_time, sgd_time)
    
    # Visualize comparison
    print("\n4. Creating comparison visualizations...")
    visualize_comparison(X, y, batch_model, sgd_model)
   
if __name__ == "__main__":
    main()