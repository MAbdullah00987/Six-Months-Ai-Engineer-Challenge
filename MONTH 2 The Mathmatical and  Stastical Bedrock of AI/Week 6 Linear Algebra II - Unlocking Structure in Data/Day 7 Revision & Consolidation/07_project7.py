
# Project 7:
# Covariance Matrix: Calculate and interpret the covariance matrix for a dataset.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, load_wine
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)

class CovarianceAnalyzer:
    """Complete covariance matrix analysis tool"""
    
    def __init__(self, data, feature_names=None):
        """
        Initialize with data
        
        Parameters:
        - data: numpy array or pandas DataFrame (n_samples × n_features)
        - feature_names: list of feature names
        """
        if isinstance(data, pd.DataFrame):
            self.feature_names = data.columns.tolist()
            self.data = data.values
        else:
            self.data = np.array(data)
            self.feature_names = feature_names or [f'Feature_{i+1}' 
                                                   for i in range(self.data.shape[1])]
        
        self.n_samples, self.n_features = self.data.shape
        self.cov_matrix = None
        self.corr_matrix = None
        self.mean = None
        self.std = None
        
    def calculate_covariance_manual(self):
        """Calculate covariance matrix manually (from scratch)"""
        print("="*80)
        print("CALCULATING COVARIANCE MATRIX (MANUAL)")
        print("="*80)
        
        # Step 1: Calculate mean of each feature
        self.mean = np.mean(self.data, axis=0)
        print("\nStep 1: Calculate mean for each feature")
        for i, name in enumerate(self.feature_names):
            print(f"  {name}: {self.mean[i]:.4f}")
        
        # Step 2: Center the data (subtract mean)
        centered_data = self.data - self.mean
        print("\nStep 2: Center the data (X - mean)")
        print(f"  Centered data shape: {centered_data.shape}")
        
        # Step 3: Calculate covariance matrix
        # Cov(X, Y) = (1/n-1) * Σ(xi - x̄)(yi - ȳ)
        print("\nStep 3: Calculate covariance matrix")
        print("  Formula: Cov = (1/(n-1)) × X_centered^T × X_centered")
        
        self.cov_matrix = (centered_data.T @ centered_data) / (self.n_samples - 1)
        
        print(f"  Result shape: {self.cov_matrix.shape}")
        print("\nCovariance Matrix:")
        self._print_matrix(self.cov_matrix)
        
        return self.cov_matrix
    
    def calculate_covariance_numpy(self):
        """Calculate covariance matrix using NumPy"""
        self.cov_matrix = np.cov(self.data.T)
        return self.cov_matrix
    
    def calculate_correlation(self):
        """Calculate correlation matrix"""
        self.corr_matrix = np.corrcoef(self.data.T)
        
        print("\n" + "="*80)
        print("CORRELATION MATRIX")
        print("="*80)
        print("Correlation = Covariance / (std_X × std_Y)")
        print("\nCorrelation Matrix:")
        self._print_matrix(self.corr_matrix)
        
        return self.corr_matrix
    
    def _print_matrix(self, matrix):
        """Print matrix in a nice format"""
        df = pd.DataFrame(matrix, 
                         columns=self.feature_names,
                         index=self.feature_names)
        print(df.to_string())
    
    def interpret_covariance(self):
        """Interpret the covariance matrix"""
        print("\n" + "="*80)
        print("INTERPRETATION OF COVARIANCE MATRIX")
        print("="*80)
        
        print("\n1. DIAGONAL ELEMENTS (Variance):")
        print("   Variance measures spread of individual features")
        for i, name in enumerate(self.feature_names):
            var = self.cov_matrix[i, i]
            std = np.sqrt(var)
            print(f"   • {name}: var={var:.4f}, std={std:.4f}")
        
        print("\n2. OFF-DIAGONAL ELEMENTS (Covariance):")
        print("   Covariance measures relationship between pairs of features")
        
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                cov = self.cov_matrix[i, j]
                corr = self.corr_matrix[i, j]
                
                # Interpret covariance
                if cov > 0:
                    direction = "POSITIVE (move together)"
                elif cov < 0:
                    direction = "NEGATIVE (move opposite)"
                else:
                    direction = "ZERO (independent)"
                
                # Interpret correlation strength
                abs_corr = abs(corr)
                if abs_corr > 0.7:
                    strength = "Strong"
                elif abs_corr > 0.4:
                    strength = "Moderate"
                elif abs_corr > 0.2:
                    strength = "Weak"
                else:
                    strength = "Very weak"
                
                print(f"\n   • {self.feature_names[i]} ↔ {self.feature_names[j]}:")
                print(f"     - Covariance: {cov:.4f} ({direction})")
                print(f"     - Correlation: {corr:.4f} ({strength})")
        
        print("\n3. KEY INSIGHTS:")
        
        # Find strongest positive correlation
        mask = np.triu(np.ones_like(self.corr_matrix, dtype=bool), k=1)
        corr_pairs = []
        for i in range(self.n_features):
            for j in range(i+1, self.n_features):
                corr_pairs.append((i, j, self.corr_matrix[i, j]))
        
        if corr_pairs:
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            i, j, corr = corr_pairs[0]
            print(f"   • Strongest correlation: {self.feature_names[i]} ↔ "
                  f"{self.feature_names[j]} (r={corr:.3f})")
            
            # Most variable feature
            var_idx = np.argmax(np.diag(self.cov_matrix))
            print(f"   • Most variable feature: {self.feature_names[var_idx]} "
                  f"(var={self.cov_matrix[var_idx, var_idx]:.3f})")
            
            # Least variable feature
            var_idx = np.argmin(np.diag(self.cov_matrix))
            print(f"   • Least variable feature: {self.feature_names[var_idx]} "
                  f"(var={self.cov_matrix[var_idx, var_idx]:.3f})")
    
    def eigendecomposition(self):
        """Perform eigendecomposition of covariance matrix"""
        print("\n" + "="*80)
        print("EIGENDECOMPOSITION OF COVARIANCE MATRIX")
        print("="*80)
        
        eigenvalues, eigenvectors = np.linalg.eig(self.cov_matrix)
        
        # Sort by eigenvalue magnitude
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print("\nEigenvalues (variance along principal components):")
        total_var = np.sum(eigenvalues)
        for i, ev in enumerate(eigenvalues):
            pct = (ev / total_var) * 100
            print(f"  PC{i+1}: {ev:.4f} ({pct:.2f}% of total variance)")
        
        print("\nEigenvectors (principal component directions):")
        for i in range(len(eigenvalues)):
            print(f"\n  PC{i+1}:")
            for j, name in enumerate(self.feature_names):
                print(f"    {name}: {eigenvectors[j, i]:.4f}")
        
        return eigenvalues, eigenvectors
    
    def analyze(self):
        """Complete analysis"""
        # Calculate covariance
        self.calculate_covariance_manual()
        
        # Verify with NumPy
        cov_numpy = self.calculate_covariance_numpy()
        print("\n✓ Verification: Manual calculation matches NumPy:",
              np.allclose(self.cov_matrix, cov_numpy))
        
        # Calculate correlation
        self.calculate_correlation()
        
        # Interpret
        self.interpret_covariance()
        
        # Eigendecomposition
        eigenvalues, eigenvectors = self.eigendecomposition()
        
        return self.cov_matrix, self.corr_matrix, eigenvalues, eigenvectors

def create_visualizations(analyzer, eigenvalues, eigenvectors):
    """Create comprehensive visualizations"""
    
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Covariance Matrix Heatmap
    ax1 = plt.subplot(3, 4, 1)
    sns.heatmap(analyzer.cov_matrix, annot=True, fmt='.2f', 
                xticklabels=analyzer.feature_names,
                yticklabels=analyzer.feature_names,
                cmap='coolwarm', center=0, square=True,
                cbar_kws={'label': 'Covariance'}, ax=ax1)
    ax1.set_title('Covariance Matrix', fontsize=13, fontweight='bold')
    
    # Plot 2: Correlation Matrix Heatmap
    ax2 = plt.subplot(3, 4, 2)
    sns.heatmap(analyzer.corr_matrix, annot=True, fmt='.2f',
                xticklabels=analyzer.feature_names,
                yticklabels=analyzer.feature_names,
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Correlation'}, ax=ax2)
    ax2.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
    
    # Plot 3: Variance of Each Feature
    ax3 = plt.subplot(3, 4, 3)
    variances = np.diag(analyzer.cov_matrix)
    colors = plt.cm.viridis(np.linspace(0, 1, len(variances)))
    bars = ax3.bar(range(len(variances)), variances, color=colors, 
                   alpha=0.7, edgecolor='black')
    ax3.set_xticks(range(len(variances)))
    ax3.set_xticklabels(analyzer.feature_names, rotation=45, ha='right')
    ax3.set_ylabel('Variance')
    ax3.set_title('Variance by Feature', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, var) in enumerate(zip(bars, variances)):
        ax3.text(bar.get_x() + bar.get_width()/2, var, f'{var:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Standard Deviation
    ax4 = plt.subplot(3, 4, 4)
    stds = np.sqrt(variances)
    bars = ax4.bar(range(len(stds)), stds, color=colors, 
                   alpha=0.7, edgecolor='black')
    ax4.set_xticks(range(len(stds)))
    ax4.set_xticklabels(analyzer.feature_names, rotation=45, ha='right')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Standard Deviation by Feature', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Pairwise Scatter Plots (for first 3 features)
    n_features_to_plot = min(3, analyzer.n_features)
    for i in range(n_features_to_plot):
        for j in range(n_features_to_plot):
            idx = i * n_features_to_plot + j + 5
            if idx > 12:
                break
            ax = plt.subplot(3, 4, idx)
            
            if i == j:
                # Diagonal: histogram
                ax.hist(analyzer.data[:, i], bins=20, color=colors[i], 
                       alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency')
                if i == n_features_to_plot - 1:
                    ax.set_xlabel(analyzer.feature_names[i])
                ax.set_title(f'{analyzer.feature_names[i]}', fontsize=10)
            else:
                # Off-diagonal: scatter plot
                ax.scatter(analyzer.data[:, j], analyzer.data[:, i], 
                          alpha=0.5, s=30, c='steelblue')
                
                # Add regression line
                z = np.polyfit(analyzer.data[:, j], analyzer.data[:, i], 1)
                p = np.poly1d(z)
                x_line = np.linspace(analyzer.data[:, j].min(), 
                                    analyzer.data[:, j].max(), 100)
                ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7)
                
                # Add correlation value
                corr = analyzer.corr_matrix[i, j]
                ax.text(0.05, 0.95, f'r={corr:.2f}', 
                       transform=ax.transAxes, fontsize=9,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                if j == 0:
                    ax.set_ylabel(analyzer.feature_names[i])
                if i == n_features_to_plot - 1:
                    ax.set_xlabel(analyzer.feature_names[j])
            
            ax.grid(True, alpha=0.3)
    
    # Plot: Eigenvalues
    if len(eigenvalues) <= 10:
        plot_idx = 3 * 4 - 1  # Last position
        ax_eigen = plt.subplot(3, 4, plot_idx)
        ax_eigen.bar(range(1, len(eigenvalues)+1), eigenvalues, 
                     color='coral', alpha=0.7, edgecolor='black')
        ax_eigen.set_xlabel('Principal Component')
        ax_eigen.set_ylabel('Eigenvalue (Variance)')
        ax_eigen.set_title('Eigenvalues of Covariance Matrix', 
                          fontsize=13, fontweight='bold')
        ax_eigen.grid(True, alpha=0.3, axis='y')
        
        # Add cumulative variance line
        cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        ax_eigen2 = ax_eigen.twinx()
        ax_eigen2.plot(range(1, len(eigenvalues)+1), cumsum * 100, 
                       'ro-', linewidth=2, markersize=8, label='Cumulative %')
        ax_eigen2.set_ylabel('Cumulative Variance (%)', color='red')
        ax_eigen2.tick_params(axis='y', labelcolor='red')
        ax_eigen2.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.show()

def demonstrate_covariance_concepts():
    """Demonstrate key covariance concepts with simple examples"""
    
    print("\n" + "="*80)
    print("DEMONSTRATION: UNDERSTANDING COVARIANCE")
    print("="*80)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    np.random.seed(42)
    n = 100
    
    # Example 1: Strong Positive Correlation
    x1 = np.random.randn(n)
    y1 = 2 * x1 + np.random.randn(n) * 0.5
    cov1 = np.cov(x1, y1)[0, 1]
    corr1 = np.corrcoef(x1, y1)[0, 1]
    
    axes[0, 0].scatter(x1, y1, alpha=0.6, s=40)
    axes[0, 0].set_title(f'Strong Positive\nCov={cov1:.2f}, r={corr1:.2f}', 
                         fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Example 2: Strong Negative Correlation
    x2 = np.random.randn(n)
    y2 = -2 * x2 + np.random.randn(n) * 0.5
    cov2 = np.cov(x2, y2)[0, 1]
    corr2 = np.corrcoef(x2, y2)[0, 1]
    
    axes[0, 1].scatter(x2, y2, alpha=0.6, s=40, color='red')
    axes[0, 1].set_title(f'Strong Negative\nCov={cov2:.2f}, r={corr2:.2f}', 
                         fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    
    # Example 3: No Correlation
    x3 = np.random.randn(n)
    y3 = np.random.randn(n)
    cov3 = np.cov(x3, y3)[0, 1]
    corr3 = np.corrcoef(x3, y3)[0, 1]
    
    axes[0, 2].scatter(x3, y3, alpha=0.6, s=40, color='green')
    axes[0, 2].set_title(f'No Correlation\nCov={cov3:.2f}, r={corr3:.2f}', 
                         fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 2].axvline(x=0, color='k', linewidth=0.5)
    
    # Example 4: Weak Positive
    x4 = np.random.randn(n)
    y4 = 0.5 * x4 + np.random.randn(n) * 2
    cov4 = np.cov(x4, y4)[0, 1]
    corr4 = np.corrcoef(x4, y4)[0, 1]
    
    axes[1, 0].scatter(x4, y4, alpha=0.6, s=40, color='orange')
    axes[1, 0].set_title(f'Weak Positive\nCov={cov4:.2f}, r={corr4:.2f}', 
                         fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    
    # Example 5: Different Scales (same correlation)
    x5 = np.random.randn(n)
    y5 = 2 * x5 + np.random.randn(n) * 0.5
    y5_scaled = y5 * 10  # Scale by 10
    cov5 = np.cov(x5, y5_scaled)[0, 1]
    corr5 = np.corrcoef(x5, y5_scaled)[0, 1]
    
    axes[1, 1].scatter(x5, y5_scaled, alpha=0.6, s=40, color='purple')
    axes[1, 1].set_title(f'Scaled Variables\nCov={cov5:.2f}, r={corr5:.2f}', 
                         fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    
    # Example 6: Non-linear relationship
    x6 = np.linspace(-3, 3, n)
    y6 = x6**2 + np.random.randn(n) * 0.5
    cov6 = np.cov(x6, y6)[0, 1]
    corr6 = np.corrcoef(x6, y6)[0, 1]
    
    axes[1, 2].scatter(x6, y6, alpha=0.6, s=40, color='brown')
    axes[1, 2].set_title(f'Non-linear\nCov={cov6:.2f}, r={corr6:.2f}\n(Correlation misleading!)', 
                         fontweight='bold', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 2].axvline(x=0, color='k', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKEY OBSERVATIONS:")
    print("• Covariance measures LINEAR relationships")
    print("• Positive covariance: variables move together")
    print("• Negative covariance: variables move opposite")
    print("• Near-zero covariance: weak/no linear relationship")
    print("• Correlation normalizes covariance (always between -1 and 1)")
    print("• Non-linear relationships may have low correlation!")

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("="*80)
print("COVARIANCE MATRIX ANALYSIS")
print("="*80)

# Load Iris dataset
data = load_iris()
X = data.data
feature_names = data.feature_names

print(f"\nDataset: Iris")
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Feature names: {feature_names}")

# Create analyzer
analyzer = CovarianceAnalyzer(X, feature_names)

# Perform complete analysis
cov_matrix, corr_matrix, eigenvalues, eigenvectors = analyzer.analyze()

# Create visualizations
create_visualizations(analyzer, eigenvalues, eigenvectors)

# Demonstrate concepts
demonstrate_covariance_concepts()

# ============================================================================
# KEY FORMULAS AND CONCEPTS
# ============================================================================

print("\n" + "="*80)
print("KEY FORMULAS AND CONCEPTS")
print("="*80)

print("""
1. COVARIANCE FORMULA:
   Cov(X, Y) = (1/(n-1)) × Σ(xi - x̄)(yi - ȳ)
   
   • Measures how two variables change together
   • Units: (unit of X) × (unit of Y)
   • Range: -∞ to +∞

2. COVARIANCE MATRIX:
   For data matrix X (n × p):
   Σ = (1/(n-1)) × X_centered^T × X_centered
   
   • Diagonal: variances of each feature
   • Off-diagonal: covariances between features
   • Symmetric matrix (Σij = Σji)

3. CORRELATION vs COVARIANCE:
   Corr(X, Y) = Cov(X, Y) / (σX × σY)
   
   • Standardized covariance
   • Range: -1 to +1
   • Scale-independent

4. INTERPRETATION:
   • Cov > 0: Positive relationship
   • Cov < 0: Negative relationship  
   • Cov = 0: No linear relationship
   • |Corr| > 0.7: Strong
   • |Corr| > 0.4: Moderate
   • |Corr| > 0.2: Weak

5. PROPERTIES:
   • Cov(X, X) = Var(X)
   • Cov(X, Y) = Cov(Y, X) (symmetric)
   • Positive semi-definite matrix
   • Eigenvalues ≥ 0

6. APPLICATIONS:
   • Principal Component Analysis (PCA)
   • Portfolio optimization (finance)
   • Feature selection
   • Multivariate analysis
   • Machine learning preprocessing
""")

print("="*80)