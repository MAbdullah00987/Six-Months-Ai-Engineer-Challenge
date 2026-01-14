
#Central Limit Theorem Simulation - Visually demonstrate CLT with various distributions

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

class CLTSimulator:
    """
    A class to simulate and visualize the Central Limit Theorem
    with various probability distributions.
    """
    
    def __init__(self, n_samples=1000, sample_sizes=[1, 5, 10, 30, 50]):
        """
        Initialize the CLT simulator.
        
        Parameters:
        -----------
        n_samples : int
            Number of samples to draw for each simulation
        sample_sizes : list
            List of sample sizes to demonstrate CLT convergence
        """
        self.n_samples = n_samples
        self.sample_sizes = sample_sizes
        
    def generate_distribution(self, dist_type, size, **params):
        """
        Generate random samples from various distributions.
        
        Parameters:
        -----------
        dist_type : str
            Type of distribution ('uniform', 'exponential', 'binomial', 'poisson', 'bimodal')
        size : int or tuple
            Size of the sample to generate
        **params : dict
            Additional parameters for specific distributions
        """
        if dist_type == 'uniform':
            return np.random.uniform(params.get('low', 0), params.get('high', 10), size)
        elif dist_type == 'exponential':
            return np.random.exponential(params.get('scale', 2), size)
        elif dist_type == 'binomial':
            return np.random.binomial(params.get('n', 10), params.get('p', 0.5), size)
        elif dist_type == 'poisson':
            return np.random.poisson(params.get('lam', 3), size)
        elif dist_type == 'bimodal':
            # Create bimodal distribution by mixing two normals
            half_size = size // 2 if isinstance(size, int) else (size[0] // 2, size[1])
            part1 = np.random.normal(-3, 1, half_size)
            part2 = np.random.normal(3, 1, size - half_size if isinstance(size, int) else (size[0] - half_size[0], size[1]))
            return np.concatenate([part1, part2])
        else:
            raise ValueError(f"Unknown distribution type: {dist_type}")
    
    def simulate_clt(self, dist_type, **params):
        """
        Simulate CLT for a given distribution across different sample sizes.
        
        Parameters:
        -----------
        dist_type : str
            Type of distribution to simulate
        **params : dict
            Distribution-specific parameters
            
        Returns:
        --------
        dict : Dictionary containing simulation results
        """
        results = {}
        
        for n in self.sample_sizes:
            # Generate n_samples means, each from n observations
            samples = self.generate_distribution(dist_type, (self.n_samples, n), **params)
            sample_means = np.mean(samples, axis=1)
            results[n] = sample_means
            
        return results
    
    def plot_clt_demonstration(self, dist_type, dist_name, **params):
        """
        Create comprehensive visualization of CLT for a given distribution.
        
        Parameters:
        -----------
        dist_type : str
            Type of distribution
        dist_name : str
            Display name for the distribution
        **params : dict
            Distribution parameters
        """
        # Simulate CLT
        results = self.simulate_clt(dist_type, **params)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, len(self.sample_sizes), hspace=0.3, wspace=0.3)
        
        # Plot original distribution
        ax_orig = fig.add_subplot(gs[0, :])
        original_sample = self.generate_distribution(dist_type, 10000, **params)
        ax_orig.hist(original_sample, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        ax_orig.set_title(f'Original {dist_name} Distribution (10,000 samples)', fontsize=14, fontweight='bold')
        ax_orig.set_xlabel('Value')
        ax_orig.set_ylabel('Density')
        ax_orig.grid(True, alpha=0.3)
        
        # Add distribution statistics
        mean_orig = np.mean(original_sample)
        std_orig = np.std(original_sample)
        ax_orig.axvline(mean_orig, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_orig:.2f}')
        ax_orig.legend()
        
        # Plot sampling distributions for different sample sizes
        for idx, n in enumerate(self.sample_sizes):
            # Histogram of sample means
            ax_hist = fig.add_subplot(gs[1, idx])
            sample_means = results[n]
            
            ax_hist.hist(sample_means, bins=30, density=True, alpha=0.7, 
                        color='lightcoral', edgecolor='black')
            
            # Overlay normal distribution
            mu = np.mean(sample_means)
            sigma = np.std(sample_means)
            x = np.linspace(sample_means.min(), sample_means.max(), 100)
            ax_hist.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', linewidth=2, 
                        label=f'Normal fit\nμ={mu:.2f}\nσ={sigma:.2f}')
            
            ax_hist.set_title(f'Sample Size n={n}', fontsize=12, fontweight='bold')
            ax_hist.set_xlabel('Sample Mean')
            ax_hist.set_ylabel('Density')
            ax_hist.legend(fontsize=8)
            ax_hist.grid(True, alpha=0.3)
            
            # Q-Q plot
            ax_qq = fig.add_subplot(gs[2, idx])
            stats.probplot(sample_means, dist="norm", plot=ax_qq)
            ax_qq.set_title(f'Q-Q Plot (n={n})', fontsize=10)
            ax_qq.grid(True, alpha=0.3)
        
        plt.suptitle(f'Central Limit Theorem Demonstration: {dist_name} Distribution', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        return fig
    
    def create_comparison_summary(self, distributions):
        """
        Create a summary comparison of CLT across multiple distributions.
        
        Parameters:
        -----------
        distributions : dict
            Dictionary with distribution types and their parameters
        """
        fig, axes = plt.subplots(len(distributions), len(self.sample_sizes), 
                                figsize=(18, 4*len(distributions)))
        
        if len(distributions) == 1:
            axes = axes.reshape(1, -1)
        
        for row_idx, (dist_info, dist_params) in enumerate(distributions.items()):
            dist_type, dist_name = dist_info
            results = self.simulate_clt(dist_type, **dist_params)
            
            for col_idx, n in enumerate(self.sample_sizes):
                ax = axes[row_idx, col_idx]
                sample_means = results[n]
                
                # Plot histogram
                ax.hist(sample_means, bins=30, density=True, alpha=0.6, 
                       color='steelblue', edgecolor='black')
                
                # Overlay normal curve
                mu, sigma = np.mean(sample_means), np.std(sample_means)
                x = np.linspace(sample_means.min(), sample_means.max(), 100)
                ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)
                
                # Formatting
                if col_idx == 0:
                    ax.set_ylabel(f'{dist_name}\nDensity', fontsize=10, fontweight='bold')
                if row_idx == 0:
                    ax.set_title(f'n = {n}', fontsize=11, fontweight='bold')
                if row_idx == len(distributions) - 1:
                    ax.set_xlabel('Sample Mean', fontsize=9)
                
                ax.grid(True, alpha=0.3)
                ax.text(0.02, 0.98, f'μ={mu:.2f}\nσ={sigma:.2f}', 
                       transform=ax.transAxes, fontsize=8,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('CLT Comparison Across Different Distributions', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def statistical_tests(self, dist_type, **params):
        """
        Perform statistical tests for normality on sampling distributions.
        
        Parameters:
        -----------
        dist_type : str
            Type of distribution
        **params : dict
            Distribution parameters
            
        Returns:
        --------
        DataFrame with test results
        """
        results = self.simulate_clt(dist_type, **params)
        test_results = []
        
        for n, sample_means in results.items():
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(sample_means[:5000])  # Shapiro limited to 5000 samples
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.kstest(sample_means, 'norm', 
                                         args=(np.mean(sample_means), np.std(sample_means)))
            
            # Anderson-Darling test
            anderson_result = stats.anderson(sample_means, dist='norm')
            
            test_results.append({
                'Sample Size (n)': n,
                'Mean': np.mean(sample_means),
                'Std Dev': np.std(sample_means),
                'Skewness': stats.skew(sample_means),
                'Kurtosis': stats.kurtosis(sample_means),
                'Shapiro-Wilk p-value': shapiro_p,
                'KS Test p-value': ks_p,
                'Normal? (p>0.05)': 'Yes' if min(shapiro_p, ks_p) > 0.05 else 'No'
            })
        
        return pd.DataFrame(test_results)


# Main execution
if __name__ == "__main__":
    # Initialize simulator
    simulator = CLTSimulator(n_samples=1000, sample_sizes=[1, 5, 10, 30, 50])
    
    # Define distributions to test
    distributions_to_test = {
        ('uniform', 'Uniform'): {'low': 0, 'high': 10},
        ('exponential', 'Exponential'): {'scale': 2},
        ('binomial', 'Binomial'): {'n': 10, 'p': 0.3},
        ('poisson', 'Poisson'): {'lam': 5},
        ('bimodal', 'Bimodal'): {}
    }
    
    # 1. Demonstrate CLT for each distribution individually
    print("="*80)
    print("CENTRAL LIMIT THEOREM SIMULATION")
    print("="*80)
    
    for (dist_type, dist_name), params in distributions_to_test.items():
        print(f"\n{'='*80}")
        print(f"Distribution: {dist_name}")
        print(f"{'='*80}")
        
        # Create visualization
        fig = simulator.plot_clt_demonstration(dist_type, dist_name, **params)
        plt.show()
        
        # Print statistical tests
        print(f"\nStatistical Tests for {dist_name} Distribution:")
        print("-" * 80)
        test_df = simulator.statistical_tests(dist_type, **params)
        print(test_df.to_string(index=False))
        print()
    
    # 2. Create comparison summary
    print("COMPARATIVE ANALYSIS")

    fig_comparison = simulator.create_comparison_summary(distributions_to_test)
    plt.show()
    
    # 3. Key insights
    print("KEY INSIGHTS FROM CLT SIMULATION")
    
    print("""
    1. CONVERGENCE TO NORMALITY:
       - Regardless of the original distribution shape, the sampling distribution
         of means approaches a normal distribution as sample size increases.
    
    2. SAMPLE SIZE EFFECT:
       - For n=1: Distribution matches the original (no averaging effect)
       - For n=5-10: Beginning to show normal characteristics
       - For n=30+: Close approximation to normal (Rule of thumb: n≥30)
    
    3. STANDARD ERROR:
       - The standard deviation of sample means decreases as n increases
       - Relationship: SE = σ/√n (where σ is population std dev)
    
    4. SKEWED DISTRIBUTIONS:
       - More skewed distributions (e.g., exponential) require larger sample
         sizes to approach normality compared to symmetric distributions.
    
    5. PRACTICAL IMPLICATIONS:
       - Justifies using normal-based inference methods (t-tests, confidence intervals)
       - Explains why means are more reliable than individual observations
       - Foundation for many statistical procedures
    """)

    print("\nSimulation complete! All visualizations have been displayed.")