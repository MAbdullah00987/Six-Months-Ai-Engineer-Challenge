#Day 3: The Central Limit Theorem (The Magic of Stats)
#Objective: Understand why the Normal Distribution is everywhere. This is the most important concept in statistics.
#* Theory: No matter what the original distribution of data is (even if it's weird and lopsided), if you take enough sample means,
#  those means will form a Normal Distribution.
#Bayes' Theorem & Conditional Probability
#Focus: Bayesian reasoning and applications
#* Study Bayes' theorem in depth
#* Review conditional probability examples
#* Understand prior, likelihood, and posterior


#CENTRAL LIMIT THEOREM - THE MAGIC OF STATISTICS
#This demonstrates why the Normal Distribution appears everywhere in nature


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, uniform, expon, poisson, binom
import pandas as pd

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


# PART 1: UNDERSTANDING THE CENTRAL LIMIT THEOREM

def demonstrate_clt_basics():
    """
    Demonstrates CLT with different original distributions
    Shows that NO MATTER the original distribution, sample means become normal
    """
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    
    # Different population distributions
    distributions = [
        ('Uniform', lambda n: np.random.uniform(0, 10, n)),
        ('Exponential', lambda n: np.random.exponential(2, n)),
        ('Bimodal', lambda n: np.concatenate([np.random.normal(3, 1, n//2), 
                                               np.random.normal(8, 1, n//2)])),
        ('Heavily Skewed', lambda n: np.random.gamma(2, 2, n))
    ]
    
    sample_sizes = [5, 30]
    n_samples = 10000
    
    for i, (dist_name, dist_func) in enumerate(distributions):
        # Original population
        population = dist_func(100000)
        
        axes[i, 0].hist(population, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i, 0].set_title(f'{dist_name} Distribution\n(Original Population)', fontsize=12, fontweight='bold')
        axes[i, 0].set_ylabel('Density')
        
        # For different sample sizes
        for j, sample_size in enumerate(sample_sizes):
            # Generate many sample means
            sample_means = []
            for _ in range(n_samples):
                sample = dist_func(sample_size)
                sample_means.append(np.mean(sample))
            
            sample_means = np.array(sample_means)
            
            # Plot distribution of sample means
            axes[i, j+1].hist(sample_means, bins=50, density=True, alpha=0.7, 
                             color='coral', edgecolor='black', label='Sample Means')
            
            # Overlay normal distribution
            mu, sigma = np.mean(sample_means), np.std(sample_means)
            x = np.linspace(sample_means.min(), sample_means.max(), 100)
            axes[i, j+1].plot(x, norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                             label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
            
            axes[i, j+1].set_title(f'Distribution of Sample Means\n(n={sample_size}, {n_samples} samples)', 
                                  fontsize=11, fontweight='bold')
            axes[i, j+1].legend()
            axes[i, j+1].set_ylabel('Density')
    
    plt.suptitle('CENTRAL LIMIT THEOREM: The Magic Transformation', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('clt_demonstration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(" KEY INSIGHT:")
    print("Notice how the rightmost columns look increasingly NORMAL,")
    print("even though the original distributions (left column) are very different!")
    print("\nThis is the MAGIC of the Central Limit Theorem!")


# PART 2: MATHEMATICAL FOUNDATION

def clt_mathematical_properties():
    """
    Demonstrates mathematical properties of CLT
    """
    
    print("\n" + "="*70)
    print("MATHEMATICAL PROPERTIES OF CLT")
    print("="*70)
    
    # Create a weird population distribution
    population = np.concatenate([
        np.random.exponential(2, 5000),
        np.random.normal(10, 1, 3000),
        np.random.uniform(15, 20, 2000)
    ])
    
    pop_mean = np.mean(population)
    pop_std = np.std(population, ddof=1)
    
    print(f"\n📊 Population Parameters:")
    print(f"   Mean (μ): {pop_mean:.4f}")
    print(f"   Std Dev (σ): {pop_std:.4f}")
    
    # Test CLT with different sample sizes
    sample_sizes = [5, 10, 30, 50, 100]
    n_samples = 5000
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Original population
    axes[0].hist(population, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
    axes[0].set_title('Original Population\n(Trimodal & Weird)', fontsize=12, fontweight='bold')
    axes[0].axvline(pop_mean, color='red', linestyle='--', linewidth=2, label=f'μ = {pop_mean:.2f}')
    axes[0].legend()
    
    for idx, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = [np.mean(np.random.choice(population, n)) for _ in range(n_samples)]
        
        # CLT predictions
        theoretical_mean = pop_mean
        theoretical_std = pop_std / np.sqrt(n)  # Standard Error
        
        # Actual measurements
        actual_mean = np.mean(sample_means)
        actual_std = np.std(sample_means, ddof=1)
        
        # Plot
        axes[idx+1].hist(sample_means, bins=50, density=True, alpha=0.7, 
                        color='coral', edgecolor='black', label='Empirical')
        
        # Overlay theoretical normal distribution
        x = np.linspace(min(sample_means), max(sample_means), 100)
        axes[idx+1].plot(x, norm.pdf(x, theoretical_mean, theoretical_std), 
                        'r-', linewidth=2, label='CLT Prediction')
        
        axes[idx+1].set_title(f'Sample Size n={n}\nSE(theory)={theoretical_std:.3f}, SE(actual)={actual_std:.3f}',
                             fontsize=11)
        axes[idx+1].legend()
        axes[idx+1].axvline(actual_mean, color='blue', linestyle='--', alpha=0.7)
        
        print(f"\n📈 Sample Size n = {n}:")
        print(f"   Theoretical SE: {theoretical_std:.4f}")
        print(f"   Actual SE: {actual_std:.4f}")
        print(f"   Difference: {abs(theoretical_std - actual_std):.4f}")
    
    plt.suptitle('CLT Mathematical Properties: Standard Error = σ/√n', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clt_mathematics.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 3: PRACTICAL APPLICATIONS

def clt_confidence_intervals():
    """
    Using CLT to construct confidence intervals
    """
    
    print("\n" + "="*70)
    print("PRACTICAL APPLICATION: CONFIDENCE INTERVALS")
    print("="*70)
    
    # Simulate: We want to estimate average height in a population
    true_population_mean = 170  # cm
    true_population_std = 10
    
    sample_size = 50
    sample = np.random.normal(true_population_mean, true_population_std, sample_size)
    
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    standard_error = sample_std / np.sqrt(sample_size)
    
    # 95% Confidence Interval
    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    margin_of_error = z_score * standard_error
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    print(f"\n🎯 From ONE sample of n={sample_size}:")
    print(f"   Sample Mean: {sample_mean:.2f} cm")
    print(f"   Sample Std Dev: {sample_std:.2f} cm")
    print(f"   Standard Error: {standard_error:.2f} cm")
    print(f"\n   95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"   True Population Mean: {true_population_mean:.2f} cm")
    print(f"   ✓ Interval captures true mean!" if ci_lower <= true_population_mean <= ci_upper else "   ✗ Missed!")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Sample distribution
    ax1.hist(sample, bins=15, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
    ax1.axvline(sample_mean, color='red', linewidth=2, label=f'Sample Mean = {sample_mean:.2f}')
    ax1.axvline(true_population_mean, color='blue', linewidth=2, linestyle='--', 
               label=f'True Mean = {true_population_mean:.2f}')
    ax1.set_xlabel('Height (cm)')
    ax1.set_ylabel('Density')
    ax1.set_title('Our Sample Data', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Right: Sampling distribution (via CLT)
    x = np.linspace(sample_mean - 4*standard_error, sample_mean + 4*standard_error, 1000)
    sampling_dist = norm.pdf(x, sample_mean, standard_error)
    
    ax2.plot(x, sampling_dist, 'b-', linewidth=2, label='Sampling Distribution of Mean')
    ax2.fill_between(x, 0, sampling_dist, where=(x >= ci_lower) & (x <= ci_upper), 
                     alpha=0.3, color='green', label='95% CI')
    ax2.axvline(sample_mean, color='red', linewidth=2, linestyle='--', label='Sample Mean')
    ax2.axvline(true_population_mean, color='orange', linewidth=2, linestyle='--', 
               label='True Mean')
    ax2.set_xlabel('Sample Mean')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Sampling Distribution (via CLT)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.suptitle('Using CLT for Confidence Intervals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clt_confidence_intervals.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Demonstrate coverage probability
    print("\n" + "="*70)
    print("TESTING: Do 95% CIs really capture the true mean 95% of the time?")
    print("="*70)
    
    n_experiments = 1000
    captures = 0
    
    for _ in range(n_experiments):
        sample = np.random.normal(true_population_mean, true_population_std, sample_size)
        sample_mean = np.mean(sample)
        se = np.std(sample, ddof=1) / np.sqrt(sample_size)
        ci_lower = sample_mean - 1.96 * se
        ci_upper = sample_mean + 1.96 * se
        
        if ci_lower <= true_population_mean <= ci_upper:
            captures += 1
    
    coverage = (captures / n_experiments) * 100
    print(f"\n✓ Out of {n_experiments} experiments:")
    print(f"  {captures} confidence intervals captured the true mean")
    print(f"  Coverage rate: {coverage:.1f}%")
    print(f"  Expected: 95.0%")
    print(f"  Difference: {abs(coverage - 95):.1f}%")


# PART 4: HYPOTHESIS TESTING WITH CLT

def clt_hypothesis_testing():
    """
    Using CLT for hypothesis testing (z-tests and t-tests)
    """
    
    print("\n" + "="*70)
    print("HYPOTHESIS TESTING USING CLT")
    print("="*70)
    
    # Scenario: A factory claims average widget weight is 100g
    # We suspect it's less
    
    claimed_mean = 100
    sample_size = 40
    
    # Generate sample (true mean is 98g)
    true_mean = 98
    true_std = 5
    sample = np.random.normal(true_mean, true_std, sample_size)
    
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    
    # Z-test (CLT application)
    standard_error = sample_std / np.sqrt(sample_size)
    z_statistic = (sample_mean - claimed_mean) / standard_error
    p_value = stats.norm.cdf(z_statistic)  # One-tailed test
    
    print(f"\n📦 Widget Weight Testing:")
    print(f"   Null Hypothesis (H₀): μ = {claimed_mean}g")
    print(f"   Alternative (H₁): μ < {claimed_mean}g")
    print(f"\n   Sample Mean: {sample_mean:.2f}g")
    print(f"   Sample Std Dev: {sample_std:.2f}g")
    print(f"   Standard Error: {standard_error:.2f}g")
    print(f"\n   Z-statistic: {z_statistic:.4f}")
    print(f"   P-value: {p_value:.4f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"\n   ✓ REJECT H₀ at α={alpha} level")
        print(f"   Evidence suggests mean weight < {claimed_mean}g")
    else:
        print(f"\n   ✗ FAIL TO REJECT H₀ at α={alpha} level")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Sample data
    ax1.hist(sample, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(sample_mean, color='red', linewidth=2, label=f'Sample Mean = {sample_mean:.2f}g')
    ax1.axvline(claimed_mean, color='green', linewidth=2, linestyle='--', 
               label=f'Claimed Mean = {claimed_mean}g')
    ax1.set_xlabel('Weight (g)')
    ax1.set_ylabel('Density')
    ax1.set_title('Sample Data', fontsize=12, fontweight='bold')
    ax1.legend()
    
    # Right: Sampling distribution under H₀
    x = np.linspace(claimed_mean - 4*standard_error, claimed_mean + 4*standard_error, 1000)
    null_dist = norm.pdf(x, claimed_mean, standard_error)
    
    ax2.plot(x, null_dist, 'b-', linewidth=2, label='Null Distribution')
    ax2.fill_between(x, 0, null_dist, where=(x <= sample_mean), 
                     alpha=0.3, color='red', label=f'P-value = {p_value:.4f}')
    ax2.axvline(sample_mean, color='red', linewidth=2, linestyle='--', 
               label=f'Observed Mean = {sample_mean:.2f}g')
    ax2.axvline(claimed_mean, color='green', linewidth=2, 
               label=f'H₀ Mean = {claimed_mean}g')
    ax2.set_xlabel('Sample Mean')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Sampling Distribution Under H₀ (via CLT)', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.suptitle('Hypothesis Testing with CLT', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('clt_hypothesis_testing.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 5: SAMPLE SIZE DETERMINATION


def clt_sample_size_calculation():
    """
    Using CLT to determine required sample size
    """
    
    print("\n" + "="*70)
    print("SAMPLE SIZE DETERMINATION USING CLT")
    print("="*70)
    
    # Question: How many samples do we need to estimate mean within ±2 units
    # with 95% confidence?
    
    population_std = 15  # Estimated from pilot study
    margin_of_error = 2
    confidence_level = 0.95
    
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Formula: n = (z * σ / E)²
    required_n = ((z_score * population_std) / margin_of_error) ** 2
    required_n = int(np.ceil(required_n))
    
    print(f"\n🎯 Planning a Study:")
    print(f"   Estimated Population Std Dev: {population_std}")
    print(f"   Desired Margin of Error: ±{margin_of_error}")
    print(f"   Confidence Level: {confidence_level*100}%")
    print(f"   Z-score: {z_score:.4f}")
    print(f"\n   📊 Required Sample Size: {required_n}")
    
    # Visualize relationship
    margins = np.linspace(0.5, 10, 100)
    sample_sizes = ((z_score * population_std) / margins) ** 2
    
    plt.figure(figsize=(12, 6))
    plt.plot(margins, sample_sizes, 'b-', linewidth=2)
    plt.axhline(required_n, color='red', linestyle='--', 
               label=f'n={required_n} for E=±{margin_of_error}')
    plt.axvline(margin_of_error, color='red', linestyle='--')
    plt.scatter([margin_of_error], [required_n], color='red', s=100, zorder=5)
    plt.xlabel('Margin of Error (±)', fontsize=12)
    plt.ylabel('Required Sample Size', fontsize=12)
    plt.title('Sample Size vs Margin of Error (95% Confidence)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('clt_sample_size.png', dpi=300, bbox_inches='tight')
    plt.show()


# RUN ALL DEMONSTRATIONS

if __name__ == "__main__":
    print("="*70)
    print("CENTRAL LIMIT THEOREM: THE MAGIC OF STATISTICS")
    print("="*70)
    
    # Part 1: Demonstration
    demonstrate_clt_basics()
    
    # Part 2: Mathematical properties
    clt_mathematical_properties()
    
    # Part 3: Confidence intervals
    clt_confidence_intervals()
    
    # Part 4: Hypothesis testing
    clt_hypothesis_testing()
    
    # Part 5: Sample size
    clt_sample_size_calculation()
    
  
    print("🎓 KEY TAKEAWAYS:")
    print("1. CLT explains why Normal distribution appears everywhere")
    print("2. Sample means are ALWAYS more normal than original data")
    print("3. Larger samples → tighter, more normal distributions")
    print("4. Standard Error = σ/√n (decreases with sample size)")
    print("5. Enables: confidence intervals, hypothesis tests, predictions")
    