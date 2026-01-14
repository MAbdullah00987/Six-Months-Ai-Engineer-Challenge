
#Day 7: Integration & Advanced Concepts
#Objective: Combine these concepts into a visual masterpiece for LinkedIn.
#Deliverable: A polished Central Limit Theorem Visualization.
#Why: It shows you can turn abstract math into concrete code.

#Focus: Correlation, causation, and project completion
#Study correlation vs. causation
#Review confounding variables
#Complete Naive Bayes implementation

#Projects
#Correlation vs. Causation Analysis - Find a dataset showing spurious correlation and write a detailed analysis
#Naive Bayes Classifier - Complete and test the spam detection implementation from Day 3

#Phase 1: Foundation - Python & NumPy
#Random Sampling with NumPy

import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

print("=" * 60)
print("NUMPY RANDOM SAMPLING - Building Blocks for CLT")
print("=" * 60)

# 1. Different Distributions for CLT Testing
print("\n1. SAMPLING FROM DIFFERENT DISTRIBUTIONS")
print("-" * 60)

# Uniform Distribution (0 to 10)
uniform_samples = np.random.uniform(0, 10, 1000)
print(f"Uniform: Mean={uniform_samples.mean():.2f}, Std={uniform_samples.std():.2f}")

# Exponential Distribution (rate=1)
exponential_samples = np.random.exponential(scale=2, size=1000)
print(f"Exponential: Mean={exponential_samples.mean():.2f}, Std={exponential_samples.std():.2f}")

# Binomial Distribution (10 trials, p=0.3)
binomial_samples = np.random.binomial(n=10, p=0.3, size=1000)
print(f"Binomial: Mean={binomial_samples.mean():.2f}, Std={binomial_samples.std():.2f}")

# Poisson Distribution (lambda=3)
poisson_samples = np.random.poisson(lam=3, size=1000)
print(f"Poisson: Mean={poisson_samples.mean():.2f}, Std={poisson_samples.std():.2f}")

# 2. Creating Sample Means (Core of CLT)
print("\n2. GENERATING SAMPLE MEANS")
print("-" * 60)

population = np.random.exponential(scale=2, size=10000)
sample_size = 30
n_samples = 1000

# Method 1: Using loop (educational)
sample_means_loop = []
for _ in range(n_samples):
    sample = np.random.choice(population, size=sample_size)
    sample_means_loop.append(sample.mean())

# Method 2: Vectorized (faster - professional approach)
samples = np.random.choice(population, size=(n_samples, sample_size))
sample_means_vectorized = samples.mean(axis=1)

print(f"Sample means (n={sample_size}): {sample_means_vectorized[:5]}")
print(f"Mean of sample means: {sample_means_vectorized.mean():.4f}")
print(f"Population mean: {population.mean():.4f}")

# 3. NumPy Array Operations for Statistics
print("\n3. NUMPY STATISTICAL OPERATIONS")
print("-" * 60)

data = np.random.normal(100, 15, 500)
print(f"Mean: {np.mean(data):.2f}")
print(f"Median: {np.median(data):.2f}")
print(f"Std Dev: {np.std(data):.2f}")
print(f"Variance: {np.var(data):.2f}")
print(f"25th Percentile: {np.percentile(data, 25):.2f}")
print(f"75th Percentile: {np.percentile(data, 75):.2f}")

# 4. Demonstrating CLT with Different Sample Sizes
print("\n4. CLT DEMONSTRATION - EFFECT OF SAMPLE SIZE")
print("-" * 60)

sample_sizes = [5, 10, 30, 50]
for n in sample_sizes:
    samples = np.random.choice(population, size=(1000, n))
    means = samples.mean(axis=1)
    print(f"n={n:2d} | Mean: {means.mean():.3f} | Std: {means.std():.3f}")

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('NumPy Sampling Foundations for CLT', fontsize=16, fontweight='bold')

# Plot 1: Original Population
axes[0, 0].hist(population, bins=50, alpha=0.7, color='coral', edgecolor='black')
axes[0, 0].set_title('Population Distribution\n(Exponential)', fontweight='bold')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Plot 2-5: Sample Means for different sample sizes
positions = [(0, 1), (0, 2), (1, 0), (1, 1)]
for idx, (n, pos) in enumerate(zip(sample_sizes, positions)):
    samples = np.random.choice(population, size=(1000, n))
    means = samples.mean(axis=1)
    
    axes[pos].hist(means, bins=40, alpha=0.7, color='skyblue', edgecolor='black')
    axes[pos].axvline(means.mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    axes[pos].set_title(f'Sample Means (n={n})', fontweight='bold')
    axes[pos].set_xlabel('Sample Mean')
    axes[pos].set_ylabel('Frequency')
    axes[pos].legend()

# Plot 6: Comparison of Standard Errors
axes[1, 2].plot(sample_sizes, 
                [np.random.choice(population, size=(1000, n)).mean(axis=1).std() 
                 for n in sample_sizes],
                marker='o', linewidth=2, markersize=8, color='green')
axes[1, 2].set_title('Standard Error vs Sample Size', fontweight='bold')
axes[1, 2].set_xlabel('Sample Size (n)')
axes[1, 2].set_ylabel('Standard Error')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('numpy_clt_foundation.png', dpi=300, bbox_inches='tight')
plt.show()

print("KEY TAKEAWAYS:")
print("- As sample size ↑, distribution becomes more normal")
print("- Standard error decreases with √n")
print("- Works regardless of population distribution shape")
