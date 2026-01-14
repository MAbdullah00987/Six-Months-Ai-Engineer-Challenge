
#Day 5: Bayesian Inference (Thinking Like a Machine)
#Objective: Learn how to update your beliefs as new data comes in.
#Concepts: Conditional Probability, Bayes' Theorem: $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$.
#Task: Project - Monty Hall Simulator. Simulate the famous game show problem where switching doors doubles your chances of winning. Prove it with code.
#Task: Project - Naive Bayes Logic. Understand how a spam filter works using Bayes' theorem (Probability of "Spam" given the word "Buy").
#Inferential Statistics & Sampling
#Focus: Drawing conclusions from samples

#Study the Central Limit Theorem
#Learn about sampling distributions
#Review confidence intervals and margin of error

#Central Limit Theorem Simulation - Visually demonstrate CLT with various distributions
#Confidence Interval Calculator - Build a tool to calculate confidence intervals for sample means


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, binom, beta
import pandas as pd
import sympy as sp
from statsmodels.stats.proportion import proportion_confint

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


print("BAYESIAN INFERENCE & INFERENTIAL STATISTICS")
print("Thinking Like a Machine: Update Beliefs with Data")



# PART 1: BAYESIAN INFERENCE - CONDITIONAL PROBABILITY
#1. Bayesian Inference 
#Medical diagnosis example: Shows how base rates matter (1% disease rate but only 8.7% chance you have it even with positive test!)
#Coin flip updating: Watch beliefs evolve from uniform prior to concentrated posterior as data accumulates
print("PART 1: BAYESIAN INFERENCE - MEDICAL DIAGNOSIS EXAMPLE")

# Example: Disease Testing
# Disease prevalence: 1% of population has the disease
# Test accuracy: 95% sensitivity (true positive), 90% specificity (true negative)

P_disease = 0.01  # Prior probability of having disease
P_no_disease = 0.99
P_pos_given_disease = 0.95  # Sensitivity
P_pos_given_no_disease = 0.10  # False positive rate (1 - specificity)

# Calculate P(positive test)
P_positive = (P_pos_given_disease * P_disease + 
              P_pos_given_no_disease * P_no_disease)

# Bayes' Theorem: P(Disease|Positive Test)
P_disease_given_pos = (P_pos_given_disease * P_disease) / P_positive

print(f"\nMedical Test Example:")
print(f"Prior P(Disease) = {P_disease:.4f} (1%)")
print(f"Test Sensitivity = {P_pos_given_disease:.2%}")
print(f"Test Specificity = {1-P_pos_given_no_disease:.2%}")
print(f"\nP(Positive Test) = {P_positive:.4f}")
print(f"P(Disease|Positive Test) = {P_disease_given_pos:.4f} ({P_disease_given_pos:.2%})")
print(f"\nInsight: Even with 95% accuracy, only {P_disease_given_pos:.1%} of positive tests")
print(f"indicate actual disease due to low base rate!")


# PART 2: BAYESIAN UPDATING - COIN FLIP EXAMPLE
#2. Symbolic Math 📐
#Uses SymPy to show Bayes' Theorem algebraically
#Demonstrates Law of Total Probability
print("PART 2: BAYESIAN UPDATING - LEARNING FROM DATA")

# Start with uniform prior (don't know if coin is fair)
# Update beliefs as we observe coin flips

# Beta distribution parameters (conjugate prior for Binomial)
# Beta(α, β) where α = heads + 1, β = tails + 1
alpha_prior = 1  # Uniform prior
beta_prior = 1

# Simulate coin flips (biased coin with p=0.7)
np.random.seed(42)
true_p = 0.7
n_flips = [0, 5, 20, 100]
results = []

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, n in enumerate(n_flips):
    if n > 0:
        flips = np.random.binomial(1, true_p, n)
        heads = np.sum(flips)
        tails = n - heads
    else:
        heads, tails = 0, 0
    
    # Update posterior
    alpha_post = alpha_prior + heads
    beta_post = beta_prior + tails
    
    # Calculate posterior distribution
    x = np.linspace(0, 1, 1000)
    prior = beta.pdf(x, alpha_prior, beta_prior)
    posterior = beta.pdf(x, alpha_post, beta_post)
    
    # Plot
    ax = axes[idx]
    if n == 0:
        ax.plot(x, prior, 'b-', lw=2, label='Prior (Uniform)')
    else:
        ax.plot(x, prior, 'b--', lw=1, alpha=0.3, label='Prior')
        ax.plot(x, posterior, 'r-', lw=3, label=f'Posterior (after {n} flips)')
    
    ax.axvline(true_p, color='green', linestyle='--', lw=2, label=f'True p={true_p}')
    ax.set_xlabel('Probability of Heads', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'After {n} flips: {heads}H, {tails}T', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Calculate credible interval
    if n > 0:
        mean_post = alpha_post / (alpha_post + beta_post)
        ci_lower, ci_upper = beta.ppf([0.025, 0.975], alpha_post, beta_post)
        print(f"\nAfter {n} flips: {heads} heads, {tails} tails")
        print(f"  Posterior mean: {mean_post:.4f}")
        print(f"  95% Credible Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

plt.suptitle('Bayesian Updating: Belief Evolution with Data', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('bayesian_updating.png', dpi=300, bbox_inches='tight')
print("\nSaved: bayesian_updating.png")


# PART 3: SYMBOLIC BAYES' THEOREM WITH SYMPY
#3. Central Limit Theorem 
#Proves sample means become normal regardless of original distribution (uniform, exponential, bimodal)
#Foundation of all inferential statistics!
print("PART 3: SYMBOLIC BAYES' THEOREM")


# Define symbols
P_A, P_B, P_B_given_A, P_A_given_B = sp.symbols('P(A) P(B) P(B|A) P(A|B)')

# Bayes' Theorem
bayes_theorem = sp.Eq(P_A_given_B, (P_B_given_A * P_A) / P_B)

print("\nBayes' Theorem (Symbolic):")
sp.pprint(bayes_theorem)

# Solve for P(B)
P_not_A = sp.Symbol('P(~A)')
P_B_given_not_A = sp.Symbol('P(B|~A)')

# Law of Total Probability
total_prob = P_B_given_A * P_A + P_B_given_not_A * P_not_A

print("\n\nLaw of Total Probability:")
print(f"P(B) = P(B|A)·P(A) + P(B|¬A)·P(¬A)")
sp.pprint(sp.Eq(P_B, total_prob))


# PART 4: CENTRAL LIMIT THEOREM
#4. Confidence Intervals 🎯
#Simulates 100 experiments showing 95% CIs
#Visualizes which intervals capture the true mean
print("PART 4: CENTRAL LIMIT THEOREM")


# Show that means of samples from ANY distribution approach normal
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Three different distributions
distributions = [
    ('Uniform', lambda: np.random.uniform(0, 10, 100000)),
    ('Exponential', lambda: np.random.exponential(2, 100000)),
    ('Bimodal', lambda: np.concatenate([np.random.normal(2, 1, 50000), 
                                         np.random.normal(8, 1, 50000)]))
]

sample_sizes = [5, 30, 100]

for col, (dist_name, dist_func) in enumerate(distributions):
    # Generate population
    population = dist_func()
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    # Plot original distribution
    axes[0, col].hist(population, bins=50, density=True, alpha=0.7, 
                      color='skyblue', edgecolor='black')
    axes[0, col].axvline(pop_mean, color='red', linestyle='--', lw=2, 
                         label=f'μ={pop_mean:.2f}')
    axes[0, col].set_title(f'{dist_name} Distribution\n(Population)', 
                           fontsize=12, fontweight='bold')
    axes[0, col].legend()
    axes[0, col].set_ylabel('Density')
    
    # Sample means distribution
    n_samples = 1000
    n = 30  # Sample size
    sample_means = [np.mean(np.random.choice(population, n)) for _ in range(n_samples)]
    
    axes[1, col].hist(sample_means, bins=40, density=True, alpha=0.7, 
                      color='lightcoral', edgecolor='black', label='Sample Means')
    
    # Overlay theoretical normal
    x_norm = np.linspace(min(sample_means), max(sample_means), 100)
    theoretical_std = pop_std / np.sqrt(n)
    y_norm = norm.pdf(x_norm, pop_mean, theoretical_std)
    axes[1, col].plot(x_norm, y_norm, 'g-', lw=3, 
                      label=f'Normal(μ={pop_mean:.2f}, σ={theoretical_std:.2f})')
    
    axes[1, col].axvline(pop_mean, color='red', linestyle='--', lw=2)
    axes[1, col].set_title(f'Sampling Distribution\n(n={n}, {n_samples} samples)', 
                           fontsize=12, fontweight='bold')
    axes[1, col].legend()
    axes[1, col].set_xlabel('Sample Mean')
    axes[1, col].set_ylabel('Density')

plt.suptitle('Central Limit Theorem: Means Approach Normal Distribution', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=300, bbox_inches='tight')
print("\n Saved: central_limit_theorem.png")

print(f"\n CLT Insight: Regardless of original distribution shape,")
print(f"   sample means form a normal distribution!")


# PART 5: CONFIDENCE INTERVALS
#5. Standard Error 
#Shows SE = σ/√n relationship
#Demonstrates how larger samples = more precision
print("PART 5: CONFIDENCE INTERVALS")


# Simulate confidence interval coverage
np.random.seed(42)
true_mean = 100
true_std = 15
n_samples = 30
n_experiments = 100
confidence_level = 0.95

# Generate multiple samples and calculate CIs
ci_results = []
for i in range(n_experiments):
    sample = np.random.normal(true_mean, true_std, n_samples)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)
    se = sample_std / np.sqrt(n_samples)
    
    # 95% CI
    ci_margin = stats.t.ppf(0.975, n_samples-1) * se
    ci_lower = sample_mean - ci_margin
    ci_upper = sample_mean + ci_margin
    
    covers_true = ci_lower <= true_mean <= ci_upper
    ci_results.append({
        'experiment': i,
        'mean': sample_mean,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'covers': covers_true
    })

ci_df = pd.DataFrame(ci_results)
coverage_rate = ci_df['covers'].mean()

print(f"\nConfidence Interval Simulation:")
print(f"True population mean: {true_mean}")
print(f"Number of experiments: {n_experiments}")
print(f"Sample size per experiment: {n_samples}")
print(f"Confidence level: {confidence_level:.0%}")
print(f"\nCoverage rate: {coverage_rate:.2%} (Expected: {confidence_level:.0%})")

# Visualize first 50 CIs
fig, ax = plt.subplots(figsize=(15, 10))

for i in range(min(50, n_experiments)):
    row = ci_df.iloc[i]
    color = 'green' if row['covers'] else 'red'
    alpha = 0.3 if row['covers'] else 0.8
    ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
            color=color, alpha=alpha, lw=2)
    ax.plot(row['mean'], i, 'o', color=color, markersize=4, alpha=alpha)

ax.axvline(true_mean, color='blue', linestyle='--', lw=3, label=f'True Mean = {true_mean}')
ax.set_xlabel('Value', fontsize=14)
ax.set_ylabel('Experiment Number', fontsize=14)
ax.set_title(f'95% Confidence Intervals: {coverage_rate:.1%} Cover True Mean\n' + 
             f'Green = Contains True Mean | Red = Misses True Mean', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_intervals.png', dpi=300, bbox_inches='tight')
print("Saved: confidence_intervals.png")


# PART 6: SAMPLING DISTRIBUTIONS & STANDARD ERROR
#6. Margin of Error 
#Poll example showing MOE decreases with sample size
#Different confidence levels (90%, 95%, 99%)
print("PART 6: SAMPLING DISTRIBUTIONS & STANDARD ERROR")


# Show how sample size affects standard error
population = np.random.normal(50, 10, 100000)
pop_mean = np.mean(population)
pop_std = np.std(population)

sample_sizes_se = [5, 10, 30, 50, 100, 200]
n_samples_per_size = 1000

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

se_results = []

for idx, n in enumerate(sample_sizes_se):
    sample_means = [np.mean(np.random.choice(population, n)) 
                    for _ in range(n_samples_per_size)]
    
    observed_se = np.std(sample_means)
    theoretical_se = pop_std / np.sqrt(n)
    
    se_results.append({
        'n': n,
        'observed_se': observed_se,
        'theoretical_se': theoretical_se
    })
    
    ax = axes[idx]
    ax.hist(sample_means, bins=40, density=True, alpha=0.7, 
            color='lightblue', edgecolor='black')
    
    # Overlay normal
    x = np.linspace(min(sample_means), max(sample_means), 100)
    y = norm.pdf(x, pop_mean, theoretical_se)
    ax.plot(x, y, 'r-', lw=3, label=f'Normal(μ={pop_mean:.1f}, SE={theoretical_se:.2f})')
    
    ax.axvline(pop_mean, color='green', linestyle='--', lw=2, label=f'True μ={pop_mean:.1f}')
    ax.set_title(f'n={n}: SE={theoretical_se:.2f}\n(σ/√n = {pop_std:.1f}/√{n})', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')

plt.suptitle('Standard Error Decreases with Sample Size: SE = σ/√n', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('standard_error.png', dpi=300, bbox_inches='tight')
print("Saved: standard_error.png")

# Print SE comparison table
se_df = pd.DataFrame(se_results)
print("\nStandard Error vs Sample Size:")
print(se_df.to_string(index=False))


# PART 7: MARGIN OF ERROR VISUALIZATION
#7. A/B Testing 
#Real-world Bayesian A/B test
#Calculates probability Version B beats Version A
#Provides decision framework

print("PART 7: MARGIN OF ERROR IN POLLS")


# Political poll example
true_support = 0.52  # True support is 52%
sample_sizes_poll = [100, 400, 1000, 2400]
confidence_levels = [0.90, 0.95, 0.99]

fig, ax = plt.subplots(figsize=(14, 8))

for conf_level in confidence_levels:
    z_score = stats.norm.ppf((1 + conf_level) / 2)
    margins = []
    
    for n in sample_sizes_poll:
        # Margin of error for proportion
        se_prop = np.sqrt(true_support * (1 - true_support) / n)
        moe = z_score * se_prop
        margins.append(moe * 100)  # Convert to percentage
    
    ax.plot(sample_sizes_poll, margins, marker='o', markersize=10, lw=3,
            label=f'{conf_level:.0%} Confidence')
    
    # Annotate points
    for n, moe in zip(sample_sizes_poll, margins):
        ax.annotate(f'±{moe:.1f}%', xy=(n, moe), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)

ax.set_xlabel('Sample Size', fontsize=14, fontweight='bold')
ax.set_ylabel('Margin of Error (%)', fontsize=14, fontweight='bold')
ax.set_title('Margin of Error Decreases with Sample Size\n' + 
             f'(True Support = {true_support:.0%})', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('margin_of_error.png', dpi=300, bbox_inches='tight')
print("Saved: margin_of_error.png")

print("\nMargin of Error Insights:")
print("- Doubling sample size reduces MOE by √2 ≈ 1.41")
print("- Quadrupling sample size cuts MOE in half")
print("- Higher confidence → Larger margin of error")



# PART 8: PRACTICAL EXAMPLE - A/B TESTING


print("\n" + "="*80)
print("PART 8: BAYESIAN A/B TESTING")
print("="*80)

# A/B test: Which website design converts better?
np.random.seed(42)

# Version A: 100 visitors, 12 conversions
# Version B: 100 visitors, 17 conversions
n_A, conversions_A = 100, 12
n_B, conversions_B = 100, 17

# Bayesian approach with Beta distribution
# Prior: Beta(1,1) - uniform
alpha_A = 1 + conversions_A
beta_A = 1 + (n_A - conversions_A)

alpha_B = 1 + conversions_B
beta_B = 1 + (n_B - conversions_B)

# Posterior distributions
x = np.linspace(0, 0.4, 1000)
posterior_A = beta.pdf(x, alpha_A, beta_A)
posterior_B = beta.pdf(x, alpha_B, beta_B)

# Monte Carlo: Probability B > A
samples_A = beta.rvs(alpha_A, beta_A, size=100000)
samples_B = beta.rvs(alpha_B, beta_B, size=100000)
prob_B_better = np.mean(samples_B > samples_A)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot posteriors
ax1.plot(x, posterior_A, 'b-', lw=3, label=f'Version A: {conversions_A}/{n_A} = {conversions_A/n_A:.1%}')
ax1.plot(x, posterior_B, 'r-', lw=3, label=f'Version B: {conversions_B}/{n_B} = {conversions_B/n_B:.1%}')
ax1.fill_between(x, posterior_A, alpha=0.3, color='blue')
ax1.fill_between(x, posterior_B, alpha=0.3, color='red')
ax1.set_xlabel('Conversion Rate', fontsize=12)
ax1.set_ylabel('Posterior Density', fontsize=12)
ax1.set_title('Posterior Distributions of Conversion Rates', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)

# Plot difference distribution
difference = samples_B - samples_A
ax2.hist(difference, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
ax2.axvline(0, color='black', linestyle='--', lw=2, label='No Difference')
ax2.axvline(np.mean(difference), color='red', linestyle='-', lw=3, 
            label=f'Mean Difference = {np.mean(difference):.3f}')
ax2.set_xlabel('Conversion Rate Difference (B - A)', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax2.set_title(f'P(B > A) = {prob_B_better:.2%}', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ab_testing.png', dpi=300, bbox_inches='tight')
print("Saved: ab_testing.png")

print(f"\nA/B Test Results:")
print(f"Version A: {conversions_A}/{n_A} = {conversions_A/n_A:.2%} conversion rate")
print(f"Version B: {conversions_B}/{n_B} = {conversions_B/n_B:.2%} conversion rate")
print(f"\nProbability that B is better than A: {prob_B_better:.2%}")
print(f"Expected improvement: {np.mean(difference)*100:.2f} percentage points")
print(f"95% Credible Interval: [{np.percentile(difference, 2.5)*100:.2f}%, {np.percentile(difference, 97.5)*100:.2f}%]")

if prob_B_better > 0.95:
    print("\n DECISION: Strong evidence to switch to Version B")
elif prob_B_better > 0.80:
    print("\n DECISION: Moderate evidence for B, consider more data")
else:
    print("\n DECISION: Insufficient evidence, continue testing")


# SUMMARY & KEY TAKEAWAYS

print("KEY TAKEAWAYS: THINKING LIKE A MACHINE")


print("""
 BAYESIAN INFERENCE:
   • Start with prior belief → Update with data → Get posterior belief
   • Bayes' Theorem: P(A|B) = P(B|A)·P(A) / P(B)
   • More data → Stronger beliefs, less uncertainty
   • Perfect for sequential learning and updating

 CENTRAL LIMIT THEOREM:
   • Sample means are normally distributed (regardless of population)
   • Enables inference even from non-normal data
   • Foundation for confidence intervals and hypothesis tests

 SAMPLING DISTRIBUTIONS:
   • Standard Error = σ/√n (decreases with larger samples)
   • Confidence Intervals: Estimate ± (Critical Value × SE)
   • 95% CI means: 95% of such intervals contain true parameter

 MARGIN OF ERROR:
   • MOE = z* × SE (or t* × SE for small samples)
   • To cut MOE in half → Need 4× the sample size
   • Higher confidence → Larger MOE (trade-off!)

 PRACTICAL APPLICATIONS:
   • Medical diagnosis (accounting for base rates)
   • A/B testing (which version performs better?)
   • Political polls (understanding uncertainty)
   • Quality control (detecting process changes)

 LOGICAL STRENGTHENING:
   • Think probabilistically, not deterministically
   • Update beliefs systematically with evidence
   • Quantify uncertainty (don't just guess!)
   • Distinguish between confidence and certainty
""")


print(" All visualizations saved successfully!")


plt.show()