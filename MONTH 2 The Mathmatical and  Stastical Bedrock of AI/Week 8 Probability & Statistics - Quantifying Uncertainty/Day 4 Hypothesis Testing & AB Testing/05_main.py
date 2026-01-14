
#Complete Statistics - Part 5: SymPy Symbolic Math

"""
PART 5: SYMBOLIC MATHEMATICS WITH SYMPY
Understanding Statistical Formulas at a Deeper Level
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sympy as sp
from sympy import symbols, sqrt, exp, pi, Sum, integrate, diff, simplify, latex
from sympy.statistics import Normal, density, E, variance
import warnings
warnings.filterwarnings('ignore')

# Initialize sympy printing
sp.init_printing(use_unicode=True)

# ============================================================================
# 1. SYMBOLIC PROBABILITY DISTRIBUTIONS
# ============================================================================

print("=" * 80)
print("1. SYMBOLIC PROBABILITY DISTRIBUTIONS")
print("=" * 80)

# Define symbolic variables
x, mu, sigma = symbols('x mu sigma', real=True)
sigma = symbols('sigma', positive=True)

print("\n📊 NORMAL DISTRIBUTION PDF")
print("-" * 80)

# Normal distribution PDF
normal_pdf = (1 / (sigma * sqrt(2 * pi))) * exp(-(x - mu)**2 / (2 * sigma**2))

print("\nSymbolic Formula:")
print(f"f(x) = {normal_pdf}")
print(f"\nLaTeX: {latex(normal_pdf)}")

# Verify properties
print("\n🔍 VERIFYING PROPERTIES:")

# 1. Integration over all space = 1
print("\n1. Total Probability = 1")
print("   ∫_{-∞}^{∞} f(x) dx = 1")
total_prob = integrate(normal_pdf, (x, -sp.oo, sp.oo))
print(f"   Result: {total_prob}")

# 2. Expected value
print("\n2. Expected Value E[X] = μ")
print("   E[X] = ∫_{-∞}^{∞} x·f(x) dx")
expected_val = integrate(x * normal_pdf, (x, -sp.oo, sp.oo))
expected_val_simplified = simplify(expected_val)
print(f"   Result: {expected_val_simplified}")

# 3. Variance calculation (symbolically)
print("\n3. Variance Var(X) = σ²")
print("   Var(X) = E[(X-μ)²] = ∫_{-∞}^{∞} (x-μ)²·f(x) dx")
variance_expr = integrate((x - mu)**2 * normal_pdf, (x, -sp.oo, sp.oo))
variance_simplified = simplify(variance_expr)
print(f"   Result: {variance_simplified}")

# ============================================================================
# 2. DERIVING STATISTICAL FORMULAS
# ============================================================================

print("\n" + "=" * 80)
print("2. DERIVING STATISTICAL FORMULAS")
print("=" * 80)

# Sample mean derivation
n = symbols('n', positive=True, integer=True)
i = symbols('i', integer=True)
x_i = symbols('x_1:%d' % 6)  # x_1, x_2, ..., x_5 for example

print("\n SAMPLE MEAN")
print("-" * 80)
sample_mean_formula = Sum(symbols('x_i'), (i, 1, n)) / n
print(f"x̄ = {sample_mean_formula}")
print("\nFor concrete example with n=5:")
concrete_mean = sum(x_i) / 5
print(f"x̄ = {concrete_mean}")

# Sample variance derivation
print("\n SAMPLE VARIANCE")
print("-" * 80)
print("s² = Σ(xᵢ - x̄)² / (n-1)")

# Expanding (x - x̄)²
x_var, x_bar = symbols('x bar_x')
variance_expansion = (x_var - x_bar)**2
expanded = sp.expand(variance_expansion)
print(f"\nExpanding (x - x̄)²:")
print(f"   {variance_expansion} = {expanded}")

# Alternative formula: s² = [Σx² - (Σx)²/n] / (n-1)
print("\n COMPUTATIONAL FORMULA (easier to calculate):")
print("s² = [Σxᵢ² - (Σxᵢ)²/n] / (n-1)")
print("\nProof: Both formulas are algebraically equivalent")

# ============================================================================
# 3. T-STATISTIC DERIVATION
# ============================================================================

print("\n" + "=" * 80)
print("3. T-STATISTIC FORMULA DERIVATION")
print("=" * 80)

x_bar, mu_0, s, n_sym = symbols('bar_x mu_0 s n', real=True)
n_sym = symbols('n', positive=True)
s = symbols('s', positive=True)

print("\n T-STATISTIC FOR ONE-SAMPLE TEST")
print("-" * 80)

# T-statistic formula
t_stat = (x_bar - mu_0) / (s / sqrt(n_sym))

print("Formula:")
print(f"t = {t_stat}")
print(f"\nLaTeX: {latex(t_stat)}")

print("\n COMPONENTS:")
print(f"   Numerator:   x̄ - μ₀  (difference from hypothesized mean)")
print(f"   Denominator: s/√n    (standard error of the mean)")

print("\n INTERPRETATION:")
print("   • Larger |t| = stronger evidence against H₀")
print("   • t measures how many standard errors x̄ is from μ₀")
print("   • Larger n → smaller standard error → larger |t| for same difference")

# Demonstrate how t changes with different inputs
print("\n NUMERICAL EXAMPLES:")
examples = [
    (85, 80, 10, 25, "Small difference, small n"),
    (85, 80, 10, 100, "Small difference, large n"),
    (90, 80, 10, 25, "Large difference, small n"),
    (90, 80, 10, 100, "Large difference, large n"),
]

print(f"\n{'x̄':>5} {'μ₀':>5} {'s':>5} {'n':>5} {'t-value':>10} {'Description'}")
print("-" * 70)

for x_val, mu_val, s_val, n_val, desc in examples:
    t_val = float(t_stat.subs([(x_bar, x_val), (mu_0, mu_val), 
                                (s, s_val), (n_sym, n_val)]))
    print(f"{x_val:>5} {mu_val:>5} {s_val:>5} {n_val:>5} {t_val:>10.4f}  {desc}")

# ============================================================================
# 4. CONFIDENCE INTERVAL DERIVATION
# ============================================================================

print("\n" + "=" * 80)
print("4. CONFIDENCE INTERVAL DERIVATION")
print("=" * 80)

t_crit = symbols('t_crit', positive=True)

print("\n CONFIDENCE INTERVAL FOR MEAN")
print("-" * 80)

# CI formula
ci_lower = x_bar - t_crit * (s / sqrt(n_sym))
ci_upper = x_bar + t_crit * (s / sqrt(n_sym))

print("95% Confidence Interval:")
print(f"Lower: {ci_lower}")
print(f"Upper: {ci_upper}")
print(f"\nCompact form: x̄ ± t_crit × (s/√n)")

print("\n WHERE:")
print("   x̄      = sample mean")
print("   t_crit = critical value from t-distribution")
print("   s      = sample standard deviation")
print("   n      = sample size")
print("   s/√n   = standard error of the mean (SEM)")

print("\n INTERPRETATION:")
print("   'We are 95% confident the true population mean μ")
print("    lies between [Lower, Upper]'")

# Numerical example
print("\n NUMERICAL EXAMPLE:")
x_bar_val, s_val, n_val = 85, 10, 25
t_crit_val = 2.064  # t-critical for 95% CI, df=24

ci_lower_val = float(ci_lower.subs([(x_bar, x_bar_val), (s, s_val), 
                                     (n_sym, n_val), (t_crit, t_crit_val)]))
ci_upper_val = float(ci_upper.subs([(x_bar, x_bar_val), (s, s_val), 
                                     (n_sym, n_val), (t_crit, t_crit_val)]))

print(f"\nGiven: x̄={x_bar_val}, s={s_val}, n={n_val}, t_crit={t_crit_val}")
print(f"95% CI: [{ci_lower_val:.2f}, {ci_upper_val:.2f}]")

# ============================================================================
# 5. EFFECT SIZE FORMULAS
# ============================================================================

print("\n" + "=" * 80)
print("5. EFFECT SIZE FORMULAS")
print("=" * 80)

mu_1, mu_2, sigma_1, sigma_2, n_1, n_2 = symbols('mu_1 mu_2 sigma_1 sigma_2 n_1 n_2', 
                                                   positive=True)

print("\n COHEN'S d (Standardized Mean Difference)")
print("-" * 80)

# Cohen's d for two groups
pooled_sd = sqrt(((n_1 - 1) * sigma_1**2 + (n_2 - 1) * sigma_2**2) / (n_1 + n_2 - 2))
cohens_d = (mu_1 - mu_2) / pooled_sd

print("Formula:")
print(f"d = (μ₁ - μ₂) / s_pooled")
print(f"\nWhere s_pooled = {pooled_sd}")
print(f"\nFull formula:")
print(f"d = {simplify(cohens_d)}")

print("\n INTERPRETATION:")
print("   d = 0.2  → Small effect")
print("   d = 0.5  → Medium effect")
print("   d = 0.8  → Large effect")
print("   d > 1.0  → Very large effect")

# Numerical example
print("\n NUMERICAL EXAMPLE:")
vals = {mu_1: 85, mu_2: 80, sigma_1: 10, sigma_2: 12, n_1: 30, n_2: 30}
d_val = float(cohens_d.subs(vals))
print(f"\nGroup 1: μ={vals[mu_1]}, σ={vals[sigma_1]}, n={vals[n_1]}")
print(f"Group 2: μ={vals[mu_2]}, σ={vals[sigma_2]}, n={vals[n_2]}")
print(f"\nCohen's d = {d_val:.4f} (Small to medium effect)")

# ============================================================================
# 6. VISUALIZING SYMBOLIC FORMULAS
# ============================================================================

print("\n" + "=" * 80)
print("6. VISUALIZING SYMBOLIC RELATIONSHIPS")
print("=" * 80)

# Plot how t-statistic changes with different parameters
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. t-statistic vs sample size (fixed effect)
x_bar_fixed, mu_0_fixed, s_fixed = 85, 80, 10
n_range = np.arange(5, 201, 5)
t_values = [(x_bar_fixed - mu_0_fixed) / (s_fixed / np.sqrt(n)) for n in n_range]

axes[0, 0].plot(n_range, t_values, linewidth=3, color='blue')
axes[0, 0].axhline(2.0, color='red', linestyle='--', linewidth=2, label='t_crit ≈ 2.0')
axes[0, 0].fill_between(n_range, 2.0, t_values, where=(np.array(t_values) > 2.0),
                         alpha=0.3, color='green')
axes[0, 0].set_xlabel('Sample Size (n)', fontsize=12)
axes[0, 0].set_ylabel('t-statistic', fontsize=12)
axes[0, 0].set_title('t-statistic vs Sample Size\n(x̄=85, μ₀=80, s=10)', 
                      fontsize=13, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. CI width vs sample size
n_range2 = np.arange(10, 201, 5)
t_crit_fixed = 2.0  # Approximate
ci_widths = [2 * t_crit_fixed * (s_fixed / np.sqrt(n)) for n in n_range2]

axes[0, 1].plot(n_range2, ci_widths, linewidth=3, color='green')
axes[0, 1].fill_between(n_range2, 0, ci_widths, alpha=0.3, color='green')
axes[0, 1].set_xlabel('Sample Size (n)', fontsize=12)
axes[0, 1].set_ylabel('CI Width', fontsize=12)
axes[0, 1].set_title('95% CI Width vs Sample Size\n(Narrower = More Precise)', 
                      fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Effect size visualization
mean_diffs = np.linspace(0, 20, 100)
effect_sizes = mean_diffs / 10  # Assuming σ = 10

axes[1, 0].plot(mean_diffs, effect_sizes, linewidth=3, color='purple')
axes[1, 0].axhline(0.2, color='orange', linestyle='--', label='Small (0.2)')
axes[1, 0].axhline(0.5, color='blue', linestyle='--', label='Medium (0.5)')
axes[1, 0].axhline(0.8, color='red', linestyle='--', label='Large (0.8)')
axes[1, 0].fill_between(mean_diffs, 0, effect_sizes, 
                         where=(effect_sizes < 0.2), alpha=0.2, color='orange')
axes[1, 0].fill_between(mean_diffs, 0, effect_sizes, 
                         where=((effect_sizes >= 0.2) & (effect_sizes < 0.5)), 
                         alpha=0.2, color='blue')
axes[1, 0].fill_between(mean_diffs, 0, effect_sizes, 
                         where=(effect_sizes >= 0.5), alpha=0.2, color='red')
axes[1, 0].set_xlabel('Mean Difference (μ₁ - μ₂)', fontsize=12)
axes[1, 0].set_ylabel("Cohen's d", fontsize=12)
axes[1, 0].set_title("Effect Size (d) vs Mean Difference\n(σ=10)", 
                      fontsize=13, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Normal distributions with different parameters
x_vals = np.linspace(-4, 4, 1000)
distributions = [
    (0, 1, 'μ=0, σ=1', 'blue'),
    (0, 0.5, 'μ=0, σ=0.5', 'green'),
    (1, 1, 'μ=1, σ=1', 'red'),
]

for mu_val, sigma_val, label, color in distributions:
    y_vals = (1 / (sigma_val * np.sqrt(2 * np.pi))) * \
             np.exp(-(x_vals - mu_val)**2 / (2 * sigma_val**2))
    axes[1, 1].plot(x_vals, y_vals, linewidth=2, label=label, color=color)

axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_ylabel('Probability Density', fontsize=12)
axes[1, 1].set_title('Normal Distributions with Different Parameters', 
                      fontsize=13, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('symbolic_visualizations.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. SYMBOLIC DIFFERENTIATION FOR OPTIMIZATION
# ============================================================================

print("\n" + "=" * 80)
print("7. MAXIMUM LIKELIHOOD ESTIMATION (MLE)")
print("=" * 80)

print("\n FINDING MLE FOR NORMAL DISTRIBUTION")
print("-" * 80)

# Log-likelihood for normal distribution
x_data = symbols('x_1:%d' % 6)
n_obs = symbols('n', positive=True, integer=True)

# Single observation likelihood
likelihood_single = (1 / (sigma * sqrt(2 * pi))) * exp(-(x - mu)**2 / (2 * sigma**2))
log_likelihood_single = sp.log(likelihood_single)

print("\nLog-likelihood for single observation:")
print(f"ℓ(μ, σ | x) = {simplify(log_likelihood_single)}")

# To find MLE for μ, take derivative and set to 0
print("\n FINDING MLE FOR μ:")
print("Take ∂ℓ/∂μ = 0")

dL_dmu = diff(log_likelihood_single, mu)
print(f"\n∂ℓ/∂μ = {simplify(dL_dmu)}")

# Solve for μ
mu_mle = sp.solve(dL_dmu, mu)
print(f"\nSetting to 0 and solving: μ̂ = {mu_mle}")
print("\n RESULT: MLE for μ is the sample mean x̄!")

print("\n FINDING MLE FOR σ²:")
print("Take ∂ℓ/∂σ² = 0")

# For variance
dL_dsigma2 = diff(log_likelihood_single, sigma**2)
print(f"\n∂ℓ/∂σ² = {simplify(dL_dsigma2)}")
print("\n RESULT: MLE for σ² is Σ(xᵢ - μ)²/n")
print("   (Note: This is biased; we use n-1 for unbiased estimator)")

print(" PART 5 COMPLETE: Symbolic Mathematics")
print("\nKey Takeaways:")
print("1. SymPy allows symbolic manipulation of formulas")
print("2. Can verify statistical properties algebraically")
print("3. Derive formulas from first principles")
print("4. Understand relationships between parameters")
print("5. Use calculus for optimization (MLE)")
print("\nYou now have a complete statistical toolkit!")
print("You've mastered descriptive statistics, hypothesis testing,")
print("A/B testing, power analysis, and symbolic mathematics!")