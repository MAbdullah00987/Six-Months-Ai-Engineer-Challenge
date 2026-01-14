
#Phase 6: Symbolic Mathematics - SymPy
#Topic 6: Mathematical Foundations with SymPy

import numpy as np
import sympy as sp
from sympy import symbols, sqrt, exp, pi, oo, integrate, diff, simplify, latex
from sympy.stats import Normal, Exponential, density, E, variance, std
import matplotlib.pyplot as plt

print("=" * 80)
print("SYMPY SYMBOLIC MATHEMATICS - CLT Theoretical Foundations")
print("=" * 80)

# 1. SYMBOLIC NORMAL DISTRIBUTION
print("\n1. NORMAL DISTRIBUTION - SYMBOLIC FORM")
print("-" * 80)

x, mu, sigma = symbols('x mu sigma', real=True, positive=True)
n = symbols('n', integer=True, positive=True)

# Define normal PDF symbolically
normal_pdf = (1 / (sigma * sqrt(2 * pi))) * exp(-((x - mu)**2) / (2 * sigma**2))

print("Normal PDF:")
sp.pprint(normal_pdf)
print("\nLaTeX form:")
print(f"f(x) = {latex(normal_pdf)}")

# Verify it integrates to 1
print("\nVerifying integral from -∞ to ∞ equals 1...")
integral_result = integrate(normal_pdf, (x, -oo, oo))
print(f"∫ f(x) dx = {integral_result}")

# 2. STANDARD ERROR FORMULA
print("\n2. STANDARD ERROR - SYMBOLIC DERIVATION")
print("-" * 80)

sigma_x = symbols('sigma_x', positive=True)
standard_error = sigma_x / sqrt(n)

print("Standard Error of the Mean:")
sp.pprint(standard_error)
print(f"\nSE(X̄) = {latex(standard_error)}")

# Show how SE changes with n
print("\nStandard Error for different sample sizes:")
for n_val in [10, 30, 50, 100]:
    se_val = standard_error.subs([(n, n_val), (sigma_x, 1)])
    print(f"  n={n_val:3d}: SE = {float(se_val):.4f}")

# 3. MOMENT GENERATING FUNCTIONS
print("\n3. MOMENT GENERATING FUNCTIONS")
print("-" * 80)

t = symbols('t', real=True)

# MGF of normal distribution
mgf_normal = exp(mu * t + (sigma**2 * t**2) / 2)
print("MGF of Normal(μ, σ²):")
sp.pprint(mgf_normal)

# MGF of sum of n independent normal variables
mgf_sum = mgf_normal**n
mgf_sum_simplified = simplify(mgf_sum)
print("\nMGF of sum of n i.i.d. Normal variables:")
sp.pprint(mgf_sum_simplified)

# MGF of sample mean
mgf_mean = mgf_sum.subs(t, t/n)
print("\nMGF of sample mean:")
sp.pprint(simplify(mgf_mean))

# 4. VARIANCE OF SAMPLE MEAN
print("\n4. VARIANCE OF SAMPLE MEAN - THEORETICAL")
print("-" * 80)

# Individual variance
var_x = sigma**2
print(f"Var(X) = {var_x}")

# Variance of sum
var_sum = n * var_x
print(f"Var(ΣX) = n·σ² = {var_sum}")

# Variance of mean
var_mean = var_sum / n**2
var_mean_simplified = simplify(var_mean)
print(f"Var(X̄) = Var(ΣX)/n² = {var_mean_simplified}")

# 5. CONFIDENCE INTERVAL FORMULA
print("\n5. CONFIDENCE INTERVAL DERIVATION")
print("-" * 80)

z_alpha, x_bar, s = symbols('z_alpha x_bar s', positive=True)

# Margin of error
margin_of_error = z_alpha * s / sqrt(n)
print("Margin of Error:")
sp.pprint(margin_of_error)

# Confidence interval bounds
ci_lower = x_bar - margin_of_error
ci_upper = x_bar + margin_of_error

print("\nConfidence Interval:")
print(f"Lower: {latex(ci_lower)}")
print(f"Upper: {latex(ci_upper)}")

# For 95% CI, z = 1.96
z_95 = 1.96
ci_lower_95 = ci_lower.subs(z_alpha, z_95)
ci_upper_95 = ci_upper.subs(z_alpha, z_95)
print(f"\n95% CI: [{latex(ci_lower_95)}, {latex(ci_upper_95)}]")

# 6. TAYLOR SERIES APPROXIMATION
print("\n6. TAYLOR SERIES - NORMAL APPROXIMATION")
print("-" * 80)

# Exponential distribution MGF: 1/(1-λt)
lam = symbols('lambda', positive=True)
mgf_exp = 1 / (1 - lam * t)

print("Exponential MGF:")
sp.pprint(mgf_exp)

# Taylor expansion around t=0
taylor_expansion = sp.series(mgf_exp, t, 0, n=4)
print("\nTaylor expansion (first 4 terms):")
sp.pprint(taylor_expansion)

# 7. LIMITING DISTRIBUTION
print("\n7. LIMITING BEHAVIOR (CLT)")
print("-" * 80)

# Standardized sample mean
x_bar_sym = symbols('bar_x')
z_score = (x_bar_sym - mu) / (sigma / sqrt(n))

print("Z-score transformation:")
sp.pprint(z_score)
print(f"\nZ = {latex(z_score)}")

# As n → ∞
print("\nAs n → ∞:")
limit_expr = sigma / sqrt(n)
limit_result = sp.limit(limit_expr, n, oo)
print(f"lim(σ/√n) = {limit_result}")
print("Therefore, Z → N(0, 1)")

# 8. PROBABILITY CALCULATIONS
print("\n8. PROBABILITY CALCULATIONS")
print("-" * 80)

from sympy.stats import P, cdf

# Define symbolic normal variable
X = Normal('X', mu, sigma)

# P(X ≤ a)
a = symbols('a', real=True)
prob_expr = cdf(X)(a)
print(f"P(X ≤ a) = {latex(prob_expr)}")

# Numerical example
mu_val, sigma_val, a_val = 100, 15, 110
prob_numerical = cdf(X)(a).subs([(mu, mu_val), (sigma, sigma_val), (a, a_val)])
print(f"\nExample: P(X ≤ 110) when μ=100, σ=15:")
print(f"  Result: {float(prob_numerical):.4f}")

# VISUALIZATION OF SYMBOLIC EXPRESSIONS
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('SymPy Mathematical Foundations for CLT', fontsize=18, fontweight='bold')

# Plot 1: Normal PDF for different parameters
ax1 = axes[0, 0]
x_vals = np.linspace(-10, 10, 1000)
for mu_val, sigma_val in [(0, 1), (0, 2), (2, 1)]:
    pdf_func = sp.lambdify(x, normal_pdf.subs([(mu, mu_val), (sigma, sigma_val)]), 'numpy')
    y_vals = pdf_func(x_vals)
    ax1.plot(x_vals, y_vals, linewidth=2, label=f'μ={mu_val}, σ={sigma_val}')
ax1.set_xlabel('x', fontweight='bold', fontsize=12)
ax1.set_ylabel('f(x)', fontweight='bold', fontsize=12)
ax1.set_title('Normal PDF: Different Parameters', fontweight='bold', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Standard Error vs Sample Size
ax2 = axes[0, 1]
n_vals = np.arange(1, 101)
se_vals = [float(standard_error.subs([(n, nv), (sigma_x, 1)])) for nv in n_vals]
ax2.plot(n_vals, se_vals, linewidth=2, color='coral')
ax2.set_xlabel('Sample Size (n)', fontweight='bold', fontsize=12)
ax2.set_ylabel('Standard Error', fontweight='bold', fontsize=12)
ax2.set_title('SE = σ/√n', fontweight='bold', fontsize=14)
ax2.grid(True, alpha=0.3)

# Add theoretical curve
ax2.plot(n_vals, 1/np.sqrt(n_vals), '--', linewidth=2, color='red', 
         alpha=0.7, label='Theoretical')
ax2.legend()

# Plot 3: Confidence Interval Widths
ax3 = axes[1, 0]
confidence_levels = [0.90, 0.95, 0.99]
z_values = [1.645, 1.96, 2.576]

for cl, z_val in zip(confidence_levels, z_values):
    ci_width = 2 * z_val / np.sqrt(n_vals)
    ax3.plot(n_vals, ci_width, linewidth=2, label=f'{cl*100:.0f}% CI')

ax3.set_xlabel('Sample Size (n)', fontweight='bold', fontsize=12)
ax3.set_ylabel('CI Width (2z·σ/√n)', fontweight='bold', fontsize=12)
ax3.set_title('Confidence Interval Width vs Sample Size', fontweight='bold', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Probability Density Evolution
ax4 = axes[1, 1]
x_range = np.linspace(-4, 4, 1000)

for n_val in [5, 10, 30, 100]:
    # Standard error decreases with n
    se = 1 / np.sqrt(n_val)
    pdf_vals = (1 / (se * np.sqrt(2 * np.pi))) * np.exp(-x_range**2 / (2 * se**2))
    ax4.plot(x_range, pdf_vals, linewidth=2, label=f'n={n_val}')

ax4.set_xlabel('Standardized Value', fontweight='bold', fontsize=12)
ax4.set_ylabel('Density', fontweight='bold', fontsize=12)
ax4.set_title('Sampling Distribution: N(0, 1/n)', fontweight='bold', fontsize=14)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sympy_mathematical_foundations.png', dpi=300, bbox_inches='tight')
plt.show()

# Export formulas for documentation
print("\n9. KEY FORMULAS (LaTeX Export)")
print("-" * 80)

formulas = {
    "Normal PDF": normal_pdf,
    "Standard Error": standard_error,
    "Variance of Mean": var_mean_simplified,
    "Z-score": z_score,
    "95% CI Lower": ci_lower_95,
    "95% CI Upper": ci_upper_95
}

for name, formula in formulas.items():
    print(f"\n{name}:")
    print(f"  LaTeX: {latex(formula)}")


print("SYMPY CAPABILITIES DEMONSTRATED:")
print("✓ Symbolic probability distributions")
print("✓ Calculus (integration, differentiation)")
print("✓ Moment generating functions")
print("✓ Limit calculations")
print("✓ Taylor series approximations")
print("✓ LaTeX formula generation")
