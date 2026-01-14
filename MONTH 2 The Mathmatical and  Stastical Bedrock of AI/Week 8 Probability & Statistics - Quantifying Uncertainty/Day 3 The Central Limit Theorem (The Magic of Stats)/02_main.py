


#Part Two: BAYES' THEOREM & BAYESIAN REASONING
#Master conditional probability and Bayesian inference


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import beta, norm, uniform, binom
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


# PART 1: UNDERSTANDING BAYES' THEOREM

def bayes_theorem_basics():
    """
    Explain and visualize Bayes' Theorem fundamentals
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    
    print("="*70)
    print("BAYES' THEOREM: THE FOUNDATION")
    print("="*70)
    
    print("\n📐 The Formula:")
    print("   P(Hypothesis|Evidence) = P(Evidence|Hypothesis) × P(Hypothesis)")
    print("                           ────────────────────────────────────────")
    print("                                    P(Evidence)")
    print("\n   Or more intuitively:")
    print("   POSTERIOR = (LIKELIHOOD × PRIOR) / MARGINAL")
    
    print("\n🔑 Components:")
    print("   • PRIOR P(H): What we believed before seeing evidence")
    print("   • LIKELIHOOD P(E|H): How probable is the evidence if hypothesis is true")
    print("   • MARGINAL P(E): Total probability of seeing the evidence")
    print("   • POSTERIOR P(H|E): Updated belief after seeing evidence")
    
    # Classic Example: Medical Test
    print("\n" + "="*70)
    print("EXAMPLE 1: MEDICAL DIAGNOSIS")
    print("="*70)
    
    # Disease prevalence (Prior)
    p_disease = 0.01  # 1% of population has disease
    p_no_disease = 1 - p_disease
    
    # Test accuracy (Likelihood)
    p_positive_given_disease = 0.95  # True Positive Rate (Sensitivity)
    p_negative_given_no_disease = 0.90  # True Negative Rate (Specificity)
    p_positive_given_no_disease = 1 - p_negative_given_no_disease  # False Positive
    
    # You test positive. What's the probability you have the disease?
    
    # Calculate P(Positive) - Marginal
    p_positive = (p_positive_given_disease * p_disease + 
                  p_positive_given_no_disease * p_no_disease)
    
    # Bayes' Theorem
    p_disease_given_positive = (p_positive_given_disease * p_disease) / p_positive
    
    print(f"\n🏥 Medical Test Scenario:")
    print(f"   Prior P(Disease): {p_disease*100:.1f}%")
    print(f"   Test Sensitivity: {p_positive_given_disease*100:.1f}%")
    print(f"   Test Specificity: {p_negative_given_no_disease*100:.1f}%")
    print(f"\n   You test POSITIVE.")
    print(f"   P(Disease|Positive) = {p_disease_given_positive*100:.2f}%")
    print(f"\n   💡 Even with a positive test, only ~{p_disease_given_positive*100:.0f}% chance!")
    print(f"      Why? The disease is rare (low prior)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Prior Distribution
    categories = ['No Disease', 'Disease']
    priors = [p_no_disease, p_disease]
    colors_prior = ['lightgreen', 'lightcoral']
    
    axes[0, 0].bar(categories, priors, color=colors_prior, edgecolor='black', linewidth=2)
    axes[0, 0].set_ylabel('Probability', fontsize=12)
    axes[0, 0].set_title('PRIOR: Before Testing\nP(Disease) = 1%', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, 1])
    for i, v in enumerate(priors):
        axes[0, 0].text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=11)
    
    # 2. Likelihood
    test_results = ['Negative', 'Positive']
    likelihood_disease = [1-p_positive_given_disease, p_positive_given_disease]
    likelihood_no_disease = [p_negative_given_no_disease, p_positive_given_no_disease]
    
    x = np.arange(len(test_results))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, likelihood_disease, width, label='Have Disease', 
                   color='lightcoral', edgecolor='black')
    axes[0, 1].bar(x + width/2, likelihood_no_disease, width, label='No Disease', 
                   color='lightgreen', edgecolor='black')
    axes[0, 1].set_ylabel('Probability', fontsize=12)
    axes[0, 1].set_title('LIKELIHOOD: Test Accuracy\nP(Test Result|Disease Status)', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(test_results)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    
    # 3. Joint Probabilities (numerator of Bayes)
    joint_positive_disease = p_positive_given_disease * p_disease
    joint_positive_no_disease = p_positive_given_no_disease * p_no_disease
    
    joints = [joint_positive_no_disease, joint_positive_disease]
    labels = ['Positive &\nNo Disease', 'Positive &\nDisease']
    colors_joint = ['orange', 'red']
    
    axes[1, 0].bar(labels, joints, color=colors_joint, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Probability', fontsize=12)
    axes[1, 0].set_title('JOINT: P(Positive AND Disease Status)\nNumerator of Bayes', 
                         fontsize=12, fontweight='bold')
    for i, v in enumerate(joints):
        axes[1, 0].text(i, v + 0.001, f'{v*100:.2f}%', ha='center', fontsize=10)
    
    # 4. Posterior Distribution
    p_no_disease_given_positive = 1 - p_disease_given_positive
    posteriors = [p_no_disease_given_positive, p_disease_given_positive]
    
    axes[1, 1].bar(categories, posteriors, color=colors_prior, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Probability', fontsize=12)
    axes[1, 1].set_title('POSTERIOR: After Positive Test\nP(Disease|Positive) = 8.7%', 
                         fontsize=12, fontweight='bold')
    axes[1, 1].set_ylim([0, 1])
    for i, v in enumerate(posteriors):
        axes[1, 1].text(i, v + 0.02, f'{v*100:.1f}%', ha='center', fontsize=11)
    
    # Add arrow annotation
    axes[1, 1].annotate('Updated from 1% to 8.7%!', 
                       xy=(1, p_disease_given_positive), 
                       xytext=(0.5, 0.5),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=11, color='red', fontweight='bold')
    
    plt.suptitle("Bayes' Theorem: Medical Diagnosis Example", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bayes_medical_example.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 2: CONDITIONAL PROBABILITY DEEP DIVE


def conditional_probability_examples():
    """
    Comprehensive conditional probability examples
    """
    
    print("\n" + "="*70)
    print("CONDITIONAL PROBABILITY: EXAMPLES")
    print("="*70)
    
    # Example 1: Card Drawing
    print("\n🎴 EXAMPLE 2: DRAWING CARDS")
    print("-" * 70)
    
    # Drawing cards without replacement
    total_cards = 52
    hearts = 13
    face_cards = 12
    heart_face_cards = 3
    
    p_heart = hearts / total_cards
    p_face_given_heart = heart_face_cards / hearts
    p_heart_given_face = heart_face_cards / face_cards
    
    print(f"   P(Heart) = {hearts}/{total_cards} = {p_heart:.4f}")
    print(f"   P(Face|Heart) = {heart_face_cards}/{hearts} = {p_face_given_heart:.4f}")
    print(f"   P(Heart|Face) = {heart_face_cards}/{face_cards} = {p_heart_given_face:.4f}")
    
    # Example 2: Weather and Traffic
    print("\n🌧️ EXAMPLE 3: WEATHER AND TRAFFIC")
    print("-" * 70)
    
    # Create contingency table
    data = {
        'Late': [30, 5],      # [Rainy, Sunny]
        'On Time': [20, 45]
    }
    
    df = pd.DataFrame(data, index=['Rainy', 'Sunny'])
    print("\n   Contingency Table:")
    print(df)
    
    total = df.sum().sum()
    p_rainy = df.loc['Rainy'].sum() / total
    p_late = df['Late'].sum() / total
    p_late_given_rainy = df.loc['Rainy', 'Late'] / df.loc['Rainy'].sum()
    p_rainy_given_late = df.loc['Rainy', 'Late'] / df['Late'].sum()
    
    print(f"\n   P(Rainy) = {p_rainy:.3f}")
    print(f"   P(Late) = {p_late:.3f}")
    print(f"   P(Late|Rainy) = {p_late_given_rainy:.3f}")
    print(f"   P(Rainy|Late) = {p_rainy_given_late:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Heatmap
    sns.heatmap(df, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0], 
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
    axes[0].set_title('Weather vs Arrival Time\nContingency Table', 
                     fontsize=12, fontweight='bold')
    
    # Conditional probabilities
    conditions = ['P(Late|Rainy)', 'P(Late|Sunny)', 'P(Rainy|Late)', 'P(Sunny|Late)']
    probs = [
        p_late_given_rainy,
        df.loc['Sunny', 'Late'] / df.loc['Sunny'].sum(),
        p_rainy_given_late,
        df.loc['Sunny', 'Late'] / df['Late'].sum()
    ]
    colors_cond = ['coral', 'lightblue', 'orange', 'yellow']
    
    axes[1].bar(conditions, probs, color=colors_cond, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_title('Conditional Probabilities', fontsize=12, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    for i, v in enumerate(probs):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('conditional_probability_examples.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 3: BAYESIAN UPDATING - COIN FLIP EXAMPLE


def bayesian_updating_coin_flip():
    """
    Demonstrates how beliefs update with evidence
    Classic coin fairness problem
    """
    
    print("\n" + "="*70)
    print("BAYESIAN UPDATING: COIN FLIP EXAMPLE")
    print("="*70)
    
    # We have a coin. Is it fair?
    # Prior: Uniform distribution over possible probabilities of heads
    
    # Generate coin flips (true probability is 0.7)
    true_prob = 0.7
    n_flips = [1, 5, 10, 50, 100, 500]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Prior belief: Beta(1, 1) = Uniform
    alpha_prior = 1
    beta_prior = 1
    
    np.random.seed(42)
    
    for idx, n in enumerate(n_flips):
        # Simulate coin flips
        flips = np.random.binomial(1, true_prob, n)
        n_heads = np.sum(flips)
        n_tails = n - n_heads
        
        # Update posterior: Beta(alpha + heads, beta + tails)
        alpha_post = alpha_prior + n_heads
        beta_post = beta_prior + n_tails
        
        # Plot
        x = np.linspace(0, 1, 1000)
        
        # Prior
        if idx == 0:
            prior_dist = beta.pdf(x, alpha_prior, beta_prior)
            axes[idx].plot(x, prior_dist, 'b--', linewidth=2, label='Prior', alpha=0.7)
        
        # Posterior
        posterior_dist = beta.pdf(x, alpha_post, beta_post)
        axes[idx].plot(x, posterior_dist, 'r-', linewidth=2, label='Posterior')
        axes[idx].fill_between(x, posterior_dist, alpha=0.3, color='red')
        
        # True value
        axes[idx].axvline(true_prob, color='green', linestyle='--', linewidth=2, 
                         label=f'True p={true_prob}')
        
        # Posterior mean
        post_mean = alpha_post / (alpha_post + beta_post)
        axes[idx].axvline(post_mean, color='orange', linestyle=':', linewidth=2, 
                         label=f'Posterior Mean={post_mean:.3f}')
        
        axes[idx].set_xlabel('Probability of Heads', fontsize=11)
        axes[idx].set_ylabel('Density', fontsize=11)
        axes[idx].set_title(f'After {n} flips: {n_heads} heads, {n_tails} tails', 
                           fontsize=11, fontweight='bold')
        axes[idx].legend()
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(True, alpha=0.3)
        
        print(f"\n   After {n:3d} flips: Posterior Mean = {post_mean:.4f}, " + 
              f"95% CI = [{beta.ppf(0.025, alpha_post, beta_post):.3f}, " + 
              f"{beta.ppf(0.975, alpha_post, beta_post):.3f}]")
    
    plt.suptitle('Bayesian Updating: Estimating Coin Bias', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bayesian_updating_coin.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n   💡 Notice how:")
    print("      • Prior is uniform (no strong belief)")
    print("      • Posterior concentrates around true value")
    print("      • More data → narrower posterior (more certainty)")


# PART 4: BAYESIAN A/B TESTING


def bayesian_ab_testing():
    """
    Compare two website designs using Bayesian methods
    """
    
    print("\n" + "="*70)
    print("BAYESIAN A/B TESTING")
    print("="*70)
    
    # Website A: 120 conversions out of 1000 visitors
    # Website B: 135 conversions out of 1000 visitors
    
    n_a, conv_a = 1000, 120
    n_b, conv_b = 1000, 135
    
    print(f"\n🌐 Website Conversion Rates:")
    print(f"   Design A: {conv_a}/{n_a} = {conv_a/n_a:.1%}")
    print(f"   Design B: {conv_b}/{n_b} = {conv_b/n_b:.1%}")
    
    # Prior: Beta(1, 1) for both
    prior_a = beta(1, 1)
    prior_b = beta(1, 1)
    
    # Posterior distributions
    post_a = beta(1 + conv_a, 1 + n_a - conv_a)
    post_b = beta(1 + conv_b, 1 + n_b - conv_b)
    
    # Sample from posteriors
    samples_a = post_a.rvs(100000)
    samples_b = post_b.rvs(100000)
    
    # Probability that B > A
    prob_b_better = np.mean(samples_b > samples_a)
    
    # Expected improvement
    improvement = samples_b - samples_a
    mean_improvement = np.mean(improvement)
    
    print(f"\n📊 Bayesian Analysis:")
    print(f"   P(B > A) = {prob_b_better:.1%}")
    print(f"   Expected Improvement: {mean_improvement:.1%}")
    print(f"   95% Credible Interval of Improvement: " + 
          f"[{np.percentile(improvement, 2.5):.1%}, {np.percentile(improvement, 97.5):.1%}]")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Posterior distributions
    x = np.linspace(0, 0.20, 1000)
    axes[0, 0].plot(x, post_a.pdf(x), 'b-', linewidth=2, label='Design A')
    axes[0, 0].plot(x, post_b.pdf(x), 'r-', linewidth=2, label='Design B')
    axes[0, 0].fill_between(x, post_a.pdf(x), alpha=0.3, color='blue')
    axes[0, 0].fill_between(x, post_b.pdf(x), alpha=0.3, color='red')
    axes[0, 0].axvline(conv_a/n_a, color='blue', linestyle='--', linewidth=2)
    axes[0, 0].axvline(conv_b/n_b, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Conversion Rate', fontsize=12)
    axes[0, 0].set_ylabel('Probability Density', fontsize=12)
    axes[0, 0].set_title('Posterior Distributions', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Difference distribution
    axes[0, 1].hist(improvement * 100, bins=50, density=True, alpha=0.7, 
                   color='purple', edgecolor='black')
    axes[0, 1].axvline(0, color='black', linestyle='--', linewidth=2, label='No difference')
    axes[0, 1].axvline(mean_improvement * 100, color='red', linewidth=2, 
                      label=f'Mean = {mean_improvement:.2%}')
    axes[0, 1].set_xlabel('Improvement (B - A) in %', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Distribution of Improvement', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Probability that B beats A
    categories = ['A is Better', 'B is Better']
    probs = [1 - prob_b_better, prob_b_better]
    colors_prob = ['lightblue', 'lightcoral']
    
    axes[1, 0].bar(categories, probs, color=colors_prob, edgecolor='black', linewidth=2)
    axes[1, 0].set_ylabel('Probability', fontsize=12)
    axes[1, 0].set_title(f'P(B > A) = {prob_b_better:.1%}', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylim([0, 1])
    for i, v in enumerate(probs):
        axes[1, 0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Decision recommendation
    if prob_b_better > 0.95:
        decision = "CHOOSE B"
        decision_color = 'green'
        reason = f"Strong evidence (>{0.95:.0%})"
    elif prob_b_better > 0.90:
        decision = "Likely B"
        decision_color = 'lightgreen'
        reason = f"Good evidence (>{0.90:.0%})"
    elif prob_b_better < 0.10:
        decision = "CHOOSE A"
        decision_color = 'red'
        reason = f"Strong evidence (>{0.90:.0%})"
    elif prob_b_better < 0.05:
        decision = "Likely A"
        decision_color = 'lightcoral'
        reason = f"Good evidence (>{0.95:.0%})"
    else:
        decision = "INCONCLUSIVE"
        decision_color = 'yellow'
        reason = "Need more data"
    
    axes[1, 1].text(0.5, 0.6, decision, ha='center', va='center', 
                   fontsize=32, fontweight='bold', color=decision_color,
                   bbox=dict(boxstyle='round', facecolor=decision_color, alpha=0.3, pad=1))
    axes[1, 1].text(0.5, 0.3, reason, ha='center', va='center', fontsize=16)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Decision', fontsize=12, fontweight='bold')
    
    plt.suptitle('Bayesian A/B Testing', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bayesian_ab_testing.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 5: BAYESIAN LINEAR REGRESSION

def bayesian_linear_regression():
    """
    Bayesian approach to linear regression with uncertainty quantification
    """
    
    print("\n" + "="*70)
    print("BAYESIAN LINEAR REGRESSION")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    n = 50
    true_slope = 2.5
    true_intercept = 1.0
    noise_std = 2.0
    
    x = np.linspace(0, 10, n)
    y = true_intercept + true_slope * x + np.random.normal(0, noise_std, n)
    
    # Frequentist fit
    from scipy.stats import linregress
    freq_result = linregress(x, y)
    
    # Bayesian fit using sampling
    # Prior: vague normal priors
    n_samples = 5000
    
    # Sample from posterior (simplified using known formulas)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ X.T @ y
    
    residuals = y - X @ beta_hat
    sigma_sq = np.sum(residuals**2) / (n - 2)
    
    # Posterior for coefficients (multivariate normal)
    cov_matrix = sigma_sq * XtX_inv
    posterior_samples = np.random.multivariate_normal(beta_hat, cov_matrix, n_samples)
    
    intercept_samples = posterior_samples[:, 0]
    slope_samples = posterior_samples[:, 1]
    
    print(f"\n📈 Regression Analysis:")
    print(f"   True slope: {true_slope:.2f}")
    print(f"   True intercept: {true_intercept:.2f}")
    print(f"\n   Frequentist:")
    print(f"      Slope: {freq_result.slope:.2f} ± {freq_result.stderr:.2f}")
    print(f"      Intercept: {freq_result.intercept:.2f}")
    print(f"\n   Bayesian:")
    print(f"      Slope: {np.mean(slope_samples):.2f} " + 
          f"[{np.percentile(slope_samples, 2.5):.2f}, {np.percentile(slope_samples, 97.5):.2f}]")
    print(f"      Intercept: {np.mean(intercept_samples):.2f} " + 
          f"[{np.percentile(intercept_samples, 2.5):.2f}, {np.percentile(intercept_samples, 97.5):.2f}]")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Data with uncertainty bands
    axes[0, 0].scatter(x, y, alpha=0.6, s=50, color='blue', edgecolor='black', 
                      label='Data', zorder=3)
    
    # Plot many posterior lines
    x_pred = np.linspace(0, 10, 100)
    for i in range(200):
        y_pred = intercept_samples[i] + slope_samples[i] * x_pred
        axes[0, 0].plot(x_pred, y_pred, 'gray', alpha=0.02, linewidth=1)
    
    # Mean prediction
    y_mean = np.mean(intercept_samples) + np.mean(slope_samples) * x_pred
    axes[0, 0].plot(x_pred, y_mean, 'r-', linewidth=3, label='Posterior Mean', zorder=4)
    
    # True line
    y_true = true_intercept + true_slope * x_pred
    axes[0, 0].plot(x_pred, y_true, 'g--', linewidth=2, label='True Line', zorder=5)
    
    axes[0, 0].set_xlabel('X', fontsize=12)
    axes[0, 0].set_ylabel('Y', fontsize=12)
    axes[0, 0].set_title('Bayesian Linear Regression\nwith Posterior Uncertainty', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Slope posterior
    axes[0, 1].hist(slope_samples, bins=50, density=True, alpha=0.7, 
                   color='coral', edgecolor='black')
    axes[0, 1].axvline(true_slope, color='green', linestyle='--', linewidth=2, 
                      label=f'True = {true_slope:.2f}')
    axes[0, 1].axvline(np.mean(slope_samples), color='red', linewidth=2, 
                      label=f'Mean = {np.mean(slope_samples):.2f}')
    axes[0, 1].set_xlabel('Slope', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Posterior Distribution of Slope', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Intercept posterior
    axes[1, 0].hist(intercept_samples, bins=50, density=True, alpha=0.7, 
                   color='skyblue', edgecolor='black')
    axes[1, 0].axvline(true_intercept, color='green', linestyle='--', linewidth=2, 
                      label=f'True = {true_intercept:.2f}')
    axes[1, 0].axvline(np.mean(intercept_samples), color='red', linewidth=2, 
                      label=f'Mean = {np.mean(intercept_samples):.2f}')
    axes[1, 0].set_xlabel('Intercept', fontsize=12)
    axes[1, 0].set_ylabel('Density', fontsize=12)
    axes[1, 0].set_title('Posterior Distribution of Intercept', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Joint posterior
    axes[1, 1].scatter(intercept_samples, slope_samples, alpha=0.1, s=1, color='purple')
    axes[1, 1].scatter([true_intercept], [true_slope], color='green', s=200, marker='*', 
                      edgecolor='black', linewidth=2, label='True Values', zorder=5)
    axes[1, 1].scatter([np.mean(intercept_samples)], [np.mean(slope_samples)], 
                      color='red', s=100, marker='o', edgecolor='black', linewidth=2, 
                      label='Posterior Mean', zorder=5)
    axes[1, 1].set_xlabel('Intercept', fontsize=12)
    axes[1, 1].set_ylabel('Slope', fontsize=12)
    axes[1, 1].set_title('Joint Posterior Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Bayesian Linear Regression', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('bayesian_linear_regression.png', dpi=300, bbox_inches='tight')
    plt.show()


# PART 6: NAIVE BAYES CLASSIFIER


def naive_bayes_classifier():
    """
    Build a Naive Bayes classifier from scratch
    """
    
    print("\n" + "="*70)
    print("NAIVE BAYES CLASSIFIER")
    print("="*70)
    
    # Email spam classification example
    # Features: contains "free", contains "money", word count
    
    # Training data
    emails = pd.DataFrame({
        'has_free': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1],
        'has_money': [1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
        'word_count': [150, 80, 200, 120, 50, 300, 60, 250, 100, 90],
        'spam': [1, 1, 0, 0, 1, 0, 1, 0, 1, 1]  # 1=spam, 0=ham
    })
    
    print("\n📧 Email Dataset:")
    print(emails)
    
    # Calculate priors
    p_spam = emails['spam'].sum() / len(emails)
    p_ham = 1 - p_spam
    
    print(f"\n📊 Prior Probabilities:")
    print(f"   P(Spam) = {p_spam:.2f}")
    print(f"   P(Ham) = {p_ham:.2f}")
    
    # Calculate likelihoods
    spam_emails = emails[emails['spam'] == 1]
    ham_emails = emails[emails['spam'] == 0]
    
    # P(has_free | spam)
    p_free_spam = spam_emails['has_free'].sum() / len(spam_emails)
    p_free_ham = ham_emails['has_free'].sum() / len(ham_emails)
    
    # P(has_money | spam)
    p_money_spam = spam_emails['has_money'].sum() / len(spam_emails)
    p_money_ham = ham_emails['has_money'].sum() / len(ham_emails)
    
    print(f"\n📈 Likelihoods:")
    print(f"   P('free'|Spam) = {p_free_spam:.2f},  P('free'|Ham) = {p_free_ham:.2f}")
    print(f"   P('money'|Spam) = {p_money_spam:.2f}, P('money'|Ham) = {p_money_ham:.2f}")
    
    # Classify new email: has "free", has "money"
    new_email = {'has_free': 1, 'has_money': 1}
    
    # P(spam | features) ∝ P(features | spam) * P(spam)
    # Assuming independence (Naive Bayes assumption)
    
    p_features_given_spam = p_free_spam * p_money_spam
    p_features_given_ham = p_free_ham * p_money_ham
    
    # Unnormalized posteriors
    spam_score = p_features_given_spam * p_spam
    ham_score = p_features_given_ham * p_ham
    
    # Normalize
    total = spam_score + ham_score
    p_spam_given_features = spam_score / total
    p_ham_given_features = ham_score / total
    
    print(f"\n🔍 Classifying new email: {new_email}")
    print(f"   P(Spam|features) = {p_spam_given_features:.3f}")
    print(f"   P(Ham|features) = {p_ham_given_features:.3f}")
    print(f"   → Classification: {'SPAM' if p_spam_given_features > 0.5 else 'HAM'}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Feature distributions
    features = ['has_free', 'has_money']
    spam_probs = [p_free_spam, p_money_spam]
    ham_probs = [p_free_ham, p_money_ham]
    
    x = np.arange(len(features))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, spam_probs, width, label='Spam', 
                   color='red', alpha=0.7, edgecolor='black')
    axes[0, 0].bar(x + width/2, ham_probs, width, label='Ham', 
                   color='green', alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('P(Feature|Class)', fontsize=12)
    axes[0, 0].set_title('Feature Likelihoods', fontsize=12, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(features)
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # 2. Prior vs Posterior
    stages = ['Prior', 'Posterior']
    spam_evolution = [p_spam, p_spam_given_features]
    ham_evolution = [p_ham, p_ham_given_features]
    
    x = np.arange(len(stages))
    axes[0, 1].bar(x - width/2, spam_evolution, width, label='Spam', 
                   color='red', alpha=0.7, edgecolor='black')
    axes[0, 1].bar(x + width/2, ham_evolution, width, label='Ham', 
                   color='green', alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('Probability', fontsize=12)
    axes[0, 1].set_title('Bayesian Updating', fontsize=12, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(stages)
    axes[0, 1].legend()
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Confusion matrix (on training data)
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import confusion_matrix
    
    X = emails[['has_free', 'has_money']].values
    y = emails['spam'].values
    
    nb = BernoulliNB()
    nb.fit(X, y)
    y_pred = nb.predict(X)
    
    cm = confusion_matrix(y, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
    axes[1, 0].set_ylabel('True Label', fontsize=12)
    axes[1, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[1, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    
    # 4. Decision boundary
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    Z = nb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    contour = axes[1, 1].contourf(xx, yy, Z, levels=20, cmap='RdYlGn_r', alpha=0.6)
    axes[1, 1].scatter(emails[emails['spam']==0]['has_free'], 
                      emails[emails['spam']==0]['has_money'],
                      c='green', s=100, edgecolor='black', linewidth=2, 
                      label='Ham', marker='o')
    axes[1, 1].scatter(emails[emails['spam']==1]['has_free'], 
                      emails[emails['spam']==1]['has_money'],
                      c='red', s=100, edgecolor='black', linewidth=2, 
                      label='Spam', marker='X')
    axes[1, 1].set_xlabel('Has "free"', fontsize=12)
    axes[1, 1].set_ylabel('Has "money"', fontsize=12)
    axes[1, 1].set_title('Decision Boundary', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    plt.colorbar(contour, ax=axes[1, 1], label='P(Spam)')
    
    plt.suptitle('Naive Bayes Classifier', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('naive_bayes_classifier.png', dpi=300, bbox_inches='tight')
    plt.show()


# RUN ALL DEMONSTRATIONS


if __name__ == "__main__":
    print("\n" + "="*70)
    print("BAYES' THEOREM & BAYESIAN REASONING")
    print("="*70)
    
    # Part 1: Basics
    bayes_theorem_basics()
    
    # Part 2: Conditional probability
    conditional_probability_examples()
    
    # Part 3: Bayesian updating
    bayesian_updating_coin_flip()
    
    # Part 4: A/B testing
    bayesian_ab_testing()
    
    # Part 5: Linear regression
    bayesian_linear_regression()
    
    # Part 6: Naive Bayes
    naive_bayes_classifier()
    
    
    print("🎓 KEY TAKEAWAYS:")
    print("1. Bayes: Posterior = (Likelihood × Prior) / Marginal")
    print("2. Update beliefs as new evidence arrives")
    print("3. Quantify uncertainty with probability distributions")
    print("4. No p-values! Direct probability statements")
    print("5. Incorporates prior knowledge naturally")
    print("6. Applications: diagnosis, A/B testing, ML, regression")
    