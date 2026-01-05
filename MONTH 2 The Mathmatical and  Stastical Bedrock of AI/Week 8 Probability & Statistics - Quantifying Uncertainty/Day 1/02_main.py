
#Part 2: Foundations of Probability - Building Intuition Through Code

#Probability Concepts
# Sample Spaces and Events

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import symbols, Eq, solve, simplify, latex
from IPython.display import display, Math

'''
def simulate_coin_flips(n_flips=10000):
    """Simulate coin flips and verify probability = 0.5"""
    flips = np.random.choice(['H', 'T'], size=n_flips)
    
    prob_heads = np.sum(flips == 'H') / n_flips
    prob_tails = np.sum(flips == 'T') / n_flips
    
    print("COIN FLIP SIMULATION")
    print(f"Number of flips: {n_flips}")
    print(f"Probability of Heads: {prob_heads:.4f}")
    print(f"Probability of Tails: {prob_tails:.4f}")
    
    # Show convergence to 0.5
    cumulative_heads = np.cumsum(flips == 'H') / np.arange(1, n_flips + 1)
    
    plt.figure(figsize=(12, 5))
    plt.plot(cumulative_heads, linewidth=0.8, alpha=0.7)
    plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='True Probability = 0.5')
    plt.xlabel('Number of Flips')
    plt.ylabel('Cumulative Probability of Heads')
    plt.title('Law of Large Numbers: Probability Converges to 0.5', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return flips

flips = simulate_coin_flips(10000)

# EXAMPLE 2: Dice Rolls
def simulate_dice_rolls(n_rolls=10000):
    """Simulate dice rolls and verify uniform distribution"""
    rolls = np.random.randint(1, 7, size=n_rolls)
    
    counts = Counter(rolls)
    probabilities = {k: v/n_rolls for k, v in counts.items()}
    
    print("\nDICE ROLL SIMULATION")
    print(f"Number of rolls: {n_rolls}")
    for face in range(1, 7):
        print(f"P(Face={face}): {probabilities[face]:.4f} (Expected: {1/6:.4f})")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart
    faces = list(range(1, 7))
    probs = [probabilities[f] for f in faces]
    axes[0].bar(faces, probs, color='lightcoral', edgecolor='black', alpha=0.7)
    axes[0].axhline(1/6, color='blue', linestyle='--', linewidth=2, label='Expected: 1/6')
    axes[0].set_xlabel('Dice Face')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Dice Roll Probabilities', fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Convergence plot
    cumulative_probs = []
    for i in range(1, n_rolls + 1):
        cumulative_probs.append(np.sum(rolls[:i] == 1) / i)
    
    axes[1].plot(cumulative_probs, linewidth=0.8, alpha=0.7)
    axes[1].axhline(1/6, color='red', linestyle='--', linewidth=2, label='Expected: 1/6')
    axes[1].set_xlabel('Number of Rolls')
    axes[1].set_ylabel('Cumulative P(Face=1)')
    axes[1].set_title('Convergence to True Probability', fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return rolls

rolls = simulate_dice_rolls(10000)
'''

#Joint Probability and Independence
##Two Dice Rolls - Joint Probability
'''
def analyze_two_dice():
    """Analyze joint probability of rolling two dice"""
    
    # Generate all possible outcomes
    die1 = range(1, 7)
    die2 = range(1, 7)
    sample_space = list(product(die1, die2))
    
    print(f"Sample Space Size: {len(sample_space)}")
    print(f"First 10 outcomes: {sample_space[:10]}")
    
    # Create joint probability table
    joint_prob_table = np.zeros((6, 6))
    for i in die1:
        for j in die2:
            joint_prob_table[i-1, j-1] = 1/36  # Each outcome equally likely
    
    # Calculate sum probabilities
    sums = [d1 + d2 for d1, d2 in sample_space]
    sum_counts = Counter(sums)
    sum_probs = {k: v/36 for k, v in sum_counts.items()}
    
    print("\nPROBABILITY OF SUMS:")
    for s in sorted(sum_probs.keys()):
        print(f"P(Sum={s:2d}) = {sum_probs[s]:.4f} ({sum_counts[s]}/36)")
    
    # Visualizations
    fig = plt.figure(figsize=(16, 5))
    
    # Joint probability heatmap
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(joint_prob_table, annot=True, fmt='.4f', cmap='YlOrRd', 
                xticklabels=range(1,7), yticklabels=range(1,7),
                cbar_kws={'label': 'Probability'})
    ax1.set_xlabel('Die 2')
    ax1.set_ylabel('Die 1')
    ax1.set_title('Joint Probability P(Die1, Die2)', fontweight='bold')
    
    # Sum distribution
    ax2 = plt.subplot(1, 3, 2)
    sums_sorted = sorted(sum_probs.keys())
    probs_sorted = [sum_probs[s] for s in sums_sorted]
    ax2.bar(sums_sorted, probs_sorted, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Sum of Two Dice')
    ax2.set_ylabel('Probability')
    ax2.set_title('Distribution of Sum', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Simulation verification
    ax3 = plt.subplot(1, 3, 3)
    n_sims = 10000
    sim_rolls = np.random.randint(1, 7, size=(n_sims, 2))
    sim_sums = sim_rolls.sum(axis=1)
    sim_counts = Counter(sim_sums)
    sim_probs = {k: v/n_sims for k in range(2, 13) if k in sim_counts}
    
    x = list(range(2, 13))
    theoretical = [sum_probs.get(i, 0) for i in x]
    simulated = [sim_probs.get(i, 0) for i in x]
    
    x_pos = np.arange(len(x))
    width = 0.35
    ax3.bar(x_pos - width/2, theoretical, width, label='Theoretical', color='green', alpha=0.7)
    ax3.bar(x_pos + width/2, simulated, width, label='Simulated', color='orange', alpha=0.7)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(x)
    ax3.set_xlabel('Sum')
    ax3.set_ylabel('Probability')
    ax3.set_title('Theoretical vs Simulated', fontweight='bold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Test independence
    print("\nINDEPENDENCE TEST:")
    print("For independent events: P(A ∩ B) = P(A) × P(B)")
    print(f"P(Die1=3) = {1/6:.4f}")
    print(f"P(Die2=5) = {1/6:.4f}")
    print(f"P(Die1=3 AND Die2=5) = {joint_prob_table[2, 4]:.4f}")
    print(f"P(Die1=3) × P(Die2=5) = {(1/6) * (1/6):.4f}")
    print(f"Independent: {np.isclose(joint_prob_table[2, 4], 1/36)}")

analyze_two_dice()
'''

#Conditional Probability
#Understanding P(A|B)


#Medical Test Accuracy
'''
def medical_test_example():
    """
    Understanding conditional probability through medical testing
    
    Given:
    - P(Disease) = 0.01 (1% of population has disease)
    - P(Positive | Disease) = 0.99 (99% sensitivity)
    - P(Negative | No Disease) = 0.95 (95% specificity)
    
    Find: P(Disease | Positive) = ?
    """
    
    # Population parameters
    population_size = 100000
    
    # Disease prevalence
    p_disease = 0.01
    p_no_disease = 1 - p_disease
    
    # Test characteristics
    sensitivity = 0.99  # P(Positive | Disease)
    specificity = 0.95  # P(Negative | No Disease)
    
    # Calculate population breakdown
    n_disease = int(population_size * p_disease)
    n_no_disease = population_size - n_disease
    
    # True Positives: Have disease AND test positive
    true_positive = int(n_disease * sensitivity)
    
    # False Negatives: Have disease BUT test negative
    false_negative = n_disease - true_positive
    
    # True Negatives: No disease AND test negative
    true_negative = int(n_no_disease * specificity)
    
    # False Positives: No disease BUT test positive
    false_positive = n_no_disease - true_negative
    
    # Total positive tests
    total_positive = true_positive + false_positive
    
    # P(Disease | Positive) - This is what we want!
    p_disease_given_positive = true_positive / total_positive
    
    print("MEDICAL TEST CONDITIONAL PROBABILITY")
    print("=" * 60)
    print(f"Population Size: {population_size:,}")
    print(f"Disease Prevalence: {p_disease:.1%}")
    print(f"Test Sensitivity: {sensitivity:.1%}")
    print(f"Test Specificity: {specificity:.1%}")
    print()
    print("POPULATION BREAKDOWN:")
    print(f"  People with disease: {n_disease:,}")
    print(f"  People without disease: {n_no_disease:,}")
    print()
    print("TEST RESULTS:")
    print(f"  True Positives (TP):  {true_positive:,}")
    print(f"  False Positives (FP): {false_positive:,}")
    print(f"  True Negatives (TN):  {true_negative:,}")
    print(f"  False Negatives (FN): {false_negative:,}")
    print()
    print("KEY QUESTION:")
    print(f"  If you test POSITIVE, what's the probability you have the disease?")
    print(f"  P(Disease | Positive) = {p_disease_given_positive:.1%}")
    print()
    print("INSIGHT: Even with 99% sensitivity, only 16.5% of positive")
    print("         tests indicate actual disease due to low prevalence!")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion Matrix
    confusion_matrix = np.array([[true_positive, false_negative],
                                  [false_positive, true_negative]])
    
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positive Test', 'Negative Test'],
                yticklabels=['Has Disease', 'No Disease'],
                cbar_kws={'label': 'Count'}, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Population = 100,000)', fontweight='bold')
    axes[0].set_ylabel('True Condition')
    axes[0].set_xlabel('Test Result')
    
    # Conditional probabilities
    categories = ['P(Positive|Disease)\n(Sensitivity)', 
                  'P(Negative|No Disease)\n(Specificity)',
                  'P(Disease|Positive)\n(PPV)']
    values = [sensitivity, specificity, p_disease_given_positive]
    colors = ['green', 'green', 'red']
    
    bars = axes[1].bar(categories, values, color=colors, edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Conditional Probabilities Comparison', fontweight='bold')
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{val:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'true_positive': true_positive,
        'false_positive': false_positive,
        'true_negative': true_negative,
        'false_negative': false_negative,
        'p_disease_given_positive': p_disease_given_positive
    }

results = medical_test_example()'''


#Bayes' Theorem - The Foundation

# Bayes' Theorem Symbolic Derivation using SymPy
def bayes_theorem_symbolic():
    """Derive Bayes' Theorem symbolically"""
    print("BAYES' THEOREM SYMBOLIC DERIVATION")
    print("=" * 60)
    
    # Define symbols
    P_A_given_B = symbols('P(A|B)', positive=True)
    P_B_given_A = symbols('P(B|A)', positive=True)
    P_A = symbols('P(A)', positive=True)
    P_B = symbols('P(B)', positive=True)
    
    # Conditional probability definition
    print("Starting from conditional probability definition:")
    print("P(A|B) = P(A ∩ B) / P(B)")
    print("P(B|A) = P(A ∩ B) / P(A)")
    print()
    
    # Since both equal P(A ∩ B), we can equate:
    print("Therefore: P(A|B) × P(B) = P(B|A) × P(A)")
    print()
    
    # Solve for P(A|B)
    print("Solving for P(A|B):")
    print("P(A|B) = [P(B|A) × P(A)] / P(B)")
    print()
    print("This is BAYES' THEOREM!")
    print()
    
    # Expanded form with law of total probability
    print("Expanded form using Law of Total Probability:")
    print("P(A|B) = [P(B|A) × P(A)] / [P(B|A)×P(A) + P(B|¬A)×P(¬A)]")
    
    return None

bayes_theorem_symbolic()

# Practical Application: Spam Email Filter
def spam_filter_bayes():
    """
    Apply Bayes' Theorem to spam detection
    
    Given:
    - P(Spam) = 0.30 (30% of emails are spam)
    - P(Contains "Win" | Spam) = 0.70
    - P(Contains "Win" | Not Spam) = 0.05
    
    Find: P(Spam | Contains "Win") = ?
    """
    
    # Prior probabilities
    p_spam = 0.30
    p_not_spam = 0.70
    
    # Likelihoods
    p_win_given_spam = 0.70
    p_win_given_not_spam = 0.05
    
    # Evidence: P(Contains "Win")
    # Using law of total probability
    p_win = (p_win_given_spam * p_spam) + (p_win_given_not_spam * p_not_spam)
    
    # Apply Bayes' Theorem
    p_spam_given_win = (p_win_given_spam * p_spam) / p_win
    
    print("\nSPAM FILTER USING BAYES' THEOREM")
    print("=" * 60)
    print("GIVEN:")
    print(f"  P(Spam) = {p_spam:.2f}")
    print(f"  P(Not Spam) = {p_not_spam:.2f}")
    print(f"  P(Contains 'Win' | Spam) = {p_win_given_spam:.2f}")
    print(f"  P(Contains 'Win' | Not Spam) = {p_win_given_not_spam:.2f}")
    print()
    print("CALCULATION:")
    print(f"  P(Contains 'Win') = {p_win:.4f}")
    print()
    print(f"  P(Spam | Contains 'Win') = {p_spam_given_win:.4f}")
    print()
    print(f"RESULT: If email contains 'Win', {p_spam_given_win:.1%} chance it's spam!")
    
    # Visualize Bayes' update
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Prior vs Posterior
    categories = ['Prior\nP(Spam)', 'Posterior\nP(Spam|"Win")']
    values = [p_spam, p_spam_given_win]
    colors = ['skyblue', 'red']
    
    bars = axes[0].bar(categories, values, color=colors, edgecolor='black', alpha=0.7, width=0.6)
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Bayesian Update: How "Win" Changes Our Belief', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.03,
                     f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Likelihood comparison
    words = ['P("Win"|Spam)', 'P("Win"|Not Spam)']
    likelihoods = [p_win_given_spam, p_win_given_not_spam]
    
    bars2 = axes[1].bar(words, likelihoods, color=['red', 'green'], 
                        edgecolor='black', alpha=0.7)
    axes[1].set_ylabel('Probability')
    axes[1].set_title('Likelihood Comparison', fontweight='bold')
    axes[1].set_ylim(0, 0.8)
    axes[1].grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars2, likelihoods):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return p_spam_given_win

prob = spam_filter_bayes()

# Multiple evidence combination
def bayes_multiple_evidence():
    """
    Extend to multiple pieces of evidence
    Email contains both "Win" and "Free"
    """
    
    # Priors
    p_spam = 0.30
    p_not_spam = 0.70
    
    # Individual likelihoods
    p_win_given_spam = 0.70
    p_win_given_not_spam = 0.05
    p_free_given_spam = 0.60
    p_free_given_not_spam = 0.10
    
    # Assuming independence of words (Naive Bayes assumption)
    # P(Win AND Free | Spam) = P(Win|Spam) × P(Free|Spam)
    p_both_given_spam = p_win_given_spam * p_free_given_spam
    p_both_given_not_spam = p_win_given_not_spam * p_free_given_not_spam
    
    # Total probability of evidence
    p_both = (p_both_given_spam * p_spam) + (p_both_given_not_spam * p_not_spam)
    
    # Apply Bayes
    p_spam_given_both = (p_both_given_spam * p_spam) / p_both
    
    print("\nNAIVE BAYES WITH MULTIPLE EVIDENCE")
    print("=" * 60)
    print("Email contains BOTH 'Win' AND 'Free'")
    print()
    print("CALCULATION:")
    print(f"  P(Win ∩ Free | Spam) = {p_both_given_spam:.4f}")
    print(f"  P(Win ∩ Free | Not Spam) = {p_both_given_not_spam:.4f}")
    print(f"  P(Win ∩ Free) = {p_both:.4f}")
    print()
    print(f"  P(Spam | Win ∩ Free) = {p_spam_given_both:.4f}")
    print()
    print(f"RESULT: {p_spam_given_both:.1%} chance it's spam!")
    print("       (Higher than single word)")
    
    # Compare single vs multiple evidence
    p_spam_given_win = 0.8571  # From previous calculation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    evidence_types = ['No Evidence\n(Prior)', 'Contains "Win"', 'Contains "Win"\nAND "Free"']
    spam_probs = [p_spam, p_spam_given_win, p_spam_given_both]
    colors = ['skyblue', 'orange', 'red']
    
    bars = ax.bar(evidence_types, spam_probs, color=colors, edgecolor='black', alpha=0.7, width=0.6)
    ax.set_ylabel('P(Spam | Evidence)', fontsize=12)
    ax.set_title('How More Evidence Strengthens Our Belief', fontweight='bold', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, spam_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=13)
    
    plt.tight_layout()
    plt.show()

bayes_multiple_evidence()