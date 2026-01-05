
#Dice Roll Probability - Calculate and visualize probability distributions for dice sums

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from itertools import product

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class DiceProbability:
    """Calculate and visualize probability distributions for dice rolls"""
    
    def __init__(self, num_dice=2, sides=6):
        """
        Initialize with number of dice and sides per die
        
        Parameters:
        -----------
        num_dice : int
            Number of dice to roll
        sides : int
            Number of sides on each die
        """
        self.num_dice = num_dice
        self.sides = sides
        self.calculate_probabilities()
    
    def calculate_probabilities(self):
        """Calculate all possible outcomes and their probabilities"""
        # Generate all possible combinations
        all_rolls = list(product(range(1, self.sides + 1), repeat=self.num_dice))
        
        # Calculate sums
        sums = [sum(roll) for roll in all_rolls]
        
        # Count occurrences
        sum_counts = Counter(sums)
        
        # Total possible outcomes
        total_outcomes = len(all_rolls)
        
        # Create probability distribution
        self.possible_sums = sorted(sum_counts.keys())
        self.counts = [sum_counts[s] for s in self.possible_sums]
        self.probabilities = [count / total_outcomes for count in self.counts]
        
        # Create DataFrame
        self.df = pd.DataFrame({
            'Sum': self.possible_sums,
            'Count': self.counts,
            'Probability': self.probabilities,
            'Percentage': [p * 100 for p in self.probabilities]
        })
        
        # Calculate statistics
        self.mean = np.average(self.possible_sums, weights=self.probabilities)
        self.variance = np.average((np.array(self.possible_sums) - self.mean)**2, 
                                   weights=self.probabilities)
        self.std = np.sqrt(self.variance)
        
        return self.df
    
    def plot_distribution(self):
        """Create comprehensive visualization of the probability distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Probability Distribution for {self.num_dice}d{self.sides}', 
                     fontsize=16, fontweight='bold')
        
        # 1. Bar plot of probabilities
        ax1 = axes[0, 0]
        bars = ax1.bar(self.possible_sums, self.probabilities, 
                       color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Sum of Dice', fontsize=12)
        ax1.set_ylabel('Probability', fontsize=12)
        ax1.set_title('Probability Mass Function', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (sum_val, prob) in enumerate(zip(self.possible_sums, self.probabilities)):
            ax1.text(sum_val, prob, f'{prob:.3f}', 
                    ha='center', va='bottom', fontsize=8)
        
        # 2. Cumulative distribution
        ax2 = axes[0, 1]
        cumulative_prob = np.cumsum(self.probabilities)
        ax2.plot(self.possible_sums, cumulative_prob, 
                marker='o', linewidth=2, markersize=6, color='darkgreen')
        ax2.fill_between(self.possible_sums, cumulative_prob, alpha=0.3, color='lightgreen')
        ax2.set_xlabel('Sum of Dice', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontsize=12)
        ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        # 3. Percentage bar chart
        ax3 = axes[1, 0]
        bars = ax3.barh(self.possible_sums, [p * 100 for p in self.probabilities], 
                        color='coral', edgecolor='black', alpha=0.7)
        ax3.set_ylabel('Sum of Dice', fontsize=12)
        ax3.set_xlabel('Percentage (%)', fontsize=12)
        ax3.set_title('Probability Distribution (Percentage)', fontsize=14, fontweight='bold')
        ax3.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (sum_val, prob) in enumerate(zip(self.possible_sums, self.probabilities)):
            ax3.text(prob * 100, sum_val, f'{prob*100:.2f}%', 
                    va='center', ha='left', fontsize=8)
        
        # 4. Statistics summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats_text = f"""
        STATISTICAL SUMMARY
        {'='*40}
        
        Number of Dice: {self.num_dice}
        Sides per Die: {self.sides}
        
        Minimum Sum: {min(self.possible_sums)}
        Maximum Sum: {max(self.possible_sums)}
        
        Expected Value (Mean): {self.mean:.3f}
        Variance: {self.variance:.3f}
        Standard Deviation: {self.std:.3f}
        
        Most Likely Sum: {self.possible_sums[np.argmax(self.probabilities)]}
        Highest Probability: {max(self.probabilities):.4f} ({max(self.probabilities)*100:.2f}%)
        
        Total Possible Outcomes: {sum(self.counts)}
        """
        
        ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def compare_normal_distribution(self):
        """Compare dice distribution with normal distribution"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot actual distribution
        ax.bar(self.possible_sums, self.probabilities, 
               alpha=0.6, label='Actual Distribution', color='steelblue', edgecolor='black')
        
        # Plot normal approximation
        x = np.linspace(min(self.possible_sums), max(self.possible_sums), 100)
        normal_dist = stats.norm.pdf(x, self.mean, self.std)
        
        # Scale normal distribution to match bar chart
        normal_dist_scaled = normal_dist * (self.possible_sums[1] - self.possible_sums[0])
        
        ax.plot(x, normal_dist_scaled, 'r-', linewidth=2, 
               label=f'Normal Approximation\n(μ={self.mean:.2f}, σ={self.std:.2f})')
        
        ax.set_xlabel('Sum of Dice', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Dice Distribution vs Normal Distribution ({self.num_dice}d{self.sides})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def simulate_rolls(self, n_simulations=10000):
        """Simulate dice rolls and compare with theoretical probabilities"""
        # Simulate rolls
        simulated_rolls = np.random.randint(1, self.sides + 1, 
                                           size=(n_simulations, self.num_dice))
        simulated_sums = simulated_rolls.sum(axis=1)
        
        # Count occurrences
        sim_counts = Counter(simulated_sums)
        sim_probs = {k: v/n_simulations for k, v in sim_counts.items()}
        
        # Create comparison plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(self.possible_sums))
        width = 0.35
        
        theoretical = ax.bar(x - width/2, self.probabilities, width, 
                           label='Theoretical', color='steelblue', alpha=0.8)
        simulated = ax.bar(x + width/2, [sim_probs.get(s, 0) for s in self.possible_sums], 
                          width, label=f'Simulated (n={n_simulations})', 
                          color='coral', alpha=0.8)
        
        ax.set_xlabel('Sum of Dice', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'Theoretical vs Simulated Probabilities ({self.num_dice}d{self.sides})', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(self.possible_sums)
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_probability_table(self):
        """Return formatted probability table"""
        return self.df


# Example Usage
if __name__ == "__main__":
    print("=" * 60)
    print("DICE ROLL PROBABILITY CALCULATOR")
    print("=" * 60)
    
    # Example 1: Classic 2d6 (two six-sided dice)
    print("\n1. Analyzing 2 six-sided dice (2d6)...")
    dice_2d6 = DiceProbability(num_dice=2, sides=6)
    print(dice_2d6.get_probability_table())
    dice_2d6.plot_distribution()
    plt.savefig('2d6_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: 2d6_distribution.png")
    
    # Example 2: Compare with normal distribution
    print("\n2. Comparing with Normal Distribution...")
    dice_2d6.compare_normal_distribution()
    plt.savefig('2d6_normal_comparison.png', dpi=300, bbox_inches='tight')
    print("   Saved: 2d6_normal_comparison.png")
    
    # Example 3: Simulate rolls
    print("\n3. Simulating 10,000 rolls...")
    dice_2d6.simulate_rolls(n_simulations=10000)
    plt.savefig('2d6_simulation.png', dpi=300, bbox_inches='tight')
    print("   Saved: 2d6_simulation.png")
    
    # Example 4: Three dice
    print("\n4. Analyzing 3 six-sided dice (3d6)...")
    dice_3d6 = DiceProbability(num_dice=3, sides=6)
    print(dice_3d6.get_probability_table().head(10))
    dice_3d6.plot_distribution()
    plt.savefig('3d6_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: 3d6_distribution.png")
    
    # Example 5: Different sided dice
    print("\n5. Analyzing 2 ten-sided dice (2d10)...")
    dice_2d10 = DiceProbability(num_dice=2, sides=10)
    print(dice_2d10.get_probability_table().head(10))
    dice_2d10.plot_distribution()
    plt.savefig('2d10_distribution.png', dpi=300, bbox_inches='tight')
    print("   Saved: 2d10_distribution.png")
    
    print("All visualizations completed!")
    plt.show()