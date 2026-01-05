

#Coin Flip Simulator - Start with fundamentals by simulating coin flips and observing convergence to theoretical probabilities

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class CoinFlipSimulator:
    def __init__(self, n_flips=1000, p_heads=0.5):
        """
        Initialize the coin flip simulator
        
        Parameters:
        -----------
        n_flips : int
            Number of coin flips to simulate
        p_heads : float
            Probability of getting heads (0.5 for fair coin)
        """
        self.n_flips = n_flips
        self.p_heads = p_heads
        self.flips = None
        self.cumulative_proportion = None
        
    def simulate_flips(self):
        """Simulate coin flips"""
        # 1 = Heads, 0 = Tails
        self.flips = np.random.binomial(1, self.p_heads, self.n_flips)
        
        # Calculate cumulative proportion of heads
        cumsum = np.cumsum(self.flips)
        flip_numbers = np.arange(1, self.n_flips + 1)
        self.cumulative_proportion = cumsum / flip_numbers
        
        return self.flips
    
    def get_statistics(self):
        """Calculate statistics for the simulation"""
        if self.flips is None:
            self.simulate_flips()
        
        n_heads = np.sum(self.flips)
        n_tails = self.n_flips - n_heads
        observed_proportion = n_heads / self.n_flips
        
        # Calculate confidence interval using Wilson score method (manual implementation)
        z = 1.96  # 95% confidence
        p_hat = observed_proportion
        n = self.n_flips
        
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) / n + z**2 / (4*n**2))) / denominator
        
        ci_low = center - margin
        ci_high = center + margin
        
        # Chi-square goodness of fit test
        expected = self.n_flips * self.p_heads
        chi2_stat = ((n_heads - expected) ** 2) / expected + ((n_tails - (self.n_flips - expected)) ** 2) / (self.n_flips - expected)
        p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        
        stats_dict = {
            'Total Flips': self.n_flips,
            'Heads': n_heads,
            'Tails': n_tails,
            'Observed Proportion (Heads)': observed_proportion,
            'Expected Proportion (Heads)': self.p_heads,
            'Difference': observed_proportion - self.p_heads,
            '95% CI Lower': ci_low,
            '95% CI Upper': ci_high,
            'Chi-square Statistic': chi2_stat,
            'P-value': p_value
        }
        
        return stats_dict
    
    def plot_convergence(self, ax=None):
        """Plot convergence to theoretical probability"""
        if self.flips is None:
            self.simulate_flips()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        flip_numbers = np.arange(1, self.n_flips + 1)
        
        # Plot cumulative proportion
        ax.plot(flip_numbers, self.cumulative_proportion, 
                label='Observed Proportion', linewidth=1.5, alpha=0.8)
        
        # Plot theoretical probability
        ax.axhline(y=self.p_heads, color='red', linestyle='--', 
                   linewidth=2, label=f'Theoretical Probability ({self.p_heads})')
        
        # Add confidence bands
        std_error = np.sqrt(self.p_heads * (1 - self.p_heads) / flip_numbers)
        ax.fill_between(flip_numbers, 
                        self.p_heads - 1.96 * std_error,
                        self.p_heads + 1.96 * std_error,
                        alpha=0.2, color='red', label='95% CI Band')
        
        ax.set_xlabel('Number of Flips', fontsize=12)
        ax.set_ylabel('Proportion of Heads', fontsize=12)
        ax.set_title('Convergence to Theoretical Probability (Law of Large Numbers)', 
                     fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def plot_distribution(self, ax=None):
        """Plot distribution of outcomes"""
        if self.flips is None:
            self.simulate_flips()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        outcomes = ['Tails', 'Heads']
        counts = [np.sum(self.flips == 0), np.sum(self.flips == 1)]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(outcomes, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}\n({count/self.n_flips*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Coin Flip Outcomes', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return ax
    
    def plot_streak_analysis(self, ax=None):
        """Analyze and plot streaks of heads/tails"""
        if self.flips is None:
            self.simulate_flips()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        # Find streaks
        streaks = []
        current_streak = 1
        
        for i in range(1, len(self.flips)):
            if self.flips[i] == self.flips[i-1]:
                current_streak += 1
            else:
                streaks.append(current_streak)
                current_streak = 1
        streaks.append(current_streak)
        
        # Plot histogram of streak lengths
        ax.hist(streaks, bins=range(1, max(streaks) + 2), 
                alpha=0.7, color='skyblue', edgecolor='black')
        
        ax.set_xlabel('Streak Length', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Distribution of Streak Lengths', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add statistics
        ax.text(0.98, 0.98, f'Max Streak: {max(streaks)}\nAvg Streak: {np.mean(streaks):.2f}',
                transform=ax.transAxes, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5), fontsize=10)
        
        return ax

def run_multiple_simulations(n_simulations=10, n_flips=1000, p_heads=0.5):
    """Run multiple simulations to show variability"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(n_simulations):
        sim = CoinFlipSimulator(n_flips, p_heads)
        sim.simulate_flips()
        flip_numbers = np.arange(1, n_flips + 1)
        ax.plot(flip_numbers, sim.cumulative_proportion, 
                alpha=0.5, linewidth=1)
    
    # Plot theoretical probability
    ax.axhline(y=p_heads, color='red', linestyle='--', 
               linewidth=3, label=f'Theoretical Probability ({p_heads})')
    
    ax.set_xlabel('Number of Flips', fontsize=12)
    ax.set_ylabel('Proportion of Heads', fontsize=12)
    ax.set_title(f'{n_simulations} Independent Simulations - Law of Large Numbers', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Main execution
if __name__ == "__main__":
    print("=" * 70)
    print("COIN FLIP SIMULATOR - Convergence to Theoretical Probabilities")
    print("=" * 70)
    
    # Create simulator with 1000 flips
    simulator = CoinFlipSimulator(n_flips=1000, p_heads=0.5)
    simulator.simulate_flips()
    
    # Print statistics
    print("\nStatistical Summary:")
    print("-" * 70)
    stats_dict = simulator.get_statistics()
    for key, value in stats_dict.items():
        if isinstance(value, (int, np.integer)):
            print(f"{key:.<40} {value}")
        else:
            print(f"{key:.<40} {value:.6f}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Convergence plot
    ax1 = plt.subplot(2, 2, 1)
    simulator.plot_convergence(ax1)
    
    # 2. Distribution plot
    ax2 = plt.subplot(2, 2, 2)
    simulator.plot_distribution(ax2)
    
    # 3. Streak analysis
    ax3 = plt.subplot(2, 2, 3)
    simulator.plot_streak_analysis(ax3)
    
    # 4. Rolling window proportion
    ax4 = plt.subplot(2, 2, 4)
    window_size = 50
    rolling_prop = pd.Series(simulator.flips).rolling(window=window_size).mean()
    ax4.plot(rolling_prop, label=f'{window_size}-flip Moving Average', linewidth=2)
    ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Expected (0.5)')
    ax4.set_xlabel('Flip Number', fontsize=12)
    ax4.set_ylabel('Proportion of Heads', fontsize=12)
    ax4.set_title(f'Rolling Average ({window_size} flips)', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('coin_flip_analysis.png', dpi=300, bbox_inches='tight')
    print("\n✓ Main analysis plot saved as 'coin_flip_analysis.png'")
    
    # Run multiple simulations
    fig2 = run_multiple_simulations(n_simulations=20, n_flips=1000)
    plt.savefig('multiple_simulations.png', dpi=300, bbox_inches='tight')
    print("✓ Multiple simulations plot saved as 'multiple_simulations.png'")
    
    plt.show()
    
    print("Simulation complete! Plots are displayed and saved.")
    
   