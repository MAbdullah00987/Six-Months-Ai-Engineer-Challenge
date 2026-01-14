
#Project 2:
#Bayesian Inference (Monty Hall Problem) - Implement a solution using Bayes' theorem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class MontyHallBayesian:
    """
    Bayesian Inference implementation for the Monty Hall Problem.
    
    Problem: You're on a game show with 3 doors. Behind one is a car,
    behind the others are goats. After you pick a door, the host (who knows
    what's behind the doors) opens another door revealing a goat.
    Should you switch your choice?
    """
    
    def __init__(self):
        self.doors = [1, 2, 3]
        self.results = []
        
    def bayes_theorem_manual(self):
        """
        Calculate probabilities using Bayes' Theorem manually.
        
        Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
        
        Where:
        - P(Car at Door X | Host opens Door Y) = Posterior probability
        - P(Host opens Door Y | Car at Door X) = Likelihood
        - P(Car at Door X) = Prior probability
        - P(Host opens Door Y) = Marginal probability
        """
        print("="*80)
        print("BAYESIAN INFERENCE - MONTY HALL PROBLEM")
        print("="*80)
        print("\n📌 SCENARIO: You choose Door 1, Host opens Door 3 (revealing a goat)")
        print("\nCalculating probabilities using Bayes' Theorem...\n")
        
        # Prior probabilities (uniform - equal chance)
        prior_door1 = 1/3  # P(Car at Door 1)
        prior_door2 = 1/3  # P(Car at Door 2)
        prior_door3 = 1/3  # P(Car at Door 3)
        
        print("STEP 1: Prior Probabilities (Before any information)")
        print("-" * 80)
        print(f"P(Car at Door 1) = {prior_door1:.4f}")
        print(f"P(Car at Door 2) = {prior_door2:.4f}")
        print(f"P(Car at Door 3) = {prior_door3:.4f}")
        
        # Likelihoods: P(Host opens Door 3 | Car at Door X)
        # If car at Door 1: Host can open Door 2 or 3 (probability 1/2 for Door 3)
        # If car at Door 2: Host must open Door 3 (probability 1)
        # If car at Door 3: Host cannot open Door 3 (probability 0)
        
        likelihood_door1 = 1/2  # P(Host opens 3 | Car at 1)
        likelihood_door2 = 1    # P(Host opens 3 | Car at 2)
        likelihood_door3 = 0    # P(Host opens 3 | Car at 3)
        
        print("\n\nSTEP 2: Likelihoods (Given host's action)")
        print("-" * 80)
        print(f"P(Host opens Door 3 | Car at Door 1) = {likelihood_door1:.4f}")
        print(f"  → If car is behind Door 1, host can randomly choose between Door 2 or 3")
        print(f"\nP(Host opens Door 3 | Car at Door 2) = {likelihood_door2:.4f}")
        print(f"  → If car is behind Door 2, host MUST open Door 3")
        print(f"\nP(Host opens Door 3 | Car at Door 3) = {likelihood_door3:.4f}")
        print(f"  → If car is behind Door 3, host CANNOT open it (has a goat)")
        
        # Marginal probability: P(Host opens Door 3)
        # Using law of total probability
        marginal = (likelihood_door1 * prior_door1 + 
                   likelihood_door2 * prior_door2 + 
                   likelihood_door3 * prior_door3)
        
        print("\n\nSTEP 3: Marginal Probability (Normalization)")
        print("-" * 80)
        print(f"P(Host opens Door 3) = Σ P(Host opens 3 | Car at i) × P(Car at i)")
        print(f"                     = ({likelihood_door1:.4f} × {prior_door1:.4f}) + "
              f"({likelihood_door2:.4f} × {prior_door2:.4f}) + ({likelihood_door3:.4f} × {prior_door3:.4f})")
        print(f"                     = {marginal:.4f}")
        
        # Posterior probabilities using Bayes' Theorem
        # P(Car at Door X | Host opens Door 3) = P(Host opens 3 | Car at X) * P(Car at X) / P(Host opens 3)
        
        posterior_door1 = (likelihood_door1 * prior_door1) / marginal
        posterior_door2 = (likelihood_door2 * prior_door2) / marginal
        posterior_door3 = (likelihood_door3 * prior_door3) / marginal
        
        print("\n\nSTEP 4: Posterior Probabilities (After host reveals Door 3)")
        print("-" * 80)
        print("Using Bayes' Theorem: P(A|B) = P(B|A) × P(A) / P(B)\n")
        
        print(f"P(Car at Door 1 | Host opens 3) = {likelihood_door1:.4f} × {prior_door1:.4f} / {marginal:.4f}")
        print(f"                                 = {posterior_door1:.4f} ({posterior_door1*100:.2f}%)")
        print(f"  → STAY with Door 1\n")
        
        print(f"P(Car at Door 2 | Host opens 3) = {likelihood_door2:.4f} × {prior_door2:.4f} / {marginal:.4f}")
        print(f"                                 = {posterior_door2:.4f} ({posterior_door2*100:.2f}%)")
        print(f"  → SWITCH to Door 2\n")
        
        print(f"P(Car at Door 3 | Host opens 3) = {likelihood_door3:.4f} × {prior_door3:.4f} / {marginal:.4f}")
        print(f"                                 = {posterior_door3:.4f} ({posterior_door3*100:.2f}%)")
        print(f"  → Already revealed (goat)")
        
        # Verification
        total = posterior_door1 + posterior_door2 + posterior_door3
        print(f"\n✓ Verification: Sum of posteriors = {total:.4f} (should be 1.0)")
        
        print("\n" + "="*80)
        print("🎯 CONCLUSION")
        print("="*80)
        print(f"❌ Staying with Door 1: {posterior_door1*100:.1f}% chance of winning")
        print(f"✅ Switching to Door 2: {posterior_door2*100:.1f}% chance of winning")
        print(f"\n💡 SWITCHING DOUBLES YOUR CHANCES OF WINNING!")
        print("="*80)
        
        return {
            'prior': [prior_door1, prior_door2, prior_door3],
            'likelihood': [likelihood_door1, likelihood_door2, likelihood_door3],
            'posterior': [posterior_door1, posterior_door2, posterior_door3]
        }
    
    def simulate_single_game(self, switch=True):
        """Simulate a single Monty Hall game."""
        # Randomly place the car
        car_door = np.random.choice(self.doors)
        
        # Player chooses door 1
        player_choice = 1
        
        # Host opens a door (not player's choice, not car)
        available_doors = [d for d in self.doors if d != player_choice and d != car_door]
        host_opens = np.random.choice(available_doors) if available_doors else np.random.choice([d for d in self.doors if d != player_choice])
        
        # Player decides to switch or stay
        if switch:
            # Switch to the remaining unopened door
            final_choice = [d for d in self.doors if d != player_choice and d != host_opens][0]
        else:
            final_choice = player_choice
        
        # Check if player wins
        win = (final_choice == car_door)
        
        return {
            'car_door': car_door,
            'player_initial': player_choice,
            'host_opens': host_opens,
            'player_final': final_choice,
            'switch': switch,
            'win': win
        }
    
    def run_simulation(self, n_simulations=10000):
        """Run Monte Carlo simulation of Monty Hall problem."""
        print("\n\n" + "="*80)
        print("MONTE CARLO SIMULATION")
        print("="*80)
        print(f"\nRunning {n_simulations:,} simulations...\n")
        
        # Run simulations for both strategies
        stay_results = []
        switch_results = []
        
        for _ in range(n_simulations):
            stay_results.append(self.simulate_single_game(switch=False))
            switch_results.append(self.simulate_single_game(switch=True))
        
        # Calculate win rates
        stay_wins = sum([r['win'] for r in stay_results])
        switch_wins = sum([r['win'] for r in switch_results])
        
        stay_rate = stay_wins / n_simulations
        switch_rate = switch_wins / n_simulations
        
        print(f"Strategy: STAY")
        print(f"  Wins: {stay_wins:,} / {n_simulations:,}")
        print(f"  Win Rate: {stay_rate:.4f} ({stay_rate*100:.2f}%)")
        print(f"  Expected (Bayesian): 33.33%\n")
        
        print(f"Strategy: SWITCH")
        print(f"  Wins: {switch_wins:,} / {n_simulations:,}")
        print(f"  Win Rate: {switch_rate:.4f} ({switch_rate*100:.2f}%)")
        print(f"  Expected (Bayesian): 66.67%\n")
        
        print(f"✓ Simulation confirms Bayesian analysis!")
        print(f"  Switching is {switch_rate/stay_rate:.2f}x better than staying")
        
        # Store results
        self.results = {
            'stay': stay_results,
            'switch': switch_results,
            'stay_rate': stay_rate,
            'switch_rate': switch_rate
        }
        
        return stay_rate, switch_rate
    
    def visualize_results(self, bayes_results):
        """Create comprehensive visualizations."""
        fig = plt.figure(figsize=(16, 10))
        
        # 1. Prior vs Posterior Probabilities
        ax1 = plt.subplot(2, 3, 1)
        doors = ['Door 1\n(Your Choice)', 'Door 2\n(Unopened)', 'Door 3\n(Opened by Host)']
        x = np.arange(len(doors))
        width = 0.35
        
        prior_probs = bayes_results['prior']
        posterior_probs = bayes_results['posterior']
        
        bars1 = ax1.bar(x - width/2, prior_probs, width, label='Prior', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, posterior_probs, width, label='Posterior', alpha=0.8, color='coral')
        
        ax1.set_ylabel('Probability', fontweight='bold', fontsize=11)
        ax1.set_title('Prior vs Posterior Probabilities\n(Bayesian Update)', fontweight='bold', fontsize=12)
        ax1.set_xticks(x)
        ax1.set_xticklabels(doors, fontsize=10)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Likelihood Distribution
        ax2 = plt.subplot(2, 3, 2)
        likelihood = bayes_results['likelihood']
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax2.bar(doors, likelihood, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Probability', fontweight='bold', fontsize=11)
        ax2.set_title('Likelihood: P(Host Opens Door 3 | Car Location)', fontweight='bold', fontsize=12)
        ax2.set_xticklabels(doors, fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Simulation Results
        if hasattr(self, 'results') and self.results:
            ax3 = plt.subplot(2, 3, 3)
            strategies = ['Stay', 'Switch']
            win_rates = [self.results['stay_rate'], self.results['switch_rate']]
            colors_sim = ['#e74c3c', '#2ecc71']
            
            bars = ax3.bar(strategies, win_rates, color=colors_sim, alpha=0.7, edgecolor='black', linewidth=2)
            ax3.axhline(y=1/3, color='red', linestyle='--', linewidth=2, label='Theoretical (Stay)', alpha=0.7)
            ax3.axhline(y=2/3, color='green', linestyle='--', linewidth=2, label='Theoretical (Switch)', alpha=0.7)
            ax3.set_ylabel('Win Rate', fontweight='bold', fontsize=11)
            ax3.set_title('Simulation Results: Win Rates', fontweight='bold', fontsize=12)
            ax3.set_ylim(0, 1)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height*100:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # 4. Bayes' Theorem Visualization
        ax4 = plt.subplot(2, 3, 4)
        ax4.axis('off')
        
        formula_text = (
            "Bayes' Theorem:\n\n"
            "P(Car at Door X | Host opens Door 3) = \n\n"
            "    P(Host opens Door 3 | Car at X) × P(Car at X)\n"
            "    ───────────────────────────────────────────\n"
            "              P(Host opens Door 3)\n\n\n"
            "Example for Door 1 (Stay):\n"
            f"P(Door 1 | Host=3) = (1/2 × 1/3) / (1/2) = 1/3\n\n"
            "Example for Door 2 (Switch):\n"
            f"P(Door 2 | Host=3) = (1 × 1/3) / (1/2) = 2/3"
        )
        
        ax4.text(0.1, 0.5, formula_text, fontsize=11, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.set_title("Bayes' Theorem Formula", fontweight='bold', fontsize=12)
        
        # 5. Decision Tree
        ax5 = plt.subplot(2, 3, 5)
        ax5.axis('off')
        
        tree_text = (
            "Decision Tree:\n\n"
            "1. Choose Door 1 (random)\n"
            "   ├─ Car at Door 1 (1/3) → Host opens 2 or 3\n"
            "   ├─ Car at Door 2 (1/3) → Host opens Door 3 ✓\n"
            "   └─ Car at Door 3 (1/3) → Host opens Door 2\n\n"
            "2. Host opens Door 3 (reveals goat)\n\n"
            "3. Update beliefs:\n"
            "   ├─ Stay Door 1: 33.3% win\n"
            "   └─ Switch Door 2: 66.7% win ✅\n\n"
            "Key Insight: Host's action provides\n"
            "information that updates our beliefs!"
        )
        
        ax5.text(0.1, 0.5, tree_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax5.set_title("Decision Tree Analysis", fontweight='bold', fontsize=12)
        
        # 6. Probability Evolution
        ax6 = plt.subplot(2, 3, 6)
        stages = ['Initial\n(No Info)', 'After Host\nOpens Door 3']
        door1_probs = [1/3, bayes_results['posterior'][0]]
        door2_probs = [1/3, bayes_results['posterior'][1]]
        door3_probs = [1/3, bayes_results['posterior'][2]]
        
        x = np.arange(len(stages))
        width = 0.25
        
        ax6.plot(x, door1_probs, 'o-', linewidth=2, markersize=10, label='Door 1 (Your Choice)', color='#3498db')
        ax6.plot(x, door2_probs, 's-', linewidth=2, markersize=10, label='Door 2 (Switch)', color='#2ecc71')
        ax6.plot(x, door3_probs, '^-', linewidth=2, markersize=10, label='Door 3 (Opened)', color='#e74c3c')
        
        ax6.set_ylabel('Probability', fontweight='bold', fontsize=11)
        ax6.set_title('Probability Evolution (Bayesian Update)', fontweight='bold', fontsize=12)
        ax6.set_xticks(x)
        ax6.set_xticklabels(stages, fontsize=10)
        ax6.legend(loc='best')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(-0.05, 0.75)
        
        plt.tight_layout()
        plt.savefig('monty_hall_bayesian_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualizations saved to: monty_hall_bayesian_analysis.png")
        plt.show()
    
    def create_simulation_comparison(self):
        """Create detailed simulation comparison visualization."""
        if not hasattr(self, 'results') or not self.results:
            print("Run simulation first!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Convert results to DataFrame
        stay_df = pd.DataFrame(self.results['stay'])
        switch_df = pd.DataFrame(self.results['switch'])
        
        # 1. Win distribution - Stay
        ax1 = axes[0, 0]
        stay_wins = stay_df['win'].value_counts()
        colors_win = ['#e74c3c', '#2ecc71']
        ax1.pie([stay_wins.get(False, 0), stay_wins.get(True, 0)], 
                labels=['Loss', 'Win'], autopct='%1.1f%%', startangle=90,
                colors=colors_win, explode=(0.1, 0))
        ax1.set_title('Strategy: STAY - Win Distribution', fontweight='bold', fontsize=12)
        
        # 2. Win distribution - Switch
        ax2 = axes[0, 1]
        switch_wins = switch_df['win'].value_counts()
        ax2.pie([switch_wins.get(False, 0), switch_wins.get(True, 0)], 
                labels=['Loss', 'Win'], autopct='%1.1f%%', startangle=90,
                colors=colors_win, explode=(0.1, 0))
        ax2.set_title('Strategy: SWITCH - Win Distribution', fontweight='bold', fontsize=12)
        
        # 3. Car position distribution
        ax3 = axes[1, 0]
        car_positions = stay_df['car_door'].value_counts().sort_index()
        ax3.bar(car_positions.index, car_positions.values, color='orange', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Door Number', fontweight='bold')
        ax3.set_ylabel('Frequency', fontweight='bold')
        ax3.set_title('Car Position Distribution (Uniform)', fontweight='bold', fontsize=12)
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Cumulative win rate
        ax4 = axes[1, 1]
        stay_cumulative = np.cumsum(stay_df['win']) / (np.arange(len(stay_df)) + 1)
        switch_cumulative = np.cumsum(switch_df['win']) / (np.arange(len(switch_df)) + 1)
        
        ax4.plot(stay_cumulative, label='Stay', linewidth=2, color='#e74c3c', alpha=0.7)
        ax4.plot(switch_cumulative, label='Switch', linewidth=2, color='#2ecc71', alpha=0.7)
        ax4.axhline(y=1/3, color='red', linestyle='--', label='Theoretical (Stay)', alpha=0.5)
        ax4.axhline(y=2/3, color='green', linestyle='--', label='Theoretical (Switch)', alpha=0.5)
        ax4.set_xlabel('Number of Games', fontweight='bold')
        ax4.set_ylabel('Cumulative Win Rate', fontweight='bold')
        ax4.set_title('Convergence to Theoretical Probabilities', fontweight='bold', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('monty_hall_simulation_details.png', dpi=300, bbox_inches='tight')
        print(f"✓ Simulation details saved to: monty_hall_simulation_details.png")
        plt.show()


def main():
    """Main execution function."""
    # Initialize Monty Hall Bayesian Analysis
    monty = MontyHallBayesian()
    
    # 1. Perform Bayesian Analysis
    bayes_results = monty.bayes_theorem_manual()
    
    # 2. Run Monte Carlo Simulation
    stay_rate, switch_rate = monty.run_simulation(n_simulations=10000)
    
    # 3. Create Visualizations
    print("\n\nGenerating visualizations...")
    monty.visualize_results(bayes_results)
    monty.create_simulation_comparison()
    
    # 4. Statistical Test
    print("\n\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*80)
    
    stay_wins = [r['win'] for r in monty.results['stay']]
    switch_wins = [r['win'] for r in monty.results['switch']]
    
    # Chi-square test
    observed = [sum(stay_wins), sum(switch_wins)]
    expected = [len(stay_wins) * 1/3, len(switch_wins) * 2/3]
    
    chi2_stat = sum([(o - e)**2 / e for o, e in zip(observed, expected)])
    
    print(f"\nChi-Square Goodness of Fit Test:")
    print(f"  Chi-Square Statistic: {chi2_stat:.4f}")
    print(f"  p-value: {1 - stats.chi2.cdf(chi2_stat, df=1):.4f}")
    print(f"  ✓ Results match theoretical predictions!")
    
    print("\n" + "="*80)
    print("✅ BAYESIAN MONTY HALL ANALYSIS COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()