
#Project - Monty Hall Simulator. Simulate the famous game show problem where switching doors doubles your chances of winning. Prove it with code.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class MontyHallSimulator:
    """Simulator for the famous Monty Hall problem"""
    
    def __init__(self, num_simulations=10000):
        self.num_simulations = num_simulations
        self.results = None
    
    def play_game(self, switch=True):
        """
        Play a single game of Monty Hall
        
        Args:
            switch: Whether to switch doors after host reveals a goat
        
        Returns:
            bool: True if won, False if lost
        """
        # Randomly place the car behind one of three doors (0, 1, 2)
        car_door = np.random.randint(0, 3)
        
        # Player makes initial choice
        initial_choice = np.random.randint(0, 3)
        
        # Host reveals a door with a goat (not the car, not the player's choice)
        available_doors = [d for d in range(3) if d != car_door and d != initial_choice]
        if len(available_doors) == 2:
            # If player initially chose a goat, host can reveal either remaining goat
            host_reveals = np.random.choice(available_doors)
        else:
            # If player initially chose the car, host reveals one of two goats
            available_doors = [d for d in range(3) if d != initial_choice and d != car_door]
            host_reveals = np.random.choice(available_doors)
        
        # Player's final choice
        if switch:
            # Switch to the remaining unopened door
            final_choice = [d for d in range(3) if d != initial_choice and d != host_reveals][0]
        else:
            # Stay with initial choice
            final_choice = initial_choice
        
        # Check if player won
        return final_choice == car_door
    
    def run_simulations(self):
        """Run simulations for both strategies"""
        print(f"Running {self.num_simulations:,} simulations...\n")
        
        # Simulate staying strategy
        stay_wins = sum(self.play_game(switch=False) for _ in range(self.num_simulations))
        stay_win_rate = stay_wins / self.num_simulations
        
        # Simulate switching strategy
        switch_wins = sum(self.play_game(switch=True) for _ in range(self.num_simulations))
        switch_win_rate = switch_wins / self.num_simulations
        
        # Store results
        self.results = pd.DataFrame({
            'Strategy': ['Stay', 'Switch'],
            'Wins': [stay_wins, switch_wins],
            'Losses': [self.num_simulations - stay_wins, self.num_simulations - switch_wins],
            'Win Rate': [stay_win_rate, switch_win_rate]
        })
        
        return self.results
    
    def run_progressive_simulation(self, max_games=10000, step=100):
        """Run simulations progressively to show convergence"""
        game_counts = np.arange(step, max_games + 1, step)
        stay_rates = []
        switch_rates = []
        
        for n in game_counts:
            stay_wins = sum(self.play_game(switch=False) for _ in range(step))
            switch_wins = sum(self.play_game(switch=True) for _ in range(step))
            stay_rates.append(stay_wins / step)
            switch_rates.append(switch_wins / step)
        
        # Calculate cumulative averages
        stay_cumulative = np.cumsum(stay_rates) / np.arange(1, len(stay_rates) + 1)
        switch_cumulative = np.cumsum(switch_rates) / np.arange(1, len(switch_rates) + 1)
        
        return game_counts, stay_cumulative, switch_cumulative
    
    def statistical_test(self):
        """Perform chi-square test to verify results"""
        if self.results is None:
            print("Run simulations first!")
            return
        
        # Expected probabilities: Stay = 1/3, Switch = 2/3
        expected_stay = self.num_simulations * (1/3)
        expected_switch = self.num_simulations * (2/3)
        
        stay_wins = self.results.loc[self.results['Strategy'] == 'Stay', 'Wins'].values[0]
        switch_wins = self.results.loc[self.results['Strategy'] == 'Switch', 'Wins'].values[0]
        
        # Chi-square test for stay strategy
        chi2_stay, p_stay = stats.chisquare([stay_wins, self.num_simulations - stay_wins],
                                            [expected_stay, self.num_simulations - expected_stay])
        
        # Chi-square test for switch strategy
        chi2_switch, p_switch = stats.chisquare([switch_wins, self.num_simulations - switch_wins],
                                                [expected_switch, self.num_simulations - expected_switch])
        
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)
        print(f"\nStay Strategy:")
        print(f"  Observed win rate: {stay_wins/self.num_simulations:.4f}")
        print(f"  Expected win rate: 0.3333")
        print(f"  Chi-square statistic: {chi2_stay:.4f}")
        print(f"  P-value: {p_stay:.4f}")
        
        print(f"\nSwitch Strategy:")
        print(f"  Observed win rate: {switch_wins/self.num_simulations:.4f}")
        print(f"  Expected win rate: 0.6667")
        print(f"  Chi-square statistic: {chi2_switch:.4f}")
        print(f"  P-value: {p_switch:.4f}")
        
        print("\nConclusion:")
        print(f"  The switching strategy wins approximately {switch_wins/stay_wins:.2f}x more often!")
    
    def visualize_results(self):
        """Create comprehensive visualizations"""
        if self.results is None:
            print("Run simulations first!")
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Win Rate Comparison (Bar Chart)
        ax1 = plt.subplot(2, 3, 1)
        bars = ax1.bar(self.results['Strategy'], self.results['Win Rate'], 
                       color=['#FF6B6B', '#4ECDC4'], alpha=0.8, edgecolor='black')
        ax1.axhline(y=1/3, color='red', linestyle='--', label='Theoretical: Stay (1/3)', linewidth=2)
        ax1.axhline(y=2/3, color='green', linestyle='--', label='Theoretical: Switch (2/3)', linewidth=2)
        ax1.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
        ax1.set_title('Win Rate by Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylim([0, 1])
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Wins vs Losses (Stacked Bar)
        ax2 = plt.subplot(2, 3, 2)
        x = np.arange(len(self.results['Strategy']))
        width = 0.6
        ax2.bar(x, self.results['Wins'], width, label='Wins', 
                color='#51CF66', alpha=0.8, edgecolor='black')
        ax2.bar(x, self.results['Losses'], width, bottom=self.results['Wins'],
                label='Losses', color='#FF6B6B', alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Number of Games', fontsize=12, fontweight='bold')
        ax2.set_title(f'Wins vs Losses ({self.num_simulations:,} games)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.results['Strategy'])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Pie Charts
        ax3 = plt.subplot(2, 3, 3)
        stay_data = [self.results.loc[0, 'Wins'], self.results.loc[0, 'Losses']]
        ax3.pie(stay_data, labels=['Wins', 'Losses'], autopct='%1.1f%%',
                colors=['#51CF66', '#FF6B6B'], startangle=90, 
                wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        ax3.set_title('Stay Strategy Results', fontsize=14, fontweight='bold')
        
        ax4 = plt.subplot(2, 3, 4)
        switch_data = [self.results.loc[1, 'Wins'], self.results.loc[1, 'Losses']]
        ax4.pie(switch_data, labels=['Wins', 'Losses'], autopct='%1.1f%%',
                colors=['#51CF66', '#FF6B6B'], startangle=90,
                wedgeprops={'edgecolor': 'black', 'linewidth': 2})
        ax4.set_title('Switch Strategy Results', fontsize=14, fontweight='bold')
        
        # 4. Convergence Plot
        ax5 = plt.subplot(2, 3, 5)
        game_counts, stay_rates, switch_rates = self.run_progressive_simulation(
            max_games=5000, step=50)
        ax5.plot(game_counts, stay_rates, label='Stay Strategy', 
                color='#FF6B6B', linewidth=2, alpha=0.8)
        ax5.plot(game_counts, switch_rates, label='Switch Strategy', 
                color='#4ECDC4', linewidth=2, alpha=0.8)
        ax5.axhline(y=1/3, color='red', linestyle='--', alpha=0.5, label='Theoretical: 1/3')
        ax5.axhline(y=2/3, color='green', linestyle='--', alpha=0.5, label='Theoretical: 2/3')
        ax5.set_xlabel('Number of Games', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Cumulative Win Rate', fontsize=12, fontweight='bold')
        ax5.set_title('Convergence to Theoretical Probabilities', 
                     fontsize=14, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 5. Difference Visualization
        ax6 = plt.subplot(2, 3, 6)
        difference = self.results.loc[1, 'Win Rate'] - self.results.loc[0, 'Win Rate']
        theoretical_diff = 2/3 - 1/3
        ax6.bar(['Observed\nDifference', 'Theoretical\nDifference'], 
               [difference, theoretical_diff],
               color=['#9B59B6', '#3498DB'], alpha=0.8, edgecolor='black')
        ax6.set_ylabel('Win Rate Difference', fontsize=12, fontweight='bold')
        ax6.set_title('Switch Advantage (Switch - Stay)', 
                     fontsize=14, fontweight='bold')
        ax6.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate([difference, theoretical_diff]):
            ax6.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('monty_hall_results.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'monty_hall_results.png'")
        plt.show()
    
    def print_results(self):
        """Print detailed results"""
        if self.results is None:
            print("Run simulations first!")
            return
        
        print("="*60)
        print("MONTY HALL PROBLEM - SIMULATION RESULTS")
        print("="*60)
        print(f"\nTotal simulations: {self.num_simulations:,}\n")
        print(self.results.to_string(index=False))
        print("\n" + "="*60)
        print("KEY FINDINGS:")
        print("="*60)
        
        stay_rate = self.results.loc[0, 'Win Rate']
        switch_rate = self.results.loc[1, 'Win Rate']
        
        print(f"\n✓ Stay Strategy wins: {stay_rate:.2%} of the time")
        print(f"  - Theoretical probability: 33.33%")
        print(f"  - Difference: {abs(stay_rate - 1/3):.2%}")
        
        print(f"\n✓ Switch Strategy wins: {switch_rate:.2%} of the time")
        print(f"  - Theoretical probability: 66.67%")
        print(f"  - Difference: {abs(switch_rate - 2/3):.2%}")
        
        print(f"\n✓ Switching is {switch_rate/stay_rate:.2f}x better than staying!")
        print(f"  - You're {((switch_rate - stay_rate) * 100):.1f}% more likely to win by switching")
        
        print("\n" + "="*60)


def explain_monty_hall():
    """Explain the Monty Hall problem"""
    explanation = """
    ╔══════════════════════════════════════════════════════════════╗
    ║           THE MONTY HALL PROBLEM - EXPLAINED                 ║
    ╚══════════════════════════════════════════════════════════════╝
    
    THE GAME:
    --------
    1. There are 3 doors: behind one is a car (prize), behind the 
       others are goats.
    2. You pick a door (say, Door 1).
    3. The host, who knows what's behind each door, opens another 
       door (say, Door 3) to reveal a goat.
    4. The host asks: "Do you want to switch to Door 2?"
    
    THE QUESTION:
    ------------
    Should you switch or stay with your original choice?
    
    THE ANSWER:
    ----------
    ALWAYS SWITCH! You double your chances of winning!
    
    WHY IT WORKS:
    ------------
    • Initial choice: 1/3 chance of picking the car
    • This means: 2/3 chance the car is behind one of the other doors
    
    • When the host reveals a goat, he's giving you information!
    • If you initially picked a goat (2/3 probability), switching wins
    • If you initially picked the car (1/3 probability), switching loses
    
    INTUITIVE EXPLANATION:
    ---------------------
    Imagine 100 doors instead of 3:
    • You pick Door 1 (1% chance it's the car)
    • Host opens 98 other doors, all with goats
    • Only Door 1 (yours) and Door 57 remain
    • Would you switch? Of course! The host essentially showed you 
      where the car is by eliminating 98 wrong choices!
    
    The Monty Hall problem is the same principle, just with 3 doors.
    """
    print(explanation)


# Main execution
if __name__ == "__main__":
    # Explain the problem
    explain_monty_hall()
    
    # Create simulator
    simulator = MontyHallSimulator(num_simulations=10000)
    
    # Run simulations
    results = simulator.run_simulations()
    
    # Print results
    simulator.print_results()
    
    # Perform statistical analysis
    simulator.statistical_test()
    
    # Create visualizations
    simulator.visualize_results()
    
    print("CONCLUSION: The simulation proves that switching doors")
    print("doubles your chances of winning from ~33% to ~67%!")
    