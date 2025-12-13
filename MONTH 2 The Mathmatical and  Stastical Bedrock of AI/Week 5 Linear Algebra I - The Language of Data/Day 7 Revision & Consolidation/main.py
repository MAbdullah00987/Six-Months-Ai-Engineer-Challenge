
#Day 6: Advanced Applications
#Read: Mathematics for ML Chapter 3 (Intro) - Eigenvalues preview
#Review Coursera Week 4 materials - Matrix inverses and special matrices

#Project 1: Affine Transformations

#Implement translation + rotation + scaling
#Use homogeneous coordinates (3×3 matrices for 2D)
#Create animation of shape moving and rotating
#Apply to letter shapes or simple graphics


#Project 2: PageRank Algorithm (Simplified)

#Create 5-6 node web graph
#Build adjacency matrix
#mplement power iteration: vₖ₊₁ = M × vₖ
#Iterate until convergence
#Rank pages by final scores


"""
Project 8: PageRank Algorithm (Simplified)
Implements PageRank using power iteration with a web graph
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch

class PageRank:
    """PageRank algorithm implementation"""
    
    def __init__(self, nodes, edges, damping_factor=0.85):
        """
        Initialize PageRank
        
        nodes: list of node names
        edges: list of (from, to) tuples
        damping_factor: probability of following links (default 0.85)
        """
        self.nodes = nodes
        self.edges = edges
        self.n = len(nodes)
        self.damping = damping_factor
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Build transition matrix
        self.M = self._build_transition_matrix()
        
    def _build_transition_matrix(self):
        """Build the transition matrix M"""
        # Initialize adjacency matrix
        adjacency = np.zeros((self.n, self.n))
        
        # Count outgoing links for each node
        outgoing = np.zeros(self.n)
        
        for from_node, to_node in self.edges:
            i = self.node_to_idx[from_node]
            j = self.node_to_idx[to_node]
            adjacency[j][i] = 1  # Column stochastic: M[j][i] means link from i to j
            outgoing[i] += 1
        
        # Build transition matrix (column stochastic)
        M = np.zeros((self.n, self.n))
        for i in range(self.n):
            if outgoing[i] > 0:
                M[:, i] = adjacency[:, i] / outgoing[i]
            else:
                # Handle dangling nodes (no outgoing links)
                M[:, i] = 1.0 / self.n
        
        return M
    
    def power_iteration(self, max_iterations=100, tolerance=1e-6):
        """
        Perform power iteration to find PageRank
        
        Returns: (scores, history)
        """
        # Initialize uniform distribution
        v = np.ones(self.n) / self.n
        history = [v.copy()]
        
        for iteration in range(max_iterations):
            # Power iteration with damping
            v_new = self.damping * (self.M @ v) + (1 - self.damping) / self.n
            
            # Normalize (should sum to 1)
            v_new = v_new / v_new.sum()
            
            history.append(v_new.copy())
            
            # Check convergence
            if np.linalg.norm(v_new - v, ord=1) < tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            v = v_new
        
        return v, history
    
    def get_rankings(self, scores):
        """Get ranked list of pages"""
        rankings = [(self.nodes[i], scores[i]) for i in range(self.n)]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


def create_web_graph():
    """Create a sample 6-node web graph"""
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    edges = [
        ('A', 'B'), ('A', 'C'), ('A', 'D'),  # A links to B, C, D
        ('B', 'C'), ('B', 'D'),               # B links to C, D
        ('C', 'A'), ('C', 'E'),               # C links to A, E
        ('D', 'E'), ('D', 'F'),               # D links to E, F
        ('E', 'F'), ('E', 'A'),               # E links to F, A
        ('F', 'C')                             # F links to C
    ]
    return nodes, edges


def visualize_graph(nodes, edges, scores=None, ax=None):
    """Visualize the web graph with PageRank scores"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create networkx graph for layout
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    # Position nodes
    pos = {
        'A': (1, 2),
        'B': (2.5, 3),
        'C': (4, 3),
        'D': (2.5, 1),
        'E': (4, 1),
        'F': (5.5, 2)
    }
    
    # Draw edges
    for edge in edges:
        from_node, to_node = edge
        from_pos = pos[from_node]
        to_pos = pos[to_node]
        
        arrow = FancyArrowPatch(
            from_pos, to_pos,
            arrowstyle='->', 
            mutation_scale=20,
            linewidth=2,
            color='gray',
            alpha=0.6,
            connectionstyle="arc3,rad=0.1"
        )
        ax.add_patch(arrow)
    
    # Draw nodes
    if scores is not None:
        for i, node in enumerate(nodes):
            x, y = pos[node]
            score = scores[i]
            
            # Node size and color based on score
            size = 1000 + score * 10000
            color_intensity = 0.3 + score * 3
            
            circle = plt.Circle((x, y), radius=0.3, 
                              color=f'C0',
                              alpha=min(color_intensity, 1.0),
                              zorder=10)
            ax.add_patch(circle)
            
            # Node label
            ax.text(x, y, node, fontsize=20, fontweight='bold',
                   ha='center', va='center', zorder=11)
            
            # Score below node
            ax.text(x, y - 0.45, f'{score*100:.1f}%',
                   fontsize=10, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        for node in nodes:
            x, y = pos[node]
            circle = plt.Circle((x, y), radius=0.3,
                              color='lightblue',
                              alpha=0.7,
                              zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, node, fontsize=20, fontweight='bold',
                   ha='center', va='center', zorder=11)
    
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return ax


def plot_convergence(history):
    """Plot convergence of PageRank scores over iterations"""
    
    history_array = np.array(history)
    iterations = len(history)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Score evolution for each node
    for i, node in enumerate(['A', 'B', 'C', 'D', 'E', 'F']):
        ax1.plot(history_array[:, i], marker='o', label=f'Node {node}', linewidth=2)
    
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('PageRank Score', fontsize=12)
    ax1.set_title('PageRank Convergence Over Iterations', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Change in scores (L1 norm)
    if len(history) > 1:
        changes = [np.linalg.norm(history[i] - history[i-1], ord=1) 
                  for i in range(1, len(history))]
        ax2.plot(range(1, len(history)), changes, marker='o', 
                color='red', linewidth=2)
        ax2.axhline(y=1e-6, color='green', linestyle='--', 
                   label='Convergence threshold (1e-6)')
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Change in Scores (L1 norm)', fontsize=12)
        ax2.set_title('Convergence Rate', fontsize=14, fontweight='bold')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def display_results_table(rankings):
    """Display results in a formatted table using pandas"""
    
    df = pd.DataFrame(rankings, columns=['Page', 'PageRank Score'])
    df['Rank'] = range(1, len(rankings) + 1)
    df['Percentage'] = (df['PageRank Score'] * 100).round(2)
    df = df[['Rank', 'Page', 'PageRank Score', 'Percentage']]
    
    print("\n" + "=" * 60)
    print("PAGERANK RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    return df


def demonstrate_step_by_step(pagerank, num_steps=5):
    """Show step-by-step iteration process"""
    
    v = np.ones(pagerank.n) / pagerank.n
    
    print("\n" + "=" * 60)
    print("STEP-BY-STEP POWER ITERATION")
    print("=" * 60)
    print(f"\nInitial vector (uniform distribution):")
    print(f"v0 = {v}")
    print(f"Sum: {v.sum():.6f}")
    
    for step in range(num_steps):
        v_new = pagerank.damping * (pagerank.M @ v) + (1 - pagerank.damping) / pagerank.n
        v_new = v_new / v_new.sum()
        
        print(f"\n--- Iteration {step + 1} ---")
        print(f"v({step + 1}) = {v_new}")
        print(f"Sum: {v_new.sum():.6f}")
        print(f"Change (L1): {np.linalg.norm(v_new - v, ord=1):.8f}")
        
        # Show rankings
        temp_rankings = [(pagerank.nodes[i], v_new[i]) for i in range(pagerank.n)]
        temp_rankings.sort(key=lambda x: x[1], reverse=True)
        print("Current rankings:", [f"{node}({score*100:.1f}%)" 
                                   for node, score in temp_rankings])
        
        v = v_new


def main():
    """Main function to run PageRank demonstration"""
    
    print("=" * 60)
    print("Project 8: PageRank Algorithm (Power Iteration)")
    print("=" * 60)
    print("\nTheory:")
    print("* PageRank uses power iteration: v_(k+1) = M * v_k")
    print("* With damping factor d=0.85: v_(k+1) = d*M*v_k + (1-d)/n*1")
    print("* Iterates until convergence (typically 20-50 iterations)")
    print("* Higher scores = more important pages")
    print("=" * 60)
    
    # Create web graph
    nodes, edges = create_web_graph()
    
    print(f"\nWeb Graph Structure:")
    print(f"Nodes: {nodes}")
    print(f"Edges ({len(edges)} links):")
    for edge in edges:
        print(f"  {edge[0]} -> {edge[1]}")
    
    # Initialize PageRank
    pr = PageRank(nodes, edges, damping_factor=0.85)
    
    # Display transition matrix
    print("\n" + "=" * 60)
    print("TRANSITION MATRIX M (Column Stochastic)")
    print("=" * 60)
    df_matrix = pd.DataFrame(pr.M, 
                            columns=nodes,
                            index=nodes)
    print(df_matrix.round(3))
    print("\nNote: Each column sums to 1.0 (probability distribution)")
    print("=" * 60)
    
    # Show step-by-step for first few iterations
    demonstrate_step_by_step(pr, num_steps=5)
    
    # Run full power iteration
    print("\n" + "=" * 60)
    print("RUNNING FULL POWER ITERATION")
    print("=" * 60)
    final_scores, history = pr.power_iteration(max_iterations=100)
    
    # Get rankings
    rankings = pr.get_rankings(final_scores)
    
    # Display results
    df_results = display_results_table(rankings)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Plot 1: Final graph with scores
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    visualize_graph(nodes, edges, final_scores, ax1)
    ax1.set_title('Web Graph with PageRank Scores\n(Node size represents importance)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Plot 2: Convergence
    fig2 = plot_convergence(history)
    
    # Plot 3: Bar chart of final rankings
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    pages = [r[0] for r in rankings]
    scores = [r[1] * 100 for r in rankings]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(pages)))
    
    bars = ax3.barh(pages, scores, color=colors, edgecolor='black', linewidth=2)
    ax3.set_xlabel('PageRank Score (%)', fontsize=12)
    ax3.set_ylabel('Page', fontsize=12)
    ax3.set_title('Final PageRank Rankings', fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (page, score) in enumerate(zip(pages, scores)):
        ax3.text(score + 0.5, i, f'{score:.2f}%', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Show all plots
    plt.show()
    
    print("\n[x] Project 8 Complete!")
    print("\nKey Insights:")
    print(f"• Most important page: {rankings[0][0]} ({rankings[0][1]*100:.2f}%)")
    print(f"• Least important page: {rankings[-1][0]} ({rankings[-1][1]*100:.2f}%)")
    print(f"• Converged in {len(history)-1} iterations")
    print(f"• Damping factor: {pr.damping}")


if __name__ == "__main__":
    main()