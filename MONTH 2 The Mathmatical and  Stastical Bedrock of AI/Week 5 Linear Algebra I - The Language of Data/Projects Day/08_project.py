
#8: PageRank Algorithm (Simplified): Implement a simplified version of the PageRank algorithm using matrix multiplication to simulate web page importance.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class PageRank:
    def __init__(self, damping_factor=0.85, max_iterations=100, tolerance=1e-6):
        """
        Initialize PageRank calculator
        
        Parameters:
        - damping_factor: probability of following a link (typically 0.85)
        - max_iterations: maximum number of iterations
        - tolerance: convergence threshold
        """
        self.d = damping_factor
        self.max_iter = max_iterations
        self.tol = tolerance
        
    def build_transition_matrix(self, adjacency_matrix):
        """
        Build the transition probability matrix from adjacency matrix
        
        Parameters:
        - adjacency_matrix: NxN matrix where A[i,j]=1 means page i links to page j
        
        Returns:
        - Transition matrix M
        """
        n = len(adjacency_matrix)
        M = adjacency_matrix.astype(float)
        
        # Normalize each row by the number of outgoing links
        row_sums = M.sum(axis=1)
        
        # Handle pages with no outgoing links (dangling nodes)
        for i in range(n):
            if row_sums[i] == 0:
                M[i, :] = 1.0 / n  # Equal probability to all pages
            else:
                M[i, :] = M[i, :] / row_sums[i]
        
        return M
    
    def calculate_pagerank(self, adjacency_matrix, page_names=None):
        """
        Calculate PageRank using power iteration method
        
        Parameters:
        - adjacency_matrix: NxN adjacency matrix
        - page_names: list of page names (optional)
        
        Returns:
        - DataFrame with PageRank scores
        """
        n = len(adjacency_matrix)
        
        if page_names is None:
            page_names = [f'Page {i+1}' for i in range(n)]
        
        # Build transition matrix
        M = self.build_transition_matrix(adjacency_matrix)
        
        # Initialize PageRank vector (uniform distribution)
        pr = np.ones(n) / n
        
        # Store history for visualization
        history = [pr.copy()]
        
        # Power iteration
        for iteration in range(self.max_iter):
            # PageRank formula: PR(t+1) = (1-d)/n + d * M^T * PR(t)
            pr_new = (1 - self.d) / n + self.d * M.T @ pr
            
            history.append(pr_new.copy())
            
            # Check convergence
            if np.linalg.norm(pr_new - pr, 1) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
            
            pr = pr_new
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Page': page_names,
            'PageRank': pr,
            'Rank': range(1, n + 1)
        })
        
        # Sort by PageRank (descending)
        results = results.sort_values('PageRank', ascending=False).reset_index(drop=True)
        results['Rank'] = range(1, n + 1)
        
        return results, np.array(history)


def visualize_network(adjacency_matrix, pagerank_scores, page_names):
    """Visualize the web graph with PageRank scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Create directed graph
    G = nx.DiGraph()
    n = len(adjacency_matrix)
    
    # Add nodes and edges
    for i in range(n):
        G.add_node(i, label=page_names[i])
        for j in range(n):
            if adjacency_matrix[i, j] == 1:
                G.add_edge(i, j)
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Node sizes based on PageRank
    node_sizes = [score * 10000 for score in pagerank_scores]
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                           node_color=pagerank_scores, 
                           cmap='YlOrRd', alpha=0.9, ax=ax1)
    nx.draw_networkx_labels(G, pos, {i: page_names[i] for i in range(n)}, 
                           font_size=10, font_weight='bold', ax=ax1)
    nx.draw_networkx_edges(G, pos, edge_color='gray', 
                           arrows=True, arrowsize=20, 
                           arrowstyle='->', ax=ax1,
                           connectionstyle='arc3,rad=0.1')
    
    ax1.set_title('Web Page Network (Node size = PageRank)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Bar chart of PageRank scores
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, n))
    sorted_indices = np.argsort(pagerank_scores)[::-1]
    sorted_scores = pagerank_scores[sorted_indices]
    sorted_names = [page_names[i] for i in sorted_indices]
    
    bars = ax2.barh(sorted_names, sorted_scores, color=colors)
    ax2.set_xlabel('PageRank Score', fontsize=12, fontweight='bold')
    ax2.set_title('PageRank Rankings', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax2.text(score, i, f' {score:.4f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    # plt.show()


def visualize_convergence(history, page_names):
    """Visualize PageRank convergence over iterations"""
    plt.figure(figsize=(12, 6))
    
    iterations = range(len(history))
    for i, page in enumerate(page_names):
        scores = [h[i] for h in history]
        plt.plot(iterations, scores, marker='o', label=page, linewidth=2)
    
    plt.xlabel('Iteration', fontsize=12, fontweight='bold')
    plt.ylabel('PageRank Score', fontsize=12, fontweight='bold')
    plt.title('PageRank Convergence Over Iterations', fontsize=14, fontweight='bold')
    plt.legend(loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.show()


# Example 1: Simple web graph
print("="*60)
print("EXAMPLE 1: Simple Web Graph")
print("="*60)

# Define adjacency matrix (A[i,j] = 1 means page i links to page j)
adjacency = np.array([
    [0, 1, 1, 0],  # Page A links to B and C
    [1, 0, 0, 0],  # Page B links to A
    [1, 0, 0, 1],  # Page C links to A and D
    [0, 1, 1, 0]   # Page D links to B and C
])

page_names = ['Page A', 'Page B', 'Page C', 'Page D']

# Calculate PageRank
pr = PageRank(damping_factor=0.85)
results, history = pr.calculate_pagerank(adjacency, page_names)

print("\nPageRank Results:")
print(results.to_string(index=False))

# Visualize
visualize_network(adjacency, results['PageRank'].values, page_names)
visualize_convergence(history, page_names)


# Example 2: More complex network
print("\n" + "="*60)
print("EXAMPLE 2: Complex Web Graph")
print("="*60)

adjacency2 = np.array([
    [0, 1, 1, 0, 0, 0],  # Page 1 -> 2, 3
    [0, 0, 1, 1, 0, 0],  # Page 2 -> 3, 4
    [1, 0, 0, 1, 1, 0],  # Page 3 -> 1, 4, 5
    [0, 0, 0, 0, 1, 1],  # Page 4 -> 5, 6
    [0, 0, 0, 0, 0, 1],  # Page 5 -> 6
    [1, 1, 1, 0, 0, 0]   # Page 6 -> 1, 2, 3
])

page_names2 = ['Home', 'About', 'Services', 'Products', 'Blog', 'Contact']

pr2 = PageRank(damping_factor=0.85)
results2, history2 = pr2.calculate_pagerank(adjacency2, page_names2)

print("\nPageRank Results:")
print(results2.to_string(index=False))

visualize_network(adjacency2, results2['PageRank'].values, page_names2)
visualize_convergence(history2, page_names2)

print("\n" + "="*60)
print("Key Insights:")
print("="*60)
print(f"• Damping factor: {pr2.d} (probability of following links)")
print(f"• Random jump probability: {1-pr2.d}")
print("• Pages with more incoming links have higher PageRank")
print("• Quality of incoming links matters (high PR pages boost others)")
print("• The algorithm converges to a stable distribution")

