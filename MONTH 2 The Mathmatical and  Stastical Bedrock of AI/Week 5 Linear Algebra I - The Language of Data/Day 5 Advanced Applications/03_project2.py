#Project 2: PageRank Algorithm (Simplified)
#Create 5-6 node web graph
#Build adjacency matrix
#Implement power iteration: vₖ₊₁ = M × vₖ
#Iterate until convergence
#Rank pages by final scores

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

print("="*80)
print("PROJECT 2: PAGERANK ALGORITHM (SIMPLIFIED)")
print("="*80)


# PART 1: UNDERSTANDING PAGERANK



print("PART 1: PAGERANK CONCEPT")

print("""
PAGERANK ALGORITHM:
- Invented by Larry Page and Sergey Brin (Google founders)
- Ranks web pages based on link structure
- Key idea: Important pages have many incoming links from other important pages

MATHEMATICAL FORMULA:
    PR(page) = (1-d)/N + d × Σ(PR(incoming_page) / outdegree(incoming_page))
    
    Where:
    - d = damping factor (usually 0.85)
    - N = total number of pages
    - outdegree = number of outgoing links

POWER ITERATION METHOD:
    v(k+1) = M × v(k)
    
    Iterate until convergence: ||v(k+1) - v(k)|| < tolerance
""")


# PART 2: CREATE WEB GRAPH



print("PART 2: CREATING A 6-NODE WEB GRAPH")


# Define our web graph (pages and their links)
pages = ['Home', 'About', 'Products', 'Blog', 'Contact', 'Support']
n_pages = len(pages)

print(f"Pages in our web graph: {pages}")
print(f"Total pages: {n_pages}\n")

# Define connections (who links to whom)
# Format: (from_page, to_page)
links = [
    ('Home', 'About'),
    ('Home', 'Products'),
    ('Home', 'Blog'),
    ('About', 'Home'),
    ('About', 'Contact'),
    ('Products', 'Home'),
    ('Products', 'Blog'),
    ('Products', 'Support'),
    ('Blog', 'Home'),
    ('Blog', 'Products'),
    ('Blog', 'About'),
    ('Contact', 'Home'),
    ('Contact', 'Support'),
    ('Support', 'Products'),
    ('Support', 'Blog')
]

print("Link structure:")
for from_page, to_page in links:
    print(f"  {from_page:12s} → {to_page}")

print(f"\nTotal links: {len(links)}")


# PART 3: BUILD ADJACENCY MATRIX



print("PART 3: BUILDING ADJACENCY MATRIX")

# Create adjacency matrix
adjacency_matrix = np.zeros((n_pages, n_pages))

# Create page index mapping
page_to_idx = {page: idx for idx, page in enumerate(pages)}

# Fill adjacency matrix
for from_page, to_page in links:
    i = page_to_idx[from_page]
    j = page_to_idx[to_page]
    adjacency_matrix[i, j] = 1

print("Adjacency Matrix (rows=from, cols=to):")
df_adjacency = pd.DataFrame(adjacency_matrix, index=pages, columns=pages)
print(df_adjacency)
print("\nNote: A[i,j] = 1 means page i links to page j")

# Calculate outdegrees
outdegrees = adjacency_matrix.sum(axis=1)
print("\nOutdegrees (number of outgoing links from each page):")
for page, outdeg in zip(pages, outdegrees):
    print(f"  {page:12s}: {int(outdeg)} links")


# PART 4: BUILD TRANSITION MATRIX



print("PART 4: BUILDING TRANSITION MATRIX")


print("""
TRANSITION MATRIX (M):
- Normalize each row by its outdegree
- M[i,j] = 1/outdegree(i) if page i links to page j
- Each row sums to 1 (probability distribution)
""")

# Create transition matrix
transition_matrix = np.zeros_like(adjacency_matrix, dtype=float)

for i in range(n_pages):
    if outdegrees[i] > 0:
        transition_matrix[i, :] = adjacency_matrix[i, :] / outdegrees[i]
    else:
        # If no outgoing links, link to all pages equally
        transition_matrix[i, :] = 1.0 / n_pages

print("\nTransition Matrix M:")
df_transition = pd.DataFrame(transition_matrix, index=pages, columns=pages)
print(df_transition.round(4))

# Verify rows sum to 1
row_sums = transition_matrix.sum(axis=1)
print(f"\nRow sums (should all be 1.0): {row_sums}")


# PART 5: ADD DAMPING FACTOR



print("PART 5: ADDING DAMPING FACTOR")


damping_factor = 0.85
teleport_prob = (1 - damping_factor) / n_pages

print(f"Damping factor (d): {damping_factor}")
print(f"Teleportation probability: {teleport_prob:.4f}")
print("""
Google Matrix (with damping):
    G = d × M + (1-d)/N × E
    
Where E is a matrix of all ones (uniform teleportation)
""")

# Create Google matrix
E = np.ones((n_pages, n_pages))
google_matrix = damping_factor * transition_matrix + teleport_prob * E

print("\nGoogle Matrix G:")
df_google = pd.DataFrame(google_matrix, index=pages, columns=pages)
print(df_google.round(4))


# PART 6: IMPLEMENT POWER ITERATION


print("\n" + "="*80)
print("PART 6: POWER ITERATION ALGORITHM")
print("="*80)

def pagerank_power_iteration(M, max_iterations=100, tolerance=1e-6, verbose=True):
    """
    Compute PageRank using power iteration
    
    Parameters:
    -----------
    M : numpy array
        Transition matrix (Google matrix)
    max_iterations : int
        Maximum number of iterations
    tolerance : float
        Convergence threshold
    verbose : bool
        Print iteration details
    
    Returns:
    --------
    pagerank : numpy array
        Final PageRank scores
    history : list
        History of PageRank vectors at each iteration
    """
    n = M.shape[0]
    
    # Initialize with uniform distribution
    v = np.ones(n) / n
    history = [v.copy()]
    
    if verbose:
        print(f"Initial PageRank (uniform): {v}")
        print(f"\nStarting power iteration...\n")
    
    for iteration in range(max_iterations):
        # Power iteration: v_new = M^T × v
        v_new = M.T @ v
        
        # Normalize (ensure sum = 1)
        v_new = v_new / v_new.sum()
        
        # Calculate change
        change = np.linalg.norm(v_new - v, ord=1)
        
        history.append(v_new.copy())
        
        if verbose and (iteration < 10 or iteration % 10 == 0):
            print(f"Iteration {iteration + 1:3d}: Change = {change:.8f}")
        
        # Check convergence
        if change < tolerance:
            if verbose:
                print(f"\n✓ Converged after {iteration + 1} iterations!")
            break
        
        v = v_new
    else:
        if verbose:
            print(f"\n⚠ Reached maximum iterations ({max_iterations})")
    
    return v_new, history

# Run PageRank
print("="*60)
pagerank_scores, history = pagerank_power_iteration(
    google_matrix, 
    max_iterations=100, 
    tolerance=1e-6,
    verbose=True
)
print("="*60)

print(f"\nFinal PageRank scores:")
print(f"Sum of scores: {pagerank_scores.sum():.6f} (should be 1.0)")


# PART 7: RANK PAGES BY SCORES



print("PART 7: RANKING PAGES")


# Create ranking DataFrame
ranking_df = pd.DataFrame({
    'Page': pages,
    'PageRank': pagerank_scores,
    'Percentage': pagerank_scores * 100,
    'Incoming_Links': adjacency_matrix.sum(axis=0),
    'Outgoing_Links': adjacency_matrix.sum(axis=1)
})

# Sort by PageRank (descending)
ranking_df = ranking_df.sort_values('PageRank', ascending=False).reset_index(drop=True)
ranking_df['Rank'] = range(1, len(ranking_df) + 1)

print("\nFinal Page Rankings:")
print(ranking_df.to_string(index=False))

print("\n" + "="*60)
print("INSIGHTS:")
print("="*60)
most_important = ranking_df.iloc[0]
print(f"Most important page: {most_important['Page']}")
print(f"  PageRank score: {most_important['PageRank']:.6f}")
print(f"  Incoming links: {int(most_important['Incoming_Links'])}")
print(f"  Outgoing links: {int(most_important['Outgoing_Links'])}")


# PART 8: VISUALIZE WEB GRAPH



print("PART 8: VISUALIZING THE WEB GRAPH")


# Create directed graph
G = nx.DiGraph()
G.add_nodes_from(pages)
G.add_edges_from(links)

# Calculate node sizes based on PageRank
node_sizes = pagerank_scores * 5000  # Scale for visibility

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Web Graph with PageRank sizing
ax = axes[0, 0]
pos = nx.spring_layout(G, seed=42, k=1.5)

# Draw nodes with size proportional to PageRank
nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                       node_color=pagerank_scores, cmap='YlOrRd',
                       ax=ax, alpha=0.9, edgecolors='black', linewidths=2)

# Draw edges
nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                       arrowsize=20, arrowstyle='->', ax=ax,
                       connectionstyle='arc3,rad=0.1', width=1.5)

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

ax.set_title('Web Graph (Node Size = PageRank Score)', fontsize=14, fontweight='bold')
ax.axis('off')

# Add colorbar
sm = plt.cm.ScalarMappable(cmap='YlOrRd', 
                           norm=plt.Normalize(vmin=pagerank_scores.min(), 
                                            vmax=pagerank_scores.max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('PageRank Score', rotation=270, labelpad=20)

# Plot 2: PageRank Bar Chart
ax = axes[0, 1]
colors = plt.cm.YlOrRd(pagerank_scores / pagerank_scores.max())
bars = ax.barh(ranking_df['Page'], ranking_df['PageRank'], color=colors, 
               edgecolor='black', linewidth=1.5)
ax.set_xlabel('PageRank Score', fontsize=12, fontweight='bold')
ax.set_title('PageRank Scores by Page', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (page, score) in enumerate(zip(ranking_df['Page'], ranking_df['PageRank'])):
    ax.text(score, i, f'{score:.4f}', va='center', ha='left', 
            fontweight='bold', fontsize=10)

# Plot 3: Convergence History
ax = axes[1, 0]
history_array = np.array(history)

for i, page in enumerate(pages):
    ax.plot(history_array[:, i], marker='o', label=page, linewidth=2, markersize=4)

ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('PageRank Score', fontsize=12, fontweight='bold')
ax.set_title('Convergence History (Power Iteration)', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Adjacency Matrix Heatmap
ax = axes[1, 1]
sns.heatmap(adjacency_matrix, annot=True, fmt='g', cmap='Blues',
            xticklabels=pages, yticklabels=pages, ax=ax,
            cbar_kws={'label': 'Link (1=yes, 0=no)'}, linewidths=0.5)
ax.set_title('Adjacency Matrix Heatmap', fontsize=14, fontweight='bold')
ax.set_xlabel('To Page', fontsize=12, fontweight='bold')
ax.set_ylabel('From Page', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('pagerank_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: pagerank_visualization.png")
plt.show()


# PART 9: CONVERGENCE ANALYSIS



print("PART 9: CONVERGENCE ANALYSIS")


# Calculate iteration-to-iteration changes
changes = []
for i in range(1, len(history)):
    change = np.linalg.norm(history[i] - history[i-1], ord=1)
    changes.append(change)

# Create convergence plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Log scale convergence
ax = axes[0]
ax.semilogy(range(1, len(changes) + 1), changes, 'b-o', linewidth=2, markersize=6)
ax.axhline(1e-6, color='r', linestyle='--', linewidth=2, label='Tolerance (1e-6)')
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Change (L1 norm)', fontsize=12, fontweight='bold')
ax.set_title('Convergence Speed (Log Scale)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

# Plot 2: Cumulative change
ax = axes[1]
cumulative_change = np.cumsum(changes)
ax.plot(range(1, len(cumulative_change) + 1), cumulative_change, 
        'g-o', linewidth=2, markersize=6)
ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Change', fontsize=12, fontweight='bold')
ax.set_title('Cumulative Change Over Iterations', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('convergence_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: convergence_analysis.png")
plt.show()

print(f"\nConvergence Statistics:")
print(f"  Total iterations: {len(changes)}")
print(f"  Initial change: {changes[0]:.8f}")
print(f"  Final change: {changes[-1]:.8f}")
print(f"  Average change: {np.mean(changes):.8f}")
print(f"  Std deviation: {np.std(changes):.8f}")


# PART 10: SENSITIVITY ANALYSIS



print("PART 10: DAMPING FACTOR SENSITIVITY ANALYSIS")


print("Testing different damping factors...")

damping_factors = [0.5, 0.7, 0.85, 0.9, 0.95]
results_by_damping = {}

for d in damping_factors:
    # Rebuild Google matrix with new damping factor
    teleport = (1 - d) / n_pages
    G_temp = d * transition_matrix + teleport * np.ones((n_pages, n_pages))
    
    # Compute PageRank
    pr_temp, _ = pagerank_power_iteration(G_temp, verbose=False)
    results_by_damping[d] = pr_temp

# Create DataFrame
sensitivity_df = pd.DataFrame(results_by_damping, index=pages)
sensitivity_df.columns = [f'd={d}' for d in damping_factors]

print("\nPageRank Scores for Different Damping Factors:")
print(sensitivity_df.round(6))

# Visualize sensitivity
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Heatmap
ax = axes[0]
sns.heatmap(sensitivity_df.T, annot=True, fmt='.4f', cmap='RdYlGn',
            ax=ax, cbar_kws={'label': 'PageRank Score'}, linewidths=0.5)
ax.set_title('PageRank Sensitivity to Damping Factor', fontsize=14, fontweight='bold')
ax.set_xlabel('Page', fontsize=12, fontweight='bold')
ax.set_ylabel('Damping Factor', fontsize=12, fontweight='bold')

# Plot 2: Line plot
ax = axes[1]
for page in pages:
    scores = [results_by_damping[d][page_to_idx[page]] for d in damping_factors]
    ax.plot(damping_factors, scores, marker='o', linewidth=2, 
            markersize=8, label=page)

ax.set_xlabel('Damping Factor', fontsize=12, fontweight='bold')
ax.set_ylabel('PageRank Score', fontsize=12, fontweight='bold')
ax.set_title('PageRank vs Damping Factor', fontsize=14, fontweight='bold')
ax.legend(loc='best', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('damping_sensitivity.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: damping_sensitivity.png")
plt.show()


# PART 11: COMPARISON WITH NETWORKX



print("PART 11: VERIFICATION WITH NETWORKX")


# Calculate PageRank using NetworkX (for verification)
nx_pagerank = nx.pagerank(G, alpha=damping_factor)
nx_scores = np.array([nx_pagerank[page] for page in pages])

# Compare results
comparison_df = pd.DataFrame({
    'Page': pages,
    'Our_Implementation': pagerank_scores,
    'NetworkX': nx_scores,
    'Difference': np.abs(pagerank_scores - nx_scores)
})

comparison_df = comparison_df.sort_values('Our_Implementation', ascending=False)

print("\nComparison with NetworkX PageRank:")
print(comparison_df.to_string(index=False))

max_diff = comparison_df['Difference'].max()
print(f"\nMaximum difference: {max_diff:.10f}")

if max_diff < 1e-5:
    print("✓ Our implementation matches NetworkX! ✓")
else:
    print("⚠ Small differences may exist due to numerical precision")

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(pages))
width = 0.35

bars1 = ax.bar(x - width/2, pagerank_scores, width, label='Our Implementation',
               color='steelblue', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, nx_scores, width, label='NetworkX',
               color='coral', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Page', fontsize=12, fontweight='bold')
ax.set_ylabel('PageRank Score', fontsize=12, fontweight='bold')
ax.set_title('Implementation Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pages, rotation=45, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('implementation_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: implementation_comparison.png")
plt.show()

# PART 12: DETAILED LINK ANALYSIS


print("PART 12: DETAILED LINK ANALYSIS")


# Calculate various metrics
link_analysis = pd.DataFrame({
    'Page': pages,
    'PageRank': pagerank_scores,
    'Rank': ranking_df['Rank'],
    'In_Degree': adjacency_matrix.sum(axis=0).astype(int),
    'Out_Degree': adjacency_matrix.sum(axis=1).astype(int),
    'In/Out_Ratio': np.where(adjacency_matrix.sum(axis=1) > 0,
                             adjacency_matrix.sum(axis=0) / adjacency_matrix.sum(axis=1),
                             adjacency_matrix.sum(axis=0))
})

link_analysis = link_analysis.sort_values('PageRank', ascending=False)

print("\nDetailed Link Analysis:")
print(link_analysis.to_string(index=False))

# Create comprehensive analysis plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: PageRank vs In-Degree
ax = axes[0, 0]
scatter = ax.scatter(link_analysis['In_Degree'], link_analysis['PageRank'],
                    s=300, c=link_analysis['PageRank'], cmap='viridis',
                    edgecolors='black', linewidths=2, alpha=0.8)
for i, page in enumerate(link_analysis['Page']):
    ax.annotate(page, 
                (link_analysis['In_Degree'].iloc[i], 
                 link_analysis['PageRank'].iloc[i]),
                fontsize=9, fontweight='bold', ha='center')
ax.set_xlabel('In-Degree (Incoming Links)', fontsize=12, fontweight='bold')
ax.set_ylabel('PageRank Score', fontsize=12, fontweight='bold')
ax.set_title('PageRank vs In-Degree', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax, label='PageRank')

# Plot 2: In-Degree vs Out-Degree
ax = axes[0, 1]
scatter = ax.scatter(link_analysis['Out_Degree'], link_analysis['In_Degree'],
                    s=link_analysis['PageRank']*3000, 
                    c=link_analysis['PageRank'], cmap='plasma',
                    edgecolors='black', linewidths=2, alpha=0.8)
for i, page in enumerate(link_analysis['Page']):
    ax.annotate(page, 
                (link_analysis['Out_Degree'].iloc[i], 
                 link_analysis['In_Degree'].iloc[i]),
                fontsize=9, fontweight='bold', ha='center')
ax.set_xlabel('Out-Degree (Outgoing Links)', fontsize=12, fontweight='bold')
ax.set_ylabel('In-Degree (Incoming Links)', fontsize=12, fontweight='bold')
ax.set_title('Link Structure (Bubble Size = PageRank)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Transition Matrix Heatmap
ax = axes[1, 0]
sns.heatmap(transition_matrix, annot=True, fmt='.3f', cmap='coolwarm',
            xticklabels=pages, yticklabels=pages, ax=ax,
            cbar_kws={'label': 'Transition Probability'}, linewidths=0.5)
ax.set_title('Transition Matrix (Normalized)', fontsize=14, fontweight='bold')
ax.set_xlabel('To Page', fontsize=12, fontweight='bold')
ax.set_ylabel('From Page', fontsize=12, fontweight='bold')

# Plot 4: Final Ranking with Metrics
ax = axes[1, 1]
y_pos = np.arange(len(link_analysis))
ax.barh(y_pos, link_analysis['PageRank'], color='steelblue', 
        edgecolor='black', linewidth=1.5)
ax.set_yticks(y_pos)
ax.set_yticklabels([f"#{row['Rank']} {row['Page']}" 
                     for _, row in link_analysis.iterrows()])
ax.set_xlabel('PageRank Score', fontsize=12, fontweight='bold')
ax.set_title('Final Rankings with Details', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add text annotations
for i, (_, row) in enumerate(link_analysis.iterrows()):
    ax.text(row['PageRank'], i, 
            f" {row['PageRank']:.4f} (In:{int(row['In_Degree'])}, Out:{int(row['Out_Degree'])})",
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('detailed_analysis.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: detailed_analysis.png")
plt.show()

# Project Notes
print("PROJECT Notes")


summary = f"""
COMPLETED ALL OBJECTIVES:

1. CREATED 6-NODE WEB GRAPH
   - Pages: {', '.join(pages)}
   - Total links: {len(links)}

2. BUILT ADJACENCY MATRIX
   - Size: {n_pages}×{n_pages}
   - Represents link structure

3. IMPLEMENTED POWER ITERATION
   - Algorithm: v(k+1) = M^T × v(k)
   - Converged in {len(changes)} iterations
   - Final convergence: {changes[-1]:.10f}

4. RANKED PAGES BY SCORES
   - Winner: {ranking_df.iloc[0]['Page']} (Score: {ranking_df.iloc[0]['PageRank']:.6f})
   - Runner-up: {ranking_df.iloc[1]['Page']} (Score: {ranking_df.iloc[1]['PageRank']:.6f})

5. ADDITIONAL ANALYSIS
   - Damping factor sensitivity (tested 5 values)
   - Convergence visualization
   - Comparison with NetworkX (verified ✓)
   - Link structure analysis

FILES GENERATED:
- pagerank_visualization.png (main results)
- convergence_analysis.png (iteration tracking)
- damping_sensitivity.png (parameter testing)
- implementation_comparison.png (verification)
- detailed_analysis.png (comprehensive metrics)

TOTAL: 5 visualization files

KEY FINDINGS:
- PageRank successfully ranks pages by importance
- Algorithm converges quickly (< {len(changes)} iterations)
- Implementation matches NetworkX (max diff: {max_diff:.10f})
- Damping factor significantly affects results
"""
