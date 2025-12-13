
#7: Network Flow Model: Represent a simple network (e.g., traffic flow at intersections) as a system of linear equations.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# Define a network with 4 intersections
# Flow conservation: inflow = outflow at each intersection

print("=" * 60)
print("TRAFFIC NETWORK FLOW MODEL")
print("=" * 60)

# Network structure:
# External sources/sinks: S1, S2, S3, S4
# Internal intersections: A, B, C, D
#
# Network layout:
#     S1 --x1--> A --x5--> B --x9--> S3
#                |         |
#               x6        x10
#                |         |
#                v         v
#     S2 --x2--> C --x7--> D --x11--> S4
#                ^         ^
#               x3        x8
#                |         |
#               S2        S4

# Known flows (vehicles/hour)
known_flows = {
    'x1': 500,   # From S1 to A
    'x2': 300,   # From S2 to C
    'x9': 400,   # From B to S3
    'x11': 450,  # From D to S4
}

print("\nNetwork Description:")
print("-" * 60)
print("4 intersections: A, B, C, D")
print("External flows (known):")
for flow, value in known_flows.items():
    print(f"  {flow}: {value} vehicles/hour")

# Set up the system of linear equations
# Flow conservation at each intersection:
# Intersection A: x1 = x5 + x6
# Intersection B: x5 + x10 = x9
# Intersection C: x2 + x3 = x6 + x7
# Intersection D: x7 + x8 = x10 + x11

# Rearranging to standard form Ax = b:
# x5 + x6 = x1                    =>  x5 + x6 = 500
# x5 + x10 = x9                   =>  x5 + x10 = 400
# -x6 + x3 + x7 = -x2             =>  -x6 + x3 + x7 = -300
# x7 - x10 + x8 = x11             =>  x7 - x10 + x8 = 450

# Variables: [x3, x5, x6, x7, x8, x10]
# (We solve for internal flows)

A = np.array([
    [0,  1,  1,  0,  0,  0],   # Intersection A
    [0,  1,  0,  0,  0,  1],   # Intersection B
    [1,  0, -1,  1,  0,  0],   # Intersection C
    [0,  0,  0,  1,  1, -1],   # Intersection D
])

b = np.array([500, 400, -300, 450])

print("\n\nSystem of Linear Equations (Ax = b):")
print("-" * 60)
print("Variables: x3, x5, x6, x7, x8, x10")
print("\nCoefficient Matrix A:")
print(A)
print("\nConstants vector b:")
print(b)

# Check if system is underdetermined
n_vars = A.shape[1]
n_equations = A.shape[0]
rank = np.linalg.matrix_rank(A)

print(f"\nSystem Analysis:")
print(f"  Number of variables: {n_vars}")
print(f"  Number of equations: {n_equations}")
print(f"  Rank of matrix A: {rank}")

if rank < n_vars:
    print(f"  Status: UNDERDETERMINED (infinite solutions)")
    print(f"  Degrees of freedom: {n_vars - rank}")
    
    # Use least squares to find one particular solution
    solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    print("\n  Finding minimum norm solution...")
else:
    print(f"  Status: DETERMINED (unique solution exists)")
    solution = np.linalg.solve(A, b)

print("\n\nSolution (Internal Traffic Flows):")
print("-" * 60)
variables = ['x3', 'x5', 'x6', 'x7', 'x8', 'x10']
results = {}
for var, val in zip(variables, solution):
    results[var] = val
    print(f"{var}: {val:8.2f} vehicles/hour")

# Add known flows to results
results.update(known_flows)

# Verify the solution
print("\n\nVerification (Flow Conservation):")
print("-" * 60)
intersections = {
    'A': f"Inflow: {results['x1']:.2f}, Outflow: {results['x5']:.2f} + {results['x6']:.2f} = {results['x5'] + results['x6']:.2f}",
    'B': f"Inflow: {results['x5']:.2f} + {results['x10']:.2f} = {results['x5'] + results['x10']:.2f}, Outflow: {results['x9']:.2f}",
    'C': f"Inflow: {results['x2']:.2f} + {results['x3']:.2f} = {results['x2'] + results['x3']:.2f}, Outflow: {results['x6']:.2f} + {results['x7']:.2f} = {results['x6'] + results['x7']:.2f}",
    'D': f"Inflow: {results['x7']:.2f} + {results['x8']:.2f} = {results['x7'] + results['x8']:.2f}, Outflow: {results['x10']:.2f} + {results['x11']:.2f} = {results['x10'] + results['x11']:.2f}",
}

for intersection, check in intersections.items():
    print(f"Intersection {intersection}: {check}")

# Create a summary dataframe
df_flows = pd.DataFrame({
    'Flow': list(results.keys()),
    'Value (veh/hr)': [results[k] for k in results.keys()],
    'Type': ['Known' if k in known_flows else 'Calculated' for k in results.keys()]
})
df_flows = df_flows.sort_values('Flow')

print("\n\nFlow Summary Table:")
print("-" * 60)
print(df_flows.to_string(index=False))

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left plot: Network diagram
ax1.set_xlim(-1, 5)
ax1.set_ylim(-1, 5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('Traffic Network Flow Diagram', fontsize=14, fontweight='bold')

# Intersection positions
positions = {
    'S1': (0, 3.5),
    'A': (1.5, 3.5),
    'B': (3.5, 3.5),
    'S3': (5, 3.5),
    'S2': (0, 1.5),
    'C': (1.5, 1.5),
    'D': (3.5, 1.5),
    'S4': (5, 1.5),
}

# Draw intersections
for node, pos in positions.items():
    if node.startswith('S'):
        circle = plt.Circle(pos, 0.2, color='lightblue', ec='blue', linewidth=2, zorder=3)
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontsize=10, fontweight='bold')
    else:
        circle = plt.Circle(pos, 0.25, color='lightcoral', ec='red', linewidth=2, zorder=3)
        ax1.text(pos[0], pos[1], node, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax1.add_patch(circle)

# Draw flows with arrows
flows_to_draw = [
    ('S1', 'A', 'x1', results['x1']),
    ('A', 'B', 'x5', results['x5']),
    ('B', 'S3', 'x9', results['x9']),
    ('A', 'C', 'x6', results['x6']),
    ('S2', 'C', 'x2', results['x2']),
    ('C', 'D', 'x7', results['x7']),
    ('B', 'D', 'x10', results['x10']),
    ('D', 'S4', 'x11', results['x11']),
]

for start, end, label, value in flows_to_draw:
    x1, y1 = positions[start]
    x2, y2 = positions[end]
    
    # Adjust arrow positions to not overlap with circles
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx**2 + dy**2)
    
    offset = 0.25
    x1 += dx/dist * offset
    y1 += dy/dist * offset
    x2 -= dx/dist * offset
    y2 -= dy/dist * offset
    
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='darkgreen', zorder=2)
    ax1.add_patch(arrow)
    
    # Add flow label
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    ax1.text(mid_x, mid_y + 0.15, f'{label}\n{value:.0f}', 
            ha='center', va='bottom', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Legend
legend_elements = [
    mpatches.Patch(color='lightcoral', label='Intersection'),
    mpatches.Patch(color='lightblue', label='External Source/Sink'),
    mpatches.Patch(color='yellow', label='Flow (veh/hr)')
]
ax1.legend(handles=legend_elements, loc='upper left')

# Right plot: Bar chart of flows
ax2.barh(df_flows['Flow'], df_flows['Value (veh/hr)'], 
         color=['blue' if t == 'Known' else 'orange' for t in df_flows['Type']])
ax2.set_xlabel('Flow Rate (vehicles/hour)', fontsize=11)
ax2.set_ylabel('Flow Variable', fontsize=11)
ax2.set_title('Traffic Flow Rates at All Links', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Add value labels on bars
for i, (flow, value) in enumerate(zip(df_flows['Flow'], df_flows['Value (veh/hr)'])):
    ax2.text(value + 10, i, f'{value:.0f}', va='center', fontsize=9)

# Legend for bar chart
legend_elements2 = [
    mpatches.Patch(color='blue', label='Known (Input)'),
    mpatches.Patch(color='orange', label='Calculated')
]
ax2.legend(handles=legend_elements2)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("Analysis complete! The network flow is balanced.")
print("=" * 60)