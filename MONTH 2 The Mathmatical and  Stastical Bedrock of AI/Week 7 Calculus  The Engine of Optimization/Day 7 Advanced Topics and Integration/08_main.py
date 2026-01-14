#Quick Reference Guide - All Topics & Libraries With the help of Claude
'''
# 📚 Optimization Study - Quick Reference Guide

## 🎯 Overview
Complete integration of advanced optimization topics with Python libraries: NumPy, Matplotlib, Seaborn, SymPy, and Pandas.

---

## 📊 Libraries & Their Roles

### 1. **NumPy** - Numerical Computing
```python
import numpy as np

# Array operations
x = np.array([1, 2, 3])
gradient = np.array([2*x[0], 2*x[1]])

# Linear algebra
hessian = np.array([[2, 0], [0, 2]])
eigenvalues = np.linalg.eigvals(hessian)
inverse = np.linalg.inv(hessian)

# Norms and distances
distance = np.linalg.norm(point - optimum)
```

**Use for:** Vector/matrix operations, gradients, Hessians, numerical computation

---

### 2. **Matplotlib** - Visualization
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 2D plotting
plt.plot(x, y, 'r-', label='Path')
plt.contour(X, Y, Z, levels=20)
plt.scatter(points[:, 0], points[:, 1])

# 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# Customization
plt.xlabel('x'), plt.ylabel('y')
plt.title('My Plot', fontweight='bold')
plt.legend(), plt.grid(True)
plt.savefig('plot.png', dpi=300)
```

**Use for:** Line plots, scatter plots, contours, 3D surfaces, animations

---

### 3. **Seaborn** - Statistical Visualization
```python
import seaborn as sns

# Style
sns.set_style("whitegrid")
sns.set_palette("husl")

# Statistical plots
sns.lineplot(data=df, x='iter', y='value', hue='method')
sns.boxplot(data=df, x='lr', y='final_value')
sns.violinplot(data=df, x='category', y='metric')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Distribution plots
sns.kdeplot(data=df, x='values', fill=True)
sns.scatterplot(data=df, x='x', y='y', hue='iteration')
```

**Use for:** Statistical analysis, correlation heatmaps, distribution plots, categorical data

---

### 4. **SymPy** - Symbolic Mathematics
```python
import sympy as sp

# Define symbols
x, y = sp.symbols('x y', real=True)

# Define function
f = (x - 2)**2 + (y + 1)**2

# Calculus
grad_x = sp.diff(f, x)  # ∂f/∂x
grad_y = sp.diff(f, y)  # ∂f/∂y
hess_xx = sp.diff(grad_x, x)  # ∂²f/∂x²

# Solve equations
critical_points = sp.solve([grad_x, grad_y], [x, y])

# Matrix operations
H = sp.Matrix([[hess_xx, hess_xy], [hess_xy, hess_yy]])
eigenvals = H.eigenvals()

# Convert to numerical function
f_numeric = sp.lambdify((x, y), f, 'numpy')
```

**Use for:** Symbolic calculus, finding critical points, exact derivatives, eigenvalue analysis

---

### 5. **Pandas** - Data Analysis
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'method': methods,
    'iterations': iters,
    'final_value': values
})

# Analysis
summary = df.groupby('method').agg({
    'iterations': ['mean', 'std'],
    'final_value': 'min'
})

# Pivot tables
pivot = df.pivot_table(
    values='final_value',
    index='method',
    columns='start_point'
)

# Export
df.to_csv('results.csv', index=False)
print(df.describe())
print(df.to_string(index=False))
```

**Use for:** Organizing results, statistical summaries, data tables, CSV export

---

## 🔍 Key Optimization Concepts

### 1. **Gradient Descent**
```python
def gradient_descent(start, lr=0.1, max_iter=100):
    x = start
    for i in range(max_iter):
        grad = compute_gradient(x)
        x = x - lr * grad
    return x
```

**Concept:** Follow negative gradient to minimize function
- **Learning rate**: Step size (too large → diverge, too small → slow)
- **Convergence**: When gradient ≈ 0

---

### 2. **Local vs Global Minima**
```python
# Convex function (one minimum)
f1 = (x - 2)**2 + (y + 1)**2  # ✓ Always finds global min

# Non-convex (multiple minima)
f2 = sin(x)*cos(y) + 0.1*(x²+y²)  # ✗ May get stuck
```

**Key Insight:**
- **Convex**: Any starting point → same solution
- **Non-convex**: Different starts → different solutions

---

### 3. **Saddle Points**
```python
# Saddle function
f = x**2 - y**2

# Hessian at (0,0)
H = [[2, 0], [0, -2]]

# Eigenvalues: [2, -2]
# Mixed signs → SADDLE POINT
```

**Classification:**
- Both λ > 0 → **Local Minimum**
- Both λ < 0 → **Local Maximum**
- Mixed signs → **Saddle Point**

---

### 4. **Second-Order Methods**
```python
# Newton's Method
def newton_method(start):
    x = start
    for i in range(max_iter):
        grad = compute_gradient(x)
        hess = compute_hessian(x)
        x = x - np.linalg.solve(hess, grad)
    return x
```

**Advantages:**
- Faster convergence (quadratic vs linear)
- Uses curvature information

**Disadvantages:**
- Expensive (compute + invert Hessian)
- Can fail if Hessian is singular

---

## 🎨 Visualization Patterns

### Pattern 1: Optimization Path
```python
# Contour + path
plt.contour(X, Y, Z, levels=20)
plt.plot(path[:, 0], path[:, 1], 'r-')
plt.plot(path[0, 0], path[0, 1], 'go')  # Start
plt.plot(path[-1, 0], path[-1, 1], 'r*')  # End
```

### Pattern 2: Convergence Curve
```python
values = [f(p) for p in path]
plt.semilogy(values)  # Log scale
plt.xlabel('Iteration')
plt.ylabel('Function Value')
```

### Pattern 3: 3D Surface
```python
ax = plt.subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
ax.plot(path[:, 0], path[:, 1], z_values, 'r-', linewidth=2)
```

### Pattern 4: Statistical Comparison
```python
sns.boxplot(data=df, x='method', y='iterations')
sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
```

---

## 💡 Best Practices

### NumPy Tips
```python
# Vectorization (fast)
result = np.sum((points - target)**2, axis=1)

# Avoid loops (slow)
for i in range(len(points)):
    result[i] = np.sum((points[i] - target)**2)
```

### Matplotlib Tips
```python
# Use tight_layout
plt.tight_layout()

# Save high-quality
plt.savefig('plot.png', dpi=300, bbox_inches='tight')

# Use gridspec for complex layouts
from matplotlib.gridspec import GridSpec
gs = GridSpec(3, 3, figure=fig)
```

### Seaborn Tips
```python
# Set style once
sns.set_style("whitegrid")
sns.set_palette("husl")

# Use long-form DataFrames
df = pd.DataFrame({'x': x_vals, 'y': y_vals, 'category': cats})
sns.lineplot(data=df, x='x', y='y', hue='category')
```

---

## 🚀 Complete Workflow Example

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sympy as sp

# 1. Define problem symbolically
x, y = sp.symbols('x y')
f_sym = (x - 2)**2 + (y + 1)**2
grad_x = sp.diff(f_sym, x)
grad_y = sp.diff(f_sym, y)

# 2. Convert to numerical
f = sp.lambdify((x, y), f_sym, 'numpy')
grad = lambda p: np.array([
    float(grad_x.subs([(x, p[0]), (y, p[1])])),
    float(grad_y.subs([(x, p[0]), (y, p[1])]))
])

# 3. Run optimization
results = []
for start in starting_points:
    path = gradient_descent(start, lr=0.1)
    results.append({
        'start': start,
        'path': path,
        'iterations': len(path)
    })

# 4. Create DataFrame
df = pd.DataFrame([{
    'start_x': r['start'][0],
    'start_y': r['start'][1],
    'iterations': r['iterations']
} for r in results])

# 5. Visualize
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Contour plot
X, Y = np.meshgrid(np.linspace(-2, 6, 100), np.linspace(-5, 3, 100))
Z = f(X, Y)
axes[0].contour(X, Y, Z, levels=20)
for r in results:
    axes[0].plot(r['path'][:, 0], r['path'][:, 1])

# Convergence
for r in results:
    values = [f(p) for p in r['path']]
    axes[1].semilogy(values)

# Statistical summary
sns.boxplot(data=df, y='iterations', ax=axes[2])

plt.tight_layout()
plt.savefig('analysis.png', dpi=300)
```

---

## 📝 Common Functions

### Task Function
```python
f(x, y) = (x - 2)² + (y + 1)²
∇f = [2(x-2), 2(y+1)]
Minimum: (2, -1)
Type: Convex (no local minima)
```

### Rosenbrock Function
```python
f(x, y) = (1-x)² + 100(y-x²)²
Minimum: (1, 1)
Type: Non-convex valley (hard to optimize)
```

### Saddle Function
```python
f(x, y) = x² - y²
Critical point: (0, 0) - saddle
Eigenvalues: [2, -2]
```

---

## ✅ Checklist for Strong Logic

- [ ] Understand NumPy broadcasting and vectorization
- [ ] Master Matplotlib subplots and 3D plots
- [ ] Use Seaborn for statistical visualizations
- [ ] Apply SymPy for symbolic analysis
- [ ] Organize results with Pandas DataFrames
- [ ] Implement gradient descent correctly
- [ ] Recognize convex vs non-convex functions
- [ ] Analyze critical points with Hessian
- [ ] Compare first-order vs second-order methods
- [ ] Visualize optimization paths effectively

---

## 🎓 Learning Path

1. **Start**: NumPy arrays, basic operations
2. **Basics**: Matplotlib line plots, scatter plots
3. **Intermediate**: 3D plots, contours, Seaborn
4. **Advanced**: SymPy calculus, Hessian analysis
5. **Expert**: Integration of all libraries, complex visualizations

---

## 📖 Additional Resources

### Documentation
- NumPy: https://numpy.org/doc/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- SymPy: https://docs.sympy.org/
- Pandas: https://pandas.pydata.org/docs/

### Key Topics
- Gradient descent optimization
- Convex optimization theory
- Second-order optimization methods
- Visualization best practices
- Scientific Python ecosystem

---

**Pro Tip:** Run each code artifact sequentially to build understanding from basics to advanced integration!'''