


{
  "day": 1,
  "topic": "Decision Trees - The Building Block",
  "learning_objectives": [
    "Understand how decision trees make splits",
    "Master Gini Impurity and Entropy calculations",
    "Visualize trees using multiple methods",
    "Implement from scratch using NumPy",
    "Create animations with Manim",
    "Perform statistical analysis with SciPy/StatsModels"
  ],
  "tools_to_master": [
    "Python (core logic)",
    "NumPy (mathematical operations)",
    "Pandas (data manipulation)",
    "Matplotlib (static visualizations)",
    "Seaborn (statistical plots)",
    "SymPy (symbolic mathematics)",
    "Manim (animations)",
    "SciPy (statistical tests)",
    "StatsModels (regression analysis)",
    "sklearn (machine learning)"
  ],
  "learning_modules": [
    {
      "module": 1,
      "title": "Mathematical Foundation with SymPy",
      "duration": "30 minutes",
      "tasks": [
        "Derive Gini Impurity formula symbolically",
        "Derive Entropy formula symbolically",
        "Calculate Information Gain mathematically",
        "Prove relationships between metrics"
      ]
    },
    {
      "module": 2,
      "title": "Core Implementation with NumPy",
      "duration": "45 minutes",
      "tasks": [
        "Calculate Gini from scratch",
        "Calculate Entropy from scratch",
        "Find best split using vectorization",
        "Build tree structure with NumPy arrays"
      ]
    },
    {
      "module": 3,
      "title": "Data Analysis with Pandas",
      "duration": "30 minutes",
      "tasks": [
        "Load and explore Iris/Mushroom datasets",
        "Calculate feature statistics",
        "Perform groupby operations for splits",
        "Create contingency tables"
      ]
    },
    {
      "module": 4,
      "title": "Statistical Validation with SciPy/StatsModels",
      "duration": "45 minutes",
      "tasks": [
        "Chi-square tests for categorical splits",
        "ANOVA for continuous features",
        "Logistic regression comparison",
        "Bootstrap confidence intervals"
      ]
    },
    {
      "module": 5,
      "title": "Visualization with Matplotlib/Seaborn",
      "duration": "45 minutes",
      "tasks": [
        "Plot decision boundaries",
        "Visualize tree structure manually",
        "Create heatmaps of impurity",
        "Compare multiple trees"
      ]
    },
    {
      "module": 6,
      "title": "Animation with Manim",
      "duration": "60 minutes",
      "tasks": [
        "Animate tree growing process",
        "Show split selection visually",
        "Animate data point traversal",
        "Create educational video"
      ]
    },
    {
      "module": 7,
      "title": "sklearn Implementation & Comparison",
      "duration": "45 minutes",
      "tasks": [
        "Train DecisionTreeClassifier",
        "Compare with custom implementation",
        "Visualize with plot_tree",
        "Export to graphviz"
      ]
    }
  ],
  "practice_exercises": [
    {
      "exercise": 1,
      "title": "From Scratch: Gini Calculator",
      "difficulty": "Medium",
      "libraries": ["numpy", "pandas"],
      "goal": "Calculate Gini impurity for every possible split"
    },
    {
      "exercise": 2,
      "title": "Statistical Tests for Splits",
      "difficulty": "Advanced",
      "libraries": ["scipy", "statsmodels"],
      "goal": "Validate splits using statistical significance tests"
    },
    {
      "exercise": 3,
      "title": "Interactive Tree Visualization",
      "difficulty": "Advanced",
      "libraries": ["matplotlib", "seaborn"],
      "goal": "Create clickable tree visualization"
    },
    {
      "exercise": 4,
      "title": "Manim Educational Video",
      "difficulty": "Expert",
      "libraries": ["manim"],
      "goal": "5-minute video explaining decision trees"
    }
  ],
  "assessment_criteria": {
    "understanding": [
      "Can explain Gini vs Entropy trade-offs",
      "Understands information gain concept",
      "Knows when to use pruning"
    ],
    "coding": [
      "Can implement Gini from scratch",
      "Uses NumPy vectorization efficiently",
      "Creates clean visualizations"
    ],
    "application": [
      "Traces predictions manually",
      "Validates results statistically",
      "Creates educational content"
    ]
  }
}