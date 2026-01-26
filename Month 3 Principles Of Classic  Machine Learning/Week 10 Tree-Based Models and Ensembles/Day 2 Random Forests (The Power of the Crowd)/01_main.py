

#Day 2: Random Forests (The Power of the Crowd)
#Objective: Understand Bagging (Bootstrap Aggregating). Why are 100 average trees better than 1 genius tree?
#Tree Visualization & Overfitting
#Goal: Visualize trees and understand overfitting issues

#Concept: Bootstrapping (sampling with replacement), Out-of-Bag (OOB) Error, Feature Randomness.

#How Decision Trees can overfit
#Pruning techniques (pre-pruning vs post-pruning)
#Regularization parameters: max_depth, min_samples_split, min_samples_leaf
#Understanding tree depth and complexity


#Task: Project - Heart Disease Prediction.

#Compare a single DecisionTree vs. a RandomForestClassifier.

#Observe how the Random Forest reduces Variance (overfitting).


#Project: Visualize a Decision Tree

#Train a Decision Tree on a simple dataset
#Use plot_tree() or graphviz to visualize
#Compare shallow vs deep trees
#Observe overfitting with different max_depth values

#Random Forests Deep Learning Prompt Structure
#Learn Prompt Structure

{
  "learning_session": "Random Forests - The Power of the Crowd",
  "duration": "Full Day Intensive",
  "objective": "Master Random Forests, Bagging, Tree Visualization, and Overfitting through Python implementation",
  
  "prerequisites": {
    "libraries_required": [
      "numpy",
      "pandas",
      "matplotlib",
      "seaborn",
      "scipy",
      "statsmodels",
      "sklearn",
      "sympy",
      "manim (optional for animations)"
    ],
    "knowledge_assumed": [
      "Basic Python syntax",
      "Basic statistics",
      "NumPy array operations",
      "Matplotlib plotting basics"
    ]
  },

  "learning_modules": [
    {
      "module_1": {
        "title": "Foundation: Understanding Decision Trees and Overfitting",
        "duration": "2 hours",
        "topics": [
          {
            "topic": "Decision Tree Fundamentals",
            "prompt": "Explain decision trees from scratch. Build a simple decision tree classifier using sklearn on the Iris dataset. Show step-by-step: 1) Load data with pandas, 2) Visualize data distribution with seaborn pairplot, 3) Train tree, 4) Visualize tree structure using plot_tree and export_text, 5) Show decision boundaries using matplotlib contour plots",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn"],
            "deliverables": [
              "Complete code with comments",
              "Tree visualization (graphical and text)",
              "Decision boundary plot",
              "Explanation of splits and Gini impurity"
            ]
          },
          {
            "topic": "Overfitting in Decision Trees",
            "prompt": "Demonstrate overfitting in decision trees: 1) Create a noisy sine wave dataset using numpy (y = sin(x) + noise), 2) Fit trees with max_depth=[1,2,3,5,10,20,None], 3) Plot all fitted curves on same matplotlib figure with training data, 4) Calculate and plot training vs validation MSE using sklearn.metrics, 5) Use seaborn to create heatmap showing how predictions change with depth",
            "libraries": ["numpy", "sklearn", "matplotlib", "seaborn"],
            "deliverables": [
              "Synthetic dataset generation code",
              "Multiple tree fits with different depths",
              "Comparison plot showing underfitting to overfitting",
              "MSE curves (train vs validation)",
              "Analysis of bias-variance tradeoff"
            ]
          },
          {
            "topic": "Tree Complexity Metrics",
            "prompt": "Analyze tree complexity: 1) Use sklearn's tree_ attribute to extract: number of nodes, depth, leaves, 2) Create pandas DataFrame tracking complexity metrics vs max_depth, 3) Use scipy.stats to calculate correlation between depth and test error, 4) Visualize with matplotlib subplots: depth vs nodes, depth vs accuracy, depth vs inference time, 5) Add statsmodels linear regression to show relationships",
            "libraries": ["sklearn", "pandas", "scipy", "statsmodels", "matplotlib"],
            "deliverables": [
              "Complexity extraction code",
              "Statistical analysis of complexity metrics",
              "Multi-panel visualization",
              "Regression analysis with confidence intervals"
            ]
          }
        ]
      }
    },
    {
      "module_2": {
        "title": "Regularization: Pruning and Control Parameters",
        "duration": "2 hours",
        "topics": [
          {
            "topic": "Pre-pruning Parameters",
            "prompt": "Implement comprehensive pre-pruning: 1) Use sklearn's DecisionTreeClassifier on breast cancer dataset, 2) Create grid of parameters: max_depth=[3,5,7,10], min_samples_split=[2,5,10,20], min_samples_leaf=[1,2,5,10], 3) Use nested loops to train all combinations, 4) Store results in pandas DataFrame with cross-validation scores, 5) Create seaborn heatmap showing accuracy for each parameter combination, 6) Use matplotlib 3D surface plot to show parameter interactions",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "numpy"],
            "deliverables": [
              "Grid search implementation",
              "Cross-validation results DataFrame",
              "Heatmap of parameter effects",
              "3D surface plot of interactions",
              "Best parameter identification"
            ]
          },
          {
            "topic": "Cost-Complexity Pruning (Post-pruning)",
            "prompt": "Demonstrate post-pruning with cost_complexity_pruning_path: 1) Train full tree on wine dataset, 2) Get pruning path using sklearn's cost_complexity_pruning_path, 3) Train tree for each alpha value, 4) Plot using matplotlib: alpha vs tree nodes, alpha vs depth, alpha vs accuracy, 5) Use numpy to find optimal alpha, 6) Compare pre-pruning vs post-pruning performance with seaborn barplot",
            "libraries": ["sklearn", "numpy", "matplotlib", "seaborn", "pandas"],
            "deliverables": [
              "Pruning path extraction",
              "Multiple trees at different alpha values",
              "Optimization curve visualization",
              "Comparison analysis",
              "Recommended pruning strategy"
            ]
          },
          {
            "topic": "Regularization Effects Visualization",
            "prompt": "Visualize regularization impact: 1) Create 2D classification dataset with sklearn.make_moons with noise, 2) Train trees with different regularization levels, 3) For each tree, create decision boundary using matplotlib contourf, 4) Calculate and overlay prediction confidence using predict_proba, 5) Use seaborn to show margin distributions, 6) Create animation showing how boundaries smooth with regularization (save as images for manual animation or use manim)",
            "libraries": ["sklearn", "matplotlib", "numpy", "seaborn"],
            "deliverables": [
              "Synthetic dataset creation",
              "Decision boundary visualizations",
              "Confidence region plots",
              "Margin analysis",
              "Progressive smoothing visualization"
            ]
          }
        ]
      }
    },
    {
      "module_3": {
        "title": "Bootstrap Aggregating (Bagging) Fundamentals",
        "duration": "2 hours",
        "topics": [
          {
            "topic": "Understanding Bootstrapping",
            "prompt": "Implement bootstrapping from scratch: 1) Create sample dataset with numpy (1000 points), 2) Write function to create bootstrap samples using numpy.random.choice with replacement, 3) Calculate statistic (mean) for each bootstrap sample, 4) Store in pandas DataFrame, 5) Visualize distribution with seaborn histplot and scipy.stats to overlay normal distribution, 6) Calculate confidence intervals using numpy.percentile, 7) Compare with statsmodels bootstrap implementation",
            "libraries": ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "statsmodels"],
            "deliverables": [
              "Bootstrap sampling function",
              "Bootstrap distribution visualization",
              "Confidence interval calculation",
              "Comparison with theoretical distribution",
              "Statistical validation"
            ]
          },
          {
            "topic": "Manual Bagging Implementation",
            "prompt": "Build bagging classifier from scratch: 1) Create classification dataset with sklearn.make_classification, 2) Implement bagging: create N bootstrap samples, train tree on each, 3) Implement prediction by majority voting using numpy, 4) Compare single tree vs bagged trees (10, 50, 100 estimators), 5) Plot learning curves with matplotlib showing train/test accuracy vs number of trees, 6) Use pandas to track individual tree predictions, 7) Visualize variance reduction with seaborn boxplots",
            "libraries": ["sklearn", "numpy", "pandas", "matplotlib", "seaborn"],
            "deliverables": [
              "Custom bagging implementation",
              "Voting mechanism code",
              "Learning curve visualization",
              "Variance reduction analysis",
              "Comparison with single model"
            ]
          },
          {
            "topic": "Variance Reduction Mathematics",
            "prompt": "Prove variance reduction mathematically: 1) Use sympy to derive variance formula for averaged predictions, 2) Show mathematically why Var(average of N) = Var(individual)/N for independent models, 3) Demonstrate with numpy simulation: create correlated predictions with different correlation levels, 4) Plot variance reduction vs correlation using matplotlib, 5) Use scipy.stats to calculate actual correlations between bagged trees, 6) Visualize with seaborn how correlation affects ensemble benefit",
            "libraries": ["sympy", "numpy", "matplotlib", "seaborn", "scipy"],
            "deliverables": [
              "Mathematical derivation using sympy",
              "Simulation code",
              "Variance vs correlation plot",
              "Tree correlation analysis",
              "Theoretical vs empirical comparison"
            ]
          }
        ]
      }
    },
    {
      "module_4": {
        "title": "Random Forests: Feature Randomness and OOB Error",
        "duration": "2 hours",
        "topics": [
          {
            "topic": "Feature Randomness Mechanism",
            "prompt": "Implement and visualize feature randomness: 1) Use sklearn RandomForestClassifier on digits dataset, 2) Track which features are used in each tree using tree.feature, 3) Create pandas DataFrame showing feature usage frequency, 4) Visualize with seaborn heatmap: trees vs features used, 5) Compare with BaggingClassifier (no feature randomness), 6) Plot correlation between trees using scipy.stats.spearmanr, 7) Show how max_features affects tree diversity",
            "libraries": ["sklearn", "pandas", "seaborn", "numpy", "scipy", "matplotlib"],
            "deliverables": [
              "Feature tracking implementation",
              "Feature usage heatmap",
              "Tree correlation analysis",
              "Diversity metrics calculation",
              "max_features parameter study"
            ]
          },
          {
            "topic": "Out-of-Bag (OOB) Error",
            "prompt": "Demonstrate OOB error estimation: 1) Train RandomForestClassifier with oob_score=True on classification dataset, 2) Manually calculate OOB predictions: for each sample, average predictions from trees that didn't see it, 3) Compare manual OOB score with sklearn's oob_score_, 4) Plot OOB error vs number of trees using matplotlib, 5) Compare OOB score with cross-validation score using pandas DataFrame, 6) Use seaborn to visualize OOB prediction confidence distribution, 7) Show with statsmodels that OOB is unbiased estimate",
            "libraries": ["sklearn", "numpy", "pandas", "matplotlib", "seaborn", "statsmodels"],
            "deliverables": [
              "Manual OOB calculation",
              "OOB vs CV comparison",
              "Convergence plot",
              "Confidence analysis",
              "Bias validation"
            ]
          },
          {
            "topic": "Why 100 Average Trees Beat 1 Genius Tree",
            "prompt": "Create comprehensive comparison: 1) Train optimal single DecisionTreeClassifier (tuned with GridSearchCV), 2) Train RandomForestClassifier with 100 trees (less tuned), 3) Use multiple datasets from sklearn.datasets, 4) Create pandas DataFrame comparing: accuracy, precision, recall, F1, training time, prediction time, 5) Use matplotlib to create multi-panel comparison plots, 6) Visualize prediction stability: make 100 predictions with bootstrap resampling, plot variance with seaborn, 7) Use scipy.stats to test if ensemble is significantly better, 8) Create decision boundary comparison on 2D data",
            "libraries": ["sklearn", "pandas", "numpy", "matplotlib", "seaborn", "scipy"],
            "deliverables": [
              "Comprehensive benchmark code",
              "Multi-metric comparison table",
              "Visualization suite",
              "Statistical significance tests",
              "Stability analysis",
              "Practical recommendations"
            ]
          }
        ]
      }
    },
    {
      "module_5": {
        "title": "Advanced Visualization and Interpretation",
        "duration": "2 hours",
        "topics": [
          {
            "topic": "Feature Importance Analysis",
            "prompt": "Deep dive into feature importance: 1) Train RandomForest on real dataset (e.g., sklearn's California housing), 2) Extract feature_importances_, 3) Calculate permutation importance using sklearn.inspection, 4) Compare both methods with pandas DataFrame, 5) Create seaborn barplot with error bars showing importance rankings, 6) Use scipy to calculate correlation between importance methods, 7) Visualize feature interactions with partial dependence plots using sklearn.inspection.PartialDependenceDisplay, 8) Create matplotlib 3D plots for 2-feature interactions",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "scipy", "numpy"],
            "deliverables": [
              "Multiple importance metrics",
              "Comparative analysis",
              "Ranking visualizations",
              "Partial dependence plots",
              "Interaction exploration"
            ]
          },
          {
            "topic": "Individual Tree Inspection",
            "prompt": "Examine trees within forest: 1) Access individual estimators from RandomForest, 2) For first 5 trees: plot structure using sklearn.tree.plot_tree, 3) Compare tree depths and node counts with pandas, 4) Extract decision paths for specific samples using decision_path, 5) Visualize paths with matplotlib, 6) Use seaborn to show prediction diversity across trees, 7) Create animation showing how different trees make different splits (save frames)",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "numpy"],
            "deliverables": [
              "Tree extraction code",
              "Multiple tree visualizations",
              "Structural comparison",
              "Decision path tracking",
              "Diversity analysis"
            ]
          },
          {
            "topic": "Error Analysis and Diagnostics",
            "prompt": "Comprehensive error analysis: 1) Get predictions and probabilities from RandomForest, 2) Create confusion matrix using sklearn.metrics, plot with seaborn heatmap, 3) Calculate per-class metrics, store in pandas, 4) Identify misclassified samples, visualize characteristics with matplotlib scatter, 5) Plot calibration curve using sklearn.calibration, 6) Use scipy.stats to analyze prediction distribution, 7) Create reliability diagram, 8) Analyze errors by tree depth using statsmodels regression",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "scipy", "statsmodels"],
            "deliverables": [
              "Confusion matrix analysis",
              "Per-class performance metrics",
              "Misclassification patterns",
              "Calibration analysis",
              "Statistical error modeling"
            ]
          }
        ]
      }
    },
    {
      "module_6": {
        "title": "Practical Implementation and Optimization",
        "duration": "2 hours",
        "topics": [
          {
            "topic": "Hyperparameter Tuning",
            "prompt": "Systematic hyperparameter optimization: 1) Define parameter grid for RandomForest: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, 2) Use sklearn.model_selection.RandomizedSearchCV with cv=5, 3) Track results in pandas DataFrame, 4) Visualize parameter importance with seaborn parallel coordinates plot, 5) Use matplotlib to plot validation curves for each parameter, 6) Apply scipy.stats to find parameter distributions that work best, 7) Compare computational cost vs performance gain",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "scipy", "numpy"],
            "deliverables": [
              "Hyperparameter search implementation",
              "Results tracking and analysis",
              "Parameter importance visualization",
              "Validation curves",
              "Cost-benefit analysis"
            ]
          },
          {
            "topic": "Ensemble Comparison",
            "prompt": "Compare ensemble methods: 1) Implement RandomForest, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier on same dataset, 2) Use sklearn.model_selection.cross_validate to get multiple metrics, 3) Create pandas DataFrame with results, 4) Visualize with seaborn: boxplots of CV scores, barplots of mean performance, 5) Use matplotlib to plot ROC curves for all methods, 6) Calculate AUC and plot comparison, 7) Analyze training/prediction time with seaborn, 8) Use scipy.stats for pairwise statistical tests",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "scipy", "numpy"],
            "deliverables": [
              "Multi-method implementation",
              "Cross-validation framework",
              "Comprehensive comparison plots",
              "ROC/AUC analysis",
              "Statistical comparison tests"
            ]
          },
          {
            "topic": "Real-World Application Pipeline",
            "prompt": "Build complete ML pipeline: 1) Load real dataset with pandas (e.g., Titanic, Heart Disease), 2) Perform EDA with seaborn: correlation heatmap, distribution plots, 3) Handle missing data with sklearn.impute, 4) Encode categorical variables with sklearn.preprocessing, 5) Create sklearn.pipeline.Pipeline with preprocessing and RandomForest, 6) Implement cross-validation, 7) Plot learning curves using sklearn.model_selection.learning_curve, 8) Analyze results with statsmodels, 9) Save model with joblib, 10) Create prediction function with confidence intervals",
            "libraries": ["sklearn", "pandas", "matplotlib", "seaborn", "scipy", "statsmodels", "numpy"],
            "deliverables": [
              "End-to-end pipeline code",
              "EDA visualizations",
              "Preprocessing pipeline",
              "Model training and validation",
              "Learning curve analysis",
              "Deployment-ready model"
            ]
          }
        ]
      }
    }
  ],

  "practice_exercises": [
    {
      "exercise_1": "Implement random forest from scratch using only numpy (no sklearn for the forest, only for datasets)",
      "exercise_2": "Create visualization comparing single deep tree vs ensemble of shallow trees on complex dataset",
      "exercise_3": "Build feature importance calculator that shows top features with confidence intervals",
      "exercise_4": "Develop diagnostic tool that plots: training curves, validation curves, OOB error, feature importance, tree depth distribution",
      "exercise_5": "Create interactive comparison tool (save multiple plots) showing effect of each hyperparameter"
    }
  ],

  "daily_schedule": {
    "hour_1-2": "Module 1 - Decision Trees and Overfitting",
    "hour_3-4": "Module 2 - Regularization and Pruning",
    "hour_5-6": "Module 3 - Bootstrap Aggregating",
    "hour_7-8": "Module 4 - Random Forests Core Concepts",
    "hour_9-10": "Module 5 - Advanced Visualization",
    "hour_11-12": "Module 6 - Practical Implementation + Exercises"
  },

  "output_format": "For each topic, provide: 1) Complete executable Python code with extensive comments, 2) Step-by-step explanation of each library usage, 3) Visualization outputs description, 4) Key insights and takeaways, 5) Common pitfalls to avoid",

  "success_criteria": [
    "Can explain why ensembles reduce variance mathematically",
    "Can implement bagging from scratch",
    "Can tune Random Forest hyperparameters effectively",
    "Can visualize and interpret tree decisions",
    "Can diagnose overfitting and apply appropriate regularization",
    "Can explain OOB error and when to use it",
    "Can compare different ensemble methods quantitatively"
  ]
}