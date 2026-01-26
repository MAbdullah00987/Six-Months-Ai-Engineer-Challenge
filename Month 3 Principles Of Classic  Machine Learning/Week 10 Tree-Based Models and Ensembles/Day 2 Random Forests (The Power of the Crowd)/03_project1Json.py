

{
  "project_title": "Heart Disease Prediction: Decision Tree vs Random Forest",
  "objective": "Compare Decision Tree and Random Forest classifiers to demonstrate how Random Forest reduces variance (overfitting)",
  
  "requirements": {
    "libraries": [
      "numpy",
      "pandas", 
      "matplotlib",
      "seaborn",
      "scipy",
      "statsmodels",
      "scikit-learn"
    ],
    "exclude": ["react", "tensorflow", "pytorch"],
    "output_format": "Python script with comprehensive comments"
  },
  
  "project_structure": {
    "steps": [
      {
        "step": 1,
        "name": "Data Loading",
        "tasks": [
          "Load heart disease dataset",
          "Display dataset shape and info",
          "Show first few rows"
        ]
      },
      {
        "step": 2,
        "name": "Exploratory Data Analysis (EDA)",
        "tasks": [
          "Check for missing values",
          "Analyze target distribution",
          "Create visualizations: pie chart, histograms, correlation heatmap",
          "Explore relationships between features and target"
        ]
      },
      {
        "step": 3,
        "name": "Statistical Analysis",
        "tasks": [
          "Use statsmodels for logistic regression",
          "Identify significant features",
          "Display statistical summary"
        ]
      },
      {
        "step": 4,
        "name": "Data Preparation",
        "tasks": [
          "Split features (X) and target (y)",
          "Create train-test split (70-30)",
          "Apply feature scaling using StandardScaler",
          "Display split sizes"
        ]
      },
      {
        "step": 5,
        "name": "Decision Tree Model",
        "tasks": [
          "Initialize DecisionTreeClassifier with hyperparameters",
          "Train on training data",
          "Make predictions on train and test sets",
          "Calculate accuracy, classification report",
          "Compute variance (train_acc - test_acc)"
        ]
      },
      {
        "step": 6,
        "name": "Random Forest Model",
        "tasks": [
          "Initialize RandomForestClassifier with 100 estimators",
          "Train on training data",
          "Make predictions on train and test sets",
          "Calculate accuracy, classification report",
          "Compute variance (train_acc - test_acc)"
        ]
      },
      {
        "step": 7,
        "name": "Model Comparison",
        "tasks": [
          "Create comparison table showing train/test accuracy",
          "Calculate variance for both models",
          "Show variance reduction by Random Forest",
          "Display percentage improvement"
        ]
      },
      {
        "step": 8,
        "name": "Cross-Validation",
        "tasks": [
          "Perform 5-fold cross-validation on both models",
          "Display CV scores with mean and standard deviation",
          "Compare model stability"
        ]
      },
      {
        "step": 9,
        "name": "Visualizations",
        "tasks": [
          "Accuracy comparison bar chart",
          "Variance (overfitting) comparison",
          "Confusion matrices for both models",
          "Feature importance chart (Random Forest)",
          "ROC curves with AUC scores"
        ]
      },
      {
        "step": 10,
        "name": "Learning Curves",
        "tasks": [
          "Generate learning curves for Decision Tree",
          "Generate learning curves for Random Forest",
          "Visualize how gap between train/validation narrows with Random Forest",
          "Demonstrate overfitting reduction"
        ]
      }
    ]
  },
  
  "key_concepts_to_explain": [
    {
      "concept": "Variance (Overfitting)",
      "explanation": "When training accuracy is much higher than test accuracy, indicating model memorizes training data"
    },
    {
      "concept": "Bias-Variance Tradeoff",
      "explanation": "Balance between model's ability to fit training data vs generalize to new data"
    },
    {
      "concept": "Ensemble Learning",
      "explanation": "Combining multiple models to improve predictions and reduce variance"
    },
    {
      "concept": "Bagging (Bootstrap Aggregating)",
      "explanation": "Training each tree on random subset of data to increase diversity"
    },
    {
      "concept": "Random Feature Selection",
      "explanation": "Selecting random subset of features at each split to decorrelate trees"
    }
  ],
  
  "visualization_requirements": {
    "plots": [
      "Target distribution pie chart",
      "Feature distributions by target class",
      "Correlation heatmap",
      "Model accuracy comparison bars",
      "Variance comparison bars",
      "Confusion matrices (both models)",
      "Feature importance chart",
      "ROC curves with AUC",
      "Learning curves (both models)"
    ],
    "style": "Professional with clear labels, titles, and legends",
    "save_format": "PNG files with 300 DPI"
  },
  
  "success_criteria": {
    "demonstrate": [
      "Random Forest has lower variance than Decision Tree",
      "Random Forest generalizes better to test data",
      "Ensemble methods reduce overfitting",
      "Learning curves show convergence of train/validation scores"
    ],
    "metrics_to_report": [
      "Training accuracy",
      "Test accuracy", 
      "Variance (train - test accuracy)",
      "Cross-validation scores",
      "AUC-ROC scores",
      "Confusion matrix values"
    ]
  },
  
  "educational_focus": {
    "beginner_friendly": true,
    "include_comments": true,
    "explain_each_step": true,
    "print_intermediate_results": true,
    "show_mathematical_concepts": false,
    "focus_on_practical_implementation": true
  },
  
  "output_preferences": {
    "code_style": "Clean, well-commented, production-ready",
    "print_statements": "Detailed progress updates and results",
    "error_handling": "Basic warnings suppression",
    "reproducibility": "Set random seeds for consistent results"
  },
  
  "learning_outcomes": [
    "Understand difference between Decision Tree and Random Forest",
    "Learn how to measure overfitting using train-test accuracy gap",
    "Master scikit-learn model training and evaluation workflow",
    "Create professional data science visualizations",
    "Interpret confusion matrices, ROC curves, and learning curves",
    "Use statsmodels for statistical analysis",
    "Apply cross-validation for robust model evaluation"
  ]
}