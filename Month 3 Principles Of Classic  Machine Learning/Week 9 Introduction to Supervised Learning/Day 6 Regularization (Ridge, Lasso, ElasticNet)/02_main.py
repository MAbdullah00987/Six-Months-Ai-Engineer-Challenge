
#JSON Learning Prompts for Regularization

{
  "regularization_learning_system": {
    "meta_info": {
      "topic": "Regularization Techniques (Ridge, Lasso, ElasticNet)",
      "goal": "Master regularization to prevent overfitting",
      "libraries": ["numpy", "pandas", "matplotlib", "seaborn", "scipy", "sympy", "sklearn", "statsmodels"],
      "difficulty": "Intermediate to Advanced",
      "estimated_time": "4-6 hours"
    },
    
    "learning_phases": [
      {
        "phase": 1,
        "name": "Mathematical Foundation",
        "duration": "45 minutes",
        "prompts": [
          {
            "prompt_id": "math_01",
            "question": "Using sympy, show me the mathematical derivation of Ridge regression loss function and its gradient. Include step-by-step symbolic differentiation.",
            "expected_libraries": ["sympy"],
            "key_concepts": ["L2 penalty", "gradient descent", "closed-form solution"]
          },
          {
            "prompt_id": "math_02",
            "question": "Explain why L1 regularization (Lasso) leads to sparse solutions while L2 (Ridge) doesn't. Use sympy to visualize the constraint regions geometrically.",
            "expected_libraries": ["sympy", "matplotlib"],
            "key_concepts": ["sparsity", "constraint optimization", "diamond vs circle"]
          },
          {
            "prompt_id": "math_03",
            "question": "Derive the optimal weight formula for Ridge regression using matrix calculus. Show the relationship between λ and the condition number of X'X.",
            "expected_libraries": ["sympy", "numpy"],
            "key_concepts": ["matrix inversion", "regularization parameter", "multicollinearity"]
          }
        ]
      },
      
      {
        "phase": 2,
        "name": "Implementation from Scratch",
        "duration": "60 minutes",
        "prompts": [
          {
            "prompt_id": "numpy_01",
            "question": "Build Ridge regression from scratch using only numpy. Implement both gradient descent and closed-form solution. Compare their results and convergence.",
            "expected_libraries": ["numpy"],
            "key_concepts": ["gradient descent", "analytical solution", "learning rate"]
          },
          {
            "prompt_id": "numpy_02",
            "question": "Implement Lasso regression from scratch using numpy and coordinate descent algorithm. Explain why standard gradient descent doesn't work well for L1.",
            "expected_libraries": ["numpy"],
            "key_concepts": ["coordinate descent", "soft thresholding", "subgradients"]
          },
          {
            "prompt_id": "numpy_03",
            "question": "Create ElasticNet from scratch. Show how the l1_ratio parameter controls the balance between L1 and L2 penalties.",
            "expected_libraries": ["numpy"],
            "key_concepts": ["mixed penalties", "feature selection vs shrinkage"]
          },
          {
            "prompt_id": "scipy_01",
            "question": "Use scipy.optimize to find optimal Ridge regression weights. Compare BFGS, L-BFGS-B, and Nelder-Mead optimizers.",
            "expected_libraries": ["scipy", "numpy"],
            "key_concepts": ["optimization algorithms", "convergence criteria"]
          }
        ]
      },
      
      {
        "phase": 3,
        "name": "Professional Implementation",
        "duration": "60 minutes",
        "prompts": [
          {
            "prompt_id": "sklearn_01",
            "question": "Using sklearn, train Ridge, Lasso, and ElasticNet models on a dataset with 50 features but only 5 are truly relevant. Compare which method best identifies the true features.",
            "expected_libraries": ["sklearn", "numpy", "pandas"],
            "key_concepts": ["feature selection", "coefficient comparison", "sparsity"]
          },
          {
            "prompt_id": "sklearn_02",
            "question": "Perform hyperparameter tuning for alpha in Ridge and Lasso using GridSearchCV and cross-validation. Visualize the regularization path.",
            "expected_libraries": ["sklearn", "matplotlib"],
            "key_concepts": ["cross-validation", "grid search", "regularization path"]
          },
          {
            "prompt_id": "sklearn_03",
            "question": "Create a pipeline with StandardScaler, PolynomialFeatures, and Ridge regression. Show how regularization prevents overfitting with high-degree polynomials.",
            "expected_libraries": ["sklearn", "matplotlib"],
            "key_concepts": ["preprocessing pipeline", "polynomial regression", "overfitting prevention"]
          },
          {
            "prompt_id": "sklearn_04",
            "question": "Compare Ridge, Lasso, and ElasticNet on a dataset with highly correlated features. Which performs best and why?",
            "expected_libraries": ["sklearn", "pandas", "seaborn"],
            "key_concepts": ["multicollinearity", "correlated features", "elastic net advantage"]
          }
        ]
      },
      
      {
        "phase": 4,
        "name": "Statistical Analysis",
        "duration": "45 minutes",
        "prompts": [
          {
            "prompt_id": "statsmodels_01",
            "question": "Use statsmodels to perform OLS regression and analyze the statistical significance of coefficients. Then apply Ridge and show how it affects p-values and confidence intervals.",
            "expected_libraries": ["statsmodels", "pandas"],
            "key_concepts": ["p-values", "confidence intervals", "statistical inference"]
          },
          {
            "prompt_id": "scipy_stats_01",
            "question": "Perform normality tests and homoscedasticity tests on residuals from Ridge regression. Use scipy.stats for Shapiro-Wilk and Breusch-Pagan tests.",
            "expected_libraries": ["scipy", "statsmodels"],
            "key_concepts": ["residual analysis", "model diagnostics", "assumptions"]
          },
          {
            "prompt_id": "statsmodels_02",
            "question": "Calculate VIF (Variance Inflation Factor) before and after Ridge regularization. Show how regularization helps with multicollinearity.",
            "expected_libraries": ["statsmodels", "pandas"],
            "key_concepts": ["VIF", "multicollinearity detection", "regularization benefits"]
          }
        ]
      },
      
      {
        "phase": 5,
        "name": "Comprehensive Visualization",
        "duration": "60 minutes",
        "prompts": [
          {
            "prompt_id": "matplotlib_01",
            "question": "Create a multi-panel visualization showing: (1) regularization paths for Ridge and Lasso, (2) coefficient magnitudes comparison, (3) train vs test performance, (4) residual plots.",
            "expected_libraries": ["matplotlib", "sklearn", "numpy"],
            "key_concepts": ["regularization paths", "model comparison", "diagnostic plots"]
          },
          {
            "prompt_id": "matplotlib_02",
            "question": "Visualize the constraint regions for Ridge (circle) and Lasso (diamond) in 2D parameter space. Show how the contours of the loss function intersect with constraints.",
            "expected_libraries": ["matplotlib", "numpy"],
            "key_concepts": ["geometric interpretation", "constraint optimization", "sparsity visualization"]
          },
          {
            "prompt_id": "seaborn_01",
            "question": "Use seaborn to create: (1) heatmap of coefficient values across different alpha values, (2) violin plots comparing coefficient distributions, (3) pairplot of performance metrics.",
            "expected_libraries": ["seaborn", "pandas", "sklearn"],
            "key_concepts": ["statistical visualization", "distribution comparison", "correlation analysis"]
          },
          {
            "prompt_id": "seaborn_02",
            "question": "Create an interactive dashboard-style visualization comparing Ridge, Lasso, and ElasticNet across multiple metrics using seaborn's FacetGrid.",
            "expected_libraries": ["seaborn", "pandas", "matplotlib"],
            "key_concepts": ["faceted plots", "multi-metric comparison", "visual storytelling"]
          }
        ]
      },
      
      {
        "phase": 6,
        "name": "Real-World Applications",
        "duration": "90 minutes",
        "prompts": [
          {
            "prompt_id": "project_01",
            "question": "Build a complete house price prediction pipeline: (1) Load dataset, (2) EDA with pandas/seaborn, (3) Feature engineering, (4) Try Linear, Ridge, Lasso, ElasticNet, (5) Compare using cross-validation, (6) Visualize results, (7) Interpret coefficients.",
            "expected_libraries": ["pandas", "sklearn", "matplotlib", "seaborn"],
            "key_concepts": ["end-to-end pipeline", "model selection", "feature importance"]
          },
          {
            "prompt_id": "project_02",
            "question": "Create a gene selection problem: Given 1000 gene expressions but only 100 samples, use Lasso to select the most predictive genes for disease classification. Visualize selected genes.",
            "expected_libraries": ["sklearn", "pandas", "seaborn", "numpy"],
            "key_concepts": ["high-dimensional data", "feature selection", "curse of dimensionality"]
          },
          {
            "prompt_id": "project_03",
            "question": "Build a time series prediction model with regularization: Use lagged features, apply Ridge/Lasso, prevent overfitting, and create forecast visualizations.",
            "expected_libraries": ["pandas", "sklearn", "matplotlib", "statsmodels"],
            "key_concepts": ["time series", "lagged features", "temporal validation"]
          }
        ]
      }
    ],
    
    "advanced_challenges": [
      {
        "challenge_id": "adv_01",
        "title": "Bayesian Interpretation of Ridge",
        "description": "Show the connection between Ridge regression and Bayesian regression with Gaussian prior. Use scipy.stats to visualize prior and posterior distributions.",
        "difficulty": "Advanced",
        "libraries": ["scipy", "numpy", "matplotlib", "sympy"]
      },
      {
        "challenge_id": "adv_02",
        "title": "Regularization Path Algorithm",
        "description": "Implement the LARS (Least Angle Regression) algorithm from scratch to compute the entire Lasso regularization path efficiently.",
        "difficulty": "Advanced",
        "libraries": ["numpy", "matplotlib"]
      },
      {
        "challenge_id": "adv_03",
        "title": "Adaptive Regularization",
        "description": "Implement adaptive Lasso where different features get different penalty weights based on OLS estimates. Compare with standard Lasso.",
        "difficulty": "Advanced",
        "libraries": ["numpy", "sklearn", "matplotlib"]
      },
      {
        "challenge_id": "adv_04",
        "title": "Group Lasso",
        "description": "Implement group Lasso for grouped feature selection (e.g., one-hot encoded categorical variables should be selected together).",
        "difficulty": "Advanced",
        "libraries": ["numpy", "scipy", "sklearn"]
      }
    ],
    
    "debugging_prompts": [
      {
        "issue": "Lasso coefficients not converging",
        "prompt": "My Lasso regression isn't converging. Show me how to diagnose this using: (1) checking data scaling, (2) increasing max_iter, (3) adjusting tolerance, (4) visualizing convergence path.",
        "libraries": ["sklearn", "matplotlib", "numpy"]
      },
      {
        "issue": "Ridge not improving over OLS",
        "prompt": "Ridge regression performs the same as OLS. Help me understand why by: (1) checking if features are already uncorrelated, (2) testing different alpha values, (3) visualizing the bias-variance tradeoff.",
        "libraries": ["sklearn", "matplotlib", "seaborn"]
      },
      {
        "issue": "ElasticNet parameter tuning",
        "prompt": "How do I choose optimal alpha and l1_ratio for ElasticNet? Show me a 2D grid search with visualization of performance across both parameters.",
        "libraries": ["sklearn", "matplotlib", "numpy"]
      }
    ],
    
    "quiz_prompts": [
      {
        "question": "Generate a synthetic dataset where Lasso outperforms Ridge. Explain why using visualizations.",
        "answer_should_include": ["sparse true weights", "many irrelevant features", "coefficient comparison plot"]
      },
      {
        "question": "Generate a synthetic dataset where Ridge outperforms Lasso. Explain why using visualizations.",
        "answer_should_include": ["all features relevant", "correlated features", "smooth coefficient shrinkage"]
      },
      {
        "question": "Create a scenario where ElasticNet is necessary (both Ridge and Lasso fail individually).",
        "answer_should_include": ["high correlation", "many features", "group selection"]
      }
    ],
    
    "daily_practice_routine": {
      "day_1": {
        "morning": "Phase 1 - Mathematical Foundation (sympy)",
        "afternoon": "Phase 2 - Numpy Implementation",
        "evening": "Review and visualize concepts with matplotlib"
      },
      "day_2": {
        "morning": "Phase 3 - sklearn Professional Implementation",
        "afternoon": "Phase 4 - Statistical Analysis",
        "evening": "Phase 5 - Comprehensive Visualization"
      },
      "day_3": {
        "morning": "Phase 6 - Real-World Project 1",
        "afternoon": "Phase 6 - Real-World Project 2",
        "evening": "Advanced Challenges"
      }
    },
    
    "resource_links": {
      "documentation": {
        "sklearn_ridge": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
        "sklearn_lasso": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
        "sklearn_elasticnet": "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
        "statsmodels": "https://www.statsmodels.org/stable/regression.html"
      },
      "theoretical": {
        "elements_of_statistical_learning": "Chapter 3.4 - Shrinkage Methods",
        "islr": "Chapter 6 - Linear Model Selection and Regularization"
      }
    },
    
    "tips_for_better_learning": [
      "Always visualize regularization paths to understand how coefficients change",
      "Use cross-validation, never test set, for hyperparameter tuning",
      "Start with simple 2D examples before moving to high dimensions",
      "Compare all three methods (Ridge, Lasso, ElasticNet) on every dataset",
      "Pay attention to feature scaling - regularization is scale-dependent",
      "Understand geometric interpretation - it builds intuition",
      "Practice implementing from scratch before using sklearn",
      "Always check residual plots to validate model assumptions"
    ],
    
    "common_mistakes_to_avoid": [
      "Forgetting to standardize features before regularization",
      "Using test set for alpha selection (use cross-validation)",
      "Comparing models on training accuracy instead of test/CV",
      "Setting alpha too high and getting useless models",
      "Not checking if Lasso converged (max_iter too small)",
      "Ignoring multicollinearity when choosing between Ridge and Lasso",
      "Applying regularization to intercept term",
      "Not understanding that Lasso can only select min(n_samples, n_features) variables"
    ]
  }
}