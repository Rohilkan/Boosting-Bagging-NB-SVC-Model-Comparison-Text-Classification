# Ensemble-NB-SVM-Model-Comparison-Text-Classification

Goal:
    Comparing **effectiveness** of **boosting/bagging ensemble model implementations** using two metrics (speed and accuracy score) and comparing discovered best model against **multinomial naive bayes** and **support vector classifier** models - algorithms more conventionally fit for text classification.

Libraries:
- Scikit-Learn
- XGBoost
- CatBoost

Models:
- Scikit-learn:
    - `Decision Tree Classifier` (Baseline)
    - `Random Forest Classifier`
    - `Naive Bayes Classifier`
    - `AdaBoost Classifier`
    - `Gradient Boosting Classifier`
    - `Histogram Gradient Boosting`
- `XGBoost Classifier`
- `CatBoost Classifier`

Data Source:
    - https://ai.stanford.edu/~amaas/data/sentiment/

- Review data read in as `.txt` segmented by **positive** (< 5) and **negative** (> 5) ratings and mapped to ratings nested in filenames
- Text reviews **TFIDF vectorized** yielding sparse matrix **bag of words**
- Data cleaned and split into **train**, **test**, and **validation sets**
- Each model implementation built; hyperparameters set to ensure performance **fairness/consistency**
- Models fit on training sets (with optional validation)
- **Accuracy** and **runtime** reported for each model on test set performance
