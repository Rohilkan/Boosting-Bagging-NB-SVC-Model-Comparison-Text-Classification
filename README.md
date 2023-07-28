# Boosting-Bagging-SVC-Model-Comparison-Text-Classification

Comparing boosting and bagging methods (Adaboost, Gradient Boosting, Random Forest) using various libraries (Scikit-Learn, XGBoost, LightGBM, CatBoost) with support vector machines for text classification on movie review data.

Goal:
    Compare effectiveness of boosting/bagging ensemble model implementations using two metrics (speed and accuracy score) and compare discovered best model against Scikit-Learn's support vector classifier, model more conventionally fit for text classification.

Models:
Scikit-learn:
    - `Decision Tree Classifier` (Baseline)
    - `Random Forest Classifier`
    - `Naive Bayes Classifier`
    - `AdaBoost Classifier`
    - `Gradient Boosting Classifier`
    - `Histogram Gradient Boosting`
`XGBoost Classifier`
`CatBoost Classifier`

- Review data read in as `.txt` segmented by positive (< 5) and negative (> 5) ratings and mapped to ratings nested in filenames
- Text reviews TFIDF vectorizered yielding sparse matrix bag of words
- Data cleaned and split into train, test, and validation sets
- Each model implementation built with set hyperparamaters
- Models fit on training sets (with optional validation)
- Accuracy and runtime reported for each model on test set performance