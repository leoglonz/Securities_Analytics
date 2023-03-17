"""
Will hold generalized stock analysis functions akin to those 'Stock_Analysis.ipynb'.

IN PROGRESS
Last Revised: 15 Mar 2023
"""

from sklearn.metrics import accuracy_score


def model_performance(model, X=X_test, y=y_test):
    """
    Get accuracy score on validation/test data from a trained model
    """
    y_pred = model.predict(X)
    return round(accuracy_score(y_pred, y),3)