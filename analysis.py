"""
Will hold generalized stock analysis functions akin to those 'Stock_Analysis.ipynb'.

IN PROGRESS
Last Revised: 15 Mar 2023
"""

import numpy as np
from sklearn.metrics import accuracy_score


from base import *


def beta(df, market=None):
    """
    Calculating betas using prices from every business day.
    If the market values are not passed, I'll assume they 
    are located in a column named 'Market'.  If not, 
    this will fail.
    """
    if market is None:
        market = df['MarketClose']
        df = df.drop('MarketClose', axis=1)
    X = market.values.reshape(-1, 1)
    X = np.concatenate([np.ones_like(X), X], axis=1)
    b = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(df.values)
    
    return float(b[1])


def addBeta(stock, market, window=252, verbose=False):
    """
    Calc beta across given df of closing prices.
    Window default is stet to calculate yearly beta.
    """
    betas = np.array([])
    data = pd.concat([stock.AdjClose, market.MarketClose], axis=1)

    for  i, sdf in enumerate(roll(data.pct_change().dropna(), window )):
        betas = np.append(betas, beta(sdf))

    full_data = data.drop(index=data.index[:window], axis=0, inplace=False)
    full_data['Beta'] = betas.tolist()

    return full_data


def get_score():
    """
    Return the average MAE over 3 CV folds of random forest model.

    n_estimators: the number of trees in the forest
    """
    
    return


def model_performance(model, X, y):
    """
    Get accuracy score on validation/test data from a trained model
    """
    y_pred = model.predict(X)
    return round(accuracy_score(y_pred, y),3)
