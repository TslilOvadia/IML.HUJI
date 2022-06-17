from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    n_samples = X.shape[0]
    n_folds = n_samples // cv
    train_score = 0
    validation_score = 0
    for i in range(cv):
        X_train_hypothetic = np.concatenate((X[:i*n_folds], X[(i + 1)*n_folds:]))
        y_train_hypothetic = np.concatenate((y[:i * n_folds], y[(i + 1) * n_folds:]))
        X_validation_hypothetic = X[i*n_folds:(i+1)*n_folds]
        y_validation_hypothetic = y[i*n_folds:(i+1)*n_folds]
        estimator.fit(X_train_hypothetic, y_train_hypothetic)
        train_score += scoring(y_train_hypothetic, estimator.predict(X_train_hypothetic))
        validation_score += scoring(y_validation_hypothetic, estimator.predict(X_validation_hypothetic))
    train_score /= cv
    validation_score /= cv
    return train_score, validation_score



