from __future__ import annotations
from typing import Tuple, NoReturn

import matplotlib.pyplot as plt
import pandas as pd

from ...base import BaseEstimator
import numpy as np
from numba import njit,jit,vectorize
from itertools import product

from ...metrics import misclassification_error


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """
    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None


    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # For each feature we check for the best score among all feature based stumps
        minimal_err_feature = np.inf
        # Check on every feature:
        for feature in range(X.shape[1]):
            values = X[:,feature]
            # Check on each sign
            for sign in [1, -1]:
                # Get the best threshold for optimal split
                thr, thr_err = self._find_threshold(values=values, labels=y, sign=sign)
                # print(f'using sign {sign} and the {feature}\'th features min err is {minimal_err_feature} and current err is {thr_err}ֿ\n')
                if thr_err < minimal_err_feature:
                    minimal_err_feature = thr_err
                    self.threshold_ = thr
                    self.j_ = feature
                    self.sign_ = sign

        # print(f'sign is {self.sign_} and best err is {minimal_err_feature} using feature number {self.j_}')

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        # Get chosen feature:
        pred = np.copy(X[:, self.j_])
        # Thresholding:
        negative_threshold_indices = np.argwhere(pred < self.threshold_)
        positive_threshold_indices = np.argwhere(pred >= self.threshold_)
        # Assign predictions accordingly:
        pred[negative_threshold_indices] = -self.sign_
        pred[positive_threshold_indices] = self.sign_
        return pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        thr, thr_err,N = 0.0 ,np.inf, labels.shape[0]
        feature_is_binary = len(np.unique(values)) == 2
        val_set = np.unique(values)
        sort_idx = np.argsort(values)
        labels = labels[sort_idx]
        values = values[sort_idx]

        for val_i in range(len(val_set)):
            cur_values = np.copy(values)
            if val_i < len(val_set)-1:
                # Find the indices of the values higher and lower than current value:
                negative_threshold_indices = cur_values < val_set[val_i] + 0.5*(val_set[val_i+1]-val_set[val_i])
                positive_threshold_indices = cur_values >= val_set[val_i] + 0.5*(val_set[val_i+1]-val_set[val_i])
                # Assign the corresponding sign to the corresponding indices:
                cur_values[positive_threshold_indices] = sign
                cur_values[negative_threshold_indices] = -sign
            # Count the errors for the current split:
            signed_labels = np.sign(labels)
            wrong_threshold_indices = np.argwhere(cur_values != signed_labels)
            err = np.sum(np.abs(labels[wrong_threshold_indices]))
            # Update the variables if there is any improvements:
            if err < thr_err:
                thr_err = err
                if val_i < len(val_set) - 1 and not feature_is_binary :
                    thr = val_set[val_i] + 0.5*(val_set[val_i+1]-val_set[val_i])
                else:
                    thr = val_set[val_i]

        return thr,thr_err

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y_true=y, y_pred=self._predict(X))
