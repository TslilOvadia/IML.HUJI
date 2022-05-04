from typing import NoReturn

import pandas as pd

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
        self.a_k, self.b_k = [],[]

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Assuming that all labels possible are in y, we'll take the set of y as the classes space:
        self.classes_,count_per_class = np.unique(y, return_counts=True)
        self.mu_ = []
        self.cov_ = np.zeros((X.shape[1],X.shape[1]))
        # Group the dataset by the different classes and calculate the mean of every class with respect to it's features
        dataset = pd.DataFrame(X)
        dataset['labels'] = y

        for y_i in self.classes_:
            df_per_class = dataset.loc[dataset['labels'] == y_i].drop(columns=['labels'])
            mu_est = np.mean(df_per_class.values,axis=0)
            self.mu_.append(mu_est)   # According to the MLE
            # for x_i in df_per_class:
            self.cov_ += np.array(df_per_class.values- mu_est).T @ np.array(df_per_class.values-mu_est)

        self.pi_ = np.asarray(count_per_class/np.shape(y)[0])
        self.cov_ = self.cov_/len(y)
        self._cov_inv = inv(self.cov_)
        # Calc the probability vector π using the formula derived from the MLE calculation:
        self.__calc_pi_vector(y)
        self.fitted_ = True

    def __calc_pi_vector(self, y: np.ndarray) -> NoReturn:
        N = len(y)  # get the total count of all samples from all the classes
        class_dict = {}
        for y_i in y:
            if y_i in class_dict.keys():
                class_dict[y_i] += 1
                continue
            else:
                class_dict[y_i] = 1
        self.pi_ = [class_dict[y_i] / N for y_i in self.classes_]

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
        """
        likelihoods = self.likelihood(X).T
        return np.argmax(likelihoods, axis=1)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        aks, bks = np.zeros((len(self.classes_),X.shape[1])),np.zeros((len(self.classes_), 1))

        for k in range(len(self.classes_)):
            a_k = self._cov_inv @ self.mu_[k].T
            b_k = np.log(self.pi_[k]) - 0.5 * self.mu_[k].T @ self._cov_inv @ self.mu_[k]
            aks[k], bks[k] = a_k,b_k
        likelihoods = aks @ X.T + bks

        return likelihoods

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
        from ...metrics import misclassification_error
        return misclassification_error(y_true=y, y_pred=self._predict(X), normalize=True)
