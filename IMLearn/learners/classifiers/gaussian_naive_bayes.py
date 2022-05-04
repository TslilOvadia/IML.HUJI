from typing import NoReturn

import pandas as pd

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None
        self.cov_ = None
        self.count_per_class__ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # Assuming that all labels possible are in y, we'll take the set of y as the classes space:
        self.classes_, self.count_per_class__ = np.unique(y, return_counts=True)
        self.mu_ = []
        self.vars_ = []
        # Group the dataset by the different classes and calculate the mean of every class with respect to it's features
        dataset = pd.DataFrame(X)
        dataset['labels'] = y

        for y_i in self.classes_:
            df_per_class = dataset.loc[dataset['labels'] == y_i].drop(columns=['labels'])
            mu_est = np.mean(df_per_class.values, axis=0)
            self.mu_.append(mu_est)  # According to the MLE
            # var = np.array(df_per_class.values - mu_est).T @ np.array(df_per_class.values - mu_est)
            var = np.std(df_per_class.values, axis=0)**2
            self.vars_.append(var)

        self.cov_ = [np.diag(self.vars_[k]) for k in range(len(self.classes_))]
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
        likelihoods = self.likelihood(X)
        return np.argmax(likelihoods, axis=0)

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
        aks, bks, cks = np.zeros((len(self.classes_),X.shape[1])),np.zeros((len(self.classes_), 1)), np.zeros((len(self.classes_),X.shape[0]))

        for k in range(len(self.classes_)):
            inv_cov_k = inv(self.cov_[k])
            a_k = inv_cov_k @ self.mu_[k]

            z_k = det(self.cov_[k]) ** 0.5 * np.power((2 * np.pi) ** 0.5, 1)     # Calc the gaussian normalization factor
            b_k = np.log(self.pi_[k]) - np.log(z_k) - 0.5 * self.mu_[k] @ inv_cov_k @ self.mu_[k]
            aks[k], bks[k] = a_k,b_k

            # Calc for the X.T * ∑ * X term
            x_sig_xt = np.zeros((X.shape[0], 1))
            for i, x_i in enumerate(X):
                x_sig_xt[i,:] = x_i[:,None].T @ inv(np.array(self.cov_[k])) @ x_i[:, None]
            cks[k,:] = x_sig_xt[:,0]

        ck = np.array(cks)
        likelihoods = aks @ X.T + bks -0.5 * ck
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
