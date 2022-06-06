import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics import misclassification_error, accuracy


class AdaBoost(BaseEstimator):
    """
    AdaBoost class for boosting a specified weak learner

    Attributes
    ----------
    self.wl_: Callable[[], BaseEstimator]
        Callable for obtaining an instance of type BaseEstimator

    self.iterations_: int
        Number of boosting iterations to perform

    self.models_: List[BaseEstimator]
        List of fitted estimators, fitted along the boosting iterations
    """

    def __init__(self, wl: Callable[[], BaseEstimator], iterations: int):
        """
        Instantiate an AdaBoost class over the specified base estimator

        Parameters
        ----------
        wl: Callable[[], BaseEstimator]
            Callable for obtaining an instance of type BaseEstimator

        iterations: int
            Number of boosting iterations to perform
        """
        super().__init__()
        self.wl_ = wl
        self.iterations_ = iterations
        self.models_, self.weights_, self.D_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an AdaBoost classifier over given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        N = y.shape[0]
        self.__init_vars(N)
        for step in range(self.iterations_):
            dstump = self.wl_()
            dstump._fit(X, y*self.D_)
            self.models_.append(dstump)
            pred = dstump._predict(X)
            stump_failures_indices = np.argwhere(pred != y)
            epsilon_t = np.sum(self.D_[stump_failures_indices])

            # w_t should represent the amount of say each stump has
            w_t = np.log((1-epsilon_t)/epsilon_t)/2
            # Update the sample weights accordingly:
            self.D_ *= np.exp(-y*w_t*pred)
            self.__norm_weights()
            # Save model weight:
            self.weights_.append(w_t)


            # print(f'model.f {dstump.threshold_} on step {step}')
            # stump_failures_indices, stump_success_indices = np.argwhere(dstump._predict(X) != y)[:,0], np.argwhere(dstump._predict(X) == y)[:,0]

    def __init_vars(self, N):
        self.models_ = []
        self.weights_ = []
        self.D_ = np.array(np.ones((N,)),dtype=np.float64)*(1/N)

    def __norm_weights(self):
        """
        Should normalize the weights s.t it summed up to 1.
        """
        self.D_ /= np.sum(self.D_)

    def _predict(self, X):
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
        h_s = np.zeros((X.shape[0],))
        for model,weight in zip(self.models_,self.weights_):
            h_s += model._predict(X)*weight
        return np.sign(h_s)

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
        return misclassification_error(y, self._predict(X))

    def partial_predict(self, X: np.ndarray, T: int) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimators

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        h_s = np.zeros((X.shape[0],))
        for model,weight in zip(self.models_,self.weights_):
            h_s += model._predict(X)*weight
            T -= 1
            if T == 0:
                return np.sign(h_s)
        return np.sign(h_s)

    def partial_loss(self, X: np.ndarray, y: np.ndarray, T: int) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        T: int
            The number of classifiers (from 1,...,T) to be used for prediction

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(y, self.partial_predict(X, T))
