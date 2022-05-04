import numpy as np
from ..base import BaseEstimator
from typing import Callable, NoReturn
from ..metrics import misclassification_error


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
        self.models_ = []
        self.__init_weights(N)
        for step in range(self.iterations_):
            dstump = self.wl_()
            dstump.fit(X,y*self.weights_)
            self.models_.append(dstump)
            epsilon_t = self.weights_ * dstump._loss(X,y)
            w_t = np.log((1-epsilon_t)/epsilon_t)/2
            self.__update_weights(w_t, dstump, X, y)
            self.__norm_weights()

    def __init_weights(self, N):
        print('Try')
        self.weights_ = 1 / N * np.ones((1, N))

    def __norm_weights(self):
        """
        Should normalize the weights s.t it summed up to 1.
        """
        self.weights_ /= np.sum(self.weights_)

    def __update_weights(self,w_t: float, dstump: BaseEstimator, X: np.ndarray, y: np.ndarray):
        """
        Calculate the next value for the weights
        """
        exp_factor = np.exp(-y * w_t * dstump._predict(X))
        self.weights_ = np.mult(self.weights_,exp_factor)

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
        for model in self.models_:
            h_s += model._predict(X)
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
        for i in range(T):
            h_s += self.models_[i]._predict(X)
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
