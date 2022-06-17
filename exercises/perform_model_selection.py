from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots



def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    f_x = lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    X = np.random.uniform(low=-1.2, high=2,size=n_samples)
    eps = np.random.normal(loc=0, scale=noise, size=n_samples)
    f_X = np.apply_along_axis(f_x, axis=0, arr=X)
    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(f_X + eps), train_proportion=2/3)
    # plot a scatter for the X dataset and train_X dataset and train_y dataset
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=f_X, mode='markers', name='Noiseless f(x)'))
    fig.add_trace(go.Scatter(x=X_train[0], y=y_train, mode='markers', name='Training data'))
    fig.add_trace(go.Scatter(x=X_test[0], y=y_test, mode='markers', name='test_X'))
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    best_validation_score = np.inf
    best_k = 0
    validation_scores=[]
    train_scores=[]
    for k in range(11):
        train_score, validation_score = cross_validate(PolynomialFitting(k=k), X_train.values[:,0], y_train.values, mean_square_error)
        train_scores.append(train_score)
        validation_scores.append(validation_score)
        if best_validation_score > validation_score:
            best_validation_score = validation_score
            best_train_score = train_score
            best_k = k
    #plot the validation and train scores for each degree
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0,11), y=validation_scores, mode='lines', name='Validation Scores'))
    fig.add_trace(go.Scatter(x=np.arange(0,11), y=train_scores, mode='lines', name='Train Scores'))
    fig.show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    poly = PolynomialFitting(k=best_k).fit(X_train.values[:,0], y_train.values)
    y_pred = poly.predict(X_test.values[:,0])
    test_err = mean_square_error(y_test.values, y_pred)
    print(f"Best value for k, aka k* is {best_k:.2f}. this value achieved test error = {test_err: .2f} while \n"
          f"the best value got from cross_validate is {best_validation_score: .2f} for validation score")




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target
    X_train, y_train, X_test, y_test = pd.DataFrame(X[:n_samples,:]), pd.Series(y[:n_samples]),\
                                       pd.DataFrame(X[n_samples:,:]),pd.Series(y[n_samples:])

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # and report the best validation error and best regularization parameter value
    r_best_validation_score = np.inf
    l_best_validation_score = np.inf
    r_best_reg_param = 0
    l_best_reg_param = 0
    ridge_validation_scores=[]
    ridge_train_scores=[]
    lasso_validation_scores=[]
    lasso_train_scores=[]
    for reg_param in np.linspace(0, 2, n_evaluations):
        ridge_train_score, ridge_validation_score = cross_validate(RidgeRegression(lam=reg_param), X_train.values, y_train.values, mean_square_error)
        lasso_train_score, lasso_validation_score = cross_validate(Lasso(alpha=reg_param), X_train.values, y_train.values, mean_square_error)

        ridge_train_scores.append(ridge_train_score)
        ridge_validation_scores.append(ridge_validation_score)
        lasso_train_scores.append(lasso_train_score)
        lasso_validation_scores.append(lasso_validation_score)
        if r_best_validation_score > ridge_validation_score:
            r_best_validation_score = ridge_validation_score
            r_best_reg_param = reg_param
        if l_best_validation_score > lasso_validation_score:
            l_best_validation_score = lasso_validation_score
            l_best_reg_param = reg_param

    #plot the validation and train scores for each regularization parameter

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.linspace(0.001, 2, n_evaluations), y=ridge_validation_scores, mode='lines', name='Ridge Validation Scores'))
    fig.add_trace(go.Scatter(x=np.linspace(0.001, 2, n_evaluations), y=ridge_train_scores, mode='lines', name='Ridge Train Scores'))
    fig.add_trace(go.Scatter(x=np.linspace(0.001, 2, n_evaluations), y=lasso_validation_scores, mode='lines', name='Lasso Validation Scores'))
    fig.add_trace(go.Scatter(x=np.linspace(0.001, 2, n_evaluations), y=lasso_train_scores, mode='lines', name='Lasso Train Scores'))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    # for the best regularization parameter value
    ridge = RidgeRegression(lam=r_best_reg_param).fit(X_train.values, y_train.values)
    lasso = Lasso(alpha=l_best_reg_param)
    lasso.fit(X_train.values, y_train.values)
    lsq = LinearRegression().fit(X_train.values, y_train.values)
    y_pred_ridge = ridge.predict(X_test.values)
    y_pred_lasso = lasso.predict(X_test.values)
    y_pred_lsq = lsq.predict(X_test.values)
    test_err_ridge = mean_square_error(y_test.values, y_pred_ridge)
    test_err_lasso = mean_square_error(y_test.values, y_pred_lasso)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X[:,3], y=y, mode='markers', name='Training data'))
    # fig.add_trace(go.Scatter(x=X_test.values, y=y_test.values, mode='markers', name='Training data'))
    fig.show()

    test_err_lsq = mean_square_error(y_test.values, y_pred_lsq)
    print(f"Best value for reg_param is {r_best_reg_param: .2f}. this value achieved test error = {test_err_ridge:.2f} while \n"
            f"the best value got from cross_validate is {r_best_validation_score:.2f} for validation score")
    print(f"Best value for reg_param is {l_best_reg_param: .2f}. this value achieved test error = {test_err_lasso:.2f} while \n"
            f"the best value got from cross_validate is {l_best_validation_score:.2f} for validation score")
    print(f"LSQ value achieved test error = {test_err_lsq:.2f} while \n")


if __name__ == '__main__':
    np.random.seed(0)
    Q123 = select_polynomial_degree(n_samples=100, noise=5)
    Q4 = select_polynomial_degree(n_samples=100, noise=0)
    Q5 = select_polynomial_degree(n_samples=1500, noise=10)
    Q678 = select_regularization_parameter(n_samples=50, n_evaluations=500)

