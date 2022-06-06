import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
from sklearn.ensemble import AdaBoostClassifier
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=100, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(wl=lambda: DecisionStump(), iterations=n_learners)
    adaboost._fit(train_X,train_y)

    preds = []
    test_loss,train_loss = [],[]
    for num_learners in range(n_learners):
        preds.append(adaboost.partial_predict(test_X,num_learners+1))
        test_loss.append(adaboost.partial_loss(test_X, test_y,num_learners+1))
        train_loss.append(adaboost.partial_loss(train_X, train_y,num_learners+1))
    test_scatter = go.Scatter(x=np.arange(len(test_loss)),y=test_loss, mode='markers+lines', name='Test Set')
    train_scatter = go.Scatter(x=np.arange(len(test_loss)), y=train_loss, mode='markers+lines', name='Train Set')
    fig = go.Figure([test_scatter,train_scatter],
                    layout=go.Layout(title=r"$\text{Loss as function of weak learners used in AdaBoost model} $",
                                     xaxis_title="$\\text{Number of weak learners}$",
                                     yaxis_title="$\\text{loss over dataset}$",
                                     height=400
                                     ))
    fig.show()
    all_model_info = [(num_learners, accuracy(test_y, adaboost.partial_predict(test_X, num_learners)), adaboost.partial_loss(test_X, test_y, num_learners))
                  for num_learners in range(250)]
    minloss = np.inf
    size_l = np.inf
    best_acc = np.inf
    for num,acc,loss in all_model_info:
        if loss < minloss:
            minloss = loss
            size_l = num
            best_acc = acc


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    model_info = [(t,accuracy(test_y, adaboost.partial_predict(test_X,t)),adaboost.partial_loss(test_X,test_y,t)) for t in T]
    model_names = [f'Using {t} learners with accuracy of {acc} and loss {loss}' for t,acc,loss in model_info]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in model_names],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    partial_models = [lambda X: adaboost.partial_predict(X,5),lambda X: adaboost.partial_predict(X,50),lambda X: adaboost.partial_predict(X,100),lambda X: adaboost.partial_predict(X,250)]

    for i in range(len(T)):
        fig.add_traces([decision_surface(partial_models[i], lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1))
                                   )],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models -  Dataset}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()
    # Question 3: Decision surface of best performing ensemble
    fig = make_subplots(rows=1, cols=1, subplot_titles=[rf"$\textbf{{{f'Using {size_l} learners with accuracy of {best_acc} and loss {minloss}'}}}$"
],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, size_l), lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y,
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1))
                               )])

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries for best ensemble}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 4: Decision surface with weighted samples
    sample_weight = 10*adaboost.D_ / np.max(adaboost.D_)
    fig = make_subplots(rows=1, cols=1, subplot_titles=[rf"$\textbf{{{f'Using {250} learners, with point size proportional to the sample weight'}}}$"],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    fig.add_traces([decision_surface(lambda X: adaboost.partial_predict(X, 250), lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=train_y,
                                           colorscale=[custom[0], custom[-1]],size=sample_weight,
                                           line=dict(color="black", width=1))
                               )])

    fig.update_layout(title=rf"$\textbf{{Decision Boundaries for best ensemble}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    noiseFactors = [0, 0.4]
    for noise_factor in noiseFactors:
        fit_and_evaluate_adaboost(noise_factor)
