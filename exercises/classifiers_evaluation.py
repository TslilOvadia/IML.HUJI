import matplotlib.pyplot as plt

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi
import os
# print(os.path.abspath("."))


SYMBOLS = np.array(["circle", "square","star"])

def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)



def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """

    for name, filename in [("Linearly Separable", "/Users/tzlilovadia/IML.HUJI/datasets/linearly_separable.npy"), ("Linearly Inseparable", "/Users/tzlilovadia/IML.HUJI/datasets/linearly_inseparable.npy")]:
        # Load dataset
        X,y = load_dataset(filename)
        # Fit Perceptron and record loss in each fit iteration
        losses = []
        def loss_callback(prcpt: Perceptron, X_i: np.ndarray, y_i: int) -> None:
            losses.append(prcpt._loss(X,y))
            # if prcpt._loss(X,y) == float('nan'):
            # print(prcpt._loss(X,y))
        perceptron = Perceptron(callback=loss_callback)
        perceptron._fit(X,y)
        symbols = np.array(["circle", "square",'star'])
        # Plot figure of loss as function of fitting iteration
        go.Figure(data=[go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=y, symbol=symbols[y], line=dict(color="black", width=1),
                                               colorscale=[custom[0], custom[-1]]))],
                  layout=go.Layout(title=rf"$\textbf{{(1) {name} Dataset}}$")).show()
        plt.plot(losses)
        plt.title(f"Loss Function for the {name} case")
        plt.ylabel("loss (normalized)")
        plt.xlabel("number of iterations")
        plt.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys,showlegend=False, mode="lines", marker_color="black")


def _get_title(dataset_name: str, model_type: str):
    return f"Classification on dataset {dataset_name} using {model_type}"

def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset

        X,y = load_dataset(f)

        # Fit models and predict over training set
        lda_model = LDA()
        gnb_model = GaussianNaiveBayes()

        lda_model._fit(X,y)
        gnb_model._fit(X,y)

        from IMLearn.metrics import accuracy
        lda_prediction = lda_model._predict(X)
        lda_loss = lda_model._loss(X, y)
        lda_acc = accuracy(y, lda_prediction)
        lda_title = f'LDA - Using dataset {f}. Performance: Accuracy = {lda_acc:.4f}, Loss = {lda_loss:.4f}'

        gnb_prediction = gnb_model._predict(X),
        gnb_loss = gnb_model._loss(X,y)
        gnb_acc = accuracy(y, gnb_prediction)
        gnb_title = f'GNB - Using dataset {f}. Performance: Accuracy = {gnb_acc:.4f}, Loss = {gnb_loss:.4f}'
        MODELS_NAME = [lda_title, gnb_title]

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in MODELS_NAME],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        models = [lda_model, gnb_model]
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])


        fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models -  Dataset}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)

        # Add traces for data-points setting symbols and colors
        for i, model in enumerate(models):
            fig.add_traces([decision_surface(model.fit(X, y).predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=True,
                                       marker=dict(color=y, symbol=SYMBOLS[y], colorscale=[custom[0], custom[-1]],
                                                   line=dict(color="black", width=1)))],
                           rows=(i // 3) + 1, cols=(i % 3) + 1)

            # Add `X` dots specifying fitted Gaussians' means
            # Add ellipses depicting the covariances of the fitted Gaussians
            for k in range(len(model.classes_)):
                mean = np.array([[model.mu_[k][0]],[model.mu_[k][1]]])
                if np.array(model.cov_).shape[0] != 2 or np.array(model.cov_).shape[1] != 2:
                    vars = np.array(model.cov_[k])
                else:
                    vars = np.array(model.cov_)
                ellipse = get_ellipse(mean, vars)
                fig.add_traces(ellipse,
                               rows=(i // 3) + 1, cols=(i % 3) + 1)
                fig.add_traces(
                    go.Marker(x=[model.mu_[k][0]],y=[model.mu_[k][1]], mode="markers", showlegend=False,
                                       marker=dict(color='black', symbol="x", size=10)

                ),rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.show()
if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    # X1=np.array([0,1,2,3,4,5,6,7])
    # y1=np.array([0,0,1,1,1,1,2,2])
    #
    # gnb1 = GaussianNaiveBayes().fit(X1,y1)
    # # print(f"mu mle is:{gnb1.mu_} and the var is {gnb1.vars_}")
    # X2 = [[1,1],[1,2],[2,3],[2,4],[3,3],[3,4]]
    # y2 = [0,0,1,1,1,1]
    # gnb2 = GaussianNaiveBayes().fit(X2,y2)
    # print(f"mu mle is:{gnb2.mu_} and the var1,0 is {gnb2.vars_[1][0]} and vars1,1 is {gnb2.vars_[1][1]} ")
num_list = [1, 2, 3, 4, 5]
num_list.remove(2)
print(num_list)
