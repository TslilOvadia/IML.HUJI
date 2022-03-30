from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samps = np.random.normal(10, 1, 1000)
    univarGaussian = UnivariateGaussian().fit(samps)
    print("mu", univarGaussian.mu_)

    # Question 2 - Empirically showing sample mean is consistent
    result = []
    for n_samples in range(10,1000,10):
        ith_gaussian = UnivariateGaussian().fit(samps[0:n_samples])
        result.append(ith_gaussian.mu_)

    go.Figure([go.Scatter(x=np.arange(10,1000,10), y=np.abs(10-np.array(result)), mode='markers+lines', name=r'$\widehat\mu$'),
               go.Scatter(x=np.arange(10,1000,10), y=[np.abs(10-np.array(result))] * len(np.arange(10,1000,10)), mode='lines', name=r'$\mu$')],
              layout=go.Layout(title=r"$\text{(3.1.2) Estimation of Expectation As Function Of Number Of Samples: f(x) = |µ-µ_est|} $",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    univarGaussian = UnivariateGaussian().fit(samps)
    pdf = univarGaussian.pdf(samps)
    plt.scatter(samps, pdf, s=0.1)
    plt.title("3.1.3: empirical PDF function under the fitted model")
    plt.xlabel("Value of x_ith drawn from Normal distributed RV with mean=10, var=1")
    plt.ylabel("The PDF value ")
    plt.show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    N = 200
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, .2, 0, .5],
                    [.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, 1000)

    mvn = MultivariateGaussian().fit(X)

    space = np.linspace(-10, 10, N)
    print(mvn.mu_)
    print(mvn.cov_)
    print("mu", mvn.mu_)
    print("cov", mvn.cov_)
    print("var", mvn.cov_[0, 0])

    # Question 5 - Likelihood evaluation
    ax = np.array([i * 0.1 for i in range(-N//2, N//2)])
    result = np.zeros((N, N))
    m_val = -np.infty
    row_i, col_j = 0, 0
    for i in range(N):
        for j in range(N):

            f1 = space[i]
            f3 = space[j]
            mu = np.array([f1, 0, f3, 0])
            val = MultivariateGaussian.log_likelihood(mu, cov, X)
            result[i, j] = val
            if m_val < val:
                m_val = val
                row_i, col_j = i, j
    df = pd.DataFrame(result, index=ax, columns=ax)
    seaborn.heatmap(df, xticklabels=10, yticklabels=10)
    plt.title("3.2.6: Likelihood(f1,f3) Values ")
    plt.xlabel("f3 values [mean]")
    plt.ylabel("f1 values [mean]")
    plt.show()

    # Question 6 - Maximum likelihood
    print(f"row_i {row_i}, col_j {col_j}")
    print(space[row_i],space[col_j],m_val)


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
