from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
from typing import NoReturn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
pio.templates.default = "simple_white"
HOUSE_PRICES_PATH = '/Users/tzlilovadia/IML.HUJI/datasets/house_prices.csv'

#########################################
############ HELPER FUNCTIONS ###########
#########################################

def find_missing_percent(df):
    """
    Returns dataframe containing the total missing values and percentage of total
    missing values of a column.
    """
    miss_df = pd.DataFrame({'ColumnName':[],'TotalMissingVals':[],'PercentMissing':[]})
    for col in df.columns:
        sum_miss_val = df[col].isnull().sum()
        percent_miss_val = round((sum_miss_val/df.shape[0])*100,2)
        miss_df = miss_df.append(dict(zip(miss_df.columns,[col,sum_miss_val,percent_miss_val])),ignore_index=True)
    return miss_df

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if filename == None or len(filename) == 0:
        raise FileNotFoundError()
    df = pd.read_csv(filename)

    #### Data Cleaning Stage ####
    # Remove unnecessary samples from the dataset:
    df.drop(columns=['long','lat'], inplace=True)

    # Remove Abnorml samples from the dataset:

    # Remove nan-valued samples from the dataset:
    df.fillna("unknown", inplace=True)
    df = df.loc[df['price'] != 'unknown']
    df = df.loc[df['date'] != 'unknown']
    df = df.loc[df['yr_built'] != 'unknown']
    df = df.loc[df['zipcode'] != 'unknown']
    df = df.loc[df['view'] != 'unknown']
    df = df.loc[df['condition'] != 'unknown']
    df = df.loc[df['sqft_above'] != 'unknown']
    df = df.loc[df['sqft_lot15'] != 'unknown']
    df = df.loc[df['sqft_basement'] != 'unknown']
    df = df.loc[df['sqft_living15'] != 'unknown']
    df = df.loc[df['grade'] != 'unknown']


    # Validate data types:
    df['id'] = df['id'].astype(str)
    df['date'] = df['date'].astype(str)
    df['price'] = df['price'].astype(np.float64)
    df['view'] = df['view'].astype(int)
    df['condition'] = df['condition'].astype(int)
    df['grade'] = df['grade'].astype(int)
    df['sqft_above'] = df['sqft_above'].astype(int)
    df['sqft_basement'] = df['sqft_basement'].astype(int)
    df['yr_built'] = df['yr_built'].astype(int)
    df['yr_renovated'] = df['yr_renovated'].astype(int)
    df['sqft_living15'] = df['sqft_living15'].astype(int)
    df['sqft_lot15'] = df['sqft_lot15'].astype(int)


    # Add Relevant Features:
    date_tuple = []
    for date in df['date'].values:
        if date == '0':
            date_tuple.append(0)
            continue
        date_tup = int(date[0:4])
        date_tuple.append(date_tup)

    df['date_tuple'] = date_tuple # (YYYY, MM, DD)
    df = df.loc[df['date'] != 0]
    df = df.loc[df['price'].astype(int) > 0]
    df = df.loc[df['sqft_above'] > 80]
    df = df.loc[df['sqft_living'] > 80]

    df = df.loc[df['sqft_living15'] > 80]
    df = df.loc[df['sqft_lot15'] > 80]
    df = df.loc[df['sqft_lot'] > df['sqft_living']]

    X = df.drop(columns=['price','id','date'])
    y = df['price']
    # extract from date_tuple and yr_built cols the age of the house, and then drop the first two cols:
    X['age'] = X['date_tuple']-X['yr_built'].astype(int)
    X.drop(columns=['yr_built'], inplace=True)

    # Encode all categorical features (both nominal and ordinal features):
    catagorical = ['zipcode']
    ordinal = ['grade', 'floors','condition']

    for label in catagorical:
        d = pd.get_dummies(X[label])
        for (columnName, columnData) in d.iteritems():
            X[f'{label}_{columnName}'] = columnData

    labelEncoder = LabelEncoder()
    for label in ordinal:
        X[label] = labelEncoder.fit_transform(X[label])
    X.drop(columns=['zipcode'], inplace=True)
    return X,y


def _pearson_corr(X: pd.DataFrame, y:pd.DataFrame):

    std_X = X.std()
    std_y = y.std()
    cov_XY = X.cov(y)
    try:
        pearson_corr = cov_XY/(std_X*std_y)
    except ZeroDivisionError:
        raise ZeroDivisionError()
    return pearson_corr

def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    pearson_correlations = []
    for feature_name in X.columns.values:
        pearson_correlations.append(_pearson_corr(X[feature_name], y))
    fig = go.Figure([go.Scatter(x=X.columns.values, y=pearson_correlations, mode='markers+lines')],
              layout=go.Layout(title=r"$\text{Correlation vs Features and the response vector} $",
                               xaxis_title="$\\text{Feature name}$",
                               yaxis_title="$\\text{Correlation of Feature and response vector}$",
                               height=900))
    # fig.show()
    minimal_effect = go.Figure([go.Scatter(x=X['sqft_living'], y=y, mode='markers')],
              layout=go.Layout(title=r"$\text{Price(sqft_living) - an example for feature with high impact on the price"
                                     r" of the house (Correlation  ~= 0.703)} $",
                               xaxis_title="$\\text{Sqft of the living room}$",
                               yaxis_title="$\\text{The price of the house in $}$",
                               height=900))
    maximal_effect = go.Figure([go.Scatter(x=X['condition'], y=y, mode='markers')
                         ],
              layout=go.Layout(title=r"$\text{Price(condition grade) - an example for feature with low impact on the price"
                                     r" of the house (Correlation ~= 0.03)} $",
                               xaxis_title="$\\text{Condition Of the House in scale of 1-10}$",
                               yaxis_title="$\\text{The price of the house in $}$",
                               height=900))
    minimal_effect.show()
    maximal_effect.show()



if __name__ == '__main__':
    # Our LR model:
    X =  np.random.normal(0, 1, size=(5,5))
    y = np.random.normal(0, 1, 5)
    ours = LinearRegression(False)
    ours._fit(X,y)
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(HOUSE_PRICES_PATH)

    # # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X,y)

    # # Question 3 - Split samples into training- and testing sets.
    X_train, y_train, X_test, y_test = split_train_test(X, y)
    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    percent = [p / 100 for p in range(10,101)]
    reg_linear_models = []
    vars = []
    gos = []
    for p in percent:
        losses = []
        for step in range(10):
            unified = pd.DataFrame(X)
            unified['response'] = y
            unified = unified.sample(frac=1).reset_index(drop=True)
            X_train,y_train = unified.drop(columns=['response']),unified['response']
            N = unified.shape[0]
            X_train_p, y_train_p,X_test_p,y_test_p = X_train[:int(N * p)], y_train[:int(N * p)], X_train[int(N * p):], y_train[int(N * p):]
            sk = LinearRegression(include_intercept=False)
            sk.fit(X_train_p.values,y_train_p.values)
            y_pred = sk.predict(X_test.values)
            losses.append(mean_square_error(y_pred,y_test)/10)
        vars.append(np.std(losses))
        reg_linear_models.append(np.mean(losses))

    graph = (go.Scatter(x=percent, y=reg_linear_models, mode="markers+lines", name="", line=dict(dash="dash"),
                marker=dict(color="green", opacity=.7)),
     go.Scatter(x=percent, y=reg_linear_models - 2 * np.array(vars), fill=None, mode="lines", line=dict(color="lightgrey"),
                showlegend=True),
     go.Scatter(x=percent, y=reg_linear_models + 2 * np.array(vars), fill='tonexty', mode="lines", line=dict(color="lightgrey"),
                showlegend=False))

    fig = go.Figure(graph,
                    layout=go.Layout(title=r"$\text{Loss vs. % of sample size} $",
                                     xaxis_title="$\\text{the ralative size from total sample}$",
                                     yaxis_title="$\\text{Loss}$",
                                     height=1100))
    fig.show()


