import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = "browser"
pio.templates.default = "simple_white"
TEMPERATURE_FILE_PATH = '/Users/tzlilovadia/IML.HUJI/datasets/City_Temperature.csv'

def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    if filename == None or len(filename) == 0:
        raise FileNotFoundError()

    df = pd.read_csv(filename, parse_dates=['Date'])
    day_of_year = []
    for date in df['Date']:
        day_of_year.append(date.day_of_year)
    df['DayOFYear'] = day_of_year

    # Check for missing data:
    # print(df.isnull().sum())
    # print(df.dtypes)
    # print(df.shape[0])
    df['Day']  = df['Day'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)
    df['Temp'] = df['Temp'].astype(float)
    df = df.loc[(df['Day'] <= 31) & (df['Day'] > 0)]
    df = df.loc[(df['Month'] <= 12) & (df['Month'] > 0)]
    df = df.loc[(df['Year'] <= 2022) & (df['Year'] > 0)]
    df = df.loc[(df['Temp'] <= 100) & (df['Temp'] > -100)]
    df.drop(columns=['Day','Date'])

    catagorical = ['City','Country']
    ordinal = ['DayOFYear','Month']

    for label in catagorical:
        d = pd.get_dummies(df[label])
        for (columnName, columnData) in d.iteritems():
            df[f'{label}_{columnName}'] = columnData

    labelEncoder = LabelEncoder()
    for label in ordinal:
        df[label] = labelEncoder.fit_transform(df[label])
    return df


def explore_country(df: pd.DataFrame, country: str):
    df_from_country = df.loc[df['Country'] == country]
    df_from_country = df_from_country.loc[df_from_country['Temp'] > -10]
    temperature_by_years = []
    for year in df_from_country.groupby(['Year']):
        std_by_month = []
        df_per_year = df_from_country.loc[df_from_country['Year'] == year[0]]

        scatter = go.Scatter(x=df_per_year['DayOFYear'], y=df_per_year['Temp'], mode='markers+lines', name=year[0])
        temperature_by_years.append(scatter)
    fig = go.Figure(temperature_by_years,
                    layout=go.Layout(title=r"$\text{Temperature measurement as function of the day of year} $",
                                     xaxis_title="$\\text{Day OF Year}$",
                                     yaxis_title="$\\text{Temperature measurement}$",
                                     height=900
                                     ))
    fig.show()
    for month in df_from_country.groupby(['Month']):
        std_by_month_i = df_from_country.loc[df_from_country['Month'] == month[0]]
        std_by_month_i = std_by_month_i['Temp'].std()
        std_by_month.append(std_by_month_i)
    plt.bar(np.arange(1, 13), std_by_month)
    plt.title('Std(Month)')
    plt.xlabel('Number of month')
    plt.ylabel('Std value')
    plt.show()


def explore_by_category(df: pd.DataFrame, groupby):
    temperature_by_ctg = {}
    for country in set(df['Country'].values):
        country_df = df.loc[df['Country'] == country]
        avg_tmp_country = []
        for month in country_df.groupby([groupby]):
            avg_tmp = country_df.loc[country_df['Month'] == month[0]]
            avg_tmp = avg_tmp['Temp'].mean()
            avg_tmp_country.append(avg_tmp)
        std_by_month_country = calc_montly_std(country_df)
        temperature_by_ctg[country] = avg_tmp_country,std_by_month_country
    for country in temperature_by_ctg.keys():
        avg_tmp, std_by_month = temperature_by_ctg[country]
        plt.errorbar(np.arange(1,13),avg_tmp, std_by_month,label=country,fmt='o')
        plt.xlabel('Month')
        plt.ylabel('Average temperature')
    plt.title('Avg temperature as function of month for different countries')
    plt.legend()
    plt.show()


def plot_std_bars(df_from_category):
    std_by_month = calc_montly_std(df_from_category)
    plt.bar(np.arange(1, 13), std_by_month)
    plt.title('Std(Month)')
    plt.xlabel('Number of month')
    plt.ylabel('Std value')
    plt.show()


def calc_montly_std(df_from_country):
    std_by_month = []
    for month in df_from_country.groupby(['Month']):
        std_by_month_i = df_from_country.loc[df_from_country['Month'] == month[0]]
        std_by_month_i = std_by_month_i['Temp'].std()
        std_by_month.append(std_by_month_i)
    return std_by_month

def fit10poly(X_train: pd.DataFrame, y_train ,X_test: pd.DataFrame, y_test:pd.DataFrame):
    losses = []
    for deg in range(1,11):
        poly = PolynomialFitting(k=deg)
        poly.fit(X_train.values,y_train.values)
        losses.append(poly.loss(X_test.values, y_test.values))
        print(f'For degree {deg} the loss is:')
        print("{:.2f}".format(losses[-1]))
        print('\n\n')
    plt.bar(np.arange(1,len(losses)+1), losses)
    plt.title("Polynomial fitting error (MSE wise) using different degree values")
    plt.xlabel('Polynomial degree')
    plt.ylabel('MSE')
    plt.show()

def fit_other_poly(X_train: pd.DataFrame, y_train ,X_test: pd.DataFrame, y_test:pd.DataFrame):
    losses = []
    for deg in range(1,11):
        poly = PolynomialFitting(k=deg)
        poly.fit(X_train.values,y_train.values)
        losses.append(poly.loss(X_test.values, y_test.values))
    plt.bar(np.arange(1,len(losses)+1), losses)
    plt.title("Error of the pol")
    plt.show()

if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(TEMPERATURE_FILE_PATH)
    df = df.loc[df['Temp'] > -10]
    # Question 2 - Exploring data for specific country
    explore_country(df,'Israel')
    # Question 3 - Exploring differences between countries
    df_from_country = df.loc[df['Temp'] > -10]
    explore_by_category(df_from_country, 'Month')
    # Question 4 - Fitting model for different values of `k`
    from_israel = df.loc[df['Country'] == 'Israel']
    X_israel, y_israel = from_israel.drop(columns=['Temp']),from_israel['Temp']
    train_X, train_y, test_X, test_y = split_train_test(X_israel,y_israel)
    fit10poly(train_X['DayOFYear'], train_y, test_X['DayOFYear'], test_y)

    # Question 5 - Evaluating fitted model on different countries
    from_israel = df.loc[df['Country'] == 'Israel']
    poly = PolynomialFitting(k=6)
    poly.fit(from_israel['DayOFYear'].values, from_israel['Temp'].values)
    other_loss = {}
    for other_country in df['Country'].unique():
        other = df.loc[df['Country'] == other_country]
        X_other,y_other = other['DayOFYear'],other['Temp']
        loss_other = poly.loss(X_other.values,y_other.values)
        other_loss[other_country] = loss_other
    plt.bar(other_loss.keys(), other_loss.values())
    plt.ylabel('Loss')
    plt.xlabel('Country')
    plt.title('Loss of predicting the temperature using Israel\'s fitted model')
    plt.show()
