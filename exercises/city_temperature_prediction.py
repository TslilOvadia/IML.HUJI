import datetime

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
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
    print(df.isnull().sum())
    print(df.dtypes)
    print(df.shape[0])
    df['Day']  = df['Day'].astype(int)
    df['Month'] = df['Month'].astype(int)
    df['Year'] = df['Year'].astype(int)
    df['Temp'] = df['Temp'].astype(float)
    df = df.loc[(df['Day'] <= 31) & (df['Day'] > 0)]
    df = df.loc[(df['Month'] <= 12) & (df['Month'] > 0)]
    df = df.loc[(df['Year'] <= 2022) & (df['Year'] > 0)]
    df = df.loc[(df['Temp'] <= 100) & (df['Temp'] > -100)]
    df_from_israel = df.loc[df['Country'] == 'Israel']
    df_from_israel = df_from_israel.loc[df_from_israel['Temp'] > -10]
    return df_from_israel

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(TEMPERATURE_FILE_PATH)

    # Question 2 - Exploring data for specific country
    t = df.groupby(['Year'])
    temperature_by_years = []
    for year in df.groupby(['Year']):
        std_by_month = []
        df_per_year = df.loc[df['Year'] == year[0]]
        stds = []
        for month in df_per_year.groupby(['Month']):
            std_by_month_i = df_per_year.loc[df_per_year['Month'] == month[0]]
            std_by_month_i = std_by_month_i['Temp'].std()
            std_by_month.append(std_by_month_i)
        for i in range(df_per_year.shape[0]):
            # stds.append(std_by_month[])
            print(f"std_by_month is {std_by_month} with len {len(std_by_month)}")
            print(df_per_year['Month'].values[i]-1)
        df_per_year['std'] = stds

        scatter = go.Scatter(x=df_per_year['DayOFYear'], y=df_per_year['Temp'], mode='markers', name=year[0], error_y=dict(type='data',array=df_per_year['std'],visible=True) )
        temperature_by_years.append(scatter)
    fig = go.Figure(temperature_by_years,
              layout=go.Layout(title=r"$\text{Temperature measurement as function of the day of year} $",
                               xaxis_title="$\\text{Day OF Year}$",
                               yaxis_title="$\\text{Temperature measurement}$",
                               height=1100))
    fig.show()
    # for month in df.groupby(['Month']):
    #     std_by_month_i = df.loc[df['Month'] == month[0]]
    #     std_by_month_i = std_by_month_i['Temp'].std()
    #     std_by_month.append(std_by_month_i)
    # Question 3 - Exploring differences between countries

    # # Question 4 - Fitting model for different values of `k`
    # raise NotImplementedError()
    #
    # # Question 5 - Evaluating fitted model on different countries
    # raise NotImplementedError()