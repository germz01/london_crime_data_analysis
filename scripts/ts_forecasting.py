import ipdb
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.tsa.seasonal import seasonal_decompose

from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")


def evaluate_model(ts, arima_order):
    train = list(ts[0:int(ts.shape[0] * 0.66)].values)
    test = list(ts[int(ts.shape[0] * 0.66):].values)
    history, predictions = train, []

    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0][0]
        predictions.append(yhat)
        history.append(test[t])

    error = mean_squared_error(test, predictions)

    return error


def evaluate_models(ts):
    best_score, best_cfg = np.Inf, None

    for test_order in itertools.product([1, 2, 3, 4], [1], [1, 2, 3, 4]):
        print('TESTING ORDER {}'.format(test_order))

        try:
            mse = evaluate_model(ts, test_order)

            if mse < best_score:
                best_score, best_cfg = mse, test_order
                print('ARIMA {} MSE = {}'.format(test_order, mse))
        except:
            continue

    print('Best ARIMA {} MSE = {}'.format(best_cfg, best_score))


df = pd.read_csv('../data/time_series/ts_total.csv', index_col='date',
                 parse_dates=True)
df = df.asfreq('MS')

df.rename(columns={'value': 'ts'}, inplace=True)

ts_log = df['ts'].apply(lambda x: np.log(x))
ts_log_diff = df['ts'].apply(lambda x: np.log(x)).diff(periods=1).dropna()

# ipdb.set_trace()

evaluate_models(ts_log_diff)
