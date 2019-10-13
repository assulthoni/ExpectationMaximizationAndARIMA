import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import datetime
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot

for a in range(0,20):
    series = pd.read_csv('cluster_'+str(a)+'_Arima.csv', usecols=['Tahun','count'],header=0, parse_dates=[0], index_col=0, squeeze=True)
    autocorrelation_plot(series)
    # plt.show()
    plt.savefig("correlation "+str(a)+".png")

    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot()
    # plt.show()
    plt.savefig("residual "+str(a)+".png")
    residuals.plot(kind='kde')
    # plt.show()
    plt.savefig("KDE "+str(a)+".png")
    print(residuals.describe())

    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    # plot
    plt.title("ARIMA Cluster "+str(a))
    plt.plot(test)
    plt.plot(predictions, color='red')
    plt.savefig("Plot prediction "+str(a)+".png")
    # plt.show()
