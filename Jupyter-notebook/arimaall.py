for a in range(20):
    datacluster = pd.read_csv("cluster_"+str(a)+"_Arima.csv")
    datacluster.Tahun = pd.to_datetime(datacluster.Tahun, format="%Y%m")
    x = datacluster['Tahun']
    y = datacluster['count']
    plt.figure(figsize=(10,10))
    plt.plot(x,y)

for a in range(20):
    series = pd.read_csv("cluster_"+str(a)+"_Arima.csv", header=0, parse_dates=[0],index_col=0, squeeze=True)
    autocorrelation_plot(series)
    plt.show()
    model = ARIMA(series,order = (3,1,1))
    model_fit = model.fit(disp=0)
    print("=====model %d=====" % a)
    print(model_fit.summary())
    residuals = pd.DataFrame(model_fit.resid)
    plt.title("residuals "+str(a))
    residuals.plot()
    plt.show()
    plt.title("KDE residuals "+str(a))
    plt.show()
    print("============================")

series = pd.read_csv("cluster_"+str(a)+"_Arima.csv", header=0, parse_dates=[0],index_col=0, squeeze=True)
autocorrelation_plot(series)
plt.show()
model = ARIMA(series,order = (3,1,1))
model_fit = model.fit(disp=0)
print("=====model %d=====" % a)
print(model_fit.summary())
residuals = pd.DataFrame(model_fit.resid)
plt.title("residuals "+str(a))
residuals.plot()
plt.show()
plt.title("KDE residuals "+str(a))
plt.show()
print("============================")
