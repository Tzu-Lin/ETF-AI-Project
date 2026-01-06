from statsmodels.tsa.arima.model import ARIMA
# 假設 data 是 Series
model = ARIMA(data, order=(5,1,0)).fit()
forecast = model.forecast(steps=10)