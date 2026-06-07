import sqlite3, warnings
import pandas as pd, numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
 
ORDER = (5, 1, 0)   # AR=5, 一階差分, MA=0
 
def load(ticker):
    conn = sqlite3.connect("etf_data.db")
    t = ticker.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn, index_col="Date", parse_dates=["Date"])
    conn.close()
    return df.sort_index()
 
def metrics(y, p):
    mse = mean_squared_error(y, p)
    return r2_score(y, p), np.sqrt(mse), mean_absolute_error(y, p)
 
def arima_walk_forward(close, n_splits=5, order=ORDER):
    close = close.reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    preds, trues = [], []
    idx = np.arange(len(close))
    for tr, te in tscv.split(idx):
        history = list(close.iloc[:tr[-1] + 1].values)
        fit = None
        for step, i in enumerate(te):
            if fit is None or step % 5 == 0:          # 每5步重新估計, 兼顧速度
                fit = ARIMA(history, order=order).fit()
            preds.append(fit.forecast(steps=1)[0])
            trues.append(close.iloc[i])
            history.append(close.iloc[i])             # one-step-ahead
    return np.array(trues), np.array(preds)
 
print(f"ARIMA{ORDER} walk-forward — 預測次日收盤價\n")
print(f"{'市場':10s}{'模型':18s}{'R2':>9s}{'RMSE':>10s}{'MAE':>10s}")
for ticker in ["QQQ", "0050.TW"]:
    close = load(ticker)["Close"]
    c = close.reset_index(drop=True)
    te_all = np.concatenate([te for _, te in TimeSeriesSplit(n_splits=5).split(np.arange(len(c)))])
    # 隨機漫步基準: 明天 = 今天
    rb = metrics(c.iloc[te_all].values, c.iloc[te_all - 1].values)
    # ARIMA
    y, p = arima_walk_forward(close)
    am = metrics(y, p)
    print(f"{ticker:10s}{'明天 = 今天':18s}{rb[0]:9.4f}{rb[1]:10.2f}{rb[2]:10.2f}")
    print(f"{ticker:10s}{'ARIMA'+str(ORDER):18s}{am[0]:9.4f}{am[1]:10.2f}{am[2]:10.2f}\n")
 