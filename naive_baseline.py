# naive_baseline.py — 迴歸基準：用今天收盤價當明天預測 (Naive Forecast)
import sqlite3, numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

DB = "etf_data.db"; WINDOW = 20; N_SPLITS = 5
MARKETS = ["SPY", "QQQ", "0050.TW"]

def load_close(tk):
    c = sqlite3.connect(DB); t = tk.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT Date,Close FROM "{t}"', c,
                           index_col="Date", parse_dates=["Date"]); c.close()
    return df["Close"].sort_index().dropna()

def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100

def make_seq(close, window=WINDOW):
    v = close.values.reshape(-1, 1)
    X_last, y = [], []       # X_last=窗口最後一天收盤價(=今天), y=次日收盤價
    for i in range(window, len(v) - 1):
        X_last.append(v[i - 1, 0])   # 今天(窗口最後一天)的收盤價
        y.append(v[i + 1, 0])        # 次日收盤價
    return np.array(X_last), np.array(y)

rows = []
for tk in MARKETS:
    close = load_close(tk)
    x_today, y_next = make_seq(close)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    tr, te = list(tscv.split(x_today))[-1]      # 對齊最後一折
    y_true = y_next[te]
    y_pred = x_today[te]                         # ★ Naive：明天預測 = 今天收盤
    mse = mean_squared_error(y_true, y_pred)
    rows.append({"市場": tk, "測試筆數": len(te),
                 "Naive_MSE": round(mse, 4),
                 "Naive_MAPE(%)": round(mape(y_true, y_pred), 2)})

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("naive_baseline_results.csv", index=False, encoding="utf-8-sig")
print("\n-> 已存 naive_baseline_results.csv")