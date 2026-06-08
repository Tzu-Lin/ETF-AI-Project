# =============================================================
#  LSTM 迴歸 — 預測次日收盤價
#  驗證: TimeSeriesSplit (與樹模型一致); ARIMA 另用 rolling forecast
#  模型: 單層 / 雙層 LSTM (迴歸版, 來自 models.py)
#  特徵: 過去 WINDOW 天收盤價序列 (MinMax 縮放, 僅用訓練集 fit)
#  輸出: 訓練集MSE / 測試集MSE / 測試集MAPE(%)
#  資料: etf_data.db    需要: tensorflow scikit-learn
# =============================================================
import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import sqlite3, warnings
import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from models import SingleLayerLSTMRegressor, DoubleLayerLSTMRegressor
warnings.filterwarnings("ignore")
tf.random.set_seed(42); np.random.seed(42)

DB = "etf_data.db"
MARKETS = ["SPY", "QQQ", "0050.TW"]
WINDOW = 20
N_SPLITS = 5
EPOCHS = 15


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100


def load_close(ticker):
    conn = sqlite3.connect(DB)
    t = ticker.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT Date, Close FROM "{t}"', conn,
                           index_col="Date", parse_dates=["Date"])
    conn.close()
    return df["Close"].sort_index().dropna()


def make_sequences(close, window=WINDOW):
    v = close.values.reshape(-1, 1)
    X, y = [], []
    for i in range(window, len(v) - 1):
        X.append(v[i - window:i, 0]); y.append(v[i + 1, 0])
    return np.array(X), np.array(y)


REGRESSORS = {"單層 LSTM": SingleLayerLSTMRegressor,
              "雙層 LSTM": DoubleLayerLSTMRegressor}


def lstm_tscv(close, RegClass, n_splits=N_SPLITS):
    X, y = make_sequences(close)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    tr_mse, te_mse, te_mape = [], [], []
    for tr, te in tscv.split(X):
        xs = MinMaxScaler(); Xtr = xs.fit_transform(X[tr]); Xte = xs.transform(X[te])
        ys = MinMaxScaler(); ytr = ys.fit_transform(y[tr].reshape(-1, 1))
        Xtr = Xtr.reshape(-1, WINDOW, 1); Xte = Xte.reshape(-1, WINDOW, 1)
        model = RegClass((WINDOW, 1))
        model.train(Xtr, ytr, epochs=EPOCHS, batch_size=32)
        ptr = ys.inverse_transform(model.predict(Xtr).reshape(-1, 1)).flatten()
        pte = ys.inverse_transform(model.predict(Xte).reshape(-1, 1)).flatten()
        tr_mse.append(mean_squared_error(y[tr], ptr))
        te_mse.append(mean_squared_error(y[te], pte))
        te_mape.append(mape(y[te], pte))
    return np.mean(tr_mse), np.mean(te_mse), np.mean(te_mape)


results = []
print("=" * 70)
print("LSTM 迴歸 — TimeSeriesSplit (5 折)")
print("=" * 70)
for tk in MARKETS:
    close = load_close(tk)
    print(f"\n● {tk}")
    for name, Reg in REGRESSORS.items():
        trm, tem, tema = lstm_tscv(close, Reg)
        print(f"  {name}: 訓練MSE={trm:.4f}  測試MSE={tem:.4f}  測試MAPE={tema:.2f}%")
        results.append({"市場": tk, "模型": name, "訓練集MSE": round(trm, 4),
                        "測試集MSE": round(tem, 4), "測試集MAPE(%)": round(tema, 2)})

pd.DataFrame(results).to_csv("lstm_regression_results.csv", index=False, encoding="utf-8-sig")
print("\n-> 已存 lstm_regression_results.csv")