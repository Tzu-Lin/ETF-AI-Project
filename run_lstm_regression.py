# =============================================================
#  LSTM 迴歸 — 預測次日收盤價（特徵一致化版：與樹模型相同的 6 維特徵）
#  驗證: TimeSeriesSplit (與樹模型一致)
#  特徵: Lag1, Lag2, Lag5, MA10, MA50, Vol10  (過去 WINDOW 天序列, MinMax, 僅訓練集 fit)
#  輸出: 訓練集MSE / 測試集MSE / 測試集MAPE(%)
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

FEATURES = ["Lag1", "Lag2", "Lag5", "MA10", "MA50", "Vol10"]   # ★與樹模型迴歸相同


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100


# ★ 原 load_close 改為 load_features：把 6 個特徵與標籤一次算好
def load_features(ticker):
    conn = sqlite3.connect(DB)
    t = ticker.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT Date, Close FROM "{t}"', conn,
                           index_col="Date", parse_dates=["Date"])
    conn.close()
    df = df["Close"].sort_index().to_frame().dropna()

    df["Lag1"]  = df["Close"].shift(1)
    df["Lag2"]  = df["Close"].shift(2)
    df["Lag5"]  = df["Close"].shift(5)
    df["MA10"]  = df["Close"].rolling(10).mean()
    df["MA50"]  = df["Close"].rolling(50).mean()
    logret      = np.log(df["Close"] / df["Close"].shift(1))
    df["Vol10"] = logret.rolling(10).std() * np.sqrt(252)   # 10日年化波動率

    df["target"] = df["Close"].shift(-1)                    # 次日收盤價 = 標籤
    return df.dropna()


# ★ make_sequences：改吃 6 維特徵，標籤用 target（次日收盤價）
def make_sequences(df, window=WINDOW):
    feat = df[FEATURES].values          # (N, 6)
    tgt  = df["target"].values          # (N,)
    X, y = [], []
    for i in range(window, len(df)):
        X.append(feat[i - window:i, :]) # (window, 6)
        y.append(tgt[i - 1])            # 視窗最後一天對應的次日收盤價
    return np.array(X), np.array(y)


REGRESSORS = {"單層 LSTM": SingleLayerLSTMRegressor,
              "雙層 LSTM": DoubleLayerLSTMRegressor}


def lstm_tscv(df, RegClass, n_splits=N_SPLITS):
    X, y = make_sequences(df)                       # X: (N, WINDOW, 6)
    n_feat = X.shape[2]                             # = 6
    tscv = TimeSeriesSplit(n_splits=n_splits)
    tr_mse, te_mse, te_mape = [], [], []
    for tr, te in tscv.split(X):
        # 攤平成 2D 縮放，再 reshape 回 3D（只用訓練集 fit，防洩漏）
        xs = MinMaxScaler()
        Xtr = xs.fit_transform(X[tr].reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
        Xte = xs.transform(X[te].reshape(-1, n_feat)).reshape(-1, WINDOW, n_feat)
        ys = MinMaxScaler(); ytr = ys.fit_transform(y[tr].reshape(-1, 1))

        model = RegClass((WINDOW, n_feat))          # ★ 傳 (20, 6)，models.py 不用改
        model.train(Xtr, ytr, epochs=EPOCHS, batch_size=32)

        ptr = ys.inverse_transform(model.predict(Xtr).reshape(-1, 1)).flatten()
        pte = ys.inverse_transform(model.predict(Xte).reshape(-1, 1)).flatten()
        tr_mse.append(mean_squared_error(y[tr], ptr))
        te_mse.append(mean_squared_error(y[te], pte))
        te_mape.append(mape(y[te], pte))
    return np.mean(tr_mse), np.mean(te_mse), np.mean(te_mape)


results = []
print("=" * 70)
print("LSTM 迴歸 — TimeSeriesSplit (5 折) — 6 維特徵一致化版")
print("=" * 70)
for tk in MARKETS:
    df = load_features(tk)              # ★ 原本是 close = load_close(tk)
    print(f"\n● {tk}")
    for name, Reg in REGRESSORS.items():
        trm, tem, tema = lstm_tscv(df, Reg)
        print(f"  {name}: 訓練MSE={trm:.4f}  測試MSE={tem:.4f}  測試MAPE={tema:.2f}%")
        results.append({"市場": tk, "模型": name, "訓練集MSE": round(trm, 4),
                        "測試集MSE": round(tem, 4), "測試集MAPE(%)": round(tema, 2)})

pd.DataFrame(results).to_csv("lstm_regression_results.csv", index=False, encoding="utf-8-sig")
print("\n-> 已存 lstm_regression_results.csv")