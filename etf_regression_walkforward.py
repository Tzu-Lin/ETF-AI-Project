# =============================================================
#  ETF 迴歸實驗 — TimeSeriesSplit walk-forward + 隨機漫步基準
#  四模型: RandomForest / XGBoost / AdaBoost / GradientBoosting
#  輸出: R2 / MSE / RMSE / MAE， 並畫出「不躺平」的實際 vs 預測圖
#  用法: 把 etf_data.db 跟本檔放同一資料夾, 直接執行
# =============================================================
import sqlite3, warnings
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor          # 若無: pip install xgboost
warnings.filterwarnings("ignore")

FEATURES = ["Lag1", "Lag2", "Lag5", "MA10", "MA50", "Vol10"]   # 特徵選擇(不含當前價)

def load(ticker):
    conn = sqlite3.connect("etf_data.db")
    t = ticker.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn, index_col="Date", parse_dates=["Date"])
    conn.close()
    return df.sort_index()

def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["Close"] = df["Close"]
    f["Lag1"]  = df["Close"].shift(1)
    f["Lag2"]  = df["Close"].shift(2)
    f["Lag5"]  = df["Close"].shift(5)
    f["MA10"]  = df["Close"].rolling(10).mean()
    f["MA50"]  = df["Close"].rolling(50).mean()
    f["Vol10"] = df["Close"].pct_change().rolling(10).std()
    f["Target"] = df["Close"].shift(-1)        # 預測次日收盤價
    return f.dropna()

def models():
    return {
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0),
        "AdaBoost":          AdaBoostRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    }

def walk_forward(X, y, model, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    ps, ts, ix = [], [], []
    for tr, te in tscv.split(X):
        m = model.__class__(**model.get_params())
        m.fit(X.iloc[tr], y.iloc[tr])
        ps.append(m.predict(X.iloc[te])); ts.append(y.iloc[te].values); ix.append(X.index[te])
    return np.concatenate(ps), np.concatenate(ts), pd.DatetimeIndex(np.concatenate(ix))

def metric_row(y, p):
    mse = mean_squared_error(y, p)
    return dict(R2=r2_score(y, p), MSE=mse, RMSE=np.sqrt(mse), MAE=mean_absolute_error(y, p))

for ticker in ["SPY", "QQQ", "0050.TW"]:
    f = make_features(load(ticker))
    X, y = f[FEATURES], f["Target"]
    print(f"\n========== {ticker} (n={len(X)}) ==========")
    print(f"{'Model':22s}{'R2':>9s}{'RMSE':>11s}{'MAE':>10s}")

    # 隨機漫步基準: 明天 = 今天的收盤價 (在 walk-forward 測試區間上評估)
    te_all = np.concatenate([te for _, te in TimeSeriesSplit(n_splits=5).split(X)])
    bm = metric_row(y.iloc[te_all].values, f["Close"].iloc[te_all].values)
    print(f"{'Random Walk (naive)':22s}{bm['R2']:9.4f}{bm['RMSE']:11.2f}{bm['MAE']:10.2f}")

    best = None
    for name, mdl in models().items():
        p, t, idx = walk_forward(X, y, mdl)
        m = metric_row(t, p)
        print(f"{name:22s}{m['R2']:9.4f}{m['RMSE']:11.2f}{m['MAE']:10.2f}")
        if name == "Random Forest":
            best = (idx, t, p)

    # 畫 SPY 的 walk-forward 圖 (不躺平)
    if ticker == "SPY":
        idx, t, p = best
        plt.figure(figsize=(11, 5))
        plt.plot(idx, t, label="Actual", color="#3b5bdb", lw=1.3)
        plt.plot(idx, p, label="Predicted (RF, walk-forward)", color="#2f9e44", lw=1.3, ls="--")
        plt.title(f"{ticker} Actual vs Predicted — TimeSeriesSplit walk-forward")
        plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.savefig("spy_walkforward.png", dpi=120)
        print("  -> saved spy_walkforward.png")
