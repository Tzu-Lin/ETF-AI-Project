# =============================================================
#  樹模型迴歸 — 預測次日收盤價
#  驗證: TimeSeriesSplit (時序交叉驗證, 5 折)
#  模型: RandomForest / XGBoost / AdaBoost / GradientBoosting
#  輸出: 各模型 訓練集MSE / 測試集MSE / 測試集MAPE(%)
#  MSE  : 同一市場內比較模型 (受股價尺度影響, 不可跨市場比)
#  MAPE : 百分比, 可跨市場比較
#  資料: etf_data.db    需要: scikit-learn xgboost
# =============================================================
import sqlite3, warnings
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

DB = "etf_data.db"
MARKETS = ["SPY", "QQQ", "0050.TW"]
FEATURES = ["Lag1", "Lag2", "Lag5", "MA10", "MA50", "Vol10"]
N_SPLITS = 5


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100


def load(ticker):
    conn = sqlite3.connect(DB)
    t = ticker.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT * FROM "{t}"', conn, index_col="Date", parse_dates=["Date"])
    conn.close()
    return df.sort_index()


def make_features(df):
    f = pd.DataFrame(index=df.index)
    f["Lag1"]  = df["Close"].shift(1)
    f["Lag2"]  = df["Close"].shift(2)
    f["Lag5"]  = df["Close"].shift(5)
    f["MA10"]  = df["Close"].rolling(10).mean()
    f["MA50"]  = df["Close"].rolling(50).mean()
    f["Vol10"] = df["Close"].pct_change().rolling(10).std()
    f["Target"] = df["Close"].shift(-1)
    return f.dropna()


def models():
    return {
        "Random Forest":     RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost":           XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0),
        "AdaBoost":          AdaBoostRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42),
    }


def tscv_eval(X, y, model, n_splits=N_SPLITS):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    tr_mse, te_mse, te_mape = [], [], []
    for tr, te in tscv.split(X):
        m = model.__class__(**model.get_params())
        m.fit(X.iloc[tr], y.iloc[tr])
        tr_mse.append(mean_squared_error(y.iloc[tr], m.predict(X.iloc[tr])))
        pred = m.predict(X.iloc[te])
        te_mse.append(mean_squared_error(y.iloc[te], pred))
        te_mape.append(mape(y.iloc[te], pred))
    return np.mean(tr_mse), np.mean(te_mse), np.mean(te_mape)


results = []
print("=" * 70)
print("樹模型迴歸 — TimeSeriesSplit (5 折)")
print("=" * 70)
for tk in MARKETS:
    f = make_features(load(tk))
    X, y = f[FEATURES], f["Target"]
    print(f"\n● {tk}  (樣本 {len(X)})")
    print(f"  {'模型':18s}{'訓練MSE':>11s}{'測試MSE':>11s}{'測試MAPE(%)':>13s}")
    for nm, m in models().items():
        trm, tem, tema = tscv_eval(X, y, m)
        print(f"  {nm:18s}{trm:11.4f}{tem:11.2f}{tema:13.2f}")
        results.append({"市場": tk, "模型": nm, "訓練集MSE": round(trm, 4),
                        "測試集MSE": round(tem, 2), "測試集MAPE(%)": round(tema, 2)})

pd.DataFrame(results).to_csv("tree_regression_results.csv", index=False, encoding="utf-8-sig")
print("\n-> 已存 tree_regression_results.csv")