import sqlite3
import numpy as np, pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

DB = "etf_data.db"
MARKETS = ["SPY", "QQQ", "0050.TW"]

def load_close(tk):
    conn = sqlite3.connect(DB)
    t = tk.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT Date, Close FROM "{t}"', conn,
                           index_col="Date", parse_dates=["Date"])
    conn.close()
    return df["Close"].sort_index().dropna()

def rsi(close, period=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(period).mean()
    dn = (-d.clip(upper=0)).rolling(period).mean()
    rs = up / dn
    return 100 - 100 / (1 + rs)

def make_features(close):
    df = pd.DataFrame(index=close.index)
    df["Return"] = close.pct_change()
    df["MA20"]   = close.rolling(20).mean()
    df["MA60"]   = close.rolling(60).mean()
    df["RSI"]    = rsi(close, 14)
    df["y"] = (close.shift(-1) > close).astype(int)   # 次日漲=1
    return df.dropna()

param_grid = {
    "rf__n_estimators": [100, 200, 300],
    "rf__max_depth":    [5, 10, 15, None],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2"],
}

print("=" * 60)
for tk in MARKETS:
    df = make_features(load_close(tk))
    X = df[["Return", "MA20", "MA60", "RSI"]].values
    y = df["y"].values

    n = len(X); cut = int(n * 0.8)          # 80/20 順序切分
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("rf", RandomForestClassifier(random_state=42))])
    grid = GridSearchCV(pipe, param_grid,
                        cv=TimeSeriesSplit(n_splits=5),
                        scoring="f1", n_jobs=-1)
    grid.fit(Xtr, ytr)

    f1_test = f1_score(yte, grid.predict(Xte))
    always_up = f1_score(yte, np.ones_like(yte))      # Always-Up 基準

    print(f"\n● {tk}")
    print("  最佳參數:", grid.best_params_)
    print(f"  測試集 F1(調參後)= {f1_test:.4f}")
    print(f"  Always-Up 基準  = {always_up:.4f}")
print("\n" + "=" * 60)