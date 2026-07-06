import sqlite3, numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit

DB = "etf_data.db"; WINDOW = 20; N_SPLITS = 5

def load_close(tk):
    c = sqlite3.connect(DB); t = tk.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT Date,Close FROM "{t}"', c,
                           index_col="Date", parse_dates=["Date"]); c.close()
    return df["Close"].sort_index().dropna()

for tk in ["SPY", "QQQ", "0050.TW"]:
    close = load_close(tk)
    seq_dates = close.index[WINDOW: len(close) - 1]
    N = len(seq_dates)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    idx = np.arange(N)
    tr, te = list(tscv.split(idx))[-1]
    d0, d1 = seq_dates[te[0]], seq_dates[te[-1]]
    print(f"{tk}: 總樣本={N}, 每折測試≈{N//(N_SPLITS+1)}, 最後一折={len(te)}筆, 期間 {d0.date()}~{d1.date()}")
    