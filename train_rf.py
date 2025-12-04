import yfinance as yf
import pandas as pd

def get_close_series(ticker: str, start="2019-01-01"):
    """
    從 Yahoo Finance 下載 ETF 的收盤價資料
    """
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        if ('Close', ticker) in df.columns:
            s = df[('Close', ticker)]
        elif (ticker, 'Close') in df.columns:
            s = df[(ticker, 'Close')]
        else:
            cols = [c for c in df.columns if (isinstance(c, tuple) and c[0] == 'Close') or c == 'Close']
            s = df[cols[0]]
    else:
        s = df['Close']

    s.name = 'Close'
    return s

def calc_rsi(s, period=14):
    """
    計算 RSI (相對強弱指標)
    用來衡量市場是否過度買進或賣出
    """
    delta = s.diff()                          # 每日價差
    gain = delta.clip(lower=0)                # 價格上漲的部分
    loss = -delta.clip(upper=0)               # 價格下跌的部分
    rs = gain.rolling(period).mean() / loss.rolling(period).mean()
    rsi = 100 - (100 / (1 + rs))              # RSI 計算公式
    return rsi


def make_features(close):
    """
    依據收盤價生成訓練用特徵與標籤
    """
    df = close.to_frame("Close")              # 轉成 DataFrame
    df["Return"] = df["Close"].pct_change()   # 日報酬率
    df["MA20"] = df["Close"].rolling(20).mean()   # 20日均線
    df["MA60"] = df["Close"].rolling(60).mean()   # 60日均線
    df["Volatility"] = df["Return"].rolling(20).std()  # 20日波動度
    df["RSI"] = calc_rsi(df["Close"])         # RSI
    df["Direction"] = (df["Return"].shift(-1) > 0).astype(int)  # 明日漲跌（標籤）

    # 移除空值
    return df.dropna()

# 指定要使用的特徵欄位
FEATURES = ["MA20", "MA60", "Volatility", "RSI"]

# 匯入必要模組
# ------------------------------------------------------------
from sklearn.pipeline import Pipeline              # 用來建立資料處理與模型串接流程
from sklearn.preprocessing import StandardScaler   # 資料標準化（Z-score）
from sklearn.ensemble import RandomForestClassifier # 隨機森林分類器
from sklearn.metrics import accuracy_score         # 評估模型準確率
from joblib import dump                            # 儲存模型用
from pathlib import Path                           # 處理檔案路徑

# === 模型訓練函式 ===
def train_one(ticker="SPY", start="2019-01-01"):
    """
    對單一 ETF 股票進行特徵生成、訓練與評估的流程。
    ticker: ETF 代碼（預設為 SPY）
    start: 下載資料的起始日期
    """
    # 1. 建立安全的檔案名稱 (與 app.py 的規則完全一致)
    safe_ticker_name = ticker.lower().replace('.', '_')
    
    # 2. 組合出完整的檔案路徑
    model_path = f"models/rf_{safe_ticker_name}.joblib" 
    
    # Step 1️⃣：取得收盤價資料
    close = get_close_series(ticker, start=start)

    # Step 2️⃣：建立技術指標與方向標籤
    df = make_features(close)

    # FEATURES 是要訓練的欄位（例如 MA20、MA60、RSI、Volatility 等）
    X, y = df[FEATURES], df["Direction"]

    # Step 3️⃣：將資料切成訓練集與測試集（時序型，不隨機打亂）
    split = int(len(df) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    # Step 4️⃣：建立 Pipeline 流程（先標準化，再放入模型）
    pipe = Pipeline([
        ("scaler", StandardScaler()),              # 對特徵進行標準化（平均=0, 標準差=1）
        ("rf", RandomForestClassifier(             # 隨機森林分類模型
            n_estimators=300,                      # 建立 300 棵決策樹
            random_state=42                        # 固定隨機種子，確保可重現
        ))
    ])

    # Step 5️⃣：模型訓練
    pipe.fit(X_tr, y_tr)

    # Step 6️⃣：測試集預測與準確率計算
    acc = accuracy_score(y_te, pipe.predict(X_te))
    print(f"📈 {ticker} 測試準確率: {acc:.3f}")

    # Step 7️⃣：模型儲存（方便之後在 app.py 匯入使用）
    Path("models").mkdir(exist_ok=True)  # 若資料夾不存在則建立
    dump({
        "model": pipe,
        "features": FEATURES,
        "ticker": ticker
    }, model_path) # 使用我們新建立的路徑變數

    print(f"✅ 已儲存模型: {model_path}")

# 執行訓練（以 SPY 為例）
train_one("SPY")

if __name__ == "__main__":
    # 訓練四支ETF
    for t in ["SPY", "QQQ", "SSO", "QLD", "0050.TW", "KRBN"]:
        train_one(t)

import matplotlib.pyplot as plt

# 抓 SPY 收盤價
spy = get_close_series("SPY")

# 畫出折線圖
plt.figure(figsize=(10, 4))
plt.plot(spy.index, spy.values, color="blue")
plt.title("SPY ETF closing price trend(2019–2025)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.grid(True)
plt.tight_layout()
plt.show()

