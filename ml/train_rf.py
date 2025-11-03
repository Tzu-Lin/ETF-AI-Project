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
    }, f"models/rf_{ticker}.joblib")

    print(f"💾 已儲存模型: models/rf_{ticker}.joblib")

# 執行訓練（以 SPY 為例）
train_one("SPY")