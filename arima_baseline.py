import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.arima import ndiffs
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import f1_score, accuracy_score

# ==========================================
# 1. 資料準備 (Data Preparation)
# ==========================================
target_symbol = 'SPY'
print(f"🚀 正在下載 {target_symbol} 資料...")

# 加入 auto_adjust=False 確保一定有 'Adj Close'，或者我們手動處理
df = yf.download(target_symbol, start='2015-01-01', end='2025-01-01', progress=False)

# --- 關鍵修正開始：處理 yfinance 欄位格式問題 ---
# 1. 如果是 MultiIndex (多層欄位)，嘗試攤平
if isinstance(df.columns, pd.MultiIndex):
    try:
        # 嘗試直接取 'Adj Close' (如果它是第一層)
        df = df['Adj Close']
    except KeyError:
        try:
            # 如果失敗，嘗試看是否在第二層 (有時候是 SPY -> Adj Close)
            df = df.xs('Adj Close', axis=1, level=1)
        except KeyError:
            # 如果還是失敗，可能是欄位名稱變了，試試看取 'Close'
            print("⚠️ 找不到 'Adj Close'，嘗試使用 'Close' 代替...")
            try:
                df = df['Close']
            except:
                # 真的沒招了，直接取第一層看看
                df = df.iloc[:, 0]

# 2. 如果不是 MultiIndex，直接檢查
elif 'Adj Close' not in df.columns:
    if 'Close' in df.columns:
        print("⚠️ 找不到 'Adj Close'，改用 'Close'")
        df = df['Close']
    else:
        # 萬一連 Close 都沒有，直接拿第一欄數據
        df = df.iloc[:, 0]
else:
    # 正常情況
    df = df['Adj Close']

# 3. 確保最後是單純的 Series 格式，並且移除空值
if isinstance(df, pd.DataFrame):
    # 如果還是 DataFrame (例如多個 ticker)，只取第一欄
    df = df.iloc[:, 0]
    
df = df.dropna()

# 分割訓練集與測試集 (80% Train, 20% Test)
train_size = int(len(df) * 0.8)
train_data, test_data = df[:train_size], df[train_size:]

print(f"📊 訓練集筆數: {len(train_data)}, 測試集筆數: {len(test_data)}")

# ==========================================
# 2. 定態檢定 (ADF Test) - 模仿文章流程
# ==========================================
# 教授很愛看這個，代表你有做統計檢定
def adf_test(timeseries):
    print("\n🔍 執行 ADF 定態檢定 (Augmented Dickey-Fuller Test):")
    result = adfuller(timeseries, autolag='AIC')
    print(f'   ADF Statistic: {result[0]:.4f}')
    print(f'   p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("   ✅ 資料是定態的 (Stationary)")
    else:
        print("   ⚠️ 資料是非定態的 (Non-Stationary) -> ARIMA 會自動做差分處理")

adf_test(train_data)

# ==========================================
# 3. 自動尋找最佳參數 (Auto-ARIMA)
# ==========================================
print("\n🤖 正在執行 Auto-ARIMA 尋找最佳參數 (可能需要一點時間)...")

# 這裡設定 m=1 (非季節性)，因為每日股價很難抓年週期，設太複雜會跑不動
model = pm.auto_arima(train_data,
                      start_p=1, start_q=1,
                      max_p=5, max_q=5,
                      m=1,              
                      d=None,           # 讓模型自動判斷差分次數
                      seasonal=False,   # 股票通常不開季節性
                      start_P=0, D=0,
                      trace=True,       # 顯示過程
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

print(f"\n🏆 最佳模型參數: {model.order}")
print(model.summary()) # 印出統計報表

# ==========================================
# 4. 模型預測 (Prediction)
# ==========================================
print(f"\n🔮 正在預測未來 {len(test_data)} 天...")
preds, conf_int = model.predict(n_periods=len(test_data), return_conf_int=True)
preds = pd.Series(preds, index=test_data.index)

# ==========================================
# 5. 轉換為漲跌訊號並計算 F1-Score
# ==========================================
# 邏輯：比較「預測價」與「前一日真實收盤價」
# 如果 預測價 > 昨日收盤價 -> 預測漲 (1)
# 如果 預測價 <= 昨日收盤價 -> 預測跌 (0)

# 取得前一日收盤價 (為了比較漲跌)
prev_close = pd.concat([train_data.iloc[-1:], test_data.iloc[:-1]])
prev_close = prev_close.values.flatten()  # 轉成一維陣列

# 真實漲跌 (Ground Truth)
actual_trend = np.where(test_data.values.flatten() > prev_close, 1, 0)

# 預測漲跌 (Predicted Trend)
pred_trend = np.where(preds.values > prev_close, 1, 0)

# 計算指標
f1 = f1_score(actual_trend, pred_trend)
acc = accuracy_score(actual_trend, pred_trend)

print("\n" + "="*30)
print(f"📊 {target_symbol} ARIMA 基準測試結果")
print("="*30)
print(f"🎯 F1-Score: {f1:.4f}  <-- 請拿這個跟你的 LSTM 比較")
print(f"🎯 Accuracy: {acc:.4f}")
print("="*30)

# ==========================================
# 6. 視覺化 (Visualization) - 模仿文章畫圖
# ==========================================
plt.figure(figsize=(12, 6))
# 為了看清楚，只畫最後 200 天
subset_test = test_data[-200:]
subset_preds = preds[-200:]

plt.plot(subset_test.index, subset_test, label='Actual Price (Ground Truth)')
plt.plot(subset_preds.index, subset_preds, label='ARIMA Prediction', color='red', linestyle='--')
plt.fill_between(subset_preds.index, 
                 conf_int[-200:, 0], 
                 conf_int[-200:, 1], 
                 color='pink', alpha=0.3, label='Confidence Interval')

plt.title(f'ARIMA Baseline: {target_symbol} Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()