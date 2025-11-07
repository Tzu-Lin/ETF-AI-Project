import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense

# --- 步驟 1: 載入並準備數據 ---

# 從資料庫讀取資料 (與您之前的程式碼相同)
DB_FILE = Path("etf_data.db")
TICKER = "SPY"
conn = sqlite3.connect(DB_FILE)
df = pd.read_sql_query(f"SELECT * FROM {TICKER}", conn, index_col='Date', parse_dates=['Date'])
conn.close()

# 選擇用於異常偵測的特徵
features = ['Close', 'Volume', 'High', 'Low'] 
data = df[features].values

# 數據標準化 (非常重要！神經網路對數據尺度敏感)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- 步驟 2: 建立滑動窗口 ---

def create_sequences(data, time_steps):
    """將數據轉換成 (樣本數, 時間步長, 特徵數) 的格式"""
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

TIME_STEPS = 30  # 設定窗口大小為 30 天
N_FEATURES = data_scaled.shape[1] # 特徵數量

X_train = create_sequences(data_scaled, TIME_STEPS)
print(f"訓練數據維度: {X_train.shape}") # 應該會是 (樣本數, 30, 4)

# --- 步驟 3: 定義並訓練模型 ---

# 這部分就是您已經完成的 model.summary() 的源頭
inputs = Input(shape=(TIME_STEPS, N_FEATURES))
encoder = LSTM(64, activation='relu')(inputs)
decoder = RepeatVector(TIME_STEPS)(encoder)
decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
output = TimeDistributed(Dense(N_FEATURES))(decoder)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mae')

# 顯示模型架構 (您已經看到這步的結果了)
print("\n--- 模型架構 ---")
model.summary()

# 【【【 核心訓練步驟 】】】
# 這一步會花一些時間，模型正在學習什麼是「正常」
print("\n--- 開始訓練模型 (這可能需要幾分鐘) ---")
history = model.fit(
    X_train, X_train,
    epochs=10, # 為了快速演示，只訓練10輪。實際應用中可能需要更多。
    batch_size=32,
    validation_split=0.1,
    shuffle=False # 時間序列數據通常不打亂
)
print("模型訓練完成！")


# --- 步驟 4: 預測與計算重建誤差 ---

print("\n--- 正在用模型進行預測（重建數據） ---")
X_pred = model.predict(X_train)

# 計算每個時間窗口的平均絕對誤差 (MAE) 作為重建誤差
mae_loss = np.mean(np.abs(X_pred - X_train), axis=(1, 2))

# --- 步驟 5: 找出正常與異常樣本 ---

normal_idx = np.argmin(mae_loss)  # 找到誤差最小的窗口索引
anomaly_idx = np.argmax(mae_loss) # 找到誤差最大的窗口索引

print(f"\n找到誤差最小的窗口 (索引 {normal_idx})，誤差值: {mae_loss[normal_idx]:.4f}")
print(f"找到誤差最大的窗口 (索引 {anomaly_idx})，誤差值: {mae_loss[anomaly_idx]:.4f}")

# --- 步驟 6: 繪製對比圖 ---

print("\n--- 正在繪製正常 vs. 異常樣本重建對比圖 ---")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
plt.suptitle('LSTM Autoencoder Reconstruction Comparison', fontsize=16)

# 選擇要繪製的特徵（0=Close, 1=Volume, ...）
FEATURE_TO_PLOT = 0 
feature_name = features[FEATURE_TO_PLOT]

# 圖一：正常樣本重建
ax1.plot(X_train[normal_idx, :, FEATURE_TO_PLOT], 'b', label='Original Data')
ax1.plot(X_pred[normal_idx, :, FEATURE_TO_PLOT], 'r--', label='Reconstructed Data')
ax1.set_title(f'Normal Period (Low Error: {mae_loss[normal_idx]:.4f})')
ax1.set_xlabel('Time Steps (Days)')
ax1.set_ylabel(f'Scaled {feature_name}')
ax1.legend()

# 圖二：異常樣本重建
ax2.plot(X_train[anomaly_idx, :, FEATURE_TO_PLOT], 'b', label='Original Data')
ax2.plot(X_pred[anomaly_idx, :, FEATURE_TO_PLOT], 'r--', label='Reconstructed Data')
ax2.set_title(f'Anomalous Period (High Error: {mae_loss[anomaly_idx]:.4f})')
ax2.set_xlabel('Time Steps (Days)')
ax2.legend()

# 【【【 顯示圖表的關鍵指令 】】】
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()