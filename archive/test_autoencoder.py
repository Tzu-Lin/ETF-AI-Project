import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping # 引入EarlyStopping

# --- 步驟 1: 載入並準備數據 ---

# 從資料庫讀取資料
DB_FILE = Path("etf_data.db")
TICKER = "SPY"

# 【防呆檢查 1】: 確保資料庫檔案存在
if not DB_FILE.exists():
    raise FileNotFoundError(f"錯誤：找不到資料庫檔案 {DB_FILE}。請先執行 update_database.py。")

conn = sqlite3.connect(DB_FILE)
df = pd.read_sql_query(f"SELECT * FROM {TICKER}", conn, index_col='Date', parse_dates=['Date'])
conn.close()

# 【防呆檢查 2】: 確保 DataFrame 不是空的
if df.empty:
    raise ValueError(f"錯誤：從資料庫中找不到 {TICKER} 的資料或資料為空。")

print(f"成功從資料庫載入 {len(df)} 筆 {TICKER} 資料。")

# 選擇用於異常偵測的特徵
features = ['Close', 'Volume', 'High', 'Low'] 
data = df[features].values

# 數據標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# --- 步驟 2: 建立滑動窗口 ---

def create_sequences(data, time_steps):
    X = []
    # 【修正】: 確保循環的邊界是正確的
    for i in range(len(data) - time_steps + 1):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

TIME_STEPS = 30
N_FEATURES = data_scaled.shape[1]

X_train = create_sequences(data_scaled, TIME_STEPS)

# 【防呆檢查 3】: 檢查 X_train 是否成功生成且維度正確
if X_train.size == 0:
    raise ValueError("錯誤：X_train 生成失敗，為空陣列。請檢查數據長度是否足夠大於 TIME_STEPS。")

print(f"訓練數據維度: {X_train.shape}")

# --- 步驟 3: 定義、編譯並訓練模型 ---

inputs = Input(shape=(TIME_STEPS, N_FEATURES))
encoder = LSTM(64, activation='relu')(inputs)
decoder = RepeatVector(TIME_STEPS)(encoder)
decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
output = TimeDistributed(Dense(N_FEATURES))(decoder)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mae')

print("\n--- 模型架構 ---")
model.summary()

print("\n--- 開始訓練模型 ---")

# 【優化】: 加入 EarlyStopping 回調函數
# 當驗證損失在 2 個 epoch 內不再下降時，就提前停止訓練，防止過擬合且節省時間
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)

# 訓練模型，並將訓練歷史儲存到 'history' 變數中
history = model.fit(
    X_train, X_train,
    epochs=20, # 稍微增加 epochs，讓 early stopping 有機會觸發
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stopping] # 應用 EarlyStopping
)
print("模型訓練完成！")


# --- 步驟 4: 繪製損失函數曲線圖 (Loss Curve) ---

print("\n--- 正在繪製模型訓練歷史曲線 ---")
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Training History (Loss over Epochs)')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.grid(True)
plt.show()


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