import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import warnings

# 忽略不必要的警告
warnings.filterwarnings('ignore')

print("--- 正在執行實作 11：1D-CNN ---")

# 1. 產生數據 (與前面一致，模擬 120 個月的數據)
np.random.seed(42)
raw_data = 10 + 0.05 * np.arange(120) + 2 * np.sin(2 * np.pi * np.arange(120) / 12) + np.random.normal(0, 0.3, 120)

# 2. 資料預處理：轉換為滑動視窗格式 (Window Size = 12)
X, y = [], []
for i in range(12, len(raw_data)):
    X.append(raw_data[i-12:i])
    y.append(raw_data[i])
X, y = np.array(X), np.array(y)

# --- 重要！CNN 需要 3D 輸入: [樣本數, 時間步, 特徵數] ---
X = X.reshape((X.shape[0], X.shape[1], 1))

X_train, X_test = X[:-12], X[-12:]
y_train, y_test = y[:-12], y[-12:]

# 3. 建立 1D-CNN 模型
model_cnn = Sequential([
    # 第一層卷積：提取局部特徵
    Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(12, 1)),
    # 池化層：降低數據維度，保留重要特徵
    MaxPooling1D(pool_size=2),
    # 攤平層：將多維轉一維
    Flatten(),
    # 全連接層與輸出
    Dense(50, activation='relu'),
    Dense(1)
])

model_cnn.compile(optimizer='adam', loss='mse')

# 4. 訓練模型 (為了展示，訓練 20 次)
print("模型訓練中...")
history = model_cnn.fit(X_train, y_train, epochs=20, verbose=0)

# 5. 預測
pred_cnn = model_cnn.predict(X_test, verbose=0)

print("\n【1D-CNN 預測結果 (未來12個月)】")
print(pred_cnn.flatten())

# 6. 繪製預測圖 (截圖放簡報)
plt.figure(figsize=(10, 5))
plt.plot(range(12), y_test, label='Actual Values', marker='o')
plt.plot(range(12), pred_cnn, label='CNN Predicted', color='red', linestyle='--', marker='x')
plt.title("Algorithm 11: 1D-CNN Forecasting")
plt.xlabel("Month (Test Data)")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()