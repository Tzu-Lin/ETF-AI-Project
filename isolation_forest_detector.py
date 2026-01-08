# isolation_forest_detector.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# --- 1. 生成帶有類孤立子波包的測試信號 ---
np.random.seed(42)
T = 1000  # 總時間點
time = np.arange(T)
# 背景雜訊
noise = 0.2 * np.random.randn(T)

# 創建兩個類孤立子波包 (使用高斯函數近似)
def gaussian_pulse(t, center, width, amplitude):
    return amplitude * np.exp(-((t - center)**2) / (2 * width**2))

pulse1 = gaussian_pulse(time, center=200, width=10, amplitude=3)
pulse2 = gaussian_pulse(time, center=750, width=15, amplitude=4)

# 合成最終信號
signal = noise + pulse1 + pulse2

# --- 2. 使用孤立森林進行異常偵測 ---
# IsolationForest 需要2D輸入, 所以我們需要重塑信號
X = signal.reshape(-1, 1)

# 初始化並訓練模型
# contamination 參數告訴模型數據中大約有多少是異常值
# 這裡我們估計兩個波包佔總長度的 (2*~30)/1000 ~= 0.06
model = IsolationForest(contamination=0.06, random_state=42)
model.fit(X)

# 進行預測 (-1 代表異常, 1 代表正常)
predictions = model.predict(X)
anomaly_indices = np.where(predictions == -1)[0]

# --- 3. 視覺化結果 ---
plt.figure(figsize=(15, 6))
plt.plot(time, signal, label='Original Signal', color='blue', alpha=0.7)
# 用紅色散點標記出被偵測為異常的點
plt.scatter(anomaly_indices, signal[anomaly_indices], 
            color='red', marker='o', s=50, label='Detected Anomalies (Soliton-like)')

plt.title("Detecting Soliton-like Patterns using Isolation Forest")
plt.xlabel("Time")
plt.ylabel("Signal Value")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()