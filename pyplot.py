import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 1. 生成模擬數據
time = np.linspace(0, 24, 144) # 模擬 24 小時
normal_traffic = 100 + 30 * np.sin(time) + np.random.normal(0, 5, 144)

# 2. 注入 DDoS 攻擊 (模擬異常能量激增)
normal_traffic[80:90] += 200 
data = normal_traffic.reshape(-1, 1)

# 3. 使用 Isolation Forest 進行偵測
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(data)
anomalies = model.predict(data)

# 4. 繪圖
plt.figure(figsize=(10, 5))
plt.plot(time, normal_traffic, label='Normal Traffic (Low Energy)', color='blue', alpha=0.6)
# 標記異常點
plt.scatter(time[anomalies==-1], normal_traffic[anomalies==-1], c='red', s=50, label='DDoS Attack (High Energy)', zorder=5)

plt.title("Traffic Anomaly Detection (Hopfield Energy View)")
plt.xlabel("Time (Hours)")
plt.ylabel("Requests / Min")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()