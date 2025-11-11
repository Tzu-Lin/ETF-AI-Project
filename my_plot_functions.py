
import pandas as pd
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
# (假設您已經訓練好了模型 model，並有了預測結果 X_pred)
# mae_loss = np.mean(np.abs(X_pred - X_train), axis=(1, 2))

# --- 找到一個低誤差（正常）和一個高誤差（異常）的樣本 ---
normal_idx = np.argmin(mae_loss) # 找到重建誤差最小的那個窗口
anomaly_idx = np.argmax(mae_loss) # 找到重建誤差最大的那個窗口

# --- 繪製對比圖 ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# 圖一：正常樣本重建
feature_to_plot = 0 # 假設我們只畫第一個特徵 (例如 'Return')
ax1.plot(X_train[normal_idx, :, feature_to_plot], label='Original (Normal)')
ax1.plot(X_pred[normal_idx, :, feature_to_plot], label='Reconstructed')
ax1.set_title(f'Normal Period (Low Reconstruction Error: {mae_loss[normal_idx]:.4f})')
ax1.legend()

# 圖二：異常樣本重建
ax2.plot(X_train[anomaly_idx, :, feature_to_plot], label='Original (Anomaly)')
ax2.plot(X_pred[anomaly_idx, :, feature_to_plot], label='Reconstructed')
ax2.set_title(f'Anomalous Period (High Reconstruction Error: {mae_loss[anomaly_idx]:.4f})')
ax2.legend()

plt.show()