# new_project.py

# --- 步驟 1: 匯入 (Import) ---
# 從 detector_class_version.py 這個檔案中，匯入我們封裝好的 LstmAutoencoderAnomalyDetector 這個類別
from detector_class_version import LstmAutoencoderAnomalyDetector
import pandas as pd
import numpy as np
import sqlite3

# --- 步驟 2: 使用 (Use) ---

# 假設您想對 QQQ 進行分析
# 1. 像往常一樣準備您的數據
print("正在載入 QQQ 的數據...")
DB_FILE = "etf_data.db"
TICKER = "QQQ" 
conn = sqlite3.connect(DB_FILE)
df_qqq = pd.read_sql_query(f"SELECT * FROM {TICKER}", conn, index_col='Date', parse_dates=['Date'])
conn.close()
features = ['Close', 'Volume', 'High', 'Low']
data_qqq = df_qqq[features].values

# 2. 建立偵測器物件
# 您可以像使用 scikit-learn 的模型一樣，輕鬆建立一個偵測器
# 這裡我們使用自訂的參數，例如窗口設為 60 天
print("正在建立 QQQ 的異常偵測器...")
qqq_detector = LstmAutoencoderAnomalyDetector(time_steps=60)

# 3. 訓練模型
# 呼叫 .fit() 方法，所有複雜的訓練流程都會在背景自動完成
print("正在訓練 QQQ 的模型...")
qqq_detector.fit(data_qqq, epochs=10) # 為了演示，只訓練10輪

# 4. 進行預測
# 呼叫 .predict() 方法，得到重建誤差
print("正在用訓練好的模型進行預測...")
qqq_mae_loss, _, _ = qqq_detector.predict(data_qqq)

# 5. 分析結果
# 現在您就可以用 qqq_mae_loss 這個陣列來做任何您想做的事
print("\nQQQ 數據的重建誤差分析：")
print(f"最大誤差 (最異常): {np.max(qqq_mae_loss):.4f}")
print(f"最小誤差 (最正常): {np.min(qqq_mae_loss):.4f}")
print(f"平均誤差: {np.mean(qqq_mae_loss):.4f}")