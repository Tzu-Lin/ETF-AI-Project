import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam #外加

class LstmAutoencoderAnomalyDetector:
    """
    一個用於時間序列異常偵測的 LSTM Autoencoder 類別。
    """
    def __init__(self, time_steps=30, latent_dim=64):
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

    def _create_sequences(self, data):
        """私有方法：建立滑動窗口"""
        X = []
        for i in range(len(data) - self.time_steps + 1):
            X.append(data[i:(i + self.time_steps)])
        return np.array(X)

    def build_model(self, n_features):
        inputs = Input(shape=(self.time_steps, n_features))
         # --- 1. 增強 Encoder ---
        # 將神經元數量從 64 增加到 128，提升學習能力
        encoder = LSTM(128, activation='tanh')(inputs)
        # 在 Encoder 的輸出後加入 Dropout，強力防止過擬合
        encoder = Dropout(0.3)(encoder) # 0.3 是一個比較強的 Dropout 率
        # --- 2. 增強 Decoder ---
        decoder = RepeatVector(self.time_steps)(encoder)
        # 同樣增加神經元數量
        decoder = LSTM(128, activation='tanh', return_sequences=True)(decoder)
        # 在 Decoder 的輸出後也加入 Dropout
        decoder = Dropout(0.3)(decoder)
        output = TimeDistributed(Dense(n_features))(decoder)
        self.model = Model(inputs=inputs, outputs=output)
        # self.model.compile(optimizer='adam', loss='mae')
        optimizer = Adam(learning_rate=0.0001) 
        self.model.compile(optimizer=optimizer, loss='mae') # 替換對抗過擬合
        # 使用 Adam 優化器，但將其學習率從預設的 0.001 降低到 0.0001
        print("--- 模型架構 ---")
        self.model.summary()
        

    def fit(self, data, epochs=20, batch_size=32):
        """訓練模型"""
        # 1. 數據標準化
        data_scaled = self.scaler.fit_transform(data)
        
        # 2. 建立滑動窗口
        X_train = self._create_sequences(data_scaled)
        if X_train.size == 0:
            raise ValueError("數據長度不足以建立滑動窗口。")
        print(f"訓練數據維度: {X_train.shape}")

        # 3. 建立模型
        n_features = X_train.shape[2]
        self.build_model(n_features)
        
        # 4. 訓練模型
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)
        print("\n--- 開始訓練模型 ---")
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            shuffle=False,
            callbacks=[early_stopping]
        )
        print("模型訓練完成！")
        return self.history

    def predict(self, data):
        """進行預測並計算重建誤差"""
        data_scaled = self.scaler.transform(data)
        sequences = self._create_sequences(data_scaled)
        reconstructions = self.model.predict(sequences)
        mae_loss = np.mean(np.abs(reconstructions - sequences), axis=(1, 2))
        return mae_loss, sequences, reconstructions

    def plot_training_history(self):
        """繪製訓練歷史曲線"""
        if self.history:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MAE)')
            plt.legend()
            plt.grid(True)
            plt.show()

# --- 主執行流程 ---
if __name__ == "__main__":
    # 1. 載入數據
    DB_FILE = Path("etf_data.db")
    TICKER = "SPY"
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(f"SELECT * FROM {TICKER}", conn, index_col='Date', parse_dates=['Date'])
    conn.close()
    
    features = ['Close', 'Volume', 'High', 'Low']
    data_values = df[features].values

    # 2. 初始化並訓練模型
    detector = LstmAutoencoderAnomalyDetector(time_steps=30)
    history = detector.fit(data_values, epochs=20)
    
    # 3. 繪製訓練歷史
    detector.plot_training_history()
    
    # 4. 進行預測並找出異常
    mae_loss, sequences, reconstructions = detector.predict(data_values)
    normal_idx = np.argmin(mae_loss)
    anomaly_idx = np.argmax(mae_loss)

     # 步驟 4.4: 加入 print 語句來顯示成果 (對應您的成果預覽)
    print("\n--- 第三階段：預測與評估結果 ---")
    print(f"最大誤差 (最異常): {np.max(mae_loss):.4f}")
    print(f"最小誤差 (最正常): {np.min(mae_loss):.4f}")
    print("------------------------------------")
    
    # 5. 繪製重建對比圖
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    feature_to_plot = 0
    feature_name = features[feature_to_plot]
    
    # 正常樣本
    ax1.plot(sequences[normal_idx, :, feature_to_plot], 'b', label='Original')
    ax1.plot(reconstructions[normal_idx, :, feature_to_plot], 'r--', label='Reconstructed')
    ax1.set_title(f'Normal Period (Low Error: {mae_loss[normal_idx]:.4f})')
    ax1.legend()
    
    # 異常樣本
    ax2.plot(sequences[anomaly_idx, :, feature_to_plot], 'b', label='Original')
    ax2.plot(reconstructions[anomaly_idx, :, feature_to_plot], 'r--', label='Reconstructed')
    ax2.set_title(f'Anomalous Period (High Error: {mae_loss[anomaly_idx]:.4f})')
    ax2.legend()
    

    plt.show()
