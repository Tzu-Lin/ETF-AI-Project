# 檔案名稱: run_anomaly_experiment.py
# 功能: 比較加入了批次正規化的 LSTM 與 GRU Autoencoder 在異常偵測任務上的表現。

# --- 0. 匯入所有必要的函式庫 ---
import pandas as pd
import numpy as np
import sqlite3
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 從 tensorflow.keras 匯入所需的所有層和工具
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, RepeatVector, TimeDistributed, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

# --- 1. 資料準備相關函式 ---

def load_data_from_db(ticker="SPY", db_file="etf_data.db"):
    """從指定的 SQLite 資料庫讀取 ETF 資料。"""
    print(f"--- 步驟 1: 正在從資料庫 '{db_file}' 讀取 '{ticker}' 的資料 ---")
    db_path = Path(db_file)
    if not db_path.exists():
        raise FileNotFoundError(f"錯誤: 找不到資料庫檔案 '{db_file}'。請先執行資料更新腳本!")
    
    conn = sqlite3.connect(db_file)
    # 將 'Date' 欄位設為索引，並解析為日期格式
    df = pd.read_sql_query(f"SELECT * FROM {ticker}", conn, index_col='Date', parse_dates=['Date'])
    conn.close()
    print("資料讀取完成。")
    return df

def create_sequences(data, time_steps=30):
    """將時間序列數據轉換為監督式學習所需的滑動窗口格式。"""
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

# --- 2. 穩定版模型建構函式 (使用 Batch Normalization) ---

def build_autoencoder_stable(model_type='lstm', timesteps=30, n_features=4, learning_rate=0.01):
    """
    建立一個包含批次正規化的 Autoencoder 模型，可選擇 LSTM 或 GRU。
    這個版本旨在解決高學習率下使用 ReLU 激活函數時的梯度爆炸問題。
    """
    inputs = Input(shape=(timesteps, n_features))

    # --- 編碼器 (Encoder) ---
    if model_type == 'lstm':
        # 將激活函數從層定義中移除，以便後續處理
        encoder_rnn = LSTM(64, return_sequences=False)(inputs)
    elif model_type == 'gru':
        encoder_rnn = GRU(64, return_sequences=False)(inputs)
    else:
        raise ValueError("model_type 必須是 'lstm' 或 'gru'")
    
    # 在 RNN 層後加入批次正規化
    encoder_bn = BatchNormalization()(encoder_rnn)
    # 再單獨加入激活函數
    encoder_act = Activation('relu')(encoder_bn)

    # --- 解碼器 (Decoder) ---
    decoder = RepeatVector(timesteps)(encoder_act)
    
    if model_type == 'lstm':
        decoder_rnn = LSTM(64, return_sequences=True)(decoder)
    elif model_type == 'gru':
        decoder_rnn = GRU(64, return_sequences=True)(decoder)
        
    decoder_bn = BatchNormalization()(decoder_rnn)
    decoder_act = Activation('relu')(decoder_bn)

    # 輸出層
    output = TimeDistributed(Dense(n_features))(decoder_act)
    
    model = Model(inputs=inputs, outputs=output)
    
    # 實例化 Adam 優化器並設定學習率
    optimizer = Adam(learning_rate=learning_rate)
    
    # 編譯模型
    model.compile(optimizer=optimizer, loss='mae')
    
    return model

# --- 3. 主實驗執行流程 ---

if __name__ == "__main__":
    # --- 參數設定 ---
    TICKER = "SPY"
    TIME_STEPS = 30
    FEATURES = ['Close', 'High', 'Low', 'Volume']
    N_FEATURES = len(FEATURES)
    LEARNING_RATE = 0.01 # 依照老師建議的學習率
    EPOCHS = 50 # 訓練 50 輪以觀察完整的收斂過程
    
    # --- 步驟 1 & 2: 載入並準備資料 ---
    df = load_data_from_db(ticker=TICKER)
    data = df[FEATURES]

    # 標準化數據
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # 建立滑動窗口序列
    X = create_sequences(data_scaled, TIME_STEPS)
    
    # 切分訓練集和測試集
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    print("\n--- 步驟 2: 資料準備完成 ---")
    print(f"訓練集形狀: {X_train.shape}")
    print(f"測試集形狀: {X_test.shape}")

    # --- 步驟 3: 進行模型比較實驗 ---
    models_to_run = ['lstm', 'gru']
    histories = {}      # 用於儲存每個模型的訓練歷史
    test_mae_loss = {}  # 用於儲存每個模型在測試集上的最終誤差

    for model_type in models_to_run:
        print(f"\n--- 步驟 3: 正在訓練 {model_type.upper()} Autoencoder 模型 ---")
        
        # 建立模型
        model = build_autoencoder_stable(
            model_type=model_type, 
            timesteps=TIME_STEPS, 
            n_features=N_FEATURES, 
            learning_rate=LEARNING_RATE
        )
        model.summary()

        # 訓練模型 (Autoencoder 的輸入和目標輸出是相同的)
        history = model.fit(
            X_train, X_train,
            epochs=EPOCHS,
            batch_size=32,
            validation_split=0.1, # 使用 10% 的訓練數據做驗證
            shuffle=True,
            verbose=1
        )
        histories[model_type] = history
        
        # 評估模型在從未見過的測試集上的表現
        loss = model.evaluate(X_test, X_test, verbose=0)
        test_mae_loss[model_type] = loss
        print(f"--- {model_type.upper()} 模型在測試集上的重建誤差 (MAE): {loss:.4f} ---")

    # --- 4. 結果總結與視覺化 ---
    print("\n\n--- 步驟 4: 實驗結果總結 ---")
    for model_type, loss in test_mae_loss.items():
        print(f"模型 {model_type.upper()} 的最終測試集 MAE: {loss:.4f}")
        
    # 繪製損失函數曲線比較圖
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    for model_type, history in histories.items():
        plt.plot(history.history['val_loss'], label=f'{model_type.upper()} Validation Loss')
    
    plt.title('Stable LSTM vs GRU Autoencoder Validation Loss Comparison (with Batch Normalization)')
    plt.ylabel('Loss (MAE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    print("\n正在顯示損失函數比較圖...")
    plt.show()