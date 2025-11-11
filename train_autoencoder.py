# ==============================================================================
# 檔案名稱: train_autoencoder.py (最終清潔版)
# ==============================================================================

# --- 步驟 0: 匯入所有必要的函式庫 ---
import pandas as pd
import pandas_ta as ta
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, GRU, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam
import sys

# (此處省略了所有函式定義，因為它們都正確無誤，直接將它們放在主程式碼區下方)

def load_data_from_db(ticker, db_path='etf_data.db'):
    # ... (此函式內容不變) ...
    print(f"正在從資料庫讀取 {ticker} 的資料...")
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(f"SELECT Date, Open, High, Low, Close, Volume FROM {ticker}", conn)
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        print("資料讀取成功！")
        return df
    except Exception as e:
        print(f"讀取資料失敗: {e}")
        return None
    finally:
        if 'conn' in locals() and conn:
            conn.close()

def create_features_with_pandas_ta(df):
    # ... (此函式內容不變) ...
    if df is None: return None
    print("正在使用 pandas-ta 計算技術指標...")
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.sma(length=20, append=True)
    df.ta.sma(length=60, append=True)
    df['return'] = df['close'].pct_change()
    print("技術指標計算完成！")
    print(f"清理前的資料筆數: {len(df)}")
    df.dropna(inplace=True)
    print(f"清理(dropna)後的資料筆數: {len(df)}")
    return df

def build_autoencoder(model_type='lstm', timesteps=30, n_features=4, learning_rate=0.01):
    # ... (此函式內容不變) ...
    inputs = Input(shape=(timesteps, n_features))
    if model_type == 'lstm':
        encoder = LSTM(64, activation='relu')(inputs)
        decoder = RepeatVector(timesteps)(encoder)
        decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
    elif model_type == 'gru':
        encoder = GRU(64, activation='relu')(inputs)
        decoder = RepeatVector(timesteps)(encoder)
        decoder = GRU(64, activation='relu', return_sequences=True)(decoder)
    else:
        raise ValueError("model_type must be 'lstm' or 'gru'")
    output = TimeDistributed(Dense(n_features))(decoder)
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mae')
    return model

def create_sequences(data, time_steps=30):
    # ... (此函式內容不變) ...
    X = []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
    return np.array(X)

# ==============================================================================
# 主程式執行區
# ==============================================================================
if __name__ == "__main__":
    
    ticker_to_analyze = 'SPY'
    ohlcv_df = load_data_from_db(ticker_to_analyze)
    feature_df = create_features_with_pandas_ta(ohlcv_df)

    if feature_df is not None:
        
        # --- 步驟 3: 特徵選擇與【自動清潔】 ---
        
        # 【關鍵步驟】：自動清潔 feature_df 的所有欄位名稱，移除前後隱形空格
        feature_df.columns = [col.strip() for col in feature_df.columns]
        
        # 我們「想要」使用的欄位名稱列表 (這裡也用.strip()清潔一遍，確保萬無一失)
        features_to_use = [
            'return'.strip(), 
            'RSI_14'.strip(), 
            'MACDh_12_26_9'.strip(),
            'BBM_20_2.0_2.0'.strip(),
            'ATRr_14'.strip(),
            'OBV'.strip()
        ]
        
        # 自我檢查
        available_columns = feature_df.columns
        missing_columns = [col for col in features_to_use if col not in available_columns]
        
        if missing_columns:
            print("\n" + "="*70)
            print("!!! 錯誤：即使在清潔後，仍然發現欄位名稱不匹配 !!!")
            print(f"以下 {len(missing_columns)} 個欄位在您的資料中找不到：")
            print(missing_columns)
            print("\n" + "-"*30 + " 請參考以下「已清潔」的可用欄位 " + "-"*30)
            print("所有可用的欄位名稱為：")
            print(list(available_columns))
            print("\n請根據上方「可用欄位」列表，修正 `features_to_use` 列表。")
            print("="*70)
            sys.exit()
            
        print(f"\n--- 所有特徵均已找到，已選定 {len(features_to_use)} 個特徵進行訓練 ---")
        print(features_to_use)
        
        data_to_train = feature_df[features_to_use]

        # --- 步驟 4: 數據標準化 ---
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_to_train)
        print(f"\n數據已完成標準化，數據形狀: {data_scaled.shape}")

        # --- 步驟 5: 建立滑動窗口序列 ---
        TIME_STEPS = 30
        X = create_sequences(data_scaled, TIME_STEPS)
        print(f"滑動窗口序列已建立，最終訓練資料形狀 (樣本數, 時間步長, 特徵數): {X.shape}")

        # 切分訓練集與驗證集
        split_index = int(X.shape[0] * 0.8)
        X_train, X_val = X[:split_index], X[split_index:]
        print(f"訓練集大小: {X_train.shape}")
        print(f"驗證集大小: {X_val.shape}")

        # --- 步驟 6: 模型訓練與比較 ---
        N_FEATURES = X.shape[2]
        
        print("\n--- 正在訓練 LSTM Autoencoder ---")
        lstm_autoencoder = build_autoencoder(model_type='lstm', timesteps=TIME_STEPS, n_features=N_FEATURES, learning_rate=0.01)
        lstm_history = lstm_autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), shuffle=False)

        print("\n--- 正在訓練 GRU Autoencoder ---")
        gru_autoencoder = build_autoencoder(model_type='gru', timesteps=TIME_STEPS, n_features=N_FEATURES, learning_rate=0.01)
        gru_history = gru_autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), shuffle=False)
        
        # --- 步驟 7: 視覺化比較訓練結果 ---
        print("\n--- 正在繪製訓練過程比較圖 ---")
        plt.figure(figsize=(12, 6))
        plt.plot(lstm_history.history['val_loss'], label='LSTM Validation Loss')
        plt.plot(gru_history.history['val_loss'], label='GRU Validation Loss')
        plt.title('Model Validation Loss Comparison')
        plt.ylabel('Loss (MAE)')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()