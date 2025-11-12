# run_experiments.py (重構升級版)

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 從 models.py 導入所有模型類別
from models import (
    RandomForestModel, 
    SingleLayerLSTM, 
    DoubleLayerLSTM, 
    SingleLayerBiLSTM, 
    DoubleLayerBiLSTM
)

# --- 函式定義區 (與您版本相同，稍作優化) ---

def load_data_from_sqlite(ticker, db_path='etf_data.db'):
    """從 SQLite 資料庫讀取特定 ETF 的資料"""
    print(f"\n{'='*20} 正在處理 {ticker} {'='*20}")
    conn = sqlite3.connect(db_path)
    table_name = ticker.lower().replace('.tw', '_tw')
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        conn.close()
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        print(f"資料讀取完畢，共 {len(df)} 筆。")
        return df
    except Exception as e:
        print(f"讀取表格 {table_name} 失敗: {e}")
        conn.close()
        return None

def feature_engineering(df):
    """執行特徵工程"""
    print("正在進行特徵工程...")
    df['Return'] = df['Close'].pct_change()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Target'] = (df['Return'].shift(-1) > 0).astype(int)
    df.dropna(inplace=True)
    
    features = ['Return', 'MA20', 'MA60', 'RSI']
    X = df[features]
    y = df['Target']
    
    return X, y

def create_sequences(X, y, time_steps=30):
    """為深度學習模型創建時間序列數據集"""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- 主程式執行區 ---

if __name__ == "__main__":
    
    TICKERS = ["SPY", "QQQ", "0050.TW"]
    TIME_STEPS = 30 # 滑動窗口大小
    all_results = []

    for ticker in TICKERS:
        raw_df = load_data_from_sqlite(ticker)
        if raw_df is None:
            continue
            
        X_raw, y_raw = feature_engineering(raw_df)

        # --- 資料準備與切割 ---
        # 1. 先切割訓練集和測試集，避免數據洩漏
        split_point = int(len(X_raw) * 0.8)
        X_train_raw, X_test_raw = X_raw[:split_point], X_raw[split_point:]
        y_train_raw, y_test_raw = y_raw[:split_point], y_raw[split_point:]

        # 2. 用訓練集的參數來標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 3. 創建給深度學習模型的 3D 時序數據
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw.values, TIME_STEPS)
        
        # 傳統機器學習模型不能使用 3D 數據，我們需要把它"壓平"
        # 我們取每個序列的最後一天作為 RandomForest 的輸入
        X_train_rf = X_train_scaled[TIME_STEPS:]
        y_train_rf = y_train_raw.values[TIME_STEPS:]
        X_test_rf = X_test_scaled[TIME_STEPS:]
        y_test_rf = y_test_raw.values[TIME_STEPS:]

        # --- 模型配置字典 ---
        # 在這裡定義所有你想跑的模型和它們需要的數據
        input_shape_seq = (X_train_seq.shape[1], X_train_seq.shape[2])
        
        models_to_run = {
            "RandomForest": {
                "model": RandomForestModel(n_estimators=100),
                "X_train": X_train_rf, "y_train": y_train_rf,
                "X_test": X_test_rf, "y_test": y_test_rf
            },
            "SingleLayerLSTM": {
                "model": SingleLayerLSTM(input_shape=input_shape_seq),
                "X_train": X_train_seq, "y_train": y_train_seq,
                "X_test": X_test_seq, "y_test": y_test_seq
            },
            "DoubleLayerLSTM": {
                "model": DoubleLayerLSTM(input_shape=input_shape_seq),
                "X_train": X_train_seq, "y_train": y_train_seq,
                "X_test": X_test_seq, "y_test": y_test_seq
            },
            "SingleLayerBiLSTM": {
                "model": SingleLayerBiLSTM(input_shape=input_shape_seq),
                "X_train": X_train_seq, "y_train": y_train_seq,
                "X_test": X_test_seq, "y_test": y_test_seq
            },
            "DoubleLayerBiLSTM": {
                "model": DoubleLayerBiLSTM(input_shape=input_shape_seq),
                "X_train": X_train_seq, "y_train": y_train_seq,
                "X_test": X_test_seq, "y_test": y_test_seq
            }
        }
        
        # --- 自動化執行迴圈 ---
        for name, config in models_to_run.items():
            print(f"--- 正在為 {ticker} 訓練 {name} ---")
            
            # 從配置中獲取模型和對應的數據
            model = config["model"]
            xtrain, ytrain = config["X_train"], config["y_train"]
            xtest, ytest = config["X_test"], config["y_test"]
            
            # 統一的訓練和預測流程
            model.train(xtrain, ytrain)
            pred, prob_up, prob_down = model.predict(xtest)
            
            acc = accuracy_score(ytest, pred)
            
            # 獲取對"明天"的預測機率 (測試集的最後一筆)
            tomorrow_up_prob = np.ravel(prob_up)[-1]
            tomorrow_down_prob = np.ravel(prob_down)[-1]

            all_results.append({
                "Ticker": ticker, 
                "Model": name, 
                "Accuracy": acc,
                "Tomorrow_Up_Prob": tomorrow_up_prob,
                "Tomorrow_Down_Prob": tomorrow_down_prob
            })
            print(f"{name} 準確率: {acc:.4f} | 明日預測: 上漲機率 {tomorrow_up_prob:.2%}, 下跌機率 {tomorrow_down_prob:.2%}")

    # --- 儲存最終結果 ---
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("experiment_results.csv", index=False)
    print("\n🎉 所有實驗完成！結果已儲存至 experiment_results.csv")
    print(results_df)