# run_experiments.py (【偵錯版】)

import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 從 models.py 導入所有模型類別
from models import (
    RandomForestModel, 
    SingleLayerLSTM, 
    DoubleLayerLSTM, 
    SingleLayerBiLSTM, 
    DoubleLayerBiLSTM
)

# --- 函式定義區 ---

def load_data_from_sqlite(ticker, db_path='etf_data.db'):
    """從 SQLite 資料庫讀取特定 ETF 的資料"""
    print(f"\n{'='*20} 正在處理 {ticker} {'='*20}")
    conn = sqlite3.connect(db_path)
    table_name = ticker.lower().replace('.tw', '_tw')
    try:
        df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
        conn.close()
        # 【偵錯點 1】: 確認原始讀取筆數
        print(f"【偵錯 1】: 從資料庫成功讀取 {table_name}，原始筆數: {len(df)}")
        if df.empty:
            print("【警告】: 讀取到的 DataFrame 為空！")
            return None
            
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
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
    
    # 【偵錯點 2】: 檢查 dropna 前的空值數量
    print(f"【偵錯 2】: 執行 dropna 之前，各欄位空值(NaN)數量:\n{df.isnull().sum()}")
    
    df.dropna(inplace=True)
    
    # 【偵錯點 3】: 確認 dropna 後的剩餘筆數 (最關鍵！)
    print(f"【偵錯 3】: 執行 dropna 之後，剩餘的資料筆數: {len(df)}")
    
    if df.empty:
        print("【嚴重錯誤】: dropna 後沒有任何數據剩餘！無法繼續處理。")
        return None, None # 回傳空值

    features = ['Return', 'MA20', 'MA60', 'RSI']
    X = df[features]
    y = df['Target']
    
    return X, y

# ... (create_sequences 函式不變) ...
def create_sequences(X, y, time_steps=30):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- 主程式執行區 ---

if __name__ == "__main__":
    
    TICKERS = ["SPY", "QQQ", "0050.TW"]
    TIME_STEPS = 30
    all_results = []

    for ticker in TICKERS:
        raw_df = load_data_from_sqlite(ticker)
        if raw_df is None:
            continue
            
        X_raw, y_raw = feature_engineering(raw_df)

        # 【偵錯點 4】: 檢查特徵工程的輸出
        if X_raw is None or y_raw is None:
            print(f"--- 因為 {ticker} 的數據在特徵工程後為空，已跳過 ---")
            continue # 跳到下一個 Ticker

        print(f"【偵錯 4】: 特徵工程成功，準備切割資料。X_raw shape: {X_raw.shape}, y_raw shape: {y_raw.shape}")

        # --- 資料準備與切割 ---
        split_point = int(len(X_raw) * 0.8)

        X_train_raw, X_test_raw = X_raw[:split_point], X_raw[split_point:]
        y_train_raw, y_test_raw = y_raw[:split_point], y_raw[split_point:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, TIME_STEPS)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw.values, TIME_STEPS)
        
        X_train_rf = X_train_scaled[TIME_STEPS:]
        y_train_rf = y_train_raw.values[TIME_STEPS:]
        X_test_rf = X_test_scaled[TIME_STEPS:]
        y_test_rf = y_test_raw.values[TIME_STEPS:]

        input_shape_seq = (X_train_seq.shape[1], X_train_seq.shape[2])
        
        models_to_run = {
            "RandomForest": {"model": RandomForestModel(n_estimators=100),"X_train": X_train_rf, "y_train": y_train_rf,"X_test": X_test_rf, "y_test": y_test_rf},
            "SingleLayerLSTM": {"model": SingleLayerLSTM(input_shape=input_shape_seq),"X_train": X_train_seq, "y_train": y_train_seq,"X_test": X_test_seq, "y_test": y_test_seq},
            "DoubleLayerLSTM": {"model": DoubleLayerLSTM(input_shape=input_shape_seq),"X_train": X_train_seq, "y_train": y_train_seq,"X_test": X_test_seq, "y_test": y_test_seq},
            "SingleLayerBiLSTM": {"model": SingleLayerBiLSTM(input_shape=input_shape_seq),"X_train": X_train_seq, "y_train": y_train_seq,"X_test": X_test_seq, "y_test": y_test_seq},
            "DoubleLayerBiLSTM": {"model": DoubleLayerBiLSTM(input_shape=input_shape_seq),"X_train": X_train_seq, "y_train": y_train_seq,"X_test": X_test_seq, "y_test": y_test_seq}
        }
        
        for name, config in models_to_run.items():
            print(f"--- 正在為 {ticker} 訓練 {name} ---")
            model = config["model"]
            xtrain, ytrain = config["X_train"], config["y_train"]
            xtest, ytest = config["X_test"], config["y_test"]
            model.train(xtrain, ytrain)
            predictions, prob_up, prob_down = model.predict(xtest)
            acc = accuracy_score(ytest, predictions)
            precision = precision_score(ytest, predictions, zero_division=0)
            recall = recall_score(ytest, predictions, zero_division=0)
            f1 = f1_score(ytest, predictions, zero_division=0)
            tomorrow_up_prob = np.ravel(prob_up)[-1]
            tomorrow_down_prob = np.ravel(prob_down)[-1]
            all_results.append({"Ticker": ticker, "Model": name, "Accuracy": acc,"Precision": precision,"Recall": recall,"F1-Score": f1, "Tomorrow_Up_Prob": tomorrow_up_prob,"Tomorrow_Down_Prob": tomorrow_down_prob})
            print(f"{name} -> Accuracy: {acc:.4f}, F1-Score: {f1:.4f} | 明日預測: 上漲機率 {tomorrow_up_prob:.2%}")

    results_df = pd.DataFrame(all_results, columns=['Ticker', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Tomorrow_Up_Prob', 'Tomorrow_Down_Prob'])
    results_df.to_csv("experiment_results.csv", index=False)
    print("\n🎉 所有實驗完成！結果已儲存至 experiment_results.csv")
    print(results_df)