import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def run_anomaly_detection(ticker="SPY"):
    """
    對指定的 ETF 進行基於孤立森林的異常偵測。
    流程包含：載入資料 -> 建立特徵 -> 訓練模型 -> 標記異常 -> 視覺化
    """
    print(f"--- 開始對 {ticker} 進行異常偵測 ---")
    
    # --- 步驟零：資料載入 ---
    DB_FILE = Path("etf_data.db")
    if not DB_FILE.exists():
        print(f"錯誤：找不到資料庫檔案 {DB_FILE}。請先執行 update_database.py。")
        return

    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {ticker}", conn, index_col='Date', parse_dates=['Date'])
    finally:
        conn.close()

    if df.empty:
        print(f"錯誤：從資料庫中找不到 {ticker} 的資料。")
        return
        
    # --- 步驟一：特徵建立 ---
    # 目標：建立能夠反映市場「行為」的每日特徵
    print("步驟 1: 正在建立行為特徵...")

    # 1. 日報酬率: 捕捉價格的劇烈變動
    df['Return'] = df['Close'].pct_change()

    # 2. 成交量變化率: 相對於前一日的成交量變化，捕捉市場參與度的異常
    df['Volume_Change'] = df['Volume'].pct_change()

    # 3. 日內波幅率: (當日最高價 - 最低價) / 收盤價，衡量多空交戰的激烈程度，並進行標準化
    df['Intraday_Range_Pct'] = (df['High'] - df['Low']) / df['Close']
    
    # 因為計算變化率會產生空值(NaN)，例如第一天沒有前一天可比較，所以要移除
    df.dropna(inplace=True)

    # 選擇要餵給模型的特徵
    features = ['Return', 'Volume_Change', 'Intraday_Range_Pct']
    X = df[features]
    print(f"特徵建立完成，使用特徵: {features}")

    # --- 步驟二：模型訓練與產出異常分數 ---
    print("步驟 2: 正在訓練 IsolationForest 模型...")
    
    # 建立模型
    # contamination 參數代表我們預期數據中有多少比例是異常值，例如 0.01 = 1%
    # random_state 確保每次執行的結果都一樣，方便研究重現
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    
    # 訓練模型
    model.fit(X)
    
    # 產出結果
    # decision_function 會給出異常分數，分數越低代表越異常
    # predict 會直接給出判斷結果：1 代表正常(inlier), -1 代表異常(outlier)
    df['Anomaly_Score'] = model.decision_function(X)
    df['Anomaly_Flag'] = model.predict(X)
    print("模型訓練與預測完成。")
    
    # --- 步驟三：結果分析與視覺化 ---
    print("步驟 3: 正在分析與視覺化結果...")

    # 篩選出被模型標記為異常的日期
    anomalies = df[df['Anomaly_Flag'] == -1]

    print(f"\n模型在 {ticker} 的歷史數據中共偵測到 {len(anomalies)} 個異常日：")
    # 顯示分數最低（最異常）的五個日期
    print(anomalies.sort_values('Anomaly_Score').head())

    # 繪製視覺化圖表
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(15, 7))
    
    # 畫出完整的收盤價曲線
    plt.plot(df.index, df['Close'], color='blue', label='SPY Close Price', alpha=0.8)
    
    # 在圖上用紅色圓點標記出異常日
    plt.scatter(anomalies.index, anomalies['Close'], color='red', s=50, label='Detected Anomaly', zorder=5)
    
    plt.title(f'{ticker} Price with Detected Anomalies (Isolation Forest)', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USD)', fontsize=12)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # 您可以修改這裡來分析不同的 ETF
    run_anomaly_detection(ticker="SPY")