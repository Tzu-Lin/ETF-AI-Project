import yfinance as yf
import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = Path("etf_data.db") 
TICKERS = ["SPY", "QQQ", "SSO", "QLD"]

def update_data_to_db():
    conn = sqlite3.connect(DB_FILE)
    print(f"成功連接到資料庫: {DB_FILE}")

    for ticker in TICKERS:
        try:
            print(f"正在從 yfinance 下載 {ticker} 的資料...")
            df = yf.download(ticker, start="2019-01-01", auto_adjust=True)

            if df.empty:
                print(f"⚠️  警告: {ticker} 沒有下載到任何資料，已跳過。")
                continue

            # 檢查欄位是否為 MultiIndex 格式
            if isinstance(df.columns, pd.MultiIndex):
                print(f"偵測到 {ticker} 的欄位為 MultiIndex，正在進行強制扁平化...")
                # 將 ('Close', 'SPY') 這樣的欄位，強制只取第一層 'Close'
                df.columns = df.columns.get_level_values(0)

            # 將日期索引轉換成一個真正的欄位
            df.index.name = 'Date'
            df.reset_index(inplace=True)
            
            # 再次確保所有欄位名稱都是首字母大寫的標準格式
            df.columns = [col.capitalize() for col in df.columns]

            # 寫入資料庫
            df.to_sql(name=ticker, con=conn, if_exists='replace', index=False)
            
            print(f"✅ 成功將 {ticker} 的資料更新至資料庫。")

        except Exception as e:
            print(f"❌ 更新 {ticker} 資料時發生錯誤: {e}")

    conn.close()
    print("資料庫連接已關閉。")

if __name__ == "__main__":
    update_data_to_db()