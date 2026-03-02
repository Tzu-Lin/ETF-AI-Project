import yfinance as yf
import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = Path("etf_data.db") 
TICKERS = ["SPY", "QQQ", "SSO", "QLD", "0050.TW", "KRBN"]

def update_data_to_db():
    conn = sqlite3.connect(DB_FILE)
    print(f"成功連接到資料庫: {DB_FILE}")

    for ticker in TICKERS:
        try:
            print(f"正在從 yfinance 下載 {ticker} 的資料 (十年期)...")
            
            # --- 修改處：將 start 改為 2015-01-01 (約十年) 或直接用 period="10y" ---
            # 使用 period="10y" 會自動抓取從今天回推十年的所有資料
            df = yf.download(ticker, period="10y", auto_adjust=True)

            if df.empty:
                print(f"⚠️  警告: {ticker} 沒有下載到任何資料，已跳過。")
                continue

            # 檢查欄位是否為 MultiIndex 格式 (yfinance 新版本常見問題)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 將日期索引轉換成欄位
            df.index.name = 'Date'
            df.reset_index(inplace=True)
            
            # 建立表格名稱
            table_name = ticker.lower().replace('.', '_')
            
            # 確保欄位名稱標準化
            df.columns = [col.capitalize() for col in df.columns]

            # 寫入資料庫 (if_exists='replace' 會洗掉舊的資料，換成新的十年份)
            df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
            
            # 印出抓取到的日期範圍，讓你自己確認
            print(f"✅ {ticker} 更新成功！資料範圍: {df['Date'].min().date()} 至 {df['Date'].max().date()}")

        except Exception as e:
            print(f"❌ 更新 {ticker} 資料時發生錯誤: {e}")

    conn.close()
    print("--- 資料庫更新作業完成 ---")

if __name__ == "__main__":
    update_data_to_db()