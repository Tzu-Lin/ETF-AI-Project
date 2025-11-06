# check_db.py

import sqlite3
import pandas as pd
from pathlib import Path

DB_FILE = Path("etf_data.db")

def check_table_schema(ticker="SPY"):
    if not DB_FILE.exists():
        print(f"錯誤：找不到資料庫檔案 {DB_FILE}！")
        return

    print(f"正在檢查資料庫 '{DB_FILE}' 中的資料表 '{ticker}'...")
    conn = sqlite3.connect(DB_FILE)
    
    try:
        # 讀取資料表的前 5 筆資料
        df = pd.read_sql_query(f"SELECT * FROM {ticker} LIMIT 5", conn)
        
        print("\n--- 資料表前 5 筆內容 ---")
        print(df)
        
        print("\n--- 偵測到的欄位名稱 ---")
        print(list(df.columns))
        
    except Exception as e:
        print(f"\n讀取資料表時發生錯誤: {e}")
        print("請確認您已經成功執行 update_database.py，並且資料庫中有 'SPY' 這張資料表。")
    finally:
        conn.close()

if __name__ == "__main__":
    check_table_schema()