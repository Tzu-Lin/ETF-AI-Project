
# 使用 OpenAI GPT 預測次日漲跌（分類任務）

import sqlite3
import pandas as pd
import numpy as np
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

# 載入環境變數（OPENAI_API_KEY）
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ========== 參數設定 ==========
TICKER = "spy"                # 可選 spy, qqq, 0050_tw
DB_PATH = "etf_data.db"
WINDOW = 60                   # 使用過去 60 天
TEST_START_DATE = "2022-01-01"
MAX_SAMPLES = 200             # 測試樣本數（可調整）

# ========== 1. 讀取資料 ==========
conn = sqlite3.connect(DB_PATH)
query = f"SELECT Date, Close FROM '{TICKER}' ORDER BY Date ASC"
df = pd.read_sql_query(query, conn, parse_dates=["Date"])
conn.close()
df = df.dropna(subset=["Close"]).reset_index(drop=True)

# ========== 2. 計算技術指標 ==========
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA60"] = df["Close"].rolling(window=60).mean()

# RSI (14日)
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))

# 波動率 (20日年化)
log_return = np.log(df["Close"] / df["Close"].shift(1))
df["Volatility"] = log_return.rolling(window=20).std() * np.sqrt(252)

df = df.dropna().reset_index(drop=True)

# ========== 3. 生成滑動窗口樣本（分類標籤：次日漲跌） ==========
X = []      # 特徵 (60天 × 5指標)
y = []      # 標籤 (1=上漲，0=下跌)
dates = []  # 窗口最後一天日期

for i in range(len(df) - WINDOW):
    feat = df.iloc[i:i+WINDOW][["Close", "MA20", "MA60", "RSI", "Volatility"]].values
    # 標籤：第61天收盤價 > 第60天收盤價
    label = 1 if df.iloc[i+WINDOW]["Close"] > df.iloc[i+WINDOW-1]["Close"] else 0
    X.append(feat)
    y.append(label)
    dates.append(df.iloc[i+WINDOW-1]["Date"])

print(f"總共生成了 {len(X)} 個樣本")

# ========== 4. 篩選測試集（2022 年以後） ==========
test_indices = [i for i, d in enumerate(dates) if d >= pd.Timestamp(TEST_START_DATE)]
if not test_indices:
    raise ValueError(f"沒有找到 {TEST_START_DATE} 之後的測試樣本")
print(f"測試集樣本數: {len(test_indices)}（從 {TEST_START_DATE} 開始）")

# 只取前 MAX_SAMPLES 個
test_indices = test_indices[:MAX_SAMPLES]
print(f"本次實驗將使用前 {len(test_indices)} 個測試樣本")

# ========== 5. 定義提示詞（分類任務） ==========
def build_prompt(features_last_5):
    """
    features_last_5: 最近 5 天的特徵陣列，形狀 (5,5)
    回傳提示詞，要求模型輸出 JSON 格式的漲跌預測
    """
    lines = ["日期(相對), 收盤價, MA20, MA60, RSI, 波動率"]
    for i, row in enumerate(features_last_5):
        lines.append(f"T-{5-i}, {row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}, {row[3]:.2f}, {row[4]:.4f}")
    table = "\n".join(lines)

    prompt = f"""
你是一個股票技術分析專家。以下是某 ETF 過去 5 個交易日的技術指標數據（過去 60 天的整體趨勢與最近 5 天類似）：

{table}

請根據這些數據，預測**下一個交易日**收盤價相對於今天收盤價是「上漲」還是「下跌」。

請只回傳一個 JSON 物件，不要有任何額外文字，格式如下：
{{"prediction": "up" 或 "down", "confidence": 0 到 100 之間的整數}}
"""
    return prompt

# ========== 6. 呼叫 OpenAI API 取得預測 ==========
results = []

for idx in test_indices:
    features_60 = X[idx]
    last_5 = features_60[-5:]   # 只取最後 5 天，節省 token
    prompt = build_prompt(last_5)

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",   # 可改用 gpt-4o-mini
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        pred = 1 if data.get("prediction") == "up" else 0
        confidence = data.get("confidence", 50)
    except Exception as e:
        print(f"樣本 {idx} 呼叫失敗: {e}")
        pred = -1
        confidence = 0

    actual = y[idx]
    results.append({
        "index": idx,
        "date": dates[idx].strftime("%Y-%m-%d"),
        "actual": actual,
        "predicted": pred,
        "confidence": confidence,
        "success": pred != -1
    })
    print(f"進度: {len(results)}/{len(test_indices)}  日期={dates[idx].strftime('%Y-%m-%d')} 實際={actual} 預測={pred}")

    time.sleep(0.5)   # 避免超過速率限制

# ========== 7. 評估分類指標 ==========
valid = [r for r in results if r["success"]]
if valid:
    y_true = [r["actual"] for r in valid]
    y_pred = [r["predicted"] for r in valid]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n========== LLM 分類預測結果 ==========")
    print(f"有效樣本數: {len(valid)} / {len(test_indices)}")
    print(f"準確率 (Accuracy): {acc:.3f}")
    print(f"精確率 (Precision): {prec:.3f}")
    print(f"召回率 (Recall): {rec:.3f}")
    print(f"F1-score: {f1:.3f}")

    print("\n混淆矩陣:")
    print("         預測下跌  預測上漲")
    print(f"實際下跌  {cm[0,0]}        {cm[0,1]}")
    print(f"實際上漲  {cm[1,0]}        {cm[1,1]}")

    # 儲存詳細結果
    df_results = pd.DataFrame(results)
    df_results.to_csv(f"llm_classification_{TICKER}.csv", index=False)
    print(f"\n詳細結果已儲存至 llm_classification_{TICKER}.csv")
else:
    print("沒有成功的預測，請檢查 API 設定")