# =============================================================
#  GPT (gpt-4o-mini) 迴歸 — 預測次日收盤價
#  對齊: 各市場 TimeSeriesSplit 最後一折的測試期間與樣本數
#  三市場自動跑完，各自輸出 MSE/RMSE/MAE/R²/MAPE，並存 CSV
# =============================================================
import sqlite3, json, time, os
import pandas as pd, numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# ========== 參數 ==========
DB_PATH = "etf_data.db"
WINDOW  = 60
# ★ 各市場對齊「最後一折」的起始日與筆數（來自 test.py 的結果）
MARKET_CONFIG = {
    "spy":     {"start": "2024-10-28", "n": 415},
    "qqq":     {"start": "2024-10-28", "n": 415},
    "0050_tw": {"start": "2024-10-30", "n": 401},
}

def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100

def load_and_feature(ticker):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT Date, Close FROM '{ticker}' ORDER BY Date ASC",
                           conn, parse_dates=["Date"])
    conn.close()
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))
    logret = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = logret.rolling(20).std() * np.sqrt(252)
    return df.dropna().reset_index(drop=True)

def build_prompt(features_last_5):
    lines = ["日期(相對), 收盤價, MA20, MA60, RSI, 波動率"]
    for i, row in enumerate(features_last_5):
        lines.append(f"T-{5-i}, {row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f}, {row[3]:.2f}, {row[4]:.4f}")
    table = "\n".join(lines)
    return f"""你是一個股票技術分析專家。以下是某 ETF 過去 5 個交易日的技術指標數據：

{table}

請根據這些數據，預測**下一個交易日**的收盤價（一個具體數值）。
請只回傳一個 JSON 物件，格式如下：
{{"predicted_close": 數值}}
不要有任何額外文字或解釋。"""

def run_market(ticker, cfg):
    df = load_and_feature(ticker)
    # 生成滑動窗口樣本
    X, y, dates = [], [], []
    for i in range(len(df) - WINDOW):
        feat = df.iloc[i:i+WINDOW][["Close", "MA20", "MA60", "RSI", "Volatility"]].values
        X.append(feat); y.append(df.iloc[i+WINDOW]["Close"])
        dates.append(df.iloc[i+WINDOW-1]["Date"])

    # 對齊最後一折：起始日之後，取 n 筆
    start = pd.Timestamp(cfg["start"])
    test_idx = [i for i, d in enumerate(dates) if d >= start][:cfg["n"]]
    print(f"\n===== {ticker.upper()} | 測試樣本 {len(test_idx)} 筆（起始 {cfg['start']}）=====")

    results = []
    for k, idx in enumerate(test_idx, 1):
        last_5 = X[idx][-5:]
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": build_prompt(last_5)}],
                temperature=0.7, response_format={"type": "json_object"})
            pred = json.loads(resp.choices[0].message.content).get("predicted_close")
        except Exception as e:
            print(f"  樣本 {idx} 失敗: {e}"); pred = None
        results.append({"date": dates[idx].strftime("%Y-%m-%d"),
                        "actual": y[idx], "pred": pred, "ok": pred is not None})
        if k % 20 == 0: print(f"  進度 {k}/{len(test_idx)}")
        time.sleep(0.4)

    valid = [r for r in results if r["ok"]]
    yt = [r["actual"] for r in valid]; yp = [r["pred"] for r in valid]
    mse = mean_squared_error(yt, yp)
    metrics = {"市場": ticker.upper(), "有效樣本": len(valid), "總樣本": len(test_idx),
               "MSE": round(mse, 2), "RMSE": round(np.sqrt(mse), 2),
               "MAE": round(mean_absolute_error(yt, yp), 2),
               "R2": round(r2_score(yt, yp), 4), "MAPE(%)": round(mape(yt, yp), 2)}
    print(f"  -> 有效 {len(valid)}/{len(test_idx)} | MAPE={metrics['MAPE(%)']}% | R²={metrics['R2']}")
    pd.DataFrame(results).to_csv(f"llm_regression_{ticker}.csv", index=False, encoding="utf-8-sig")
    return metrics

all_metrics = [run_market(tk, cfg) for tk, cfg in MARKET_CONFIG.items()]
summary = pd.DataFrame(all_metrics)
summary.to_csv("llm_regression_summary.csv", index=False, encoding="utf-8-sig")
print("\n========== GPT 迴歸總結（對齊最後一折）==========")
print(summary.to_string(index=False))
print("\n-> 已存 llm_regression_summary.csv")