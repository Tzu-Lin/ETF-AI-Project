# =============================================================
#  ARIMA 迴歸基準 — 預測次日收盤價
#  驗證: rolling forecast (逐日滾動, ARIMA 標準作法)
#  輸出: 最佳參數 / 訓練集MSE / 測試集MSE / 測試集MAPE(%)
#  資料: etf_data.db    需要: pmdarima statsmodels scikit-learn
# =============================================================
import sqlite3, warnings
import pandas as pd, numpy as np
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

DB = "etf_data.db"
MARKETS = ["SPY", "QQQ", "0050.TW"]


def mape(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return np.mean(np.abs((y - p) / y)) * 100


def load_close(ticker):
    conn = sqlite3.connect(DB)
    t = ticker.lower().replace(".", "_")
    df = pd.read_sql_query(f'SELECT Date, Close FROM "{t}"', conn,
                           index_col="Date", parse_dates=["Date"])
    conn.close()
    return df["Close"].sort_index().dropna()


def adf_test(series, name):
    r = adfuller(series, autolag="AIC")
    status = "定態" if r[1] <= 0.05 else "非定態 (需差分)"
    print(f"   [{name}] ADF p-value = {r[1]:.4f}  ->  {status}")


results = []
print("=" * 60)
print("ARIMA 迴歸基準 — rolling forecast (讀 etf_data.db)")
print("=" * 60)
for ticker in MARKETS:
    s = load_close(ticker)
    n = int(len(s) * 0.8)
    train, test = s.iloc[:n], s.iloc[n:]
    print(f"\n● {ticker}  (訓練 {len(train)} / 測試 {len(test)})")
    adf_test(train, ticker)

    model = pm.auto_arima(train, start_p=1, start_q=1, max_p=5, max_q=5,
                          d=None, seasonal=False, stepwise=True,
                          error_action="ignore", suppress_warnings=True)

    train_pred = np.asarray(model.predict_in_sample())
    train_mse = mean_squared_error(train.values, train_pred)

    # rolling: 每步用真實值更新後預測下一天
    preds = []
    for actual in test.values:
        preds.append(np.asarray(model.predict(n_periods=1))[0])
        model.update(actual)
    preds = np.asarray(preds)
    test_mse = mean_squared_error(test.values, preds)
    test_mape = mape(test.values, preds)

    print(f"   最佳參數 ARIMA{model.order}")
    print(f"   訓練集 MSE  = {train_mse:.4f}")
    print(f"   測試集 MSE  = {test_mse:.4f}")
    print(f"   測試集 MAPE = {test_mape:.2f}%")
    results.append({"市場": ticker, "參數": str(model.order),
                    "訓練集MSE": round(train_mse, 4), "測試集MSE": round(test_mse, 4),
                    "測試集MAPE(%)": round(test_mape, 2)})

print("\n" + "=" * 60 + "\n彙總表\n" + "=" * 60)
dfres = pd.DataFrame(results)
print(dfres.to_string(index=False))
dfres.to_csv("arima_regression_results.csv", index=False, encoding="utf-8-sig")
print("\n-> 已存 arima_regression_results.csv")