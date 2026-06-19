# =============================================================
#  ARIMA 迴歸基準 — 預測次日收盤價
#  驗證: TimeSeriesSplit (5 折) + 每折內 rolling forecast (逐日 1 步)
#        → 與樹模型 / LSTM 迴歸統一驗證結構, 公平比較
#  輸出: 各折參數 / 平均訓練MSE / 平均測試MSE / 平均測試MAPE(%)
#  資料: etf_data.db    需要: pmdarima statsmodels scikit-learn
# =============================================================
import sqlite3, warnings
import pandas as pd, numpy as np
import pmdarima as pm
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
warnings.filterwarnings("ignore")

DB = "etf_data.db"
MARKETS = ["SPY", "QQQ", "0050.TW"]
N_SPLITS = 5


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
print("ARIMA 迴歸基準 — TimeSeriesSplit(5 折) + 每折 rolling forecast")
print("=" * 60)

for ticker in MARKETS:
    s = load_close(ticker)
    print(f"\n● {ticker}  (總樣本 {len(s)})")
    adf_test(s, ticker)                      # 對整段序列做一次 ADF 檢定

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_train_mse, fold_test_mse, fold_test_mape, orders = [], [], [], []

    for k, (tr, te) in enumerate(tscv.split(s.values), start=1):
        train, test = s.iloc[tr], s.iloc[te]

        model = pm.auto_arima(train, start_p=1, start_q=1, max_p=5, max_q=5,
                              d=None, seasonal=False, stepwise=True,
                              error_action="ignore", suppress_warnings=True)
        orders.append(str(model.order))

        # 訓練集 in-sample MSE
        train_pred = np.asarray(model.predict_in_sample())
        fold_train_mse.append(mean_squared_error(train.values, train_pred))

        # 測試集: 每折內逐日 1 步滾動, 用真值更新
        preds = []
        for actual in test.values:
            preds.append(np.asarray(model.predict(n_periods=1))[0])
            model.update(actual)
        preds = np.asarray(preds)
        fold_test_mse.append(mean_squared_error(test.values, preds))
        fold_test_mape.append(mape(test.values, preds))

        print(f"   折 {k}: ARIMA{model.order}  訓練MSE={fold_train_mse[-1]:.4f}  "
              f"測試MSE={fold_test_mse[-1]:.4f}  測試MAPE={fold_test_mape[-1]:.2f}%")

    avg_train_mse = np.mean(fold_train_mse)
    avg_test_mse  = np.mean(fold_test_mse)
    avg_test_mape = np.mean(fold_test_mape)
    print(f"   ──> 五折平均  訓練MSE={avg_train_mse:.4f}  "
          f"測試MSE={avg_test_mse:.4f}  測試MAPE={avg_test_mape:.2f}%")

    results.append({"市場": ticker,
                    "各折參數": " / ".join(orders),
                    "平均訓練MSE": round(avg_train_mse, 4),
                    "平均測試MSE": round(avg_test_mse, 4),
                    "平均測試MAPE(%)": round(avg_test_mape, 2)})

print("\n" + "=" * 60 + "\n彙總表\n" + "=" * 60)
dfres = pd.DataFrame(results)
print(dfres.to_string(index=False))
dfres.to_csv("arima_regression_results.csv", index=False, encoding="utf-8-sig")
print("\n-> 已存 arima_regression_results.csv")