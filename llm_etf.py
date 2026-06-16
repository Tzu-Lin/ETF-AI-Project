# ==================== 1. 匯入所有必要的函式庫 ====================
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import os
import openai
from datetime import timedelta, datetime
from dotenv import load_dotenv
import yfinance as yf

# 機器學習：分類與迴歸
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error, r2_score

# ==================== 2. 初始化設定 ====================
load_dotenv()                     # 載入 .env 檔案中的環境變數（如 API Key）
openai.api_key = os.getenv("OPENAI_API_KEY")   # 設定 OpenAI API 金鑰

# Streamlit 頁面設定：標題、圖示、版面寬度
st.set_page_config(page_title="LLM-ETF：跨市場ETF趨勢預測平台", page_icon="📈", layout="wide")

# 自訂 CSS 樣式（美化頁面）
st.markdown("""
<style>
    .custom-title { font-size: 32px !important; font-weight: 700; margin-bottom: 10px; }
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; color: #aaaaaa; }
    .stMetric { background-color: #1E2127; padding: 10px 15px; border-radius: 8px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# ==================== 3. 資料載入與特徵工程 ====================
# 從 SQLite 資料庫讀取指定 ETF 的歷史日線資料，並計算技術指標
@st.cache_data(ttl=3600)   # 快取 1 小時，避免重複讀取
def load_and_prepare_data(ticker):
    DB_FILE = Path("etf_data.db").resolve()   # 取得資料庫絕對路徑
    if not DB_FILE.exists():
        return None
    conn = sqlite3.connect(DB_FILE)
    try:
        # 表格名稱轉換：例如 "0050.TW" -> "0050_tw"
        table_name = ticker.lower().replace('.', '_')
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    except:
        return None
    finally:
        conn.close()

    # 計算報酬率
    df["Return"] = df["Close"].pct_change()
    # 簡單移動平均線
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    # 波動率 (20 天報酬率的標準差)
    df["Volatility"] = df["Return"].rolling(20).std()

    # RSI (相對強弱指標) 計算函數
    def calc_rsi(s, period=14):
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(period).mean() / loss.rolling(period).mean()
        return 100 - (100 / (1 + rs))
    df["RSI"] = calc_rsi(df["Close"])
    # 刪除含有 NaN 的列（因計算移動平均初期會產生空值）
    df.dropna(inplace=True)
    return df

# ==================== 4. 分類模型訓練（輔助任務） ====================
# 目標：預測次日收盤價是漲還是跌（二元分類）
# 輸入特徵：MA20, MA60, Volatility, RSI（四項）
# 輸出：模型效能指標（F1, precision, recall）及全序列上漲機率
def train_classification_model(df, model_name):
    feature_cols = ["MA20", "MA60", "Volatility", "RSI"]
    X = df[feature_cols].copy()
    # 目標：次日收盤價 > 當日收盤價 -> 1 (上漲) ，否則 0 (下跌)
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    # 最後一筆資料沒有次日標籤，移除
    X = X.iloc[:-1]
    y = y.iloc[:-1]

    # 依時間順序分割 80% 訓練、20% 測試（不使用隨機打亂）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 標準化：讓每個特徵平均值為 0，標準差為 1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)   # 用於取得全序列上漲機率

    # 根據選擇的模型名稱建立對應的分類器
    if "Random Forest" in model_name:
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    elif "XGBoost" in model_name:
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    elif "AdaBoost" in model_name:
        model = AdaBoostClassifier(n_estimators=50, random_state=42)
    elif "SVR" in model_name:   # 這裡其實是 SVC（分類器）
        model = SVC(kernel='rbf', C=1e3, gamma=0.1, probability=True, random_state=42)
    else:  # Deep Learning (MLP)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # 計算評估指標
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    # 全序列上漲機率（用於圖表）
    proba_full = model.predict_proba(X_full_scaled)[:, 1]

    return {
        "name": model_name,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "full_proba": pd.Series(proba_full, index=X.index),
        "model": model,
        "scaler": scaler
    }

# ==================== 5. 迴歸模型訓練（主要任務）【已修正】 ====================
# 修正重點：
#   舊版直接用 RandomForestRegressor 預測「絕對收盤價」，因樹模型無法外推，
#   且特徵(MA/RSI/波動率)標準化後不含「現價」資訊，導致預測掉到訓練範圍內(如477)、R² 為負。
#   新版改為「預測次日報酬率」，再用今日收盤價換算回價格：
#       明日預測價 = 今日收盤價 × (1 + 預測報酬率)
#   這樣預測值必然落在今日價格附近(合理)，且 R² 反映報酬率的真實可預測性。
@st.cache_resource(ttl=3600)   # 快取模型，避免每次互動都重新訓練
def train_regression_model(df, model_name="Random Forest Regressor"):
    feature_cols = ["MA20", "MA60", "Volatility", "RSI"]
    X = df[feature_cols].copy()

    # ✅ 目標改為「次日報酬率」（而非絕對收盤價）
    y = df['Close'].pct_change().shift(-1)

    # 去除無效列（最後一筆無次日標籤，開頭可能有 NaN）
    valid = (~X.isnull().any(axis=1)) & (~y.isnull())
    X = X[valid]
    y = y[valid]

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if "Random Forest" in model_name:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "Linear Regression" in model_name:
        model = LinearRegression()
    else:  # MLPRegressor
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    model.fit(X_train_scaled, y_train)

    # 測試集評估（在「報酬率」空間計算 R² / MSE）
    y_pred_test = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)

    # ✅ 預測「次日報酬率」，再用今日收盤價換算回明日價格
    last_features = X.iloc[[-1]]
    last_scaled = scaler.transform(last_features)
    pred_return = model.predict(last_scaled)[0]          # 預測的次日報酬率
    today_close = df['Close'].iloc[-1]                   # 今日(最新)收盤價
    next_day_price = today_close * (1 + pred_return)     # 換算回價格

    return {
        "r2": r2,
        "mse": mse,
        "pred_return": pred_return,
        "next_day_price": next_day_price,
        "model": model,
        "scaler": scaler
    }

# ==================== 6. AI 風險摘要（OpenAI GPT） ====================
# 根據迴歸預測結果產生簡短風險提示
def get_llm_summary(ticker, next_price, r2, mae):
    if not openai.api_key:
        return "⚠️ 未設定 OpenAI API Key，無法產生摘要。"
    prompt = f"""
    你是 ETF 投資分析師。請根據以下數據給出 30 字以內的風險摘要（繁體中文）：
    - ETF {ticker} 明日預測收盤價：{next_price:.2f}
    - 迴歸模型 R²：{r2:.3f}
    - 平均絕對誤差：{mae:.2f}
    請只回傳一句話，不要廢話。
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"摘要生成失敗：{e}"

# ==================== 7. 台股 vs 美股走勢對比（QQQ vs 0050.TW） ====================
# 抓取 QQQ 與 0050.TW 的收盤價，正規化至基期 100，方便比較相對強弱
@st.cache_data(ttl=3600)
def fetch_tw_us_comparison():
    try:
        qqq_data = load_and_prepare_data("QQQ")
        tw_data = load_and_prepare_data("0050.TW")
        if qqq_data is None or tw_data is None:
            return None, None
        # 取兩者共同日期區間
        common_idx = qqq_data.index.intersection(tw_data.index)
        qqq_close = qqq_data.loc[common_idx, 'Close']
        tw_close = tw_data.loc[common_idx, 'Close']
        # 正規化：第一天的價格設為 100
        qqq_norm = qqq_close / qqq_close.iloc[0] * 100
        tw_norm = tw_close / tw_close.iloc[0] * 100
        return qqq_norm, tw_norm
    except:
        return None, None

# ==================== 8. Streamlit 主界面 ====================
st.markdown('<p class="custom-title">📊 LLM-ETF：基於大語言模型輔助之跨市場ETF趨勢預測平台</p>', unsafe_allow_html=True)
st.markdown("**迴歸預測（收盤價）為主軸，分類結果（漲跌）僅供對照**")

# ------ 側邊欄設定 ------
with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_ticker = st.selectbox("投資標的 ETF", ("SPY", "QQQ", "0050.TW"))
    st.markdown("---")
    st.write("🤖 **分類模型設定 (對照組)**")
    main_model_name = st.selectbox("主圖表顯示模型 (分類)",
                                   ["Random Forest", "XGBoost", "AdaBoost", "SVR", "Deep Learning (MLP)"])
    st.info("本平台以迴歸預測次日收盤價為主，分類漲跌僅供參考。")

# 載入所選 ETF 的資料
raw_main_data = load_and_prepare_data(selected_ticker)
if raw_main_data is None:
    st.error("❌ 無法讀取資料庫。請確認 etf_data.db 存在於專案根目錄，且包含 SPY, QQQ, 0050.TW 表格。")
    st.stop()

# ------ 訓練分類模型（輔助）並顯示趨勢信心 ------
with st.spinner(f"正在訓練分類模型 {main_model_name} ..."):
    cls_result = train_classification_model(raw_main_data, main_model_name)
    proba_series = cls_result["full_proba"]
    last_proba = proba_series.iloc[-1]
    main_trend = "看漲 📈" if last_proba >= 0.5 else "看跌 📉"
    main_conf = max(last_proba, 1-last_proba) * 100

# ------ 頂部儀表板：顯示最新收盤價、分類趨勢、迴歸預測 ------
col1, col2, col3 = st.columns(3)
col1.metric(f"{selected_ticker} 最新收盤價", f"${raw_main_data['Close'].iloc[-1]:.2f}")
col2.metric("分類趨勢 (輔助)", main_trend, delta=f"信心度 {main_conf:.1f}%", delta_color="normal")

# 訓練迴歸模型並顯示預測值
with st.spinner("訓練迴歸模型 (Random Forest Regressor)..."):
    reg_result = train_regression_model(raw_main_data, "Random Forest Regressor")
# ✅ 同時顯示預測報酬率，讓數字更直觀
col3.metric("📈 迴歸預測明日收盤價",
            f"${reg_result['next_day_price']:.2f}",
            delta=f"預測報酬率 {reg_result['pred_return']*100:+.2f}%")

# 展開顯示迴歸模型詳細評估指標
with st.expander("📊 迴歸模型評估指標 (測試集，報酬率空間)"):
    st.write(f"**R² (決定係數)**: {reg_result['r2']:.4f}")
    st.write(f"**MSE (均方誤差)**: {reg_result['mse']:.6f}")
    st.write(f"**RMSE**: {np.sqrt(reg_result['mse']):.6f}")
    st.caption("註：本平台改為預測『次日報酬率』再換算回價格，避免樹模型無法外推導致的不合理預測。"
               "報酬率本身難以預測，故 R² 偏低屬正常現象，亦呼應本研究結論。")

# AI 摘要按鈕
if st.button("🤖 生成 AI 風險摘要 (OpenAI GPT)"):
    with st.spinner("AI 分析中..."):
        summary = get_llm_summary(selected_ticker, reg_result['next_day_price'], reg_result['r2'], np.sqrt(reg_result['mse']))
        st.info(summary)

st.markdown("---")

# ------ 時間範圍選擇滑桿（影響圖表顯示區間）------
time_range = st.select_slider("⏳ 圖表時間範圍", options=["1M", "6M", "1Y", "3Y", "5Y", "ALL"], value="1Y")
end_date = raw_main_data.index.max()
if time_range == "1M": start_date = end_date - timedelta(days=30)
elif time_range == "6M": start_date = end_date - timedelta(days=180)
elif time_range == "1Y": start_date = end_date - timedelta(days=365)
elif time_range == "3Y": start_date = end_date - timedelta(days=365*3)
elif time_range == "5Y": start_date = end_date - timedelta(days=365*5)
else: start_date = raw_main_data.index.min()

# ------ 分頁選單 ------
tab_chart, tab_arena, tab_data, tab_compare = st.tabs(["📈 趨勢與機率圖", "🏆 分類模型競技場", "📊 數據詳情", "🌍 台股 vs 美股走勢對比"])

# --- 分頁 1：股價走勢 + 分類上漲機率線 ---
with tab_chart:
    st.subheader(f"{selected_ticker} 收盤價走勢與分類上漲機率 (對照)")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_data = raw_main_data.loc[start_date:]
    # 主 Y 軸：股價
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], name=f"{selected_ticker} 收盤價", line=dict(color='white', width=2)), secondary_y=False)
    # 次 Y 軸：上漲機率（縮放到股價範圍內以利視覺重疊）
    proba_plot = proba_series.loc[start_date:]
    fig.add_trace(go.Scatter(x=proba_plot.index, y=proba_plot*plot_data['Close'].max(), name="上漲機率 (分類模型)", line=dict(color='#00D4FF', dash='dash')), secondary_y=False)
    fig.update_layout(height=500, template="plotly_dark", hovermode="x unified")
    fig.update_yaxes(title_text="股價 (USD)", secondary_y=False)
    st.plotly_chart(fig, use_container_width=True)

# --- 分頁 2：分類模型競技場（比較多種分類器的 F1-score）---
with tab_arena:
    st.subheader("🏆 分類模型效能比較 (F1-score) — 僅供參考")
    if st.button("🚀 開始分類模型競賽"):
        models_to_test = ["Random Forest", "XGBoost", "AdaBoost", "SVR", "Deep Learning (MLP)"]
        results = []
        progress_bar = st.progress(0)
        status = st.empty()
        for i, mname in enumerate(models_to_test):
            status.text(f"訓練 {mname} ...")
            res = train_classification_model(raw_main_data, mname)
            results.append({"模型": mname, "F1-score": res["f1"], "精確率": res["precision"], "召回率": res["recall"]})
            progress_bar.progress((i+1)/len(models_to_test))
        status.text("競賽完成！")
        df_res = pd.DataFrame(results).sort_values("F1-score", ascending=False)
        st.dataframe(df_res.style.highlight_max(subset=["F1-score"]), use_container_width=True)
        fig_bar = go.Figure(go.Bar(x=df_res["模型"], y=df_res["F1-score"], text=df_res["F1-score"].round(3), textposition="auto"))
        fig_bar.update_layout(title="分類模型 F1-score 比較 (漲跌預測)", template="plotly_dark", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

# --- 分頁 3：資料詳情（最新 20 筆原始資料）---
with tab_data:
    st.subheader(f"📋 {selected_ticker} 最近 20 筆資料")
    st.dataframe(raw_main_data.tail(20).sort_index(ascending=False), use_container_width=True)

# --- 分頁 4：台股 vs 美股走勢對比（QQQ vs 0050.TW）---
with tab_compare:
    st.subheader("🌍 台股 (0050.TW) 與美股科技 (QQQ) 近五年走勢對比")
    qqq_norm, tw_norm = fetch_tw_us_comparison()
    if qqq_norm is not None and tw_norm is not None:
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=qqq_norm.index, y=qqq_norm, name="QQQ (美股科技)", line=dict(color='#00D4FF', width=2)))
        fig_comp.add_trace(go.Scatter(x=tw_norm.index, y=tw_norm, name="0050.TW (台股)", line=dict(color='#FFA500', width=2)))
        fig_comp.update_layout(title="台股與美股科技指數價格指數化比較 (基期=100)", xaxis_title="日期", yaxis_title="指數 (基期100)", template="plotly_dark", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("圖表顯示 QQQ (Nasdaq-100) 與 0050.TW 近五年收盤價走勢，正規化至相同起始點 100，方便比較相對強弱。")
    else:
        st.warning("無法取得台股或美股科技數據，請確認資料庫包含 0050.TW 與 QQQ 表格。")