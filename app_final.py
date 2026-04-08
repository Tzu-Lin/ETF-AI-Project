# --- 1. 匯入所有必要的函式庫 ---
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, mean_squared_error, r2_score

# --- 2. 初始化設定 ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AEGIS 氣候金融風險分析平台 v5.0 (分類主+回歸輔助)", page_icon="🌍", layout="wide")

# CSS 樣式 (沿用 V2)
st.markdown("""
<style>
    .custom-title { font-size: 32px !important; font-weight: 700; margin-bottom: 10px; }
    [data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 600; }
    [data-testid="stMetricLabel"] { font-size: 14px !important; color: #aaaaaa; }
    .stMetric { background-color: #1E2127; padding: 10px 15px; border-radius: 8px; border: 1px solid #333; }
</style>
""", unsafe_allow_html=True)

# --- 3. 資料載入與特徵工程 (與 V2 相同) ---
@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker):
    DB_FILE = Path("etf_data.db").resolve()
    if not DB_FILE.exists(): return None
    conn = sqlite3.connect(DB_FILE)
    try:
        table_name = ticker.lower().replace('.', '_')
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    except: return None
    finally: conn.close()
    
    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    
    def calc_rsi(s, period=14):
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(period).mean() / loss.rolling(period).mean()
        return 100 - (100 / (1 + rs))
    df["RSI"] = calc_rsi(df["Close"])
    df.dropna(inplace=True)
    return df

# --- 4. 分類模型訓練 (主要任務) ---
def train_classification_model(df, model_name):
    """訓練分類模型，預測次日漲跌方向，回傳 F1, precision, recall, 以及全序列預測機率"""
    feature_cols = ["MA20", "MA60", "Volatility", "RSI"]
    X = df[feature_cols].copy()
    # 目標：次日收盤價 > 當日收盤價 -> 1 (上漲)
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    # 移除最後一行（無次日標籤）
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)
    
    if "Random Forest" in model_name:
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    elif "XGBoost" in model_name:
        model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    elif "SVR" in model_name:
        model = SVC(kernel='rbf', C=1e3, gamma=0.1, probability=True, random_state=42)
    else:  # Deep Learning (MLP)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    # 全序列預測機率 (上漲機率)
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

# --- 5. 輔助回歸模型訓練 (僅在點擊時訓練) ---
@st.cache_resource(ttl=3600)
def train_regression_model(df, model_name="Random Forest Regressor"):
    """訓練迴歸模型，預測次日收盤價數值 (輔助用)"""
    feature_cols = ["MA20", "MA60", "Volatility", "RSI"]
    X = df[feature_cols].copy()
    # 目標：次日收盤價
    y = df['Close'].shift(-1)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)
    
    if "Random Forest" in model_name:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "Linear Regression" in model_name:
        model = LinearRegression()
    else:
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    
    model.fit(X_train_scaled, y_train)
    y_pred_test = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    # 最後一天的特徵預測次日收盤價
    last_features = X.iloc[[-1]]
    last_scaled = scaler.transform(last_features)
    next_day_price = model.predict(last_scaled)[0]
    
    return {
        "r2": r2,
        "mse": mse,
        "next_day_price": next_day_price,
        "model": model,
        "scaler": scaler
    }

# --- 6. 其他原有功能 (AI 報告、自選股比較等) ---
def generate_mock_response(ticker, ticker_pred, krbn_pred, correlation):
    if correlation < 0:
        hedge_text = f"KRBN 與 {ticker} 呈負相關，具備顯著氣候避險效果。"
        action = "建議配置 10-15% 資金於碳權以對沖風險。"
    else:
        hedge_text = f"KRBN 與 {ticker} 走勢同步，避險效果有限。"
        action = "建議順勢操作，關注碳價突破訊號。"
    return f"""
    (⚠️ 注意：此為 Demo 模擬分析，因 API 額度不足自動切換)
    1. 🎯 **避險判斷**：{hedge_text}
    2. ⚡ **趨勢訊號**：AI 模型預測 {ticker} 目前{ticker_pred}，且碳權市場亦{krbn_pred}，顯示氣候政策對股價有連動影響。
    3. 💡 **操作建議**：{action} 當前 RSI 指標顯示動能強勁，可分批佈局。
    """

def get_climate_gpt_summary(ticker, ticker_pred, ticker_conf, krbn_pred, krbn_conf, latest_data, correlation):
    if not openai.api_key:
        return generate_mock_response(ticker, ticker_pred, krbn_pred, correlation)
    prompt = f"""
    你是專業操盤手。請根據數據直接給出 **3 點關鍵操作結論**，**嚴禁廢話**，總字數控制在 150 字內：
    [市場數據]
    - 標的 ({ticker}) 預測：{ticker_pred} (信心 {ticker_conf}%)
    - 碳權 (KRBN) 預測：{krbn_pred} (信心 {krbn_conf}%)
    - 兩者相關係數：{correlation:.2f} (正值=同步, 負值=避險)
    - 標的 RSI：{latest_data['RSI']:.0f}
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except:
        return generate_mock_response(ticker, ticker_pred, krbn_pred, correlation)

# 自選股比較功能 (從 V3 移植)
@st.cache_data(ttl=3600)
def fetch_custom_ticker_data(ticker, start_date, end_date):
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            close_series = df.xs('Close', level=0, axis=1).iloc[:, 0]
        elif 'Close' in df.columns:
            close_series = df['Close']
        else:
            close_series = df.iloc[:, 0]
        close_series.index = pd.to_datetime(close_series.index)
        return close_series
    except:
        return None

def calculate_performance_metrics(price_series, initial_value=10000):
    if price_series is None or len(price_series) < 2: return None
    start_price = price_series.iloc[0]
    end_price = price_series.iloc[-1]
    total_return = (end_price - start_price) / start_price * 100
    daily_ret = price_series.pct_change().dropna()
    volatility = daily_ret.std() * np.sqrt(252) * 100
    cummax = price_series.cummax()
    drawdown = (price_series - cummax) / cummax
    max_drawdown = drawdown.min() * 100
    return {"total_return": total_return, "volatility": volatility, "max_drawdown": max_drawdown}

# --- 7. Streamlit 介面 ---
st.markdown('<p class="custom-title">🌍 AEGIS：氣候金融風險分析平台 (分類主 + 回歸輔助)</p>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_ticker = st.selectbox("投資標的 ETF:", ("SPY", "QQQ", "0050.TW"))
    st.markdown("---")
    st.write("🤖 **分類模型設定**")
    main_model_name = st.selectbox("主圖表顯示模型 (分類)", 
                                   ["Random Forest", "XGBoost", "SVR", "Deep Learning (MLP)"])
    st.info("系統已自動載入 KRBN (碳權) 數據進行交叉比對。")

raw_main_data = load_and_prepare_data(selected_ticker)
raw_krbn_data = load_and_prepare_data("KRBN")

if raw_main_data is None:
    st.error("❌ 無法讀取資料庫。請確認 etf_data.db 存在。")
    st.stop()

# 訓練分類模型 (主要)
with st.spinner(f"正在訓練分類模型 {main_model_name} ..."):
    cls_result = train_classification_model(raw_main_data, main_model_name)
    proba_series = cls_result["full_proba"]
    last_proba = proba_series.iloc[-1]
    main_trend = "看漲 📈" if last_proba >= 0.5 else "看跌 📉"
    main_conf = max(last_proba, 1-last_proba) * 100

# 訓練 KRBN 分類模型 (用於趨勢)
if raw_krbn_data is not None:
    krbn_cls = train_classification_model(raw_krbn_data, "Random Forest")
    krbn_proba = krbn_cls["full_proba"].iloc[-1]
    krbn_trend = "看漲 📈" if krbn_proba >= 0.5 else "看跌 📉"
    krbn_conf = max(krbn_proba, 1-krbn_proba) * 100
else:
    krbn_trend = "資料不足"
    krbn_conf = 0

# 相關係數 (收盤價)
correlation = 0.0
corr_desc = "資料不足"
corr_color = "gray"
if raw_krbn_data is not None:
    common_idx = raw_main_data.index.intersection(raw_krbn_data.index)
    if len(common_idx) > 30:
        correlation = raw_main_data.loc[common_idx, 'Close'].corr(raw_krbn_data.loc[common_idx, 'Close'])
        if correlation > 0.5:
            corr_desc = "高度正相關"
            corr_color = "#ff4b4b"
        elif correlation < -0.3:
            corr_desc = "負相關 (避險效果)"
            corr_color = "#09ab3b"
        else:
            corr_desc = "低度相關"
            corr_color = "gray"

# 頂部儀表板
col1, col2, col3, col4 = st.columns(4)
col1.metric(f"{selected_ticker} 最新價", f"${raw_main_data['Close'].iloc[-1]:.2f}")
col2.metric("AI 趨勢預測 (分類)", main_trend, delta=f"信心度 {main_conf:.1f}%", delta_color="normal")
col3.metric("KRBN 碳權趨勢", krbn_trend)
col4.markdown(f"""
<div style="text-align: center;">
    <p style="margin: 0px; font-size: 14px; color: #aaaaaa;">與碳權相關性</p>
    <p style="margin: 0px; font-size: 26px; color: {corr_color}; font-weight: 600;">{correlation:.2f}</p>
    <p style="margin: 0px; font-size: 12px; color: #888;">{corr_desc}</p>
</div>
""", unsafe_allow_html=True)

# 時間範圍選擇
st.write("###")
time_range = st.select_slider("⏳ 圖表時間範圍", options=["1M", "6M", "1Y", "3Y", "5Y", "ALL"], value="1Y")
end_date = raw_main_data.index.max()
if time_range == "1M": start_date = end_date - timedelta(days=30)
elif time_range == "6M": start_date = end_date - timedelta(days=180)
elif time_range == "1Y": start_date = end_date - timedelta(days=365)
elif time_range == "3Y": start_date = end_date - timedelta(days=365*3)
elif time_range == "5Y": start_date = end_date - timedelta(days=365*5)
else: start_date = raw_main_data.index.min()

st.markdown("---")

# 分頁
tab_chart, tab_arena, tab_ai, tab_data, tab_custom = st.tabs(["📈 雙軌趨勢圖", "🏆 模型競技場 (分類)", "🤖 AI 決策報告", "📊 數據詳情", "🔍 自選股比較"])

# Tab 1: 雙軌圖 (顯示分類的上漲機率或擬合值? 為了視覺化，仍顯示真實股價，但可加機率線)
with tab_chart:
    st.subheader(f"{selected_ticker} vs KRBN 走勢對照")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    plot_data = raw_main_data.loc[start_date:]
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], name=f"{selected_ticker} 真實股價", line=dict(color='white', width=2)), secondary_y=False)
    # 可選擇顯示上漲機率 (以陰影表示)
    proba_plot = proba_series.loc[start_date:]
    fig.add_trace(go.Scatter(x=proba_plot.index, y=proba_plot*plot_data['Close'].max(), name="上漲機率 (分類)", line=dict(color='#00D4FF', dash='dash')), secondary_y=False)
    if raw_krbn_data is not None:
        krbn_plot = raw_krbn_data.loc[start_date:]
        fig.add_trace(go.Scatter(x=krbn_plot.index, y=krbn_plot['Close'], name="KRBN 碳權", line=dict(color='orange', width=1.5, dash='dot')), secondary_y=True)
    fig.update_layout(height=500, template="plotly_dark", hovermode="x unified")
    fig.update_yaxes(title_text="股價 (USD)", secondary_y=False)
    fig.update_yaxes(title_text="碳權 (USD)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: 模型競技場 (分類 F1)
with tab_arena:
    st.subheader("🏆 分類模型效能大亂鬥 (F1-score)")
    if st.button("🚀 開始分類模型競賽"):
        models_to_test = ["Random Forest", "XGBoost", "SVR", "Deep Learning (MLP)"]
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
        fig_bar.update_layout(title="各分類模型 F1-score 比較", template="plotly_dark", height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

# Tab 3: AI 報告 (與之前相同)
with tab_ai:
    st.subheader("🤖 智能風險評估")
    if st.button("生成 AI 分析報告"):
        with st.spinner("AI 分析中..."):
            summary = get_climate_gpt_summary(selected_ticker, main_trend, main_conf, krbn_trend, krbn_conf, raw_main_data.iloc[-1], correlation)
            st.success("分析完成")
            st.info(summary)

# Tab 4: 數據詳情
with tab_data:
    st.subheader(f"📊 {selected_ticker} 詳細數據")
    st.dataframe(raw_main_data.tail(20).sort_index(ascending=False), use_container_width=True)

# Tab 5: 自選股比較
with tab_custom:
    st.subheader("🔍 自選股比較")
    custom_symbol = st.text_input("股票代碼", value="NVDA", help="美股或台股 ex: AAPL, 2330.TW")
    compare_period = st.selectbox("比較期間", ["1Y", "2Y", "3Y", "5Y"], index=0)
    end_today = datetime.today()
    if compare_period == "1Y": start_custom = end_today - timedelta(days=365)
    elif compare_period == "2Y": start_custom = end_today - timedelta(days=730)
    elif compare_period == "3Y": start_custom = end_today - timedelta(days=1095)
    else: start_custom = end_today - timedelta(days=1825)
    if st.button("📈 載入自選股", key="load_custom"):
        if custom_symbol:
            custom_series = fetch_custom_ticker_data(custom_symbol.strip(), start_custom, end_today)
            main_etf_data = raw_main_data.loc[start_custom:end_today] if raw_main_data is not None else None
            if custom_series is None or main_etf_data is None:
                st.error("無法取得資料")
            else:
                main_norm = (main_etf_data['Close'] / main_etf_data['Close'].iloc[0]) * 100
                custom_norm = (custom_series / custom_series.iloc[0]) * 100
                fig_custom = go.Figure()
                fig_custom.add_trace(go.Scatter(x=main_norm.index, y=main_norm, name=f"{selected_ticker} (正規化)", line=dict(color='#00D4FF')))
                fig_custom.add_trace(go.Scatter(x=custom_norm.index, y=custom_norm, name=f"{custom_symbol} (正規化)", line=dict(color='#FFA500')))
                fig_custom.update_layout(title="價格指數化比較 (基期=100)", template="plotly_dark", height=450)
                st.plotly_chart(fig_custom, use_container_width=True)
                metrics_main = calculate_performance_metrics(main_etf_data['Close'])
                metrics_custom = calculate_performance_metrics(custom_series)
                if metrics_main and metrics_custom:
                    comp_df = pd.DataFrame({
                        "指標": ["報酬率(%)", "年化波動率(%)", "最大回撤(%)"],
                        selected_ticker: [f"{metrics_main['total_return']:.2f}", f"{metrics_main['volatility']:.2f}", f"{metrics_main['max_drawdown']:.2f}"],
                        custom_symbol: [f"{metrics_custom['total_return']:.2f}", f"{metrics_custom['volatility']:.2f}", f"{metrics_custom['max_drawdown']:.2f}"]
                    })
                    st.dataframe(comp_df, use_container_width=True)

# --- 輔助回歸區塊 (摺疊，不影響主流程) ---
with st.expander("🔍 輔助：顯示回歸預測價格 (點擊展開)"):
    st.markdown("此為輔助功能，使用 **Random Forest 迴歸** 預測明日收盤價，不影響主要分類評估。")
    if st.button("📊 訓練回歸模型並預測明日價格"):
        with st.spinner("訓練回歸模型中..."):
            reg_result = train_regression_model(raw_main_data, "Random Forest Regressor")
        st.success(f"📈 根據回歸模型，**{selected_ticker} 明日預測收盤價：${reg_result['next_day_price']:.2f}**")
        st.caption(f"回歸模型評估：R² = {reg_result['r2']:.3f}, MSE = {reg_result['mse']:.2f}")