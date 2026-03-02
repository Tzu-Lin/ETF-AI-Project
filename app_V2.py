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
from datetime import timedelta
from dotenv import load_dotenv

# 引入機器學習庫
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# --- 2. 初始化設定 ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AEGIS 氣候金融風險分析平台", page_icon="🌍", layout="wide")

# === CSS 樣式設定 (字體縮小版) ===
st.markdown("""
<style>
    /* 1. 調整標題 (Title) 的字體大小 */
    .custom-title {
        font-size: 32px !important;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* 2. 調整 Metric 指標數值 */
    [data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 600;
    }

    /* 3. 調整 Metric 標籤 */
    [data-testid="stMetricLabel"] {
        font-size: 14px !important;
        color: #aaaaaa;
    }
    
    /* 4. Metric 背景優化 */
    .stMetric {
        background-color: #1E2127;
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心功能函式 ---

@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker):
    """從 SQLite 讀取資料並準備特徵"""
    DB_FILE = Path("etf_data.db").resolve()
    if not DB_FILE.exists(): return None
    
    conn = sqlite3.connect(DB_FILE)
    try:
        table_name = ticker.lower().replace('.', '_')
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    except: return None
    finally: conn.close()
    
    # 特徵工程
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

def train_model(df, model_name):
    """訓練單一模型並回傳結果"""
    feature_cols = ["MA20", "MA60", "Volatility", "RSI"]
    X = df[feature_cols]
    y = df["Close"]
    
    # 切分訓練集與測試集
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_full_scaled = scaler.transform(X)
    
    # 模型選擇
    if "Random Forest" in model_name:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "Linear Regression" in model_name:
        model = LinearRegression()
    elif "SVR" in model_name:
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    elif "XGBoost" in model_name:
        model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    else: # Deep Learning (MLP)
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    model.fit(X_train_scaled, y_train)
    
    # 評估
    y_pred_test = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    
    # 全局預測
    full_prediction = model.predict(X_full_scaled)
    
    return {
        "name": model_name,
        "r2": r2,
        "mse": mse,
        "full_series": pd.Series(full_prediction, index=df.index),
        "model": model
    }

def generate_mock_response(ticker, ticker_pred, krbn_pred, correlation):
    """生成模擬的 AI 分析報告 (Demo 用，當 API 沒錢時觸發)"""
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
    """生成氣候金融風險摘要 (含 Demo 模式)"""
    if not openai.api_key:
        return generate_mock_response(ticker, ticker_pred, krbn_pred, correlation)
    
    prompt = f"""
    你是專業操盤手。請根據數據直接給出 **3 點關鍵操作結論**，**嚴禁廢話**，總字數控制在 150 字內：

    [市場數據]
    - 標的 ({ticker}) 預測：{ticker_pred} (信心 {ticker_conf}%)
    - 碳權 (KRBN) 預測：{krbn_pred} (信心 {krbn_conf}%)
    - 兩者相關係數：{correlation:.2f} (正值=同步, 負值=避險)
    - 標的 RSI：{latest_data['RSI']:.0f}

    [輸出格式]
    1. 🎯 **避險判斷**：(一句話判定 KRBN 是否能保護 {ticker})
    2. ⚡ **趨勢訊號**：(解讀兩者方向一致或相反的意義)
    3. 💡 **操作建議**：(直接給出加碼、減碼或觀望建議)
    """
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"OpenAI Error: {e}")
        return generate_mock_response(ticker, ticker_pred, krbn_pred, correlation)

# --- 4. Streamlit 介面佈局 ---

# 使用自訂 CSS 的標題
st.markdown('<p class="custom-title">🌍 AEGIS：基於生成式 AI 之碳權與美股雙軌分析平台</p>', unsafe_allow_html=True)

# === 側邊欄 ===
with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_ticker = st.selectbox("請選擇投資標的 ETF:", ("SPY", "QQQ", "0050.TW"))
    
    st.markdown("---")
    st.write("🤖 **模型設定**")
    main_model_name = st.selectbox("主圖表顯示模型", 
                                   ["Random Forest", "XGBoost", "Linear Regression", "SVR", "Deep Learning (MLP)"])
    
    st.info("系統已自動載入 KRBN (碳權) 數據進行交叉比對。")

# === 主程式邏輯 ===

# 1. 載入資料
raw_main_data = load_and_prepare_data(selected_ticker)
raw_krbn_data = load_and_prepare_data("KRBN")

if raw_main_data is not None:
    
    # --- 2. 訓練主模型 ---
    with st.spinner(f"正在使用 {main_model_name} 運算 {selected_ticker} 數據..."):
        main_result = train_model(raw_main_data, main_model_name)
        display_ai_pred = main_result["full_series"]
        main_r2 = main_result["r2"]

    # --- 3. 訓練 KRBN 模型 (用於判斷趨勢) ---
    if raw_krbn_data is not None:
        krbn_result = train_model(raw_krbn_data, "Random Forest")
        display_krbn_pred = krbn_result["full_series"]
        krbn_r2 = krbn_result["r2"]
    else:
        display_krbn_pred = None
        krbn_r2 = 0

    # --- 4. 計算相關係數 ---
    correlation = 0.0
    corr_desc = "資料不足"
    corr_color = "gray"
    
    if raw_krbn_data is not None:
        common_idx = raw_main_data.index.intersection(raw_krbn_data.index)
        if len(common_idx) > 30:
            correlation = raw_main_data.loc[common_idx, 'Close'].corr(raw_krbn_data.loc[common_idx, 'Close'])
            
            if correlation > 0.5:
                corr_desc = "高度正相關 (同步波動)"
                corr_color = "#ff4b4b" # 紅色
            elif correlation < -0.3:
                corr_desc = "負相關 (具避險效果)"
                corr_color = "#09ab3b" # 綠色
            else:
                corr_desc = "低度相關 (走勢脫鉤)"
                corr_color = "gray"

    # --- 5. 頂部儀表板 (數值顯示區) ---
    latest_close = raw_main_data['Close'].iloc[-1]
    pred_close = display_ai_pred.iloc[-1]
    
    main_trend = "看漲 📈" if pred_close > latest_close else "看跌 📉"
    krbn_trend = "看漲 📈" if (display_krbn_pred is not None and display_krbn_pred.iloc[-1] > raw_krbn_data['Close'].iloc[-1]) else "看跌 📉"
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(f"{selected_ticker} 最新價", f"${latest_close:.2f}")
    col2.metric("AI 預測趨勢", main_trend, f"R²: {main_r2:.2f}")
    col3.metric("KRBN 碳權趨勢", krbn_trend)
    
    # 顯示相關係數 (HTML)
    col4.markdown(f"""
    <div style="text-align: center;">
        <p style="margin: 0px; font-size: 14px; color: #aaaaaa;">與碳權相關性</p>
        <p style="margin: 0px; font-size: 26px; color: {corr_color}; font-weight: 600;">{correlation:.2f}</p>
        <p style="margin: 0px; font-size: 12px; color: #888;">{corr_desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # --- 6. 時間範圍選擇器 ---
    st.write("###")
    col_time, _ = st.columns([2, 1])
    with col_time:
        time_range = st.select_slider(
            "⏳ 選擇圖表顯示的時間範圍",
            options=["1M", "6M", "1Y", "3Y", "5Y", "ALL"],
            value="1Y"
        )

    # 計算顯示範圍
    end_date = raw_main_data.index.max()
    if time_range == "1M": start_date = end_date - timedelta(days=30)
    elif time_range == "6M": start_date = end_date - timedelta(days=180)
    elif time_range == "1Y": start_date = end_date - timedelta(days=365)
    elif time_range == "3Y": start_date = end_date - timedelta(days=365*3)
    elif time_range == "5Y": start_date = end_date - timedelta(days=365*5)
    else: start_date = raw_main_data.index.min()

    st.markdown("---")

    # --- 7. 分頁功能 (新增了「數據詳情」分頁) ---
    tab_chart, tab_arena, tab_ai, tab_data = st.tabs(["📈 雙軌趨勢圖", "🏆 模型競技場 (Model Arena)", "🤖 AI 決策報告", "📊 數據詳情"])

    # === TAB 1: 雙軌圖表 ===
    with tab_chart:
        st.subheader(f"{selected_ticker} vs KRBN 走勢對照 ({time_range})")
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 根據時間軸裁切資料
        plot_data = raw_main_data.loc[start_date:]
        plot_pred = display_ai_pred.loc[start_date:]
        
        # 主標的
        fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Close'], name=f"{selected_ticker} 真實股價", line=dict(color='white', width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=plot_pred.index, y=plot_pred, name=f"AI 擬合 ({main_model_name})", line=dict(color='#00D4FF', dash='dash')), secondary_y=False)
        
        # 碳權 (KRBN)
        if raw_krbn_data is not None:
            krbn_plot = raw_krbn_data.loc[start_date:]
            fig.add_trace(go.Scatter(x=krbn_plot.index, y=krbn_plot['Close'], name="KRBN 碳權", line=dict(color='orange', width=1.5, dash='dot')), secondary_y=True)

        fig.update_layout(height=500, template="plotly_dark", hovermode="x unified")
        fig.update_yaxes(title_text="股價 (USD)", secondary_y=False)
        fig.update_yaxes(title_text="碳權 (USD)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 2: 模型競技場 ===
    with tab_arena:
        st.subheader("🏆 模型效能大亂鬥")
        st.write("同時訓練 5 種 AI 模型，比較準確度 (R² Score) 與誤差 (MSE)。")
        
        if st.button("🚀 開始模型競賽"):
            models_to_test = ["Random Forest", "XGBoost", "Linear Regression", "SVR", "Deep Learning (MLP)"]
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, m_name in enumerate(models_to_test):
                status_text.text(f"正在訓練 {m_name}...")
                res = train_model(raw_main_data, m_name)
                results.append({
                    "模型名稱": m_name,
                    "R2 Score (準確度)": res['r2'],
                    "MSE (誤差值)": res['mse']
                })
                progress_bar.progress((i + 1) / len(models_to_test))
            
            status_text.text("競賽結束！")
            
            res_df = pd.DataFrame(results).sort_values(by="R2 Score (準確度)", ascending=False)
            best_model = res_df.iloc[0]
            
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.success(f"👑 冠軍模型：\n\n**{best_model['模型名稱']}**")
                st.metric("最高準確度", f"{best_model['R2 Score (準確度)']:.4f}")
            
            with col_b:
                st.dataframe(res_df.style.highlight_max(axis=0, subset=["R2 Score (準確度)"]), use_container_width=True)
                
            st.bar_chart(res_df.set_index("模型名稱")["R2 Score (準確度)"])

    # === TAB 3: AI 報告 ===
    with tab_ai:
        st.subheader("🤖 智能風險評估")
        
        main_conf_str = f"{main_r2*100:.1f}"
        krbn_conf_str = f"{krbn_r2*100:.1f}"
        
        if st.button("生成 AI 分析報告"):
            with st.spinner("AI 正在分析市場數據..."):
                summary = get_climate_gpt_summary(
                    ticker=selected_ticker,
                    ticker_pred=main_trend,
                    ticker_conf=main_conf_str,
                    krbn_pred=krbn_trend,
                    krbn_conf=krbn_conf_str,
                    latest_data=raw_main_data.iloc[-1],
                    correlation=correlation
                )
                st.success("分析完成")
                st.info(summary)

    # === TAB 4: 數據詳情 (新增) ===
    with tab_data:
        st.subheader(f"📊 {selected_ticker} 詳細交易數據")
        st.write("以下顯示最近 20 筆交易日資料，包含技術指標數值：")
        # 顯示最後 20 筆資料，並按日期倒序排列 (最新的在上面)
        st.dataframe(raw_main_data.tail(20).sort_index(ascending=False), use_container_width=True)

else:
    st.error("❌ 無法讀取資料庫。請確認 etf_data.db 是否存在且有資料。")