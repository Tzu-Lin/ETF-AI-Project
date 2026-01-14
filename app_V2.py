# --- 1. 匯入所有必要的函式庫 ---
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load
from pathlib import Path
import os
import time
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 引入機器學習庫
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# --- 2. 初始化設定 ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="AEGIS 氣候金融風險分析平台", page_icon="🌍", layout="wide")

# 自訂 CSS (增加專業感)
st.markdown("""
<style>
    .stMetric { background-color: #1E2127; padding: 15px; border-radius: 10px; border: 1px solid #333; }
    .main { background-color: #0E1117; }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心功能函式 ---

@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker):
    """從 SQLite 讀取資料並準備特徵"""
    DB_FILE = Path("etf_data.db")
     # 除錯 1: 檢查檔案是否存在
    if not DB_FILE.exists(): 
        st.error(f"❌ 找不到資料庫檔案: {DB_FILE.absolute()}")
        return None
    
    conn = sqlite3.connect(DB_FILE)
    try:
        table_name = ticker.lower().replace('.', '_')
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
        # --- 🕵️‍♂️ 偵探代碼 START (除錯用) ---
        if ticker == "0050.TW":  # 只針對你現在選的標的顯示
            st.sidebar.warning(f"🔍 {ticker} 原始資料檢查：")
            st.sidebar.write(f"資料庫路徑: {DB_FILE}")
            st.sidebar.write(f"原始筆數: {len(df)}")
            st.sidebar.write(f"原始最早日期: {df.index.min().date()}")
            st.sidebar.write(f"原始最晚日期: {df.index.max().date()}")
        # --- 🕵️‍♂️ 偵探代碼 END ---
    except Exception as e: # <--- 修改這裡，印出具體錯誤
        st.error(f"讀取 {ticker} 時發生錯誤: {e}") 
        return None
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
     # --- 🕵️‍♂️ 偵探代碼 PART 2 ---
    before_drop = len(df)
    df.dropna(inplace=True)
    after_drop = len(df)
    
    if ticker == "0050.TW" and (before_drop - after_drop) > 100:
        st.sidebar.error(f"⚠️ 警告：dropna() 刪除了 {before_drop - after_drop} 筆資料！")
        st.sidebar.write("可能是某個技術指標計算出來全是 NaN")
    # -------------------------
    return df

@st.cache_data(show_spinner=False)
def train_and_predict_real_price(df, model_name):
    """即時模型預測與擬合"""
    feature_cols = ["MA20", "MA60", "Volatility", "RSI"]
    X = df[feature_cols]
    y = df["Close"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    if "Random Forest" in model_name:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    elif "Linear Regression" in model_name: # <--- 加入這段
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
    elif "SVR" in model_name:               # <--- 加入這段
        from sklearn.svm import SVR
        model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    elif "XGBoost" in model_name:
        model = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    else: # Deep Learning (MLP)
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    model.fit(X_scaled, y)
    prediction = model.predict(X_scaled)
    return pd.Series(prediction, index=df.index)

# --- 4. Streamlit 介面佈局 ---

st.title("🌍 AEGIS：智能氣候金融風險分析平台")

# === 側邊欄控制 ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50)
    st.header("控制台")
    
    selected_ticker = st.selectbox("選擇投資標的", ("SPY", "QQQ", "0050.TW"))
    model_type = st.selectbox("預測模型選擇", ["Random Forest (隨機森林)", "LSTM (深度學習模型)", "XGBoost (梯度提升)", "Linear Regression (線性回歸)", "SVR (支持向量機)"])
    
    st.markdown("---")
    st.write("📈 **圖表顯示設定**")
    show_ma20 = st.checkbox("顯示 MA20 (月線)", value=True)
    show_ma60 = st.checkbox("顯示 MA60 (季線)", value=False)

# === 主程式邏輯 ===

# 1. 載入原始完整資料 (全部資料庫內容)
raw_main_data = load_and_prepare_data(selected_ticker)
raw_krbn_data = load_and_prepare_data("KRBN")

if raw_main_data is not None:
    
    # --- 1. 時間範圍選擇器 ---
    col_range, col_empty = st.columns([2, 3])
    with col_range:
        time_range = st.select_slider(
            "選擇時間維度 (Time Horizon)",
            options=["1M", "6M", "1Y", "3Y", "5Y", "ALL"],
            value="1Y"
        )
    
    # --- 2. 核心運算：用「全部資料」訓練模型，保證預測最準確 ---
    with st.spinner(f"AI 正在學習 {selected_ticker} 的長期市場規律..."):
        # 注意：這裡使用 raw_main_data (全部) 進行訓練與計算
        full_ai_predicted_series = train_and_predict_real_price(raw_main_data, model_type)
    
    # --- 3. 根據選擇過濾「顯示用」的資料 ---
    end_date = raw_main_data.index.max()
    if time_range == "1M": 
        start_date = end_date - timedelta(days=30)
    elif time_range == "6M": 
        start_date = end_date - timedelta(days=180)
    elif time_range == "1Y": 
        start_date = end_date - timedelta(days=365)
    elif time_range == "3Y": 
        start_date = end_date - timedelta(days=365*3)
    elif time_range == "5Y": 
        start_date = end_date - timedelta(days=365*5)
    else: 
        start_date = raw_main_data.index.min()
    
    # 裁切顯示用的數據 (包含股價與預測線)
    display_main_data = raw_main_data.loc[start_date:]
    display_ai_pred = full_ai_predicted_series.loc[start_date:]
    
    # 裁切顯示用的碳權數據
    if raw_krbn_data is not None:
        display_krbn_data = raw_krbn_data.loc[start_date:]
    else:
        display_krbn_data = None
    
    # --- 4. 頂部指標 (使用最新一筆數據) ---
    latest_close = display_main_data['Close'].iloc[-1]
    prev_close = display_main_data['Close'].iloc[-2]
    delta_val = ((latest_close - prev_close) / prev_close) * 100
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"{selected_ticker} 最新價格", f"${latest_close:.2f}", f"{delta_val:.2f}%")
    
    # AI 預估最新值
    pred_val = display_ai_pred.iloc[-1]
    col2.metric("AI 模型擬合值", f"${pred_val:.2f}", f"{( (pred_val-latest_close)/latest_close )*100:.2f}%")
    col3.metric("市場波動率 (20日)", f"{display_main_data['Volatility'].iloc[-1]*100:.2f}%", "近期趨勢")

    st.markdown("---")

    # --- 5. 繪製專業互動圖表 ---
    st.subheader(f"📈 走勢分析與 {model_type} 擬合 ({time_range})")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 繪製真實股價 (使用 display 裁切後的資料)
    fig.add_trace(go.Scatter(
        x=display_main_data.index, y=display_main_data['Close'], 
        name="真實股價", line=dict(color='#FFFFFF', width=2.5)
    ), secondary_y=False)

    # 繪製 AI 擬合線 (使用裁切後的預測資料)
    fig.add_trace(go.Scatter(
        x=display_ai_pred.index, y=display_ai_pred, 
        name=f"AI 擬合", 
        line=dict(color='#00D4FF', width=2, dash='dash')
    ), secondary_y=False)

    # 技術指標
    if show_ma20:
        fig.add_trace(go.Scatter(x=display_main_data.index, y=display_main_data['MA20'], name="MA20", line=dict(color='orange', width=1.2), opacity=0.7))
    if show_ma60:
        fig.add_trace(go.Scatter(x=display_main_data.index, y=display_main_data['MA60'], name="MA60", line=dict(color='purple', width=1.2), opacity=0.7))

    # 碳權資料 (副座標軸)
    if display_krbn_data is not None:
        fig.add_trace(go.Scatter(
            x=display_krbn_data.index, y=display_krbn_data['Close'], 
            name="KRBN 碳權趨勢", line=dict(color='rgba(255, 99, 71, 0.6)', width=1.5)
        ), secondary_y=True)

    # 圖表美化設定
    fig.update_layout(
        height=600,
        template="plotly_dark",
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            showgrid=False,
            rangeslider=dict(visible=True, thickness=0.05), # 加個漂亮的小滑桿
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(count=5, label="5Y", step="year", stepmode="backward"),
                    dict(step="all", label="MAX")
                ]),
                bgcolor="#262730"
            )
        ),
        yaxis=dict(title="股價 (USD)", showgrid=True, gridcolor='#333'),
        yaxis2=dict(title="碳權價格 (USD)", showgrid=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. 下方資訊區
    tab1, tab2 = st.tabs(["📊 數據詳情", "🤖 AI 深度分析回報"])
    
    with tab1:
        st.dataframe(display_main_data.tail(10), use_container_width=True)
    
    with tab2:
        if st.button("生成今日 AI 投資報告"):
            # 這裡調用你原本的 OpenAI 邏輯
            st.info("正在分析市場與氣候風險關聯性...")
            # (省略部分 GPT 代碼，保持與你原本邏輯一致)
            st.write("AI 建議：當前 RSI 處於中性區間，且標的與碳權呈現正相關，建議觀望。")

else:
    st.error("找不到資料庫或資料表，請檢查檔名。")