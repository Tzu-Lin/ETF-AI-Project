# --- 1. 匯入所有必要的函式庫 ---
import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from joblib import load
from pathlib import Path
import os
import openai
from dotenv import load_dotenv

# --- 2. 初始化設定 ---

# 載入 .env 檔案中的環境變數
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 設定 Streamlit 頁面
st.set_page_config(page_title="ETF AI 分析平台", page_icon="📈", layout="wide")

# --- 3. 核心功能函式 (帶有快取功能) ---
@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker):
    """從 SQLite 讀取資料並準備特徵"""
    DB_FILE = Path("etf_data.db")
    if not DB_FILE.exists():
        st.error("錯誤：找不到資料庫 etf_data.db。請先執行 update_database.py！")
        
        return None
    conn = sqlite3.connect(DB_FILE)
    try:
        # 1. 對傳入的 ticker (例如 "0050.TW") 進行同樣的名稱轉換
        table_name = ticker.lower().replace('.', '_')

        # 2. 使用轉換後安全的名稱來查詢表格
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    finally:
        conn.close()
    
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

@st.cache_resource
def load_model(ticker):
    """載入對應的模型"""
    model_path = Path("models") / f"rf_{ticker}.joblib"
    if model_path.exists():
        return load(model_path)['model']
    return None

def get_gpt_summary(ticker, latest_data, prediction, probability):
    """生成 GPT 自然語言摘要"""
    if not openai.api_key:
        return "錯誤：OpenAI API Key 未設定。請在 .env 檔案中設定。"
    
    pred_text = "上漲" if prediction == 1 else "下跌"
    prompt = f"""
    您是一位專業的金融數據分析師。請根據以下 ETF 最新數據和 AI 模型預測結果，生成一段客觀、中立的市場摘要分析。
    請不要提供任何投資建議，並在結尾加上免責聲明。

    - ETF 代碼: {ticker}
    - 20日均線: {latest_data['MA20']:.2f}
    - 60日均線: {latest_data['MA60']:.2f}
    - RSI 指標: {latest_data['RSI']:.2f}
    - 機器學習模型預測次日方向: {pred_text} (信心度: {probability*100:.1f}%)

    請基於以上數據，撰寫一段約 100-150 字的摘要。
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 摘要生成失敗：{e}"

# --- 4. Streamlit 介面佈局 ---

st.title("📈 ETF AI 分析與風險視覺化平台")

with st.sidebar:
    st.header("⚙️ 控制面板")
    selected_ticker = st.selectbox("請選擇要分析的 ETF:", ("SPY", "QQQ", "SSO", "QLD", "0050_TW"))

data = load_and_prepare_data(selected_ticker)
model = load_model(selected_ticker)

if data is not None and model is not None:
    # 執行預測
    features = ["MA20", "MA60", "Volatility", "RSI"]
    latest_features = data[features].iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    probability = model.predict_proba(latest_features)[0]
    confidence = probability[prediction]

    # 顯示預測結果指標卡
    st.header(f"{selected_ticker} AI 趨勢預測")
    col1, col2 = st.columns(2)
    with col1:
        if prediction == 1:
            st.metric(label="AI 預測次日趨勢", value="看漲 ▲", delta="趨勢向上")
        else:
            st.metric(label="AI 預測次日趨勢", value="看跌 ▼", delta="趨勢向下", delta_color="inverse")
    with col2:
        st.metric(label="模型信心度", value=f"{confidence:.2%}")

    st.markdown("---")

    # 顯示互動式圖表
    st.subheader("歷史價格與技術指標")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='收盤價'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA20'], mode='lines', name='20日均線', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA60'], mode='lines', name='60日均線', line=dict(dash='dot')))
    fig.update_layout(title=f'{selected_ticker} 價格走勢', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 顯示 GPT 摘要
    st.subheader("🤖 AI 生成摘要說明")
    with st.spinner("AI 正在分析數據並生成摘要..."):
        summary = get_gpt_summary(selected_ticker, data.iloc[-1], prediction, confidence)
        st.info(summary)
else:
    st.warning("無法載入資料或模型，請檢查設定與檔案。")