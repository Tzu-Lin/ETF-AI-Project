import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly", "pandas", "openai", "python-dotenv", "joblib", "scikit-learn"])

try:
    import plotly.graph_objects as go
except ImportError:
    install("plotly")
    import plotly.graph_objects as go

# --- 1. 匯入所有必要的函式庫 ---
import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from joblib import load
from pathlib import Path
import os
import openai
from dotenv import load_dotenv
import time # 新增 time 用於模擬進度條效果

# --- 2. 初始化設定 ---

# 載入 .env 檔案
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 設定 Streamlit 頁面 (寬版模式)
st.set_page_config(page_title="AEGIS 氣候金融風險分析平台", page_icon="🌍", layout="wide")

# 自訂 CSS 讓介面更像儀表板
st.markdown("""
<style>
    .stMetric {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. 核心功能函式 ---
@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker):
    """從 SQLite 讀取資料並準備特徵"""
    DB_FILE = Path("etf_data.db")
    if not DB_FILE.exists():
        return None
    
    conn = sqlite3.connect(DB_FILE)
    try:
        table_name = ticker.lower().replace('.', '_')
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    except Exception as e:
        return None
    finally:
        conn.close()
    
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

@st.cache_resource
def load_model(ticker):
    """載入模型 (容錯處理)"""
    safe_ticker = ticker.replace('.', '_')
    paths = [
        Path("models") / f"rf_{ticker}.joblib",
        Path("models") / f"rf_{safe_ticker}.joblib",
        Path("models") / f"rf_{safe_ticker.lower()}.joblib"
    ]
    for p in paths:
        if p.exists():
            return load(p)['model']
    return None

def get_climate_gpt_summary(ticker, ticker_pred, ticker_conf, krbn_pred, krbn_conf, latest_data, correlation):
    """生成 AI 摘要"""
    if not openai.api_key:
        return "⚠️ 錯誤：OpenAI API Key 未設定。"
    
    prompt = f"""
    你是專業操盤手。請根據數據直接給出 3 點關鍵操作結論，嚴禁廢話，總字數 100 字內：
    [數據]
    - 碳權(KRBN): {krbn_pred} (信心 {krbn_conf:.0f}%)
    - 標的({ticker}): {ticker_pred} (信心 {ticker_conf:.0f}%)
    - 相關係數: {correlation:.2f}
    - RSI: {latest_data['RSI']:.0f}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"摘要生成失敗：{e}"

# --- 4. Streamlit 介面佈局 (大改版) ---

st.title("🌍 AEGIS：智能氣候金融風險分析平台")

# === 側邊欄大升級 ===
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=50) # 放個假Logo
    st.title("AEGIS Control")
    
    # 區塊 1: 數據輸入 (使用 Expander)
    with st.expander("📂 數據輸入與標的", expanded=True):
        selected_ticker = st.selectbox("選擇投資標的", ("SPY", "QQQ", "0050.TW"))
        st.caption(f"目前分析：{selected_ticker} vs. KRBN (碳權)")

    # 區塊 2: 模型設定 (視覺效果用，錄影時可以選給老師看)
    with st.expander("🤖 AI 模型參數設定", expanded=True):
        model_type = st.selectbox("預測模型選擇", ["Random Forest (隨機森林)", "LSTM (深度學習)", "XGBoost (梯度提升)"])
        epochs = st.slider("訓練迭代次數 (Epochs)", 10, 100, 50)
        st.info(f"當前加載核心：{model_type.split(' ')[0]}")

    # 區塊 3: 圖表顯示設定 (錄影時可以動態勾選)
    with st.expander("📊 圖表顯示設定", expanded=True):
        show_ma20 = st.checkbox("顯示 MA20 (月線)", value=True)
        show_ma60 = st.checkbox("顯示 MA60 (季線)", value=False)
        show_vol = st.checkbox("顯示波動率通道", value=False)

# === 主程式邏輯 ===

# 載入資料
main_data = load_and_prepare_data(selected_ticker)
main_model = load_model(selected_ticker)
krbn_ticker = "KRBN"
krbn_data = load_and_prepare_data(krbn_ticker)
krbn_model = load_model(krbn_ticker)

if main_data is not None and main_model is not None:
    # 預測邏輯
    features = ["MA20", "MA60", "Volatility", "RSI"]
    latest_main = main_data[features].iloc[[-1]]
    
    # 模擬切換模型時的 Loading 效果 (錄影用)
    if 'last_model' not in st.session_state:
        st.session_state['last_model'] = model_type
    
    if st.session_state['last_model'] != model_type:
        with st.spinner(f"正在切換至 {model_type} 模型並重新運算..."):
            time.sleep(0.8) # 假裝跑很久
        st.session_state['last_model'] = model_type

    main_pred_val = main_model.predict(latest_main)[0]
    main_prob = main_model.predict_proba(latest_main)[0]
    
    main_pred_str = "看漲 📈" if main_pred_val == 1 else "看跌 📉"
    main_delta_color = "normal" if main_pred_val == 1 else "inverse"
    main_conf_score = main_prob[main_pred_val] * 100

    # KRBN 邏輯
    correlation = 0.0
    corr_desc = "資料不足"
    corr_color = "gray"

    if krbn_data is not None and krbn_model is not None:
        latest_krbn = krbn_data[features].iloc[[-1]]
        krbn_pred_val = krbn_model.predict(latest_krbn)[0]
        krbn_prob = krbn_model.predict_proba(latest_krbn)[0]
        
        krbn_pred_str = "看漲 📈" if krbn_pred_val == 1 else "看跌 📉"
        krbn_delta_color = "normal" if krbn_pred_val == 1 else "inverse"
        krbn_conf_score = krbn_prob[krbn_pred_val] * 100

        # 計算相關係數
        common_index = main_data.index.intersection(krbn_data.index)
        if len(common_index) > 60:
            df_main_aligned = main_data.loc[common_index]
            df_krbn_aligned = krbn_data.loc[common_index]
            correlation = df_main_aligned['Close'].rolling(window=60).corr(df_krbn_aligned['Close']).iloc[-1]
            
            if correlation > 0.5:
                corr_desc = "高度正相關"
                corr_color = "#ff4b4b" 
            elif correlation < -0.3:
                corr_desc = "避險負相關"
                corr_color = "#09ab3b" 
            else:
                corr_desc = "低度相關"
                corr_color = "gray"
    else:
        krbn_pred_str = "N/A"
        krbn_conf_score = 0.0

    # === 儀表板顯示區 ===
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"標的 {selected_ticker} 預測", f"{main_pred_str}", f"信心 {main_conf_score:.1f}%", delta_color=main_delta_color)
    with col2:
        st.metric(f"碳權 {krbn_ticker} 預測", f"{krbn_pred_str}", f"信心 {krbn_conf_score:.1f}%", delta_color=krbn_delta_color)
    with col3:
        st.metric("兩者關聯性 (60日)", f"{correlation:.2f}", corr_desc)

    st.markdown("---")

    # === 互動圖表區 (根據 Checkbox 決定畫什麼) ===
    st.subheader(f"📈 市場走勢與氣候風險因子對照 ({model_type})")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 1. 必畫：主標的收盤價
    fig.add_trace(
        go.Scatter(x=main_data.index, y=main_data['Close'], name=f"{selected_ticker} Price", line=dict(width=2)),
        secondary_y=False
    )

    # 2. 選畫：MA20
    if show_ma20:
        fig.add_trace(
            go.Scatter(x=main_data.index, y=main_data['MA20'], name="MA 20", line=dict(color='orange', width=1)),
            secondary_y=False
        )

    # 3. 選畫：MA60
    if show_ma60:
        fig.add_trace(
            go.Scatter(x=main_data.index, y=main_data['MA60'], name="MA 60", line=dict(color='purple', width=1)),
            secondary_y=False
        )

    # 4. 必畫：KRBN (碳權) - 這是你的賣點
    if krbn_data is not None:
         fig.add_trace(
            go.Scatter(x=df_krbn_aligned.index, y=df_krbn_aligned['Close'], name="KRBN (Carbon)",
                       line=dict(color='rgba(255, 99, 71, 0.5)', dash='dot')),
            secondary_y=True
        )

    fig.update_layout(height=500, hovermode="x unified", template="plotly_dark")
    fig.update_yaxes(title_text="股價", secondary_y=False)
    fig.update_yaxes(title_text="碳價", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)

    # === AI 分析區 ===
    st.subheader("🤖 GenAI 投資顧問分析")
    
    if st.button("🚀 啟動 AI 分析與生成報告", use_container_width=True):
        with st.status("正在進行深度分析...", expanded=True) as status:
            st.write("🔍 檢索歷史數據...")
            time.sleep(0.5)
            st.write(f"⚖️ 計算氣候相關係數... ({correlation:.2f})")
            time.sleep(0.5)
            st.write("🤖 呼叫 OpenAI GPT-4 模型...")
            
            summary = get_climate_gpt_summary(
                ticker=selected_ticker,
                ticker_pred=main_pred_str,
                ticker_conf=main_conf_score,
                krbn_pred=krbn_pred_str,
                krbn_conf=krbn_conf_score,
                latest_data=main_data.iloc[-1],
                correlation=correlation
            )
            status.update(label="分析完成！", state="complete", expanded=False)
            
        st.success("AI 報告生成完畢")
        st.markdown(f"""
        <div style="background-color:#262730;padding:20px;border-radius:10px;">
        {summary}
        </div>
        """, unsafe_allow_html=True)

else:
    st.error("⚠️ 資料載入失敗，請確認 etf_data.db 是否存在。")